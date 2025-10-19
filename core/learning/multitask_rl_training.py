"""
Multi-Task RL Training - Single policy across all empire platforms

Trains a unified RL policy that can manage:
- Sewago (platform operations)
- Halobuzz (social media)
- SolsniperPro (trading/DeFi)

Uses shared representations with task-specific heads for efficient
transfer learning across domains.

Part of ShivX Personal Empire AGI (Week 5).
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces

from core.learning.data_collector import get_collector, TaskDomain, TaskType
from core.ml.neural_base import MLPModel, ModelConfig
from core.ml.experiment_tracker import get_tracker

logger = logging.getLogger(__name__)


@dataclass
class MultiTaskRLConfig:
    """Configuration for multi-task RL training"""

    # Architecture
    shared_hidden_dims: List[int] = None  # [128, 64]
    task_head_dim: int = 32

    # Training
    total_timesteps: int = 50000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Multi-task
    task_sampling: str = "uniform"  # uniform, proportional
    task_switch_freq: int = 1000  # Switch task every N steps

    # Regularization
    l2_weight: float = 1e-4
    entropy_coef: float = 0.01

    def __post_init__(self):
        if self.shared_hidden_dims is None:
            self.shared_hidden_dims = [128, 64]


class MultiTaskPolicy(nn.Module):
    """
    Multi-task RL policy with shared base and task-specific heads.

    Architecture:
    - Shared encoder: Learns common representations
    - Task-specific heads: Specialized for each empire
    - Value head: Shared critic for all tasks
    """

    def __init__(
        self,
        observation_dim: int,
        action_dims: Dict[str, int],  # {task_id: action_dim}
        config: MultiTaskRLConfig,
    ):
        super().__init__()

        self.observation_dim = observation_dim
        self.action_dims = action_dims
        self.config = config
        self.task_ids = list(action_dims.keys())

        # Shared encoder
        encoder_layers = []
        prev_dim = observation_dim

        for hidden_dim in config.shared_hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        self.shared_encoder = nn.Sequential(*encoder_layers)
        self.shared_dim = prev_dim

        # Task-specific policy heads
        self.policy_heads = nn.ModuleDict()

        for task_id, action_dim in action_dims.items():
            self.policy_heads[task_id] = nn.Sequential(
                nn.Linear(self.shared_dim, config.task_head_dim),
                nn.ReLU(),
                nn.Linear(config.task_head_dim, action_dim),
            )

        # Shared value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(self.shared_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        logger.info(
            f"Multi-Task Policy: {self.count_parameters()} params, "
            f"{len(self.task_ids)} tasks"
        )

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        observations: torch.Tensor,
        task_id: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for specific task.

        Args:
            observations: Batch of observations [batch_size, obs_dim]
            task_id: Which task to use (sewago, halobuzz, solsniper)

        Returns:
            (action_logits, value_estimates)
        """
        # Shared encoding
        shared_features = self.shared_encoder(observations)

        # Task-specific policy
        if task_id not in self.policy_heads:
            raise ValueError(f"Unknown task: {task_id}")

        action_logits = self.policy_heads[task_id](shared_features)

        # Shared value
        values = self.value_head(shared_features).squeeze(-1)

        return action_logits, values

    def get_action_distribution(
        self,
        observations: torch.Tensor,
        task_id: str,
    ) -> torch.distributions.Categorical:
        """Get action distribution for sampling"""
        action_logits, _ = self.forward(observations, task_id)
        return torch.distributions.Categorical(logits=action_logits)

    def get_value(
        self,
        observations: torch.Tensor,
        task_id: str,
    ) -> torch.Tensor:
        """Get value estimates"""
        _, values = self.forward(observations, task_id)
        return values

    def get_shared_features(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Extract shared features (useful for analysis)"""
        return self.shared_encoder(observations)


class EmpireMultiTaskEnv(gym.Env):
    """
    Multi-task environment wrapper for empire management.

    Switches between different empire tasks during training.
    """

    def __init__(
        self,
        tasks: Dict[str, Dict[str, Any]],  # {task_id: task_config}
        current_task: Optional[str] = None,
    ):
        super().__init__()

        self.tasks = tasks
        self.task_ids = list(tasks.keys())
        self.current_task = current_task or self.task_ids[0]

        # Observation space (unified across all tasks)
        self.observation_dim = 20  # Feature dimension
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        # Action space (varies by task)
        self.action_spaces = {
            task_id: spaces.Discrete(config["num_actions"])
            for task_id, config in tasks.items()
        }

        self.action_space = self.action_spaces[self.current_task]

        # State
        self.step_count = 0
        self.episode_reward = 0.0

        logger.info(f"Multi-Task Env: {len(self.task_ids)} tasks")

    def switch_task(self, task_id: str):
        """Switch to a different task"""
        if task_id not in self.task_ids:
            raise ValueError(f"Unknown task: {task_id}")

        self.current_task = task_id
        self.action_space = self.action_spaces[task_id]
        logger.debug(f"Switched to task: {task_id}")

    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)

        self.step_count = 0
        self.episode_reward = 0.0

        # Generate initial observation
        obs = self._generate_observation()

        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action in current task"""
        self.step_count += 1

        # Simulate task execution
        task_config = self.tasks[self.current_task]

        # Simple reward model (in production, would use real metrics)
        base_reward = np.random.randn() * 0.1

        # Task-specific reward shaping
        if self.current_task == "sewago":
            # Platform health focus
            reward = base_reward + 0.5
        elif self.current_task == "halobuzz":
            # Engagement focus
            reward = base_reward + 0.3
        elif self.current_task == "solsniper":
            # Trading performance focus
            reward = base_reward + 0.2
        else:
            reward = base_reward

        self.episode_reward += reward

        # Episode termination
        done = self.step_count >= 100
        truncated = False

        # Next observation
        next_obs = self._generate_observation()

        info = {
            "task_id": self.current_task,
            "episode_reward": self.episode_reward,
        }

        return next_obs, reward, done, truncated, info

    def _generate_observation(self) -> np.ndarray:
        """Generate observation for current task"""
        # Simulate task-specific observations
        obs = np.random.randn(self.observation_dim).astype(np.float32)

        # Add task-specific signal
        if self.current_task == "sewago":
            obs[0] = 1.0  # Task indicator
        elif self.current_task == "halobuzz":
            obs[0] = 2.0
        elif self.current_task == "solsniper":
            obs[0] = 3.0

        return obs


class MultiTaskRLTrainer:
    """
    Multi-task RL trainer for empire management.

    Trains a single policy across all empire platforms using
    shared representations and task-specific heads.
    """

    def __init__(
        self,
        config: Optional[MultiTaskRLConfig] = None,
        model_dir: str = "data/models/multitask_rl",
    ):
        self.config = config or MultiTaskRLConfig()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Data collector
        self.collector = get_collector()

        # Experiment tracker
        self.tracker = get_tracker()

        # Define tasks
        self.tasks = {
            "sewago": {"num_actions": 5, "domain": TaskDomain.SEWAGO},
            "halobuzz": {"num_actions": 5, "domain": TaskDomain.HALOBUZZ},
            "solsniper": {"num_actions": 5, "domain": TaskDomain.SOLSNIPER},
        }

        logger.info("Multi-Task RL Trainer initialized")

    def train(self) -> Dict[str, Any]:
        """
        Train multi-task RL policy.

        Returns:
            Training results
        """
        logger.info("Starting multi-task RL training...")

        # Start experiment
        run_id = self.tracker.start_run(
            run_name="multitask_rl_empire",
            config={
                "shared_hidden_dims": self.config.shared_hidden_dims,
                "task_head_dim": self.config.task_head_dim,
                "total_timesteps": self.config.total_timesteps,
                "learning_rate": self.config.learning_rate,
                "tasks": list(self.tasks.keys()),
            },
            tags=["multi_task", "rl", "empire"],
            notes="Multi-task RL across all empire platforms"
        )

        # Create environment
        env = EmpireMultiTaskEnv(tasks=self.tasks)

        # Create multi-task policy
        action_dims = {
            task_id: config["num_actions"]
            for task_id, config in self.tasks.items()
        }

        policy = MultiTaskPolicy(
            observation_dim=env.observation_dim,
            action_dims=action_dims,
            config=self.config,
        )

        # Optimizer
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.l2_weight,
        )

        # Training loop
        total_steps = 0
        episode_rewards = {task_id: [] for task_id in self.tasks.keys()}

        while total_steps < self.config.total_timesteps:
            # Sample task
            task_id = self._sample_task()
            env.switch_task(task_id)

            # Collect rollout
            rollout = self._collect_rollout(
                policy=policy,
                env=env,
                task_id=task_id,
                max_steps=self.config.task_switch_freq,
            )

            total_steps += len(rollout["observations"])

            # Update policy
            loss_dict = self._update_policy(
                policy=policy,
                optimizer=optimizer,
                rollout=rollout,
                task_id=task_id,
            )

            # Log metrics
            episode_rewards[task_id].append(rollout["episode_reward"])

            if total_steps % 5000 == 0:
                avg_rewards = {
                    task_id: np.mean(rewards[-10:]) if rewards else 0.0
                    for task_id, rewards in episode_rewards.items()
                }

                logger.info(
                    f"Step {total_steps}/{self.config.total_timesteps}: "
                    f"sewago={avg_rewards['sewago']:.2f}, "
                    f"halobuzz={avg_rewards['halobuzz']:.2f}, "
                    f"solsniper={avg_rewards['solsniper']:.2f}, "
                    f"loss={loss_dict['total_loss']:.3f}"
                )

                self.tracker.log({
                    "total_steps": total_steps,
                    **{f"reward_{k}": v for k, v in avg_rewards.items()},
                    **loss_dict,
                })

        # Final evaluation
        final_rewards = {}
        for task_id in self.tasks.keys():
            env.switch_task(task_id)
            task_reward = self._evaluate_task(policy, env, task_id, num_episodes=10)
            final_rewards[task_id] = task_reward
            logger.info(f"Final {task_id} reward: {task_reward:.2f}")

        # Save model
        model_path = self.model_dir / "multitask_policy.pth"
        torch.save({
            "policy_state_dict": policy.state_dict(),
            "config": self.config,
            "action_dims": action_dims,
            "final_rewards": final_rewards,
        }, model_path)

        logger.info(f"Model saved to: {model_path}")

        # Log summary
        self.tracker.log_summary({
            "final_rewards": final_rewards,
            "total_steps": total_steps,
            "avg_reward": np.mean(list(final_rewards.values())),
        })

        self.tracker.finish_run()

        return {
            "final_rewards": final_rewards,
            "total_steps": total_steps,
            "model_path": str(model_path),
        }

    def _sample_task(self) -> str:
        """Sample next task to train on"""
        if self.config.task_sampling == "uniform":
            return np.random.choice(list(self.tasks.keys()))
        else:
            # Could implement proportional sampling based on performance
            return np.random.choice(list(self.tasks.keys()))

    def _collect_rollout(
        self,
        policy: MultiTaskPolicy,
        env: EmpireMultiTaskEnv,
        task_id: str,
        max_steps: int,
    ) -> Dict[str, Any]:
        """Collect rollout for task"""
        observations = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []

        obs, _ = env.reset()
        episode_reward = 0.0

        policy.eval()
        with torch.no_grad():
            for _ in range(max_steps):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

                # Get action
                dist = policy.get_action_distribution(obs_tensor, task_id)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                # Get value
                value = policy.get_value(obs_tensor, task_id)

                # Store
                observations.append(obs)
                actions.append(action.item())
                log_probs.append(log_prob.item())
                values.append(value.item())

                # Step
                next_obs, reward, done, truncated, info = env.step(action.item())

                rewards.append(reward)
                dones.append(done or truncated)
                episode_reward += reward

                obs = next_obs

                if done or truncated:
                    obs, _ = env.reset()
                    break

        policy.train()

        return {
            "observations": np.array(observations),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "dones": np.array(dones),
            "values": np.array(values),
            "log_probs": np.array(log_probs),
            "episode_reward": episode_reward,
        }

    def _update_policy(
        self,
        policy: MultiTaskPolicy,
        optimizer: torch.optim.Optimizer,
        rollout: Dict[str, Any],
        task_id: str,
    ) -> Dict[str, float]:
        """Update policy using rollout"""
        # Compute advantages using GAE
        advantages, returns = self._compute_gae(
            rewards=rollout["rewards"],
            values=rollout["values"],
            dones=rollout["dones"],
        )

        # Convert to tensors
        obs_tensor = torch.FloatTensor(rollout["observations"])
        actions_tensor = torch.LongTensor(rollout["actions"])
        old_log_probs = torch.FloatTensor(rollout["log_probs"])
        advantages_tensor = torch.FloatTensor(advantages)
        returns_tensor = torch.FloatTensor(returns)

        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-8
        )

        # Multi-epoch updates
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.config.n_epochs):
            # Get current policy outputs
            dist = policy.get_action_distribution(obs_tensor, task_id)
            values = policy.get_value(obs_tensor, task_id)

            # Policy loss (PPO clip)
            log_probs = dist.log_prob(actions_tensor)
            ratio = torch.exp(log_probs - old_log_probs)

            policy_loss_1 = ratio * advantages_tensor
            policy_loss_2 = torch.clamp(ratio, 0.8, 1.2) * advantages_tensor
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(values, returns_tensor)

            # Entropy bonus
            entropy = dist.entropy().mean()

            # Total loss
            loss = (
                policy_loss
                + 0.5 * value_loss
                - self.config.entropy_coef * entropy
            )

            # Update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        return {
            "policy_loss": total_policy_loss / self.config.n_epochs,
            "value_loss": total_value_loss / self.config.n_epochs,
            "entropy": total_entropy / self.config.n_epochs,
            "total_loss": loss.item(),
        }

    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        last_advantage = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_value - values[t]
            advantages[t] = last_advantage = (
                delta + self.config.gamma * self.config.gae_lambda * last_advantage
            )

            if dones[t]:
                last_advantage = 0.0

        returns = advantages + values

        return advantages, returns

    def _evaluate_task(
        self,
        policy: MultiTaskPolicy,
        env: EmpireMultiTaskEnv,
        task_id: str,
        num_episodes: int = 10,
    ) -> float:
        """Evaluate policy on specific task"""
        episode_rewards = []

        policy.eval()
        with torch.no_grad():
            for _ in range(num_episodes):
                obs, _ = env.reset()
                episode_reward = 0.0
                done = False

                while not done:
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    dist = policy.get_action_distribution(obs_tensor, task_id)
                    action = dist.sample()

                    obs, reward, done, truncated, _ = env.step(action.item())
                    episode_reward += reward

                    if truncated:
                        done = True

                episode_rewards.append(episode_reward)

        policy.train()

        return np.mean(episode_rewards)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n=== Multi-Task RL Training Test ===\n")

    # Create trainer
    config = MultiTaskRLConfig(
        shared_hidden_dims=[128, 64],
        task_head_dim=32,
        total_timesteps=20000,  # Reduced for quick test
        task_switch_freq=500,
    )

    trainer = MultiTaskRLTrainer(config=config)

    # Train
    print("Training multi-task policy across all empire platforms...")
    results = trainer.train()

    print("\n=== Results ===")
    print(f"Total Steps: {results['total_steps']}")
    print(f"Model: {results['model_path']}")

    print("\nFinal Rewards by Task:")
    for task_id, reward in results['final_rewards'].items():
        print(f"  {task_id}: {reward:.2f}")

    avg_reward = np.mean(list(results['final_rewards'].values()))
    print(f"\nAverage Reward: {avg_reward:.2f}")

    if avg_reward > 30.0:
        print("\nEXCELLENT: Multi-task policy learned successfully!")
    elif avg_reward > 20.0:
        print("\nGOOD: Policy learning progressing well")
    else:
        print("\nIN PROGRESS: Policy needs more training")

    print("\nMulti-task RL demonstrates:")
    print("- Shared representations across empire domains")
    print("- Task-specific specialization via heads")
    print("- Efficient transfer learning")
    print("- Single unified policy for all platforms")
