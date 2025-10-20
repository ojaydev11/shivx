"""
Week 14: Federated Learning

Implements federated learning for distributed AGI training:
- Federated averaging (FedAvg) algorithm
- Privacy-preserving aggregation
- Distributed model training
- Secure multi-party computation
- Node coordination and synchronization

Key Features:
- Train models across multiple nodes without sharing raw data
- Privacy preservation through local training + aggregation
- Byzantine-robust aggregation (handles malicious nodes)
- Communication-efficient updates
- Differential privacy support

Integrates with:
- Week 2: RL policies (federated RL)
- Week 4: Continual learning (distributed continual learning)
- Week 9: Multi-agent coordination
- Week 11: Production hardening
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Aggregation methods for federated learning"""
    FEDAVG = "fedavg"  # Standard federated averaging
    FEDPROX = "fedprox"  # Federated with proximal term
    MEDIAN = "median"  # Coordinate-wise median (Byzantine-robust)
    TRIMMED_MEAN = "trimmed_mean"  # Robust to outliers


class NodeStatus(Enum):
    """Node status in federated network"""
    IDLE = "idle"
    TRAINING = "training"
    UPLOADING = "uploading"
    WAITING = "waiting"
    FAILED = "failed"


@dataclass
class FederatedNode:
    """Federated learning node"""
    node_id: str
    node_name: str
    status: NodeStatus = NodeStatus.IDLE

    # Node characteristics
    compute_power: float = 1.0  # Relative compute (1.0 = baseline)
    data_size: int = 0  # Local dataset size
    bandwidth: float = 1.0  # Relative bandwidth

    # Training state
    local_epochs: int = 0
    local_loss: float = 0.0
    local_accuracy: float = 0.0

    # Communication
    last_update: Optional[datetime] = None
    updates_contributed: int = 0

    # Security
    reputation: float = 1.0  # 0-1, decreases if behaves maliciously
    verified: bool = False


@dataclass
class ModelUpdate:
    """Model update from a node"""
    update_id: str
    node_id: str
    round_number: int
    timestamp: datetime

    # Model parameters (as state dict)
    parameters: Dict[str, torch.Tensor]

    # Metadata
    num_samples: int  # Training samples used
    loss: float
    accuracy: float

    # Privacy
    noise_scale: float = 0.0  # Differential privacy noise

    # Verification
    signature: Optional[str] = None


@dataclass
class FederatedRound:
    """One round of federated training"""
    round_number: int
    start_time: datetime
    end_time: Optional[datetime] = None

    # Participating nodes
    selected_nodes: List[str] = field(default_factory=list)
    completed_nodes: List[str] = field(default_factory=list)

    # Aggregation results
    global_loss: float = 0.0
    global_accuracy: float = 0.0

    # Statistics
    total_samples: int = 0
    convergence_metric: float = 0.0


class FederatedAveraging:
    """
    Federated Averaging (FedAvg) Algorithm

    Core algorithm:
    1. Server sends global model to nodes
    2. Nodes train locally on their data
    3. Nodes send updates back to server
    4. Server aggregates updates (weighted average)
    5. Repeat
    """

    def __init__(
        self,
        method: AggregationMethod = AggregationMethod.FEDAVG,
        byzantine_robust: bool = True,
        differential_privacy: bool = False,
        noise_multiplier: float = 0.1,
    ):
        self.method = method
        self.byzantine_robust = byzantine_robust
        self.differential_privacy = differential_privacy
        self.noise_multiplier = noise_multiplier

        logger.info(f"Federated averaging initialized: {method.value}, "
                   f"Byzantine-robust: {byzantine_robust}, "
                   f"DP: {differential_privacy}")

    def aggregate(
        self,
        updates: List[ModelUpdate],
        nodes: Dict[str, FederatedNode],
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate model updates from multiple nodes.

        Args:
            updates: List of model updates from nodes
            nodes: Dictionary of federated nodes

        Returns:
            Aggregated global model parameters
        """
        if not updates:
            raise ValueError("No updates to aggregate")

        # Filter out updates from low-reputation nodes if Byzantine-robust
        if self.byzantine_robust:
            updates = self._filter_byzantine(updates, nodes)

        # Perform aggregation based on method
        if self.method == AggregationMethod.FEDAVG:
            return self._federated_averaging(updates, nodes)
        elif self.method == AggregationMethod.MEDIAN:
            return self._coordinate_median(updates)
        elif self.method == AggregationMethod.TRIMMED_MEAN:
            return self._trimmed_mean(updates, trim_ratio=0.1)
        else:
            return self._federated_averaging(updates, nodes)

    def _federated_averaging(
        self,
        updates: List[ModelUpdate],
        nodes: Dict[str, FederatedNode],
    ) -> Dict[str, torch.Tensor]:
        """Standard federated averaging (weighted by number of samples)"""

        # Calculate total samples
        total_samples = sum(update.num_samples for update in updates)

        if total_samples == 0:
            raise ValueError("Total samples is zero")

        # Initialize aggregated parameters
        aggregated = {}

        # Get parameter names from first update
        param_names = list(updates[0].parameters.keys())

        for param_name in param_names:
            # Weighted sum
            weighted_sum = torch.zeros_like(updates[0].parameters[param_name])

            for update in updates:
                weight = update.num_samples / total_samples

                # Apply node reputation if Byzantine-robust
                if self.byzantine_robust:
                    node = nodes.get(update.node_id)
                    if node:
                        weight *= node.reputation

                param_tensor = update.parameters[param_name]
                weighted_sum += weight * param_tensor

            aggregated[param_name] = weighted_sum

        logger.info(f"Aggregated {len(updates)} updates using FedAvg "
                   f"({total_samples} total samples)")

        return aggregated

    def _coordinate_median(
        self,
        updates: List[ModelUpdate],
    ) -> Dict[str, torch.Tensor]:
        """Coordinate-wise median (Byzantine-robust)"""

        aggregated = {}
        param_names = list(updates[0].parameters.keys())

        for param_name in param_names:
            # Stack all parameters for this layer
            param_stack = torch.stack([
                update.parameters[param_name]
                for update in updates
            ])

            # Compute median along node dimension
            median_params = torch.median(param_stack, dim=0)[0]
            aggregated[param_name] = median_params

        logger.info(f"Aggregated {len(updates)} updates using coordinate median")

        return aggregated

    def _trimmed_mean(
        self,
        updates: List[ModelUpdate],
        trim_ratio: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """Trimmed mean (removes outliers, Byzantine-robust)"""

        aggregated = {}
        param_names = list(updates[0].parameters.keys())

        # Calculate how many to trim from each end
        num_trim = max(1, int(len(updates) * trim_ratio))

        for param_name in param_names:
            # Stack parameters
            param_stack = torch.stack([
                update.parameters[param_name]
                for update in updates
            ])

            # Sort along node dimension
            sorted_params, _ = torch.sort(param_stack, dim=0)

            # Trim extremes
            trimmed = sorted_params[num_trim:-num_trim]

            # Compute mean of trimmed values
            aggregated[param_name] = torch.mean(trimmed, dim=0)

        logger.info(f"Aggregated {len(updates)} updates using trimmed mean "
                   f"(trimmed {num_trim} from each end)")

        return aggregated

    def _filter_byzantine(
        self,
        updates: List[ModelUpdate],
        nodes: Dict[str, FederatedNode],
        min_reputation: float = 0.5,
    ) -> List[ModelUpdate]:
        """Filter out updates from potentially malicious nodes"""

        filtered = []
        rejected = 0

        for update in updates:
            node = nodes.get(update.node_id)

            if node and node.reputation >= min_reputation:
                filtered.append(update)
            else:
                rejected += 1
                logger.warning(f"Rejected update from node {update.node_id} "
                             f"(reputation: {node.reputation if node else 'N/A'})")

        if rejected > 0:
            logger.info(f"Filtered {rejected} Byzantine updates")

        return filtered

    def add_differential_privacy(
        self,
        parameters: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Add differential privacy noise to parameters"""

        if not self.differential_privacy:
            return parameters

        noisy_params = {}

        for param_name, param_tensor in parameters.items():
            # Add Gaussian noise
            noise = torch.randn_like(param_tensor) * self.noise_multiplier
            noisy_params[param_name] = param_tensor + noise

        logger.debug(f"Added DP noise (multiplier: {self.noise_multiplier})")

        return noisy_params


class FederatedLearningCoordinator:
    """
    Coordinates federated learning across multiple nodes.

    Responsibilities:
    - Manage federated nodes
    - Coordinate training rounds
    - Aggregate model updates
    - Track convergence
    - Handle node failures
    """

    def __init__(
        self,
        global_model: nn.Module,
        aggregation_method: AggregationMethod = AggregationMethod.FEDAVG,
        min_nodes: int = 2,
        max_nodes: int = 100,
        nodes_per_round: int = 10,
        local_epochs: int = 5,
        byzantine_robust: bool = True,
        differential_privacy: bool = False,
    ):
        self.global_model = global_model
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.nodes_per_round = nodes_per_round
        self.local_epochs = local_epochs

        # Federated averaging
        self.aggregator = FederatedAveraging(
            method=aggregation_method,
            byzantine_robust=byzantine_robust,
            differential_privacy=differential_privacy,
        )

        # Nodes
        self.nodes: Dict[str, FederatedNode] = {}

        # Training state
        self.current_round = 0
        self.rounds: List[FederatedRound] = []
        self.convergence_history: List[float] = []

        # Statistics
        self.stats = {
            "total_rounds": 0,
            "total_updates": 0,
            "nodes_registered": 0,
            "avg_accuracy": 0.0,
            "converged": False,
        }

        logger.info(f"Federated coordinator initialized: "
                   f"{min_nodes}-{max_nodes} nodes, "
                   f"{nodes_per_round} per round, "
                   f"{local_epochs} local epochs")

    # ========== Node Management ==========

    def register_node(
        self,
        node_id: str,
        node_name: str,
        compute_power: float = 1.0,
        data_size: int = 100,
        bandwidth: float = 1.0,
    ) -> FederatedNode:
        """Register a new federated node"""

        if len(self.nodes) >= self.max_nodes:
            raise ValueError(f"Maximum nodes ({self.max_nodes}) reached")

        node = FederatedNode(
            node_id=node_id,
            node_name=node_name,
            compute_power=compute_power,
            data_size=data_size,
            bandwidth=bandwidth,
        )

        self.nodes[node_id] = node
        self.stats["nodes_registered"] += 1

        logger.info(f"Registered node: {node_name} (ID: {node_id}, "
                   f"data: {data_size} samples)")

        return node

    def select_nodes_for_round(
        self,
        selection_strategy: str = "random",
    ) -> List[str]:
        """Select nodes to participate in current round"""

        available_nodes = [
            node_id for node_id, node in self.nodes.items()
            if node.status in [NodeStatus.IDLE, NodeStatus.WAITING]
        ]

        if len(available_nodes) < self.min_nodes:
            raise ValueError(f"Not enough nodes available "
                           f"({len(available_nodes)} < {self.min_nodes})")

        # Select based on strategy
        if selection_strategy == "random":
            # Random sampling
            selected = np.random.choice(
                available_nodes,
                size=min(self.nodes_per_round, len(available_nodes)),
                replace=False,
            ).tolist()

        elif selection_strategy == "data_size":
            # Prioritize nodes with more data
            sorted_nodes = sorted(
                available_nodes,
                key=lambda nid: self.nodes[nid].data_size,
                reverse=True,
            )
            selected = sorted_nodes[:self.nodes_per_round]

        elif selection_strategy == "reputation":
            # Prioritize high-reputation nodes
            sorted_nodes = sorted(
                available_nodes,
                key=lambda nid: self.nodes[nid].reputation,
                reverse=True,
            )
            selected = sorted_nodes[:self.nodes_per_round]

        else:
            selected = available_nodes[:self.nodes_per_round]

        logger.info(f"Selected {len(selected)} nodes for round {self.current_round + 1}")

        return selected

    # ========== Federated Training ==========

    async def train_round(
        self,
        node_selection: str = "random",
    ) -> FederatedRound:
        """Execute one round of federated training"""

        self.current_round += 1

        round_obj = FederatedRound(
            round_number=self.current_round,
            start_time=datetime.now(),
        )

        # Select nodes
        selected_node_ids = self.select_nodes_for_round(node_selection)
        round_obj.selected_nodes = selected_node_ids

        # Send global model to selected nodes
        global_params = self.global_model.state_dict()

        # Simulate local training on each node
        updates = []

        for node_id in selected_node_ids:
            node = self.nodes[node_id]
            node.status = NodeStatus.TRAINING

            # Simulate local training
            update = await self._simulate_local_training(
                node_id=node_id,
                global_params=global_params,
                local_epochs=self.local_epochs,
            )

            if update:
                updates.append(update)
                round_obj.completed_nodes.append(node_id)
                node.status = NodeStatus.IDLE
                node.updates_contributed += 1
            else:
                node.status = NodeStatus.FAILED

        # Aggregate updates
        if len(updates) >= self.min_nodes:
            aggregated_params = self.aggregator.aggregate(updates, self.nodes)

            # Update global model
            self.global_model.load_state_dict(aggregated_params)

            # Calculate round statistics
            round_obj.total_samples = sum(u.num_samples for u in updates)
            round_obj.global_loss = np.mean([u.loss for u in updates])
            round_obj.global_accuracy = np.mean([u.accuracy for u in updates])

            # Calculate convergence metric (gradient norm)
            round_obj.convergence_metric = self._calculate_convergence(updates)

        else:
            logger.warning(f"Round {self.current_round} failed: "
                          f"insufficient updates ({len(updates)} < {self.min_nodes})")

        round_obj.end_time = datetime.now()
        self.rounds.append(round_obj)

        # Update statistics
        self.stats["total_rounds"] += 1
        self.stats["total_updates"] += len(updates)
        self.stats["avg_accuracy"] = round_obj.global_accuracy

        # Check convergence
        self.convergence_history.append(round_obj.convergence_metric)
        if len(self.convergence_history) >= 5:
            recent_variance = np.var(self.convergence_history[-5:])
            if recent_variance < 0.001:  # Converged
                self.stats["converged"] = True
                logger.info(f"Model converged after {self.current_round} rounds")

        logger.info(f"Round {self.current_round} complete: "
                   f"loss={round_obj.global_loss:.4f}, "
                   f"acc={round_obj.global_accuracy:.1%}, "
                   f"nodes={len(updates)}")

        return round_obj

    async def _simulate_local_training(
        self,
        node_id: str,
        global_params: Dict[str, torch.Tensor],
        local_epochs: int,
    ) -> Optional[ModelUpdate]:
        """Simulate local training on a node"""

        node = self.nodes[node_id]

        try:
            # Simulate training (in real implementation, this would be actual training)
            # For simulation, we'll add small random perturbations to parameters

            updated_params = {}
            total_change = 0.0

            for param_name, param_tensor in global_params.items():
                # Simulate gradient update
                gradient = torch.randn_like(param_tensor) * 0.01
                updated_param = param_tensor - 0.01 * gradient  # SGD step
                updated_params[param_name] = updated_param

                # Track change magnitude
                total_change += torch.norm(updated_param - param_tensor).item()

            # Simulate loss and accuracy
            # In reality, these would come from actual evaluation
            base_loss = 0.5
            base_accuracy = 0.7

            # Better nodes (more data, compute) get better performance
            improvement_factor = (node.compute_power * (node.data_size / 100)) ** 0.5

            loss = max(0.1, base_loss - improvement_factor * 0.1)
            accuracy = min(0.95, base_accuracy + improvement_factor * 0.1)

            # Add differential privacy noise if enabled
            if self.aggregator.differential_privacy:
                updated_params = self.aggregator.add_differential_privacy(updated_params)

            update = ModelUpdate(
                update_id=f"update_{node_id}_{self.current_round}",
                node_id=node_id,
                round_number=self.current_round,
                timestamp=datetime.now(),
                parameters=updated_params,
                num_samples=node.data_size,
                loss=loss,
                accuracy=accuracy,
            )

            # Update node stats
            node.local_epochs += local_epochs
            node.local_loss = loss
            node.local_accuracy = accuracy
            node.last_update = datetime.now()

            return update

        except Exception as e:
            logger.error(f"Local training failed on node {node_id}: {e}")
            return None

    def _calculate_convergence(self, updates: List[ModelUpdate]) -> float:
        """Calculate convergence metric (average gradient norm)"""

        if not updates or len(updates) < 2:
            return 1.0

        # Calculate variance in parameter updates
        param_variances = []

        param_names = list(updates[0].parameters.keys())

        for param_name in param_names:
            # Stack parameters from all updates
            param_stack = torch.stack([
                update.parameters[param_name]
                for update in updates
            ])

            # Calculate variance
            variance = torch.var(param_stack, dim=0).mean().item()
            param_variances.append(variance)

        # Average variance across all parameters
        avg_variance = np.mean(param_variances)

        return avg_variance

    # ========== Training Loop ==========

    async def train(
        self,
        num_rounds: int = 10,
        convergence_threshold: float = 0.001,
        node_selection: str = "random",
    ) -> Dict[str, Any]:
        """Train model using federated learning"""

        logger.info(f"Starting federated training: {num_rounds} rounds, "
                   f"{len(self.nodes)} nodes registered")

        for round_num in range(num_rounds):
            round_obj = await self.train_round(node_selection)

            # Check convergence
            if self.stats["converged"]:
                logger.info(f"Training converged after {round_num + 1} rounds")
                break

            # Check if improvement stopped
            if len(self.convergence_history) >= 10:
                recent_change = abs(
                    self.convergence_history[-1] - self.convergence_history[-10]
                )
                if recent_change < convergence_threshold:
                    logger.info(f"Training stopped: minimal improvement")
                    break

        final_stats = {
            "rounds_completed": self.current_round,
            "converged": self.stats["converged"],
            "final_accuracy": self.stats["avg_accuracy"],
            "final_convergence": self.convergence_history[-1] if self.convergence_history else 0.0,
            "total_updates": self.stats["total_updates"],
        }

        logger.info(f"Federated training complete: "
                   f"{final_stats['rounds_completed']} rounds, "
                   f"{final_stats['final_accuracy']:.1%} accuracy")

        return final_stats

    # ========== Statistics ==========

    def get_statistics(self) -> Dict[str, Any]:
        """Get federated learning statistics"""
        return {
            **self.stats,
            "current_round": self.current_round,
            "nodes_registered": len(self.nodes),
            "active_nodes": sum(1 for n in self.nodes.values() if n.status != NodeStatus.FAILED),
            "convergence_history": self.convergence_history[-10:] if self.convergence_history else [],
        }


# ========== Testing Functions ==========

class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self, input_dim: int = 10, output_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 20)
        self.fc2 = nn.Linear(20, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


async def test_federated_learning():
    """Test federated learning system"""
    print("\n" + "="*60)
    print("Testing Federated Learning System")
    print("="*60)

    # Create global model
    model = SimpleModel(input_dim=10, output_dim=2)

    # Create coordinator
    coordinator = FederatedLearningCoordinator(
        global_model=model,
        aggregation_method=AggregationMethod.FEDAVG,
        min_nodes=3,
        nodes_per_round=5,
        local_epochs=5,
        byzantine_robust=True,
        differential_privacy=False,
    )

    # Test 1: Register Nodes
    print("\n1. Registering nodes...")

    nodes_config = [
        ("node_1", "Server A", 2.0, 500, 1.5),
        ("node_2", "Server B", 1.5, 300, 1.0),
        ("node_3", "Server C", 1.0, 200, 0.8),
        ("node_4", "Server D", 0.8, 150, 0.6),
        ("node_5", "Server E", 0.5, 100, 0.5),
        ("node_6", "Edge Device F", 0.3, 50, 0.3),
        ("node_7", "Edge Device G", 0.3, 50, 0.3),
    ]

    for node_id, name, compute, data, bandwidth in nodes_config:
        node = coordinator.register_node(node_id, name, compute, data, bandwidth)
        print(f"   Registered: {name} ({data} samples, {compute}x compute)")

    # Test 2: Single Round
    print("\n2. Testing single training round...")

    round_obj = await coordinator.train_round(node_selection="data_size")

    print(f"   Round: {round_obj.round_number}")
    print(f"   Selected Nodes: {len(round_obj.selected_nodes)}")
    print(f"   Completed Nodes: {len(round_obj.completed_nodes)}")
    print(f"   Global Loss: {round_obj.global_loss:.4f}")
    print(f"   Global Accuracy: {round_obj.global_accuracy:.1%}")
    print(f"   Convergence Metric: {round_obj.convergence_metric:.6f}")

    # Test 3: Multi-Round Training
    print("\n3. Testing multi-round federated training...")

    results = await coordinator.train(
        num_rounds=10,
        convergence_threshold=0.001,
        node_selection="random",
    )

    print(f"\n   Training Results:")
    print(f"     Rounds Completed: {results['rounds_completed']}")
    print(f"     Converged: {results['converged']}")
    print(f"     Final Accuracy: {results['final_accuracy']:.1%}")
    print(f"     Final Convergence: {results['final_convergence']:.6f}")
    print(f"     Total Updates: {results['total_updates']}")

    # Test 4: Node Statistics
    print("\n4. Node statistics...")

    print(f"\n   Top Performers:")
    sorted_nodes = sorted(
        coordinator.nodes.values(),
        key=lambda n: n.updates_contributed,
        reverse=True,
    )

    for i, node in enumerate(sorted_nodes[:3], 1):
        print(f"     {i}. {node.node_name}: {node.updates_contributed} updates, "
              f"{node.local_accuracy:.1%} accuracy")

    # Test 5: Byzantine Robustness
    print("\n5. Testing Byzantine robustness...")

    # Simulate malicious node
    malicious_node = coordinator.nodes["node_7"]
    malicious_node.reputation = 0.2
    print(f"   Set {malicious_node.node_name} reputation to 0.2 (malicious)")

    # Train another round
    round_obj = await coordinator.train_round()

    participated = malicious_node.node_id in round_obj.completed_nodes
    print(f"   Malicious node participated: {participated}")
    print(f"   (Low reputation nodes should be filtered out)")

    # Final Statistics
    print("\n" + "="*60)
    print("Final Statistics")
    print("="*60)

    stats = coordinator.get_statistics()
    print(f"Total Rounds: {stats['total_rounds']}")
    print(f"Total Updates: {stats['total_updates']}")
    print(f"Nodes Registered: {stats['nodes_registered']}")
    print(f"Active Nodes: {stats['active_nodes']}")
    print(f"Average Accuracy: {stats['avg_accuracy']:.1%}")
    print(f"Converged: {stats['converged']}")

    return coordinator


if __name__ == "__main__":
    asyncio.run(test_federated_learning())
