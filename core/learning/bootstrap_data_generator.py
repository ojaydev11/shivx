"""
Bootstrap Data Generator - Create synthetic training data for cold start

Generates realistic training examples when production data is sparse.
Used to bootstrap RL policies before sufficient real data is collected.

Part of ShivX Personal Empire AGI (Week 3).
"""

import logging
import random
from typing import List, Dict, Any
from datetime import datetime, timedelta

from core.learning.data_collector import (
    DataCollector,
    TaskDomain,
    TaskType,
    TaskExample,
)

logger = logging.getLogger(__name__)


class BootstrapDataGenerator:
    """
    Generates synthetic but realistic training examples.

    This helps bootstrap ML models when real production data is sparse.
    Examples are based on common empire operations and typical user requests.
    """

    def __init__(self):
        self.collector = DataCollector(storage_dir="data/agi_training")

        # Define realistic scenarios for each domain
        self.scenarios = self._build_scenarios()

        logger.info("Bootstrap Data Generator initialized")

    def _build_scenarios(self) -> Dict[TaskDomain, List[Dict[str, Any]]]:
        """Build realistic scenarios for each empire domain"""
        return {
            TaskDomain.SEWAGO: [
                {
                    "query": "Deploy authentication update to production",
                    "task_type": TaskType.DECISION_MAKING,
                    "action": "Ran tests, checked staging, deployed to production",
                    "reasoning": "Tests passed on staging, ready for production deployment",
                    "success": True,
                    "confidence": 0.9,
                },
                {
                    "query": "Fix user login error affecting 5% of users",
                    "task_type": TaskType.BUG_FIXING,
                    "action": "Identified session timeout issue, increased timeout to 30min",
                    "reasoning": "Logs showed session expiry errors, timeout was too short",
                    "success": True,
                    "confidence": 0.85,
                },
                {
                    "query": "Optimize database queries for user dashboard",
                    "task_type": TaskType.SYSTEM_OPTIMIZATION,
                    "action": "Added indexes on user_id and created_at columns",
                    "reasoning": "Query analysis showed full table scans, indexes will speed up",
                    "success": True,
                    "confidence": 0.8,
                },
                {
                    "query": "Add email notification feature for new messages",
                    "task_type": TaskType.CODE_GENERATION,
                    "action": "Created email service, integrated with message queue",
                    "reasoning": "Users requested notifications, built async email system",
                    "success": True,
                    "confidence": 0.75,
                },
                {
                    "query": "Scale server capacity for weekend traffic spike",
                    "task_type": TaskType.SYSTEM_OPTIMIZATION,
                    "action": "Increased instance count from 2 to 4, enabled autoscaling",
                    "reasoning": "Historical data shows 3x traffic on weekends",
                    "success": True,
                    "confidence": 0.9,
                },
            ],
            TaskDomain.HALOBUZZ: [
                {
                    "query": "Create engaging LinkedIn post about AI trends",
                    "task_type": TaskType.CONTENT_CREATION,
                    "action": "Generated post highlighting GPT-4 and autonomous agents",
                    "reasoning": "AI and automation trending in tech professional circles",
                    "success": True,
                    "confidence": 0.85,
                },
                {
                    "query": "Schedule social media posts for next week",
                    "task_type": TaskType.DECISION_MAKING,
                    "action": "Scheduled 5 posts across LinkedIn, Twitter at peak engagement times",
                    "reasoning": "Optimal posting times: 9am, 12pm, 5pm based on analytics",
                    "success": True,
                    "confidence": 0.8,
                },
                {
                    "query": "Analyze competitor social media strategy",
                    "task_type": TaskType.DECISION_MAKING,
                    "action": "Scraped competitor posts, identified content gaps and opportunities",
                    "reasoning": "Competitors focus on product features, we can highlight customer success",
                    "success": True,
                    "confidence": 0.75,
                },
                {
                    "query": "Generate Twitter thread about startup lessons",
                    "task_type": TaskType.CONTENT_CREATION,
                    "action": "Created 8-tweet thread with actionable startup advice",
                    "reasoning": "Thread format performs well on Twitter, actionable content drives engagement",
                    "success": True,
                    "confidence": 0.8,
                },
            ],
            TaskDomain.SOLSNIPER: [
                {
                    "query": "Scan for arbitrage opportunities on Solana DEXes",
                    "task_type": TaskType.TRADING_DECISION,
                    "action": "Found 3 opportunities: SOL/USDC (2.3%), RAY/USDC (1.8%), ORCA/USDC (1.5%)",
                    "reasoning": "Price discrepancies between Raydium and Orca DEXes",
                    "success": True,
                    "confidence": 0.9,
                },
                {
                    "query": "Should I buy SOL at current price of $150?",
                    "task_type": TaskType.TRADING_DECISION,
                    "action": "Recommended wait, price showing resistance at $150",
                    "reasoning": "Technical indicators show overbought (RSI 75), wait for pullback",
                    "success": True,
                    "confidence": 0.7,
                },
                {
                    "query": "Execute arbitrage trade: buy Raydium, sell Orca",
                    "task_type": TaskType.TRADING_DECISION,
                    "action": "Executed trade, profit: 2.1% after fees",
                    "reasoning": "Spread wide enough to cover gas fees and slippage",
                    "success": True,
                    "confidence": 0.85,
                },
                {
                    "query": "Monitor wallet for suspicious transactions",
                    "task_type": TaskType.DECISION_MAKING,
                    "action": "No suspicious activity detected, all transactions verified",
                    "reasoning": "Checked transaction signatures, all match expected patterns",
                    "success": True,
                    "confidence": 0.95,
                },
                {
                    "query": "Set stop loss for SOL position at $140",
                    "task_type": TaskType.TRADING_DECISION,
                    "action": "Configured stop loss order, monitoring price",
                    "reasoning": "7% downside protection, preserves capital if market drops",
                    "success": True,
                    "confidence": 0.9,
                },
            ],
            TaskDomain.SHIVX_CORE: [
                {
                    "query": "Explain how vector memory works",
                    "task_type": TaskType.USER_INTERACTION,
                    "action": "Provided explanation of FAISS, embeddings, and semantic search",
                    "reasoning": "User needs understanding of core ShivX technology",
                    "success": True,
                    "confidence": 0.85,
                },
                {
                    "query": "Update Python dependencies to latest versions",
                    "task_type": TaskType.SYSTEM_OPTIMIZATION,
                    "action": "Updated 12 packages, ran tests, all passed",
                    "reasoning": "Security patches available, tests confirm compatibility",
                    "success": True,
                    "confidence": 0.8,
                },
                {
                    "query": "Configure Prometheus monitoring for API endpoints",
                    "task_type": TaskType.SYSTEM_OPTIMIZATION,
                    "action": "Set up Prometheus metrics, created Grafana dashboard",
                    "reasoning": "Need visibility into API performance and errors",
                    "success": True,
                    "confidence": 0.75,
                },
            ],
        }

    def generate_examples(self, count: int = 100) -> List[TaskExample]:
        """
        Generate synthetic training examples.

        Args:
            count: Number of examples to generate

        Returns:
            List of TaskExample objects
        """
        examples = []

        # Calculate distribution across domains
        domain_weights = {
            TaskDomain.SEWAGO: 0.3,  # 30% Sewago
            TaskDomain.HALOBUZZ: 0.2,  # 20% Halobuzz
            TaskDomain.SOLSNIPER: 0.3,  # 30% SolsniperPro (high volume)
            TaskDomain.SHIVX_CORE: 0.2,  # 20% Core operations
        }

        for i in range(count):
            # Select domain based on weights
            domain = random.choices(
                list(domain_weights.keys()),
                weights=list(domain_weights.values()),
            )[0]

            # Select scenario from domain
            scenarios = self.scenarios.get(domain, [])
            if not scenarios:
                continue

            scenario = random.choice(scenarios)

            # Add some variation
            success = scenario["success"]
            if random.random() < 0.1:  # 10% failure rate
                success = False

            confidence = scenario["confidence"] + random.uniform(-0.1, 0.1)
            confidence = max(0.5, min(0.99, confidence))  # Clamp to [0.5, 0.99]

            # Simulate realistic duration
            duration = self._simulate_duration(scenario["task_type"])

            # Create example
            example = TaskExample(
                id=f"bootstrap_{i:04d}",
                domain=domain,
                task_type=scenario["task_type"],
                context={
                    "source": "bootstrap_generator",
                    "scenario_id": i,
                    "synthetic": True,
                },
                query=scenario["query"],
                action_taken=scenario["action"],
                reasoning=scenario["reasoning"],
                alternatives_considered=[],
                outcome=scenario["action"] if success else "Operation failed due to external constraints",
                success=success,
                user_feedback=None,
                timestamp=datetime.utcnow() - timedelta(days=random.randint(0, 30)),
                confidence=confidence,
                duration_seconds=duration,
                reward=self._compute_reward(success, duration, confidence),
            )

            examples.append(example)

        logger.info(f"Generated {len(examples)} synthetic training examples")

        return examples

    def _simulate_duration(self, task_type: TaskType) -> float:
        """Simulate realistic task duration"""
        # Different task types have different typical durations
        duration_ranges = {
            TaskType.USER_INTERACTION: (5, 60),  # 5s - 1min
            TaskType.DECISION_MAKING: (10, 120),  # 10s - 2min
            TaskType.TRADING_DECISION: (2, 30),  # 2s - 30s (fast decisions)
            TaskType.CODE_GENERATION: (60, 600),  # 1min - 10min
            TaskType.BUG_FIXING: (120, 1800),  # 2min - 30min
            TaskType.SYSTEM_OPTIMIZATION: (300, 3600),  # 5min - 1hr
            TaskType.CONTENT_CREATION: (30, 300),  # 30s - 5min
        }

        min_dur, max_dur = duration_ranges.get(task_type, (10, 300))
        return random.uniform(min_dur, max_dur)

    def _compute_reward(self, success: bool, duration: float, confidence: float) -> float:
        """Compute reward signal (same formula as real data collector)"""
        reward = 1.0 if success else -1.0

        # Speed bonus/penalty
        if duration < 60:
            reward += 0.5
        elif duration > 300:
            reward -= 0.2

        # Confidence bonus/penalty
        if success and confidence > 0.8:
            reward += 0.3
        if not success and confidence > 0.8:
            reward -= 0.3

        return reward

    def save_to_collector(self, examples: List[TaskExample]):
        """Save generated examples to data collector"""
        for example in examples:
            self.collector.current_dataset.add_example(example)

        self.collector.save_dataset("dataset_bootstrap.json")

        logger.info(f"Saved {len(examples)} examples to collector")

    def generate_and_save(self, count: int = 100) -> str:
        """
        Generate examples and save to collector.

        Args:
            count: Number of examples to generate

        Returns:
            Path to saved dataset
        """
        examples = self.generate_examples(count)
        self.save_to_collector(examples)

        # Get stats
        stats = self.collector.get_stats()

        logger.info(
            f"Bootstrap complete: {stats['dataset']['total']} total examples, "
            f"{stats['dataset']['success_rate']:.1%} success rate"
        )

        return str(self.collector.storage_dir / "dataset_bootstrap.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n=== Bootstrap Data Generator Test ===\n")

    generator = BootstrapDataGenerator()

    # Generate 100 examples
    dataset_path = generator.generate_and_save(count=100)

    print(f"\n✅ Generated 100 training examples")
    print(f"   Saved to: {dataset_path}")

    # Show stats
    stats = generator.collector.get_stats()
    print(f"\nDataset Statistics:")
    print(f"  Total examples: {stats['dataset']['total']}")
    print(f"  Success rate: {stats['dataset']['success_rate']:.1%}")

    print(f"\n  Domain distribution:")
    for domain, count in stats['dataset']['domains'].items():
        print(f"    {domain}: {count}")

    print(f"\n  Task type distribution:")
    for task_type, count in stats['dataset']['task_types'].items():
        print(f"    {task_type}: {count}")

    # Export for training
    training_data = generator.collector.export_for_training()
    print(f"\n  Training examples ready: {len(training_data)}")
    print(f"  Successful examples: {len(generator.collector.export_for_training(success_only=True))}")
