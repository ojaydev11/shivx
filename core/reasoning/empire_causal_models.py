"""
Empire Causal Models - Domain-specific causal graphs

Defines causal structures for each empire platform:
- Sewago: Platform operations causality
- Halobuzz: Social media causality
- SolsniperPro: Trading causality

Enables:
- Understanding WHY interventions work
- Predicting effects of actions
- Identifying optimal intervention points
- Avoiding spurious correlations

Part of ShivX Personal Empire AGI (Week 6).
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

from core.reasoning.causal_inference import (
    CausalInferenceEngine,
    CausalGraph,
    Intervention,
    CounterfactualQuery,
    get_causal_engine,
)
from core.learning.data_collector import get_collector, TaskDomain

logger = logging.getLogger(__name__)


@dataclass
class CausalInsight:
    """Actionable insight from causal analysis"""

    domain: str
    insight_type: str  # intervention, counterfactual, confounder
    description: str
    expected_impact: float
    confidence: float
    action_required: Optional[str] = None


class EmpireCausalModels:
    """
    Domain-specific causal models for empire platforms.

    Provides causal reasoning capabilities for each platform:
    - Sewago: Platform health causality
    - Halobuzz: Engagement causality
    - SolsniperPro: Trading causality
    """

    def __init__(self):
        self.engine = get_causal_engine()
        self.collector = get_collector()

        # Domain-specific causal structures
        self.domain_structures = {
            "sewago": self._build_sewago_causal_structure(),
            "halobuzz": self._build_halobuzz_causal_structure(),
            "solsniper": self._build_solsniper_causal_structure(),
        }

        logger.info("Empire Causal Models initialized")

    def _build_sewago_causal_structure(self) -> Dict[str, Any]:
        """
        Build causal structure for Sewago platform.

        Causal relationships:
        - error_rate -> performance (negative)
        - performance -> user_satisfaction (positive)
        - user_satisfaction -> active_users (positive)
        - active_users -> revenue (positive)
        - deployment_frequency -> error_rate (positive if poor quality)
        """
        return {
            "variables": [
                "error_rate",
                "performance",
                "user_satisfaction",
                "active_users",
                "revenue",
                "deployment_frequency",
            ],
            "edges": [
                {"cause": "error_rate", "effect": "performance", "strength": -0.8},
                {"cause": "performance", "effect": "user_satisfaction", "strength": 0.7},
                {"cause": "user_satisfaction", "effect": "active_users", "strength": 0.6},
                {"cause": "active_users", "effect": "revenue", "strength": 0.9},
                {"cause": "deployment_frequency", "effect": "error_rate", "strength": 0.3},
            ],
            "key_outcomes": ["revenue", "active_users", "user_satisfaction"],
            "intervention_targets": ["error_rate", "performance", "deployment_frequency"],
        }

    def _build_halobuzz_causal_structure(self) -> Dict[str, Any]:
        """
        Build causal structure for Halobuzz platform.

        Causal relationships:
        - content_quality -> engagement_rate (positive)
        - posting_frequency -> visibility (positive)
        - visibility -> impressions (positive)
        - impressions -> engagement_rate (positive)
        - engagement_rate -> follower_growth (positive)
        - follower_growth -> reach (positive)
        """
        return {
            "variables": [
                "content_quality",
                "posting_frequency",
                "visibility",
                "impressions",
                "engagement_rate",
                "follower_growth",
                "reach",
            ],
            "edges": [
                {"cause": "content_quality", "effect": "engagement_rate", "strength": 0.8},
                {"cause": "posting_frequency", "effect": "visibility", "strength": 0.6},
                {"cause": "visibility", "effect": "impressions", "strength": 0.7},
                {"cause": "impressions", "effect": "engagement_rate", "strength": 0.5},
                {"cause": "engagement_rate", "effect": "follower_growth", "strength": 0.9},
                {"cause": "follower_growth", "effect": "reach", "strength": 0.7},
            ],
            "key_outcomes": ["follower_growth", "engagement_rate", "reach"],
            "intervention_targets": ["content_quality", "posting_frequency"],
        }

    def _build_solsniper_causal_structure(self) -> Dict[str, Any]:
        """
        Build causal structure for SolsniperPro platform.

        Causal relationships:
        - market_volatility -> arbitrage_opportunities (positive)
        - arbitrage_opportunities -> trade_frequency (positive)
        - trade_frequency -> transaction_costs (positive)
        - transaction_costs -> net_profit (negative)
        - risk_level -> position_size (negative)
        - position_size -> potential_profit (positive)
        - position_size -> potential_loss (positive)
        """
        return {
            "variables": [
                "market_volatility",
                "arbitrage_opportunities",
                "trade_frequency",
                "transaction_costs",
                "risk_level",
                "position_size",
                "potential_profit",
                "potential_loss",
                "net_profit",
            ],
            "edges": [
                {"cause": "market_volatility", "effect": "arbitrage_opportunities", "strength": 0.7},
                {"cause": "arbitrage_opportunities", "effect": "trade_frequency", "strength": 0.6},
                {"cause": "trade_frequency", "effect": "transaction_costs", "strength": 0.8},
                {"cause": "transaction_costs", "effect": "net_profit", "strength": -0.5},
                {"cause": "risk_level", "effect": "position_size", "strength": -0.7},
                {"cause": "position_size", "effect": "potential_profit", "strength": 0.9},
                {"cause": "position_size", "effect": "potential_loss", "strength": 0.9},
                {"cause": "potential_profit", "effect": "net_profit", "strength": 0.8},
            ],
            "key_outcomes": ["net_profit", "potential_loss"],
            "intervention_targets": ["risk_level", "position_size", "trade_frequency"],
        }

    def initialize_causal_graph(self, domain: str) -> CausalGraph:
        """
        Initialize causal graph for domain with expert knowledge.

        Args:
            domain: Domain name

        Returns:
            Initialized causal graph
        """
        if domain not in self.domain_structures:
            logger.error(f"Unknown domain: {domain}")
            return CausalGraph()

        structure = self.domain_structures[domain]

        graph = CausalGraph()

        # Add nodes
        for var in structure["variables"]:
            graph.add_node(var)

        # Add edges
        for edge in structure["edges"]:
            graph.add_edge(
                cause=edge["cause"],
                effect=edge["effect"],
                strength=edge["strength"],
                confidence=1.0,  # Expert knowledge
            )

        logger.info(f"Initialized causal graph for {domain}: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

        return graph

    async def analyze_intervention_impact(
        self,
        domain: str,
        intervention_var: str,
        intervention_value: float,
        outcome_vars: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Analyze impact of intervention across multiple outcomes.

        Args:
            domain: Domain name
            intervention_var: Variable to intervene on
            intervention_value: Value to set
            outcome_vars: Outcomes to measure (default: all key outcomes)

        Returns:
            Dictionary of outcome -> expected change
        """
        logger.info(f"Analyzing intervention impact: do({intervention_var}={intervention_value})")

        # Get or initialize graph
        if domain not in self.engine.graphs:
            graph = self.initialize_causal_graph(domain)
            self.engine.graphs[domain] = graph

        # Get outcomes
        if outcome_vars is None:
            structure = self.domain_structures.get(domain, {})
            outcome_vars = structure.get("key_outcomes", [])

        # Compute effects
        intervention = Intervention(variable=intervention_var, value=intervention_value)
        effects = {}

        for outcome in outcome_vars:
            effect = self.engine.compute_intervention(
                domain=domain,
                intervention=intervention,
                target=outcome,
            )
            effects[outcome] = effect

        logger.info(f"Intervention effects: {effects}")

        return effects

    async def find_optimal_intervention(
        self,
        domain: str,
        outcome: str,
        intervention_candidates: List[str],
    ) -> Dict[str, Any]:
        """
        Find optimal intervention to maximize outcome.

        Args:
            domain: Domain name
            outcome: Target outcome to optimize
            intervention_candidates: List of variables to consider

        Returns:
            Best intervention details
        """
        logger.info(f"Finding optimal intervention for {outcome} in {domain}")

        # Get or initialize graph
        if domain not in self.engine.graphs:
            graph = self.initialize_causal_graph(domain)
            self.engine.graphs[domain] = graph

        best_intervention = None
        best_effect = -np.inf

        # Test different intervention values
        for var in intervention_candidates:
            # Try increasing the variable
            for value in [0.5, 1.0, 1.5, 2.0]:
                intervention = Intervention(variable=var, value=value)
                effect = self.engine.compute_intervention(
                    domain=domain,
                    intervention=intervention,
                    target=outcome,
                )

                if effect > best_effect:
                    best_effect = effect
                    best_intervention = {
                        "variable": var,
                        "value": value,
                        "expected_effect": effect,
                    }

        logger.info(f"Best intervention: {best_intervention}")

        return best_intervention

    async def generate_causal_insights(
        self,
        domain: str,
    ) -> List[CausalInsight]:
        """
        Generate actionable insights from causal analysis.

        Args:
            domain: Domain name

        Returns:
            List of causal insights
        """
        logger.info(f"Generating causal insights for {domain}")

        insights = []

        # Get or initialize graph
        if domain not in self.engine.graphs:
            graph = self.initialize_causal_graph(domain)
            self.engine.graphs[domain] = graph
        else:
            graph = self.engine.graphs[domain]

        structure = self.domain_structures.get(domain, {})
        intervention_targets = structure.get("intervention_targets", [])
        key_outcomes = structure.get("key_outcomes", [])

        # Analyze each intervention target
        for target in intervention_targets:
            for outcome in key_outcomes:
                # Compute causal effect
                causal_effect = self.engine.get_causal_effect(
                    domain=domain,
                    cause=target,
                    effect=outcome,
                )

                # Check for indirect effects
                descendants = graph.get_descendants(target)

                if outcome in descendants or causal_effect is not None:
                    # There's a causal path
                    if causal_effect is None:
                        causal_effect = 0.5  # Indirect effect

                    # Generate insight
                    if causal_effect > 0.5:
                        insight = CausalInsight(
                            domain=domain,
                            insight_type="intervention",
                            description=f"Improving {target} will likely improve {outcome}",
                            expected_impact=abs(causal_effect),
                            confidence=0.8,
                            action_required=f"Focus on {target} optimization",
                        )
                        insights.append(insight)
                    elif causal_effect < -0.5:
                        insight = CausalInsight(
                            domain=domain,
                            insight_type="intervention",
                            description=f"Reducing {target} will likely improve {outcome}",
                            expected_impact=abs(causal_effect),
                            confidence=0.8,
                            action_required=f"Focus on {target} reduction",
                        )
                        insights.append(insight)

        # Identify confounders for key relationships
        if len(key_outcomes) >= 2:
            confounders = self.engine.identify_confounders(
                domain=domain,
                treatment=intervention_targets[0] if intervention_targets else list(graph.nodes)[0],
                outcome=key_outcomes[0],
            )

            if confounders:
                insight = CausalInsight(
                    domain=domain,
                    insight_type="confounder",
                    description=f"Confounders detected: {confounders}",
                    expected_impact=0.7,
                    confidence=0.6,
                    action_required="Control for confounders in analysis",
                )
                insights.append(insight)

        logger.info(f"Generated {len(insights)} causal insights")

        return insights

    async def compare_scenarios(
        self,
        domain: str,
        scenario_a: Dict[str, float],
        scenario_b: Dict[str, float],
        outcome: str,
    ) -> Dict[str, Any]:
        """
        Compare two scenarios using counterfactual analysis.

        Args:
            domain: Domain name
            scenario_a: First scenario (variable -> value)
            scenario_b: Second scenario (variable -> value)
            outcome: Outcome to compare

        Returns:
            Comparison results
        """
        logger.info(f"Comparing scenarios for {outcome}")

        # Get or initialize graph
        if domain not in self.engine.graphs:
            graph = self.initialize_causal_graph(domain)
            self.engine.graphs[domain] = graph

        # Compute outcomes under each scenario
        outcome_a = 0.0
        outcome_b = 0.0

        for var, value in scenario_a.items():
            intervention = Intervention(variable=var, value=value)
            effect = self.engine.compute_intervention(
                domain=domain,
                intervention=intervention,
                target=outcome,
            )
            outcome_a += effect

        for var, value in scenario_b.items():
            intervention = Intervention(variable=var, value=value)
            effect = self.engine.compute_intervention(
                domain=domain,
                intervention=intervention,
                target=outcome,
            )
            outcome_b += effect

        difference = outcome_b - outcome_a
        better_scenario = "B" if difference > 0 else "A"

        results = {
            "outcome_a": outcome_a,
            "outcome_b": outcome_b,
            "difference": difference,
            "better_scenario": better_scenario,
            "interpretation": (
                f"Scenario {better_scenario} is expected to be "
                f"{abs(difference):.2f} better for {outcome}"
            ),
        }

        logger.info(f"Scenario comparison: {results['interpretation']}")

        return results


async def main():
    """Test empire causal models"""
    logging.basicConfig(level=logging.INFO)

    print("\n=== Empire Causal Models Test ===\n")

    models = EmpireCausalModels()

    # Test Sewago causal analysis
    print("=== Sewago Platform Analysis ===\n")

    # Initialize graph
    sewago_graph = models.initialize_causal_graph("sewago")
    print(f"Sewago causal graph: {len(sewago_graph.nodes)} nodes, {len(sewago_graph.edges)} edges\n")

    # Analyze intervention: reduce error rate
    print("Intervention: Reduce error_rate to 0.1")
    effects = await models.analyze_intervention_impact(
        domain="sewago",
        intervention_var="error_rate",
        intervention_value=0.1,
    )
    print(f"Expected effects: {effects}\n")

    # Find optimal intervention
    print("Finding optimal intervention for revenue...")
    best = await models.find_optimal_intervention(
        domain="sewago",
        outcome="revenue",
        intervention_candidates=["error_rate", "performance", "deployment_frequency"],
    )
    print(f"Best intervention: {best}\n")

    # Generate insights
    print("Generating causal insights...")
    insights = await models.generate_causal_insights(domain="sewago")
    print(f"\nCausal Insights ({len(insights)}):")
    for insight in insights:
        print(f"  - {insight.description}")
        if insight.action_required:
            print(f"    Action: {insight.action_required}")

    # Test Halobuzz causal analysis
    print("\n\n=== Halobuzz Platform Analysis ===\n")

    halobuzz_graph = models.initialize_causal_graph("halobuzz")
    print(f"Halobuzz causal graph: {len(halobuzz_graph.nodes)} nodes, {len(halobuzz_graph.edges)} edges\n")

    # Compare scenarios
    print("Comparing content strategies...")
    scenario_a = {"content_quality": 0.7, "posting_frequency": 2.0}
    scenario_b = {"content_quality": 0.9, "posting_frequency": 1.5}

    comparison = await models.compare_scenarios(
        domain="halobuzz",
        scenario_a=scenario_a,
        scenario_b=scenario_b,
        outcome="follower_growth",
    )
    print(f"Scenario A (high frequency): {scenario_a}")
    print(f"Scenario B (high quality): {scenario_b}")
    print(f"Result: {comparison['interpretation']}\n")

    # Test SolsniperPro causal analysis
    print("\n=== SolsniperPro Platform Analysis ===\n")

    solsniper_graph = models.initialize_causal_graph("solsniper")
    print(f"Solsniper causal graph: {len(solsniper_graph.nodes)} nodes, {len(solsniper_graph.edges)} edges\n")

    # Analyze risk intervention
    print("Intervention: Reduce risk_level to 0.3")
    effects = await models.analyze_intervention_impact(
        domain="solsniper",
        intervention_var="risk_level",
        intervention_value=0.3,
    )
    print(f"Expected effects: {effects}\n")

    insights = await models.generate_causal_insights(domain="solsniper")
    print(f"\nCausal Insights ({len(insights)}):")
    for insight in insights[:5]:  # Top 5
        print(f"  - {insight.description}")

    print("\n=== Empire Causal Models Ready ===")
    print("Causal reasoning enabled for all empire platforms!")


if __name__ == "__main__":
    asyncio.run(main())
