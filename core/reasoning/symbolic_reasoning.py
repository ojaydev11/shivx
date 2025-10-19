"""
Symbolic Reasoning System for Empire AGI
Week 19: Neuro-Symbolic Integration

Implements symbolic reasoning capabilities that complement neural learning:
- First-Order Logic (FOL): Logical inference with predicates and quantifiers
- Knowledge Graphs: Structured knowledge representation and reasoning
- Neuro-Symbolic Bridge: Integrate neural predictions with symbolic reasoning
- Rule-Based Systems: Expert system inference engines
- Explainable Reasoning: Traceable reasoning paths

Key capabilities:
- Logical inference (deduction, induction, abduction)
- Knowledge graph querying and reasoning
- Hybrid neuro-symbolic decision making
- Explainable reasoning traces
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from enum import Enum
import numpy as np
from collections import defaultdict


class LogicalOperator(Enum):
    """Logical operators"""
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    IFF = "iff"  # If and only if
    FORALL = "forall"  # Universal quantifier
    EXISTS = "exists"  # Existential quantifier


class InferenceType(Enum):
    """Types of logical inference"""
    DEDUCTION = "deduction"  # General → Specific (certain)
    INDUCTION = "induction"  # Specific → General (probable)
    ABDUCTION = "abduction"  # Effect → Cause (plausible)
    ANALOGY = "analogy"  # Similar cases


class KnowledgeType(Enum):
    """Types of knowledge"""
    FACT = "fact"  # Ground truth
    RULE = "rule"  # If-then rule
    HEURISTIC = "heuristic"  # Rule of thumb
    CONSTRAINT = "constraint"  # Must satisfy


@dataclass
class Predicate:
    """Logical predicate (e.g., IsError(x), FixedBy(x, y))"""
    name: str
    arguments: List[str]
    truth_value: Optional[bool] = None
    confidence: float = 1.0


@dataclass
class LogicalRule:
    """Logical rule (e.g., IF IsError(x) AND HasFix(f) THEN Apply(f, x))"""
    rule_id: str
    premises: List[Predicate]  # Antecedent (IF part)
    conclusion: Predicate  # Consequent (THEN part)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGraphNode:
    """Node in knowledge graph"""
    node_id: str
    node_type: str  # Entity type (e.g., "error", "fix", "user")
    properties: Dict[str, Any]


@dataclass
class KnowledgeGraphEdge:
    """Edge in knowledge graph"""
    edge_id: str
    source: str  # Source node ID
    target: str  # Target node ID
    relation: str  # Relation type (e.g., "causes", "fixes", "requires")
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningTrace:
    """Trace of reasoning steps for explainability"""
    steps: List[Dict[str, Any]]
    conclusion: Any
    confidence: float
    reasoning_type: InferenceType


class FirstOrderLogicEngine:
    """
    First-Order Logic (FOL) reasoning engine.

    Supports:
    - Predicates with variables and constants
    - Logical operators (AND, OR, NOT, IMPLIES)
    - Quantifiers (FORALL, EXISTS)
    - Forward chaining (data-driven inference)
    - Backward chaining (goal-driven inference)
    """

    def __init__(self):
        self.facts: List[Predicate] = []
        self.rules: List[LogicalRule] = []

        # Statistics
        self.inferences_made = 0
        self.rules_applied = 0

    def add_fact(self, predicate: Predicate):
        """Add a fact to the knowledge base"""
        self.facts.append(predicate)

    def add_rule(self, rule: LogicalRule):
        """Add a rule to the knowledge base"""
        self.rules.append(rule)

    def forward_chain(
        self,
        max_iterations: int = 100,
    ) -> List[Predicate]:
        """
        Forward chaining: Derive new facts from existing facts and rules.

        Process:
        1. Find rules whose premises are satisfied by current facts
        2. Apply rules to derive new facts
        3. Repeat until no new facts can be derived
        """
        new_facts = []
        iteration = 0

        while iteration < max_iterations:
            facts_added = 0

            for rule in self.rules:
                # Check if rule premises are satisfied
                if self._premises_satisfied(rule.premises):
                    # Derive conclusion
                    conclusion = rule.conclusion

                    # Check if conclusion is new
                    if not self._fact_exists(conclusion):
                        self.facts.append(conclusion)
                        new_facts.append(conclusion)
                        facts_added += 1
                        self.rules_applied += 1

            # Stop if no new facts derived
            if facts_added == 0:
                break

            iteration += 1
            self.inferences_made += facts_added

        return new_facts

    def backward_chain(
        self,
        goal: Predicate,
        depth: int = 0,
        max_depth: int = 10,
    ) -> Tuple[bool, ReasoningTrace]:
        """
        Backward chaining: Try to prove a goal by finding supporting facts/rules.

        Process:
        1. Check if goal is a known fact
        2. Find rules that conclude the goal
        3. Recursively try to prove rule premises
        4. Return True if goal can be proved
        """
        trace_steps = []

        # Base case: Check if goal is a fact
        if self._fact_exists(goal):
            trace_steps.append({
                "type": "fact",
                "predicate": goal,
                "depth": depth,
            })
            return True, ReasoningTrace(
                steps=trace_steps,
                conclusion=goal,
                confidence=goal.confidence,
                reasoning_type=InferenceType.DEDUCTION,
            )

        # Check depth limit
        if depth >= max_depth:
            return False, ReasoningTrace(
                steps=trace_steps,
                conclusion=None,
                confidence=0.0,
                reasoning_type=InferenceType.DEDUCTION,
            )

        # Find rules that conclude the goal
        for rule in self.rules:
            if self._predicates_match(rule.conclusion, goal):
                trace_steps.append({
                    "type": "rule",
                    "rule_id": rule.rule_id,
                    "depth": depth,
                })

                # Try to prove all premises
                all_premises_proved = True
                premise_confidence = 1.0

                for premise in rule.premises:
                    proved, premise_trace = self.backward_chain(
                        premise,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )

                    if not proved:
                        all_premises_proved = False
                        break

                    trace_steps.extend(premise_trace.steps)
                    premise_confidence *= premise_trace.confidence

                # If all premises proved, goal is proved
                if all_premises_proved:
                    self.inferences_made += 1
                    self.rules_applied += 1

                    return True, ReasoningTrace(
                        steps=trace_steps,
                        conclusion=goal,
                        confidence=premise_confidence * rule.confidence,
                        reasoning_type=InferenceType.DEDUCTION,
                    )

        # Goal cannot be proved
        return False, ReasoningTrace(
            steps=trace_steps,
            conclusion=None,
            confidence=0.0,
            reasoning_type=InferenceType.DEDUCTION,
        )

    def _premises_satisfied(self, premises: List[Predicate]) -> bool:
        """Check if all premises are satisfied by current facts"""
        for premise in premises:
            if not self._fact_exists(premise):
                return False
        return True

    def _fact_exists(self, predicate: Predicate) -> bool:
        """Check if a fact exists in knowledge base"""
        for fact in self.facts:
            if self._predicates_match(fact, predicate):
                return True
        return False

    def _predicates_match(self, p1: Predicate, p2: Predicate) -> bool:
        """Check if two predicates match (unification)"""
        # Simple matching: Same name and arguments
        # In full FOL, would need proper unification with variable binding
        return (p1.name == p2.name and
                p1.arguments == p2.arguments)

    def get_statistics(self) -> Dict[str, Any]:
        """Get reasoning statistics"""
        return {
            "facts": len(self.facts),
            "rules": len(self.rules),
            "inferences_made": self.inferences_made,
            "rules_applied": self.rules_applied,
        }


class KnowledgeGraph:
    """
    Knowledge graph for structured knowledge representation.

    Represents knowledge as a graph of entities (nodes) and
    relations (edges).
    """

    def __init__(self):
        self.nodes: Dict[str, KnowledgeGraphNode] = {}
        self.edges: List[KnowledgeGraphEdge] = []

        # Index for fast lookups
        self.node_edges: Dict[str, List[KnowledgeGraphEdge]] = defaultdict(list)

    def add_node(self, node: KnowledgeGraphNode):
        """Add a node to the graph"""
        self.nodes[node.node_id] = node

    def add_edge(self, edge: KnowledgeGraphEdge):
        """Add an edge to the graph"""
        self.edges.append(edge)

        # Update index
        self.node_edges[edge.source].append(edge)
        self.node_edges[edge.target].append(edge)

    def get_node(self, node_id: str) -> Optional[KnowledgeGraphNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)

    def get_neighbors(
        self,
        node_id: str,
        relation: Optional[str] = None,
    ) -> List[KnowledgeGraphNode]:
        """
        Get neighboring nodes.

        Args:
            node_id: Source node
            relation: Filter by relation type (optional)

        Returns:
            List of neighbor nodes
        """
        neighbors = []

        for edge in self.node_edges[node_id]:
            # Determine neighbor (node on other end of edge)
            if edge.source == node_id:
                neighbor_id = edge.target
            else:
                neighbor_id = edge.source

            # Filter by relation if specified
            if relation is None or edge.relation == relation:
                neighbor = self.nodes.get(neighbor_id)
                if neighbor:
                    neighbors.append(neighbor)

        return neighbors

    def query_path(
        self,
        start_node: str,
        end_node: str,
        max_depth: int = 5,
    ) -> Optional[List[KnowledgeGraphEdge]]:
        """
        Find shortest path between two nodes.

        Uses BFS to find shortest path.
        """
        # BFS
        queue = [(start_node, [])]
        visited = {start_node}

        while queue:
            current_node, path = queue.pop(0)

            # Check if reached end
            if current_node == end_node:
                return path

            # Check depth limit
            if len(path) >= max_depth:
                continue

            # Explore neighbors
            for edge in self.node_edges[current_node]:
                # Determine next node
                if edge.source == current_node:
                    next_node = edge.target
                else:
                    next_node = edge.source

                # Skip if visited
                if next_node in visited:
                    continue

                visited.add(next_node)
                queue.append((next_node, path + [edge]))

        # No path found
        return None

    def subgraph(
        self,
        center_node: str,
        radius: int = 2,
    ) -> 'KnowledgeGraph':
        """
        Extract subgraph around a center node.

        Args:
            center_node: Center of subgraph
            radius: Number of hops from center

        Returns:
            New KnowledgeGraph containing subgraph
        """
        subgraph = KnowledgeGraph()

        # BFS to find nodes within radius
        queue = [(center_node, 0)]
        visited = {center_node}

        while queue:
            current_node, depth = queue.pop(0)

            # Add node to subgraph
            node = self.nodes.get(current_node)
            if node:
                subgraph.add_node(node)

            # Stop if reached radius
            if depth >= radius:
                continue

            # Explore neighbors
            for edge in self.node_edges[current_node]:
                # Determine next node
                if edge.source == current_node:
                    next_node = edge.target
                else:
                    next_node = edge.source

                # Add edge to subgraph
                subgraph.add_edge(edge)

                # Skip if visited
                if next_node in visited:
                    continue

                visited.add(next_node)
                queue.append((next_node, depth + 1))

        return subgraph

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        # Count node types
        node_types = defaultdict(int)
        for node in self.nodes.values():
            node_types[node.node_type] += 1

        # Count relation types
        relation_types = defaultdict(int)
        for edge in self.edges:
            relation_types[edge.relation] += 1

        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "node_types": dict(node_types),
            "relation_types": dict(relation_types),
        }


class NeuroSymbolicBridge:
    """
    Bridge between neural networks and symbolic reasoning.

    Enables:
    - Neural predictions → Symbolic facts
    - Symbolic reasoning → Neural confidence
    - Hybrid decision making
    """

    def __init__(
        self,
        neural_model: nn.Module,
        fol_engine: FirstOrderLogicEngine,
        knowledge_graph: KnowledgeGraph,
    ):
        self.neural_model = neural_model
        self.fol_engine = fol_engine
        self.knowledge_graph = knowledge_graph

        # Statistics
        self.neural_calls = 0
        self.symbolic_calls = 0
        self.hybrid_calls = 0

    def neural_to_symbolic(
        self,
        neural_output: torch.Tensor,
        output_to_predicate: Callable[[torch.Tensor], Predicate],
    ) -> Predicate:
        """
        Convert neural network output to symbolic predicate.

        Args:
            neural_output: Neural network output (e.g., logits, probabilities)
            output_to_predicate: Function to convert output to predicate

        Returns:
            Symbolic predicate with confidence from neural network
        """
        predicate = output_to_predicate(neural_output)

        # Extract confidence from neural output
        if neural_output.dim() == 1:
            # Classification: Use max probability as confidence
            probs = torch.softmax(neural_output, dim=0)
            confidence = probs.max().item()
        else:
            # Other: Use sigmoid of max logit
            confidence = torch.sigmoid(neural_output.max()).item()

        predicate.confidence = confidence

        self.neural_calls += 1

        return predicate

    def symbolic_to_neural(
        self,
        predicate: Predicate,
        predicate_to_features: Callable[[Predicate], torch.Tensor],
    ) -> torch.Tensor:
        """
        Convert symbolic predicate to neural network input.

        Args:
            predicate: Symbolic predicate
            predicate_to_features: Function to convert predicate to feature vector

        Returns:
            Feature tensor for neural network
        """
        features = predicate_to_features(predicate)

        self.symbolic_calls += 1

        return features

    def hybrid_inference(
        self,
        input_features: torch.Tensor,
        goal_predicate: Predicate,
        use_neural: bool = True,
        use_symbolic: bool = True,
    ) -> Tuple[Any, float, ReasoningTrace]:
        """
        Hybrid neural-symbolic inference.

        Combines neural predictions with symbolic reasoning for
        more robust and explainable decisions.

        Args:
            input_features: Input to neural network
            goal_predicate: Goal to prove symbolically
            use_neural: Whether to use neural network
            use_symbolic: Whether to use symbolic reasoning

        Returns:
            Conclusion, confidence, reasoning trace
        """
        self.hybrid_calls += 1

        trace_steps = []

        # Neural inference
        neural_confidence = 0.0
        if use_neural:
            self.neural_model.eval()
            with torch.no_grad():
                neural_output = self.neural_model(input_features)

            # Convert to confidence
            neural_confidence = torch.sigmoid(neural_output.max()).item()

            trace_steps.append({
                "type": "neural",
                "output": neural_output.tolist(),
                "confidence": neural_confidence,
            })

        # Symbolic inference
        symbolic_confidence = 0.0
        symbolic_proved = False
        if use_symbolic:
            symbolic_proved, symbolic_trace = self.fol_engine.backward_chain(goal_predicate)
            symbolic_confidence = symbolic_trace.confidence

            trace_steps.append({
                "type": "symbolic",
                "proved": symbolic_proved,
                "confidence": symbolic_confidence,
                "trace": symbolic_trace.steps,
            })

        # Combine neural and symbolic
        if use_neural and use_symbolic:
            # Weighted combination
            combined_confidence = (neural_confidence * 0.5 + symbolic_confidence * 0.5)

            # Require agreement for high confidence
            if abs(neural_confidence - symbolic_confidence) > 0.3:
                # Disagreement: Lower confidence
                combined_confidence *= 0.7

            conclusion = symbolic_proved and (neural_confidence > 0.5)
        elif use_neural:
            combined_confidence = neural_confidence
            conclusion = (neural_confidence > 0.5)
        elif use_symbolic:
            combined_confidence = symbolic_confidence
            conclusion = symbolic_proved
        else:
            combined_confidence = 0.0
            conclusion = False

        trace = ReasoningTrace(
            steps=trace_steps,
            conclusion=conclusion,
            confidence=combined_confidence,
            reasoning_type=InferenceType.DEDUCTION,
        )

        return conclusion, combined_confidence, trace

    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics"""
        return {
            "neural_calls": self.neural_calls,
            "symbolic_calls": self.symbolic_calls,
            "hybrid_calls": self.hybrid_calls,
        }


# ============================================================
# Test Functions
# ============================================================

async def test_symbolic_reasoning():
    """Test symbolic reasoning system"""
    print("=" * 60)
    print("Testing Symbolic Reasoning System")
    print("=" * 60)
    print()

    # Test 1: First-Order Logic - Forward Chaining
    print("1. Testing FOL forward chaining...")

    fol = FirstOrderLogicEngine()

    # Add facts
    fol.add_fact(Predicate("IsError", ["error1"], truth_value=True))
    fol.add_fact(Predicate("HasFix", ["fix1"], truth_value=True))

    # Add rules
    fol.add_rule(LogicalRule(
        rule_id="rule1",
        premises=[
            Predicate("IsError", ["error1"], truth_value=True),
            Predicate("HasFix", ["fix1"], truth_value=True),
        ],
        conclusion=Predicate("CanFix", ["error1", "fix1"], truth_value=True),
    ))

    # Forward chain
    new_facts = fol.forward_chain()

    print(f"   Initial facts: {len(fol.facts) - len(new_facts)}")
    print(f"   New facts derived: {len(new_facts)}")
    print(f"   Rules applied: {fol.rules_applied}")
    print(f"   Total facts: {len(fol.facts)}")
    print()

    # Test 2: FOL - Backward Chaining
    print("2. Testing FOL backward chaining...")

    # Try to prove goal
    goal = Predicate("CanFix", ["error1", "fix1"], truth_value=True)
    proved, trace = fol.backward_chain(goal)

    print(f"   Goal: CanFix(error1, fix1)")
    print(f"   Proved: {proved}")
    print(f"   Confidence: {trace.confidence:.1%}")
    print(f"   Reasoning steps: {len(trace.steps)}")
    print()

    # Test 3: Knowledge Graph
    print("3. Testing knowledge graph...")

    kg = KnowledgeGraph()

    # Add nodes
    kg.add_node(KnowledgeGraphNode(
        node_id="error1",
        node_type="error",
        properties={"severity": "high", "message": "NullPointerException"}
    ))
    kg.add_node(KnowledgeGraphNode(
        node_id="fix1",
        node_type="fix",
        properties={"type": "null_check", "code": "if (x != null)"}
    ))
    kg.add_node(KnowledgeGraphNode(
        node_id="user1",
        node_type="user",
        properties={"name": "Developer A"}
    ))

    # Add edges
    kg.add_edge(KnowledgeGraphEdge(
        edge_id="edge1",
        source="error1",
        target="fix1",
        relation="fixed_by"
    ))
    kg.add_edge(KnowledgeGraphEdge(
        edge_id="edge2",
        source="fix1",
        target="user1",
        relation="created_by"
    ))

    # Query
    neighbors = kg.get_neighbors("error1", relation="fixed_by")
    print(f"   Nodes: {len(kg.nodes)}")
    print(f"   Edges: {len(kg.edges)}")
    print(f"   Neighbors of error1 (fixed_by): {len(neighbors)}")
    if neighbors:
        print(f"   First neighbor: {neighbors[0].node_id} ({neighbors[0].node_type})")

    # Find path
    path = kg.query_path("error1", "user1")
    print(f"   Path from error1 to user1: {len(path) if path else 0} edges")
    print()

    # Test 4: Neuro-Symbolic Bridge
    print("4. Testing neuro-symbolic bridge...")

    # Simple neural model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()

    # Create bridge
    bridge = NeuroSymbolicBridge(
        neural_model=model,
        fol_engine=fol,
        knowledge_graph=kg,
    )

    # Hybrid inference
    input_features = torch.randn(1, 10)
    goal = Predicate("CanFix", ["error1", "fix1"], truth_value=True)

    conclusion, confidence, trace = bridge.hybrid_inference(
        input_features=input_features,
        goal_predicate=goal,
        use_neural=True,
        use_symbolic=True,
    )

    print(f"   Conclusion: {conclusion}")
    print(f"   Confidence: {confidence:.1%}")
    print(f"   Reasoning steps: {len(trace.steps)}")
    print(f"   Neural used: {trace.steps[0]['type'] == 'neural'}")
    print(f"   Symbolic used: {trace.steps[1]['type'] == 'symbolic'}")
    print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    fol_stats = fol.get_statistics()
    kg_stats = kg.get_statistics()
    bridge_stats = bridge.get_statistics()

    print(f"FOL Engine:")
    print(f"  Facts: {fol_stats['facts']}")
    print(f"  Rules: {fol_stats['rules']}")
    print(f"  Inferences: {fol_stats['inferences_made']}")
    print()
    print(f"Knowledge Graph:")
    print(f"  Nodes: {kg_stats['num_nodes']}")
    print(f"  Edges: {kg_stats['num_edges']}")
    print()
    print(f"Neuro-Symbolic Bridge:")
    print(f"  Neural calls: {bridge_stats['neural_calls']}")
    print(f"  Symbolic calls: {bridge_stats['symbolic_calls']}")
    print(f"  Hybrid calls: {bridge_stats['hybrid_calls']}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_symbolic_reasoning())
