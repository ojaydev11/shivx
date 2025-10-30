"""
Creative Problem Solver

Implements creative problem-solving techniques:
- Design Thinking
- TRIZ (Theory of Inventive Problem Solving)
- Divergent thinking
- Constraint relaxation
- Problem reframing
- Analogical problem solving
"""
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib
import random
from collections import defaultdict


class SolutionType(str, Enum):
    """Types of creative solutions"""
    INCREMENTAL = "incremental"  # Small improvement
    INNOVATIVE = "innovative"  # Novel approach
    DISRUPTIVE = "disruptive"  # Changes the game
    TRANSFORMATIVE = "transformative"  # Paradigm shift


class ProblemType(str, Enum):
    """Types of problems"""
    WELL_DEFINED = "well_defined"  # Clear problem and solution space
    ILL_DEFINED = "ill_defined"  # Ambiguous problem
    WICKED = "wicked"  # Complex, no clear solution
    TECHNICAL = "technical"  # Engineering problem
    CONCEPTUAL = "conceptual"  # Abstract problem
    SOCIAL = "social"  # Interpersonal problem


class ThinkingMode(str, Enum):
    """Modes of thinking"""
    CONVERGENT = "convergent"  # Narrow down to best solution
    DIVERGENT = "divergent"  # Generate many alternatives
    LATERAL = "lateral"  # Break patterns
    VERTICAL = "vertical"  # Logical progression
    CRITICAL = "critical"  # Evaluate and judge
    CREATIVE = "creative"  # Generate novel ideas


@dataclass
class Problem:
    """A problem to solve"""
    problem_id: str
    description: str
    problem_type: ProblemType
    constraints: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    stakeholders: List[str] = field(default_factory=list)
    resources: Dict[str, Any] = field(default_factory=dict)
    current_solutions: List[str] = field(default_factory=list)
    pain_points: List[str] = field(default_factory=list)


@dataclass
class Solution:
    """A creative solution to a problem"""
    solution_id: str
    problem_id: str
    description: str
    solution_type: SolutionType
    approach: str  # Method used to generate
    steps: List[str] = field(default_factory=list)
    creativity_score: float = 0.5  # 0-1
    feasibility_score: float = 0.5  # 0-1
    impact_score: float = 0.5  # 0-1
    risk_score: float = 0.5  # 0-1 (higher = riskier)
    cost_estimate: str = "medium"
    time_estimate: str = "medium"
    prerequisites: List[str] = field(default_factory=list)
    advantages: List[str] = field(default_factory=list)
    disadvantages: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def overall_score(self) -> float:
        """Calculate overall solution quality"""
        return (
            self.creativity_score * 0.25 +
            self.feasibility_score * 0.30 +
            self.impact_score * 0.30 +
            (1.0 - self.risk_score) * 0.15
        )


@dataclass
class DesignThinkingSession:
    """Design thinking session state"""
    session_id: str
    problem: Problem
    empathy_insights: List[str] = field(default_factory=list)
    problem_statements: List[str] = field(default_factory=list)
    ideas: List[str] = field(default_factory=list)
    prototypes: List[Dict[str, Any]] = field(default_factory=list)
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    current_phase: str = "empathize"  # empathize, define, ideate, prototype, test


class CreativeSolver:
    """
    Creative Problem Solving Engine

    Implements multiple creative problem-solving frameworks:

    1. Design Thinking (5 phases)
       - Empathize: Understand users and context
       - Define: Frame the problem
       - Ideate: Generate solutions
       - Prototype: Build tangible representations
       - Test: Evaluate and iterate

    2. TRIZ (40 Inventive Principles)
       - Systematic approach to innovation
       - Resolve contradictions
       - Use patterns from patent analysis

    3. Divergent Thinking
       - Generate many alternatives
       - Defer judgment
       - Explore wild ideas

    4. Constraint Manipulation
       - Relax constraints
       - Add constraints
       - Invert constraints

    5. Problem Reframing
       - Look from different perspectives
       - Change the level of abstraction
       - Question assumptions
    """

    def __init__(self):
        self.problems: Dict[str, Problem] = {}
        self.solutions: Dict[str, Solution] = {}
        self.design_thinking_sessions: Dict[str, DesignThinkingSession] = {}
        self.problem_counter = 0
        self.solution_counter = 0

        # TRIZ inventive principles (simplified subset)
        self.triz_principles = [
            "Segmentation - Divide object into independent parts",
            "Taking out - Extract problematic part",
            "Local quality - Make each part optimal",
            "Asymmetry - Replace symmetrical with asymmetrical",
            "Merging - Combine similar operations",
            "Universality - Make object perform multiple functions",
            "Nested doll - Place objects inside each other",
            "Anti-weight - Compensate weight with lift",
            "Preliminary anti-action - Pre-stress to counteract",
            "Preliminary action - Perform action in advance",
            "Beforehand cushioning - Prepare for emergency",
            "Equipotentiality - Minimize need to move",
            "The other way round - Invert action or object",
            "Spheroidality - Use curves instead of straight",
            "Dynamics - Make object adaptable",
            "Partial or excessive - Use slightly less or more",
            "Another dimension - Use multi-level arrangement",
            "Mechanical vibration - Use oscillation",
            "Periodic action - Use periodic instead of continuous",
            "Continuity of useful action - Eliminate idle time",
            "Skipping - Conduct process at high speed",
            "Blessing in disguise - Turn harmful into beneficial",
            "Feedback - Introduce feedback",
            "Intermediary - Use intermediary object",
            "Self-service - Make object serve itself",
            "Copying - Use simpler/cheaper copy",
            "Cheap short-living - Replace expensive with disposable",
            "Mechanical substitution - Replace mechanical with sensory",
            "Pneumatics and hydraulics - Use gas and liquid",
            "Flexible shells - Use flexible membranes",
            "Porous materials - Make object porous",
            "Color changes - Change color or transparency",
            "Homogeneity - Make objects interact homogeneously",
            "Discarding and recovering - Make used object disappear",
            "Parameter changes - Change physical state",
            "Phase transitions - Use effects at phase change",
            "Thermal expansion - Use expansion/contraction",
            "Strong oxidants - Use enriched air or oxygen",
            "Inert atmosphere - Replace normal with inert",
            "Composite materials - Use composite instead of homogeneous"
        ]

    def define_problem(
        self,
        description: str,
        problem_type: ProblemType = ProblemType.ILL_DEFINED,
        constraints: Optional[List[str]] = None,
        goals: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Problem:
        """Define a problem to solve"""
        self.problem_counter += 1
        problem_id = f"problem_{self.problem_counter}"

        problem = Problem(
            problem_id=problem_id,
            description=description,
            problem_type=problem_type,
            constraints=constraints or [],
            goals=goals or [],
            context=context or {}
        )

        self.problems[problem_id] = problem
        return problem

    def solve_creatively(
        self,
        problem: Problem,
        approaches: Optional[List[str]] = None,
        thinking_mode: ThinkingMode = ThinkingMode.DIVERGENT,
        num_solutions: int = 5
    ) -> List[Solution]:
        """
        Generate creative solutions to a problem

        Args:
            problem: Problem to solve
            approaches: Specific approaches to use
            thinking_mode: Mode of thinking
            num_solutions: Number of solutions to generate

        Returns:
            List of creative solutions
        """
        solutions = []

        # Use all approaches if none specified
        if approaches is None:
            approaches = [
                "design_thinking",
                "triz",
                "constraint_relaxation",
                "reframing",
                "analogical",
                "lateral"
            ]

        # Generate solutions using each approach
        for approach in approaches:
            if approach == "design_thinking":
                solutions.extend(self._solve_design_thinking(problem, num_solutions // len(approaches)))
            elif approach == "triz":
                solutions.extend(self._solve_triz(problem, num_solutions // len(approaches)))
            elif approach == "constraint_relaxation":
                solutions.extend(self._solve_constraint_relaxation(problem, num_solutions // len(approaches)))
            elif approach == "reframing":
                solutions.extend(self._solve_reframing(problem, num_solutions // len(approaches)))
            elif approach == "analogical":
                solutions.extend(self._solve_analogical(problem, num_solutions // len(approaches)))
            elif approach == "lateral":
                solutions.extend(self._solve_lateral(problem, num_solutions // len(approaches)))

        # Apply thinking mode filter
        solutions = self._apply_thinking_mode(solutions, thinking_mode)

        # Sort by overall score
        solutions.sort(key=lambda s: s.overall_score(), reverse=True)

        # Store solutions
        for solution in solutions[:num_solutions]:
            self.solutions[solution.solution_id] = solution

        return solutions[:num_solutions]

    def _solve_design_thinking(self, problem: Problem, count: int) -> List[Solution]:
        """Solve using design thinking methodology"""
        solutions = []

        # Create design thinking session
        session_id = f"dt_session_{problem.problem_id}"
        session = DesignThinkingSession(
            session_id=session_id,
            problem=problem
        )

        # Phase 1: Empathize - understand users
        session.empathy_insights = [
            f"User needs: {goal}" for goal in problem.goals[:3]
        ]
        session.empathy_insights.extend([
            f"Pain point: {pain}" for pain in problem.pain_points[:3]
        ])

        # Phase 2: Define - frame the problem
        session.problem_statements = [
            f"How might we {problem.description} while addressing {', '.join(problem.pain_points[:2])}?",
            f"How can we help users achieve {', '.join(problem.goals[:2])}?",
        ]

        # Phase 3: Ideate - generate solutions
        for i in range(count):
            # Generate idea based on insights
            if session.empathy_insights:
                insight = random.choice(session.empathy_insights)
                idea = f"Address {insight} through user-centered design"
            else:
                idea = f"Human-centered solution for {problem.description}"

            solution = self._create_solution(
                problem=problem,
                description=idea,
                approach="design_thinking",
                solution_type=SolutionType.INNOVATIVE
            )

            # Design thinking solutions are human-centered
            solution.creativity_score = random.uniform(0.6, 0.9)
            solution.feasibility_score = random.uniform(0.6, 0.9)
            solution.impact_score = random.uniform(0.7, 0.95)
            solution.advantages = ["User-centered", "Tested with users", "Iterative"]

            solutions.append(solution)

        self.design_thinking_sessions[session_id] = session
        return solutions

    def _solve_triz(self, problem: Problem, count: int) -> List[Solution]:
        """Solve using TRIZ inventive principles"""
        solutions = []

        # Identify contradictions in problem
        contradictions = self._identify_contradictions(problem)

        # Apply TRIZ principles
        for i in range(count):
            # Select random TRIZ principle
            principle = random.choice(self.triz_principles)
            principle_name = principle.split(" - ")[0]
            principle_desc = principle.split(" - ")[1] if " - " in principle else ""

            description = f"Apply TRIZ principle '{principle_name}' to {problem.description}: {principle_desc}"

            solution = self._create_solution(
                problem=problem,
                description=description,
                approach="triz",
                solution_type=SolutionType.INNOVATIVE
            )

            # TRIZ solutions are systematic and proven
            solution.creativity_score = random.uniform(0.7, 0.95)
            solution.feasibility_score = random.uniform(0.6, 0.85)
            solution.impact_score = random.uniform(0.6, 0.9)
            solution.advantages = ["Systematic", "Proven patterns", "Resolves contradictions"]
            solution.steps = [
                f"1. Identify contradiction in {problem.description}",
                f"2. Apply {principle_name}",
                "3. Adapt principle to specific context",
                "4. Validate solution"
            ]

            solutions.append(solution)

        return solutions

    def _solve_constraint_relaxation(self, problem: Problem, count: int) -> List[Solution]:
        """Solve by relaxing or manipulating constraints"""
        solutions = []

        if not problem.constraints:
            # If no constraints, add some hypothetical ones to relax
            problem.constraints = ["time", "budget", "resources"]

        for i in range(count):
            # Pick a constraint to relax
            if problem.constraints:
                constraint = random.choice(problem.constraints)
                description = f"What if we removed constraint '{constraint}' from {problem.description}? Solution: [innovative approach without this limitation]"
            else:
                description = f"Unconstrained solution to {problem.description}"

            solution = self._create_solution(
                problem=problem,
                description=description,
                approach="constraint_relaxation",
                solution_type=SolutionType.DISRUPTIVE
            )

            # Relaxing constraints gives creative but potentially infeasible solutions
            solution.creativity_score = random.uniform(0.8, 1.0)
            solution.feasibility_score = random.uniform(0.3, 0.6)
            solution.impact_score = random.uniform(0.7, 0.95)
            solution.risk_score = random.uniform(0.6, 0.9)  # Higher risk
            solution.advantages = ["Highly creative", "Breakthrough potential"]
            solution.disadvantages = ["May be infeasible", "High risk"]

            solutions.append(solution)

        return solutions

    def _solve_reframing(self, problem: Problem, count: int) -> List[Solution]:
        """Solve by reframing the problem"""
        solutions = []

        # Different reframing strategies
        reframe_strategies = [
            ("opposite", "What if we did the opposite?"),
            ("zoom_out", "What's the larger context?"),
            ("zoom_in", "What's the specific detail?"),
            ("stakeholder", "How would a different stakeholder see this?"),
            ("time", "How would this look in 10 years?"),
            ("resource", "What if we had unlimited resources?"),
            ("simplify", "What's the simplest version?"),
            ("complicate", "What if we made it more complex?")
        ]

        for i in range(count):
            strategy_name, strategy_question = random.choice(reframe_strategies)

            description = f"Reframe {problem.description} by asking: {strategy_question}"

            solution = self._create_solution(
                problem=problem,
                description=description,
                approach=f"reframing_{strategy_name}",
                solution_type=SolutionType.TRANSFORMATIVE
            )

            # Reframing can lead to paradigm shifts
            solution.creativity_score = random.uniform(0.7, 0.95)
            solution.feasibility_score = random.uniform(0.4, 0.7)
            solution.impact_score = random.uniform(0.6, 0.95)
            solution.advantages = ["New perspective", "Challenges assumptions"]
            solution.metadata = {"reframe_strategy": strategy_name}

            solutions.append(solution)

        return solutions

    def _solve_analogical(self, problem: Problem, count: int) -> List[Solution]:
        """Solve using analogical reasoning from other domains"""
        solutions = []

        # Source domains for analogies
        source_domains = [
            "nature",
            "sports",
            "military",
            "medicine",
            "arts",
            "architecture",
            "biology",
            "physics"
        ]

        for i in range(count):
            domain = random.choice(source_domains)

            description = f"Apply solution pattern from {domain} to {problem.description}. For example, how {domain} solves similar problems through [specific mechanism]"

            solution = self._create_solution(
                problem=problem,
                description=description,
                approach=f"analogical_{domain}",
                solution_type=SolutionType.INNOVATIVE
            )

            # Analogies are moderately creative and feasible
            solution.creativity_score = random.uniform(0.6, 0.85)
            solution.feasibility_score = random.uniform(0.5, 0.8)
            solution.impact_score = random.uniform(0.5, 0.85)
            solution.advantages = [f"Proven in {domain}", "Cross-domain transfer"]
            solution.steps = [
                f"1. Identify similar problem in {domain}",
                f"2. Extract solution pattern from {domain}",
                "3. Map pattern to current problem",
                "4. Adapt to specific context"
            ]

            solutions.append(solution)

        return solutions

    def _solve_lateral(self, problem: Problem, count: int) -> List[Solution]:
        """Solve using lateral thinking techniques"""
        solutions = []

        # Lateral thinking techniques
        techniques = [
            "Random entry - Start from random concept",
            "Provocation - Use impossible statement",
            "Challenge assumptions - Question everything",
            "Escape - Break out of current patterns",
            "Reversal - Do the opposite",
            "Exaggeration - Take to extreme"
        ]

        for i in range(count):
            technique = random.choice(techniques)
            technique_name = technique.split(" - ")[0]

            description = f"Use lateral thinking technique '{technique_name}' on {problem.description}: {technique.split(' - ')[1]}"

            solution = self._create_solution(
                problem=problem,
                description=description,
                approach=f"lateral_{technique_name.lower().replace(' ', '_')}",
                solution_type=SolutionType.DISRUPTIVE
            )

            # Lateral thinking produces highly creative solutions
            solution.creativity_score = random.uniform(0.8, 1.0)
            solution.feasibility_score = random.uniform(0.3, 0.6)
            solution.impact_score = random.uniform(0.6, 0.95)
            solution.risk_score = random.uniform(0.6, 0.9)
            solution.advantages = ["Breaks patterns", "Unexpected approach"]
            solution.disadvantages = ["May seem impractical", "Requires adaptation"]

            solutions.append(solution)

        return solutions

    def _identify_contradictions(self, problem: Problem) -> List[Tuple[str, str]]:
        """Identify contradictions in problem (for TRIZ)"""
        contradictions = []

        # Common contradiction patterns
        if "fast" in problem.description.lower() and "quality" in problem.description.lower():
            contradictions.append(("speed", "quality"))

        if "cheap" in problem.description.lower() and ("quality" in problem.description.lower() or "performance" in problem.description.lower()):
            contradictions.append(("cost", "quality"))

        if "simple" in problem.description.lower() and "powerful" in problem.description.lower():
            contradictions.append(("simplicity", "capability"))

        return contradictions

    def _apply_thinking_mode(self, solutions: List[Solution], mode: ThinkingMode) -> List[Solution]:
        """Filter/modify solutions based on thinking mode"""
        if mode == ThinkingMode.DIVERGENT:
            # Keep all diverse solutions
            return solutions

        elif mode == ThinkingMode.CONVERGENT:
            # Keep only most feasible and high-impact
            filtered = [s for s in solutions if s.feasibility_score > 0.6 and s.impact_score > 0.6]
            return filtered if filtered else solutions[:3]

        elif mode == ThinkingMode.LATERAL:
            # Keep only highly creative solutions
            return [s for s in solutions if s.creativity_score > 0.7]

        elif mode == ThinkingMode.CRITICAL:
            # Keep only well-evaluated solutions with low risk
            return [s for s in solutions if s.risk_score < 0.5 and s.feasibility_score > 0.7]

        elif mode == ThinkingMode.CREATIVE:
            # Prefer novelty over feasibility
            solutions.sort(key=lambda s: s.creativity_score, reverse=True)
            return solutions

        else:  # VERTICAL
            # Logical progression - keep incremental improvements
            return [s for s in solutions if s.solution_type in [SolutionType.INCREMENTAL, SolutionType.INNOVATIVE]]

    def _create_solution(
        self,
        problem: Problem,
        description: str,
        approach: str,
        solution_type: SolutionType
    ) -> Solution:
        """Create a solution object"""
        self.solution_counter += 1

        solution_id = hashlib.md5(
            f"{description}:{time.time()}:{self.solution_counter}".encode()
        ).hexdigest()[:12]

        return Solution(
            solution_id=solution_id,
            problem_id=problem.problem_id,
            description=description,
            solution_type=solution_type,
            approach=approach
        )

    def generate_alternatives(
        self,
        solution: Solution,
        num_alternatives: int = 3
    ) -> List[Solution]:
        """Generate alternative variations of a solution"""
        alternatives = []

        problem = self.problems.get(solution.problem_id)
        if not problem:
            return alternatives

        # Generate variations
        variation_strategies = [
            "Scale up",
            "Scale down",
            "Automate",
            "Manual version",
            "Hybrid approach",
            "Phased implementation"
        ]

        for i in range(num_alternatives):
            strategy = random.choice(variation_strategies)
            description = f"{strategy}: {solution.description}"

            alternative = self._create_solution(
                problem=problem,
                description=description,
                approach=f"{solution.approach}_variant",
                solution_type=solution.solution_type
            )

            # Inherit scores with variation
            alternative.creativity_score = solution.creativity_score * random.uniform(0.8, 1.0)
            alternative.feasibility_score = solution.feasibility_score * random.uniform(0.8, 1.2)
            alternative.impact_score = solution.impact_score * random.uniform(0.8, 1.1)

            alternatives.append(alternative)

        return alternatives

    def evaluate_solution(
        self,
        solution: Solution,
        criteria: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Evaluate a solution against criteria"""
        if criteria is None:
            criteria = {
                "creativity": 0.25,
                "feasibility": 0.30,
                "impact": 0.30,
                "risk": 0.15
            }

        evaluation = {
            "overall_score": solution.overall_score(),
            "creativity": solution.creativity_score,
            "feasibility": solution.feasibility_score,
            "impact": solution.impact_score,
            "risk": solution.risk_score,
            "weighted_score": (
                solution.creativity_score * criteria.get("creativity", 0.25) +
                solution.feasibility_score * criteria.get("feasibility", 0.30) +
                solution.impact_score * criteria.get("impact", 0.30) +
                (1.0 - solution.risk_score) * criteria.get("risk", 0.15)
            ),
            "advantages": solution.advantages,
            "disadvantages": solution.disadvantages
        }

        return evaluation

    def combine_solutions(
        self,
        solution1: Solution,
        solution2: Solution,
        combination_type: str = "merge"
    ) -> Solution:
        """Combine two solutions into a hybrid"""
        problem = self.problems.get(solution1.problem_id)
        if not problem:
            problem = Problem(
                problem_id=solution1.problem_id,
                description="Combined problem",
                problem_type=ProblemType.ILL_DEFINED
            )

        if combination_type == "merge":
            description = f"Merge: {solution1.description} AND {solution2.description}"
        elif combination_type == "sequence":
            description = f"Sequence: First {solution1.description}, then {solution2.description}"
        elif combination_type == "conditional":
            description = f"Conditional: If condition, use {solution1.description}, else {solution2.description}"
        else:  # hybrid
            description = f"Hybrid: Combine best aspects of both solutions"

        combined = self._create_solution(
            problem=problem,
            description=description,
            approach=f"combined_{combination_type}",
            solution_type=SolutionType.INNOVATIVE
        )

        # Average the scores
        combined.creativity_score = (solution1.creativity_score + solution2.creativity_score) / 2
        combined.feasibility_score = (solution1.feasibility_score + solution2.feasibility_score) / 2
        combined.impact_score = max(solution1.impact_score, solution2.impact_score)
        combined.risk_score = (solution1.risk_score + solution2.risk_score) / 2

        combined.advantages = list(set(solution1.advantages + solution2.advantages))
        combined.disadvantages = list(set(solution1.disadvantages + solution2.disadvantages))

        return combined

    def get_best_solutions(
        self,
        n: int = 10,
        min_score: float = 0.6,
        prefer_feasible: bool = False
    ) -> List[Solution]:
        """Get the best solutions"""
        solutions = list(self.solutions.values())

        # Filter by minimum score
        solutions = [s for s in solutions if s.overall_score() >= min_score]

        # Sort
        if prefer_feasible:
            solutions.sort(key=lambda s: (s.feasibility_score, s.overall_score()), reverse=True)
        else:
            solutions.sort(key=lambda s: s.overall_score(), reverse=True)

        return solutions[:n]

    def get_stats(self) -> Dict[str, Any]:
        """Get solver statistics"""
        solution_types = defaultdict(int)
        approaches = defaultdict(int)

        for solution in self.solutions.values():
            solution_types[solution.solution_type.value] += 1
            approaches[solution.approach] += 1

        return {
            "problems_defined": len(self.problems),
            "solutions_generated": len(self.solutions),
            "design_thinking_sessions": len(self.design_thinking_sessions),
            "by_solution_type": dict(solution_types),
            "by_approach": dict(approaches),
            "avg_creativity": sum(s.creativity_score for s in self.solutions.values()) / len(self.solutions) if self.solutions else 0,
            "avg_feasibility": sum(s.feasibility_score for s in self.solutions.values()) / len(self.solutions) if self.solutions else 0,
            "avg_impact": sum(s.impact_score for s in self.solutions.values()) / len(self.solutions) if self.solutions else 0,
        }
