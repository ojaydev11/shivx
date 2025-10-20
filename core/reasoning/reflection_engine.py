"""
Reflection Engine - Self-correction and learning from failures.

This module enables ShivX to:
- Analyze why failures happened (root cause analysis)
- Generate alternative strategies when something fails
- Learn patterns from mistakes to avoid repeating them
- Improve decision-making over time

Part of ShivX 9/10 Agentic AI transformation (Phase 1).
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from core.autonomy.models import ReflectionResult, FixStrategy

logger = logging.getLogger(__name__)


class ReflectionEngine:
    """
    Analyzes failures and generates corrective strategies.

    This is the core of self-improvement - when something goes wrong,
    the system reflects on WHY and HOW to fix it, rather than just retrying.
    """

    def __init__(self, llm_client=None, brain=None):
        """
        Initialize Reflection Engine.

        Args:
            llm_client: LLM for deep failure analysis
            brain: Brain for storing learned error patterns
        """
        self.llm = llm_client
        self.brain = brain

        # Cache of known error patterns
        self.error_patterns: Dict[str, List[Dict]] = {}

        # Statistics
        self.total_reflections = 0
        self.successful_fixes = 0

    async def reflect_on_failure(
        self,
        action: Dict[str, Any],
        error: str,
        context: Dict[str, Any],
        use_llm: bool = True,
    ) -> ReflectionResult:
        """
        Deep analysis of why something failed.

        This is where the "thinking about thinking" happens - meta-cognition.

        Args:
            action: What was attempted (tool, params, etc.)
            error: Error message or description
            context: Surrounding context (previous steps, system state, etc.)
            use_llm: Whether to use LLM for deep analysis

        Returns:
            ReflectionResult with root cause and alternatives
        """
        self.total_reflections += 1

        logger.info(f"🤔 Reflecting on failure: {error[:100]}...")

        # First, check if we've seen this error pattern before
        known_pattern = await self._check_known_patterns(error, action, context)
        if known_pattern:
            logger.info(f"✓ Found known error pattern: {known_pattern['category']}")
            return self._pattern_to_reflection(known_pattern)

        # New error - perform deep analysis
        if use_llm and self.llm:
            reflection = await self._llm_reflect(action, error, context)
        else:
            reflection = await self._rule_based_reflect(action, error, context)

        # Learn this pattern for future
        await self.learn_from_reflection(reflection)

        return reflection

    async def _check_known_patterns(
        self,
        error: str,
        action: Dict,
        context: Dict
    ) -> Optional[Dict]:
        """Check if this error matches a known pattern"""

        # Load error patterns from brain if not cached
        if not self.error_patterns and self.brain:
            await self._load_error_patterns()

        # Match against known patterns
        for pattern_key, patterns in self.error_patterns.items():
            for pattern in patterns:
                if self._matches_pattern(error, action, pattern):
                    return pattern

        return None

    def _matches_pattern(
        self,
        error: str,
        action: Dict,
        pattern: Dict
    ) -> bool:
        """Check if error matches a learned pattern"""

        # Simple keyword matching for now
        # TODO: Use embeddings for semantic matching
        error_keywords = pattern.get("error_keywords", [])
        if any(kw.lower() in error.lower() for kw in error_keywords):
            # Check if action type also matches
            if action.get("tool") == pattern.get("tool"):
                return True

        return False

    def _pattern_to_reflection(self, pattern: Dict) -> ReflectionResult:
        """Convert known pattern to reflection result"""
        return ReflectionResult(
            root_cause=pattern.get("root_cause", "Known error pattern"),
            error_category=pattern.get("category", "unknown"),
            wrong_assumptions=pattern.get("wrong_assumptions", []),
            missing_information=pattern.get("missing_information", []),
            alternatives=pattern.get("alternatives", []),
            prevention_strategies=pattern.get("prevention_strategies", []),
            learned_patterns=[pattern],
            confidence=pattern.get("confidence", 0.7),
            analysis_method="pattern_match",
        )

    async def _llm_reflect(
        self,
        action: Dict,
        error: str,
        context: Dict
    ) -> ReflectionResult:
        """Use LLM for deep failure analysis"""

        prompt = f"""Analyze this failure and provide a detailed reflection:

**Action Attempted:**
```json
{json.dumps(action, indent=2)}
```

**Error Encountered:**
{error}

**Context:**
```json
{json.dumps(context, indent=2)}
```

Please provide a thorough analysis:

1. **Root Cause**: What fundamentally went wrong? (1-2 sentences)

2. **Error Category**: Classify this error
   - tool_failure: The tool didn't work as expected
   - planning_error: The plan was flawed
   - resource_constraint: Insufficient resources (memory, permissions, etc.)
   - external_dependency: External service failed
   - input_error: Invalid input or parameters
   - assumption_error: Wrong assumption about system state

3. **Wrong Assumptions**: What assumptions were incorrect?

4. **Missing Information**: What information would have prevented this?

5. **Alternative Approaches**: What are 2-3 different ways to achieve the goal?

6. **Prevention Strategies**: How can we avoid this in the future?

Return a JSON object with these fields.
"""

        try:
            response = await self.llm.chat(prompt, temperature=0.3)

            # Parse LLM response
            analysis = json.loads(response)

            return ReflectionResult(
                root_cause=analysis.get("root_cause", "Unknown"),
                error_category=analysis.get("error_category", "unknown"),
                wrong_assumptions=analysis.get("wrong_assumptions", []),
                missing_information=analysis.get("missing_information", []),
                alternatives=analysis.get("alternative_approaches", []),
                prevention_strategies=analysis.get("prevention_strategies", []),
                learned_patterns=[],
                confidence=0.8,  # High confidence in LLM analysis
                analysis_method="llm_reflection",
            )

        except Exception as e:
            logger.error(f"LLM reflection failed: {e}")
            # Fall back to rule-based
            return await self._rule_based_reflect(action, error, context)

    async def _rule_based_reflect(
        self,
        action: Dict,
        error: str,
        context: Dict
    ) -> ReflectionResult:
        """Simple rule-based reflection when LLM is unavailable"""

        # Categorize error based on keywords
        error_lower = error.lower()

        if "permission" in error_lower or "access denied" in error_lower:
            category = "resource_constraint"
            root_cause = "Insufficient permissions to perform action"
            alternatives = [
                "Request elevated permissions",
                "Use alternative tool that doesn't require permissions",
                "Modify approach to work within permission constraints"
            ]
        elif "not found" in error_lower or "404" in error_lower:
            category = "input_error"
            root_cause = "Resource or file does not exist"
            alternatives = [
                "Verify the resource exists before accessing",
                "Create the resource if it should exist",
                "Use a different resource"
            ]
        elif "timeout" in error_lower:
            category = "external_dependency"
            root_cause = "Operation took too long to complete"
            alternatives = [
                "Increase timeout duration",
                "Break operation into smaller chunks",
                "Use asynchronous processing"
            ]
        elif "memory" in error_lower or "oom" in error_lower:
            category = "resource_constraint"
            root_cause = "Insufficient memory to complete operation"
            alternatives = [
                "Process data in smaller batches",
                "Optimize memory usage",
                "Increase available memory"
            ]
        else:
            category = "tool_failure"
            root_cause = "Tool execution failed"
            alternatives = [
                "Try a different tool",
                "Modify tool parameters",
                "Check tool availability and configuration"
            ]

        return ReflectionResult(
            root_cause=root_cause,
            error_category=category,
            wrong_assumptions=[f"Assumed {action.get('tool', 'operation')} would succeed"],
            missing_information=["Error details", "System state at time of failure"],
            alternatives=alternatives,
            prevention_strategies=[
                f"Add pre-check for {category}",
                "Add error handling for this case"
            ],
            learned_patterns=[],
            confidence=0.5,  # Lower confidence for rule-based
            analysis_method="rule_based",
        )

    async def generate_fix_strategies(
        self,
        reflection: ReflectionResult,
        max_strategies: int = 3
    ) -> List[FixStrategy]:
        """
        Convert reflection into concrete fix strategies.

        Args:
            reflection: Analysis of what went wrong
            max_strategies: Maximum number of strategies to generate

        Returns:
            List of FixStrategy objects, sorted by score (best first)
        """
        logger.info(f"Generating fix strategies for {reflection.error_category}...")

        strategies = []

        # Generate strategy for each alternative approach
        for i, alternative in enumerate(reflection.alternatives[:max_strategies]):
            # Estimate success probability based on reflection confidence
            estimated_success = reflection.confidence * 0.9  # Slightly conservative

            # Estimate cost (higher for later alternatives)
            estimated_cost = 1.0 + (i * 0.5)

            # Assess risk
            risk_level = self._assess_risk(reflection.error_category, alternative)

            # Generate concrete steps
            steps = await self._alternative_to_steps(alternative, reflection)

            strategy = FixStrategy(
                approach=alternative,
                steps=steps,
                estimated_success=estimated_success,
                estimated_cost=estimated_cost,
                risk_level=risk_level,
            )

            strategies.append(strategy)

        # Sort by score (best first)
        strategies.sort(key=lambda s: s.score(), reverse=True)

        logger.info(f"Generated {len(strategies)} fix strategies")

        return strategies

    def _assess_risk(self, error_category: str, alternative: str) -> str:
        """Assess risk level of a fix strategy"""

        # High-risk indicators
        if any(word in alternative.lower() for word in ["delete", "drop", "remove", "reset"]):
            return "high"

        # Medium-risk indicators
        if any(word in alternative.lower() for word in ["modify", "change", "update"]):
            return "medium"

        # Low-risk by default (read-only, retry, etc.)
        return "low"

    async def _alternative_to_steps(
        self,
        alternative: str,
        reflection: ReflectionResult
    ) -> List[Dict[str, Any]]:
        """Convert alternative approach into concrete steps"""

        # Use LLM to generate steps if available
        if self.llm:
            try:
                prompt = f"""Convert this alternative approach into concrete steps:

Alternative: {alternative}
Context: {reflection.root_cause}

Generate 2-5 specific, actionable steps.
Return JSON array of steps, each with:
- action: What to do
- tool: Which tool to use (if applicable)
- params: Parameters for the tool

Example:
[
  {{"action": "Check file exists", "tool": "read", "params": {{"path": "/path/to/file"}}}},
  {{"action": "Create file if missing", "tool": "write", "params": {{"path": "/path/to/file", "content": ""}}}}
]
"""
                response = await self.llm.chat(prompt, temperature=0.2)
                steps = json.loads(response)
                return steps if isinstance(steps, list) else []

            except Exception as e:
                logger.warning(f"LLM step generation failed: {e}")

        # Fallback: Create generic step
        return [
            {
                "action": alternative,
                "tool": "auto",  # Auto-select tool
                "params": {}
            }
        ]

    async def learn_from_reflection(self, reflection: ReflectionResult):
        """
        Store reflection as a learned pattern.

        This is how the system improves over time - by remembering failures.
        """
        if not self.brain:
            logger.warning("No brain available for learning")
            return

        try:
            # Store as error pattern
            pattern_data = {
                "root_cause": reflection.root_cause,
                "category": reflection.error_category,
                "alternatives": reflection.alternatives,
                "prevention_strategies": reflection.prevention_strategies,
                "wrong_assumptions": reflection.wrong_assumptions,
                "confidence": reflection.confidence,
                "learned_at": datetime.utcnow().isoformat(),
            }

            await self.brain.learn_pattern(
                pattern_type="error_pattern",
                pattern_data=pattern_data,
                confidence=reflection.confidence,
            )

            # Update cache
            category = reflection.error_category
            if category not in self.error_patterns:
                self.error_patterns[category] = []
            self.error_patterns[category].append(pattern_data)

            logger.info(f"✓ Learned error pattern: {reflection.error_category}")

        except Exception as e:
            logger.error(f"Failed to learn from reflection: {e}")

    async def _load_error_patterns(self):
        """Load error patterns from brain into cache"""
        if not self.brain:
            return

        try:
            # Query brain for error patterns
            patterns = await self.brain.get_patterns_by_type("error_pattern")

            for pattern in patterns:
                category = pattern.get("category", "unknown")
                if category not in self.error_patterns:
                    self.error_patterns[category] = []
                self.error_patterns[category].append(pattern)

            logger.info(f"Loaded {len(patterns)} error patterns from brain")

        except Exception as e:
            logger.error(f"Failed to load error patterns: {e}")

    def record_fix_success(self, fix_strategy: FixStrategy, success: bool):
        """
        Record whether a fix strategy worked.

        This feedback loop improves future fix generation.
        """
        if success:
            self.successful_fixes += 1
            logger.info(f"✅ Fix successful: {fix_strategy.approach}")
        else:
            logger.warning(f"❌ Fix failed: {fix_strategy.approach}")

        # TODO: Update strategy success rates in brain
        # This enables meta-learning about which fixes work best

    def get_stats(self) -> Dict[str, Any]:
        """Get reflection statistics"""
        success_rate = (
            self.successful_fixes / max(self.total_reflections, 1)
        )

        return {
            "total_reflections": self.total_reflections,
            "successful_fixes": self.successful_fixes,
            "success_rate": success_rate,
            "known_patterns": sum(len(p) for p in self.error_patterns.values()),
            "pattern_categories": list(self.error_patterns.keys()),
        }


# Convenience function for quick reflection
async def quick_reflect(
    error: str,
    action: Dict[str, Any],
    llm_client=None
) -> ReflectionResult:
    """
    Quick reflection without full engine setup.

    Useful for one-off error analysis.
    """
    engine = ReflectionEngine(llm_client=llm_client)
    return await engine.reflect_on_failure(
        action=action,
        error=error,
        context={},
        use_llm=llm_client is not None
    )
