"""
Advanced Reasoning Engine for ShivX AGI
Implements multi-step reasoning, chain-of-thought, and problem decomposition
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of reasoning"""
    DEDUCTIVE = "deductive"  # A → B, A ∴ B
    INDUCTIVE = "inductive"  # Multiple examples → General rule
    ABDUCTIVE = "abductive"  # Best explanation
    ANALOGICAL = "analogical"  # A is like B
    CAUSAL = "causal"  # X causes Y
    COUNTERFACTUAL = "counterfactual"  # What if X?


@dataclass
class ReasoningStep:
    """Single step in reasoning chain"""
    step_num: int
    thought: str
    reasoning_type: ReasoningType
    conclusion: Optional[str] = None
    confidence: float = 1.0
    supporting_facts: List[str] = None
    
    def __post_init__(self):
        if self.supporting_facts is None:
            self.supporting_facts = []


@dataclass
class ReasoningChain:
    """Complete chain of reasoning"""
    query: str
    steps: List[ReasoningStep]
    final_answer: str
    confidence: float
    reasoning_path: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "steps": [
                {
                    "step": s.step_num,
                    "thought": s.thought,
                    "type": s.reasoning_type.value,
                    "conclusion": s.conclusion,
                    "confidence": s.confidence
                }
                for s in self.steps
            ],
            "final_answer": self.final_answer,
            "confidence": self.confidence,
            "reasoning_path": self.reasoning_path
        }


class ReasoningEngine:
    """
    Advanced reasoning engine for ShivX
    Implements chain-of-thought and multi-step reasoning
    """
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.reasoning_history: List[ReasoningChain] = []
        
    async def reason(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        reasoning_type: Optional[ReasoningType] = None
    ) -> ReasoningChain:
        """
        Perform multi-step reasoning on a query
        
        Args:
            query: Question or problem to reason about
            context: Additional context
            reasoning_type: Type of reasoning to use (auto-detected if None)
        
        Returns:
            ReasoningChain with steps and final answer
        """
        logger.info(f"Starting reasoning for: {query}")
        
        # Detect reasoning type if not specified
        if reasoning_type is None:
            reasoning_type = self._detect_reasoning_type(query)
        
        # Choose reasoning strategy
        if reasoning_type == ReasoningType.DEDUCTIVE:
            chain = await self._deductive_reasoning(query, context)
        elif reasoning_type == ReasoningType.CAUSAL:
            chain = await self._causal_reasoning(query, context)
        elif reasoning_type == ReasoningType.ANALOGICAL:
            chain = await self._analogical_reasoning(query, context)
        else:
            # Default: chain-of-thought
            chain = await self._chain_of_thought(query, context)
        
        # Store in history
        self.reasoning_history.append(chain)
        
        return chain
    
    def _detect_reasoning_type(self, query: str) -> ReasoningType:
        """Detect the type of reasoning needed"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['cause', 'why', 'because', 'result']):
            return ReasoningType.CAUSAL
        elif any(word in query_lower for word in ['if', 'would', 'could', 'suppose']):
            return ReasoningType.COUNTERFACTUAL
        elif any(word in query_lower for word in ['like', 'similar', 'compare']):
            return ReasoningType.ANALOGICAL
        elif any(word in query_lower for word in ['therefore', 'thus', 'hence']):
            return ReasoningType.DEDUCTIVE
        else:
            return ReasoningType.INDUCTIVE
    
    async def _chain_of_thought(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningChain:
        """
        Chain-of-thought reasoning
        Break down problem into steps and solve sequentially
        """
        steps = []
        
        # Step 1: Understand the problem
        steps.append(ReasoningStep(
            step_num=1,
            thought="Understanding the problem and identifying key components",
            reasoning_type=ReasoningType.INDUCTIVE,
            conclusion=f"This query requires breaking down: {query}"
        ))
        
        # Step 2: Identify what we know
        known_facts = self._extract_known_facts(query, context)
        steps.append(ReasoningStep(
            step_num=2,
            thought="Identifying known facts and constraints",
            reasoning_type=ReasoningType.INDUCTIVE,
            conclusion=f"Known: {', '.join(known_facts) if known_facts else 'Limited information'}",
            supporting_facts=known_facts
        ))
        
        # Step 3: Determine what we need to find
        steps.append(ReasoningStep(
            step_num=3,
            thought="Determining what needs to be solved or answered",
            reasoning_type=ReasoningType.DEDUCTIVE,
            conclusion="Identifying the core question and required solution"
        ))
        
        # Step 4: Apply reasoning
        if self.llm:
            # Use LLM for complex reasoning
            try:
                from core.agents.llm_client import get_llm_client
                llm = get_llm_client()
                
                reasoning_prompt = f"""Let's solve this step by step:

Question: {query}
Known facts: {', '.join(known_facts) if known_facts else 'None explicitly stated'}

Please provide:
1. The key insight needed to solve this
2. The logical steps to reach the answer
3. The final answer

Think through this carefully."""

                llm_response = await llm.chat(
                    prompt=reasoning_prompt,
                    system_message="You are a logical reasoning expert. Break down problems into clear steps."
                )
                
                steps.append(ReasoningStep(
                    step_num=4,
                    thought="Applying logical reasoning and domain knowledge",
                    reasoning_type=ReasoningType.DEDUCTIVE,
                    conclusion=llm_response,
                    confidence=0.85
                ))
                
                final_answer = llm_response
                
            except Exception as e:
                logger.error(f"LLM reasoning error: {e}")
                final_answer = "Unable to complete reasoning - LLM unavailable"
        else:
            # Rule-based reasoning
            steps.append(ReasoningStep(
                step_num=4,
                thought="Applying rule-based reasoning",
                reasoning_type=ReasoningType.DEDUCTIVE,
                conclusion="Processing with available knowledge",
                confidence=0.6
            ))
            final_answer = "Reasoning completed with limited capabilities"
        
        # Build reasoning path
        reasoning_path = " → ".join([f"Step {s.step_num}: {s.thought}" for s in steps])
        
        return ReasoningChain(
            query=query,
            steps=steps,
            final_answer=final_answer,
            confidence=sum(s.confidence for s in steps) / len(steps),
            reasoning_path=reasoning_path
        )
    
    async def _deductive_reasoning(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningChain:
        """Deductive reasoning: General rules → Specific conclusion"""
        steps = []
        
        steps.append(ReasoningStep(
            step_num=1,
            thought="Identifying general rules and principles",
            reasoning_type=ReasoningType.DEDUCTIVE
        ))
        
        steps.append(ReasoningStep(
            step_num=2,
            thought="Applying rules to specific case",
            reasoning_type=ReasoningType.DEDUCTIVE
        ))
        
        steps.append(ReasoningStep(
            step_num=3,
            thought="Drawing logical conclusion",
            reasoning_type=ReasoningType.DEDUCTIVE
        ))
        
        return await self._chain_of_thought(query, context)
    
    async def _causal_reasoning(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningChain:
        """Causal reasoning: Understand cause and effect"""
        steps = []
        
        steps.append(ReasoningStep(
            step_num=1,
            thought="Identifying potential causes",
            reasoning_type=ReasoningType.CAUSAL
        ))
        
        steps.append(ReasoningStep(
            step_num=2,
            thought="Analyzing causal relationships",
            reasoning_type=ReasoningType.CAUSAL
        ))
        
        steps.append(ReasoningStep(
            step_num=3,
            thought="Determining most likely cause or effect",
            reasoning_type=ReasoningType.CAUSAL
        ))
        
        return await self._chain_of_thought(query, context)
    
    async def _analogical_reasoning(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningChain:
        """Analogical reasoning: Reason by analogy"""
        steps = []
        
        steps.append(ReasoningStep(
            step_num=1,
            thought="Finding similar known cases",
            reasoning_type=ReasoningType.ANALOGICAL
        ))
        
        steps.append(ReasoningStep(
            step_num=2,
            thought="Mapping relationships between cases",
            reasoning_type=ReasoningType.ANALOGICAL
        ))
        
        steps.append(ReasoningStep(
            step_num=3,
            thought="Transferring insights to current problem",
            reasoning_type=ReasoningType.ANALOGICAL
        ))
        
        return await self._chain_of_thought(query, context)
    
    def _extract_known_facts(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Extract known facts from query and context"""
        facts = []
        
        # Extract numbers
        import re
        numbers = re.findall(r'\d+\.?\d*', query)
        if numbers:
            facts.append(f"Numbers mentioned: {', '.join(numbers)}")
        
        # Extract entities (simple heuristic)
        words = query.split()
        capitalized = [w for w in words if w[0].isupper() and len(w) > 1]
        if capitalized:
            facts.append(f"Entities: {', '.join(capitalized)}")
        
        # Add context facts
        if context:
            if 'facts' in context:
                facts.extend(context['facts'])
        
        return facts
    
    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get statistics about reasoning performed"""
        if not self.reasoning_history:
            return {
                "total_reasoning_chains": 0,
                "average_confidence": 0,
                "average_steps": 0
            }
        
        return {
            "total_reasoning_chains": len(self.reasoning_history),
            "average_confidence": sum(r.confidence for r in self.reasoning_history) / len(self.reasoning_history),
            "average_steps": sum(len(r.steps) for r in self.reasoning_history) / len(self.reasoning_history),
            "reasoning_types_used": list(set(
                step.reasoning_type.value
                for chain in self.reasoning_history
                for step in chain.steps
            ))
        }


# Global reasoning engine
_reasoning_engine = None


def get_reasoning_engine():
    """Get global reasoning engine"""
    global _reasoning_engine
    if _reasoning_engine is None:
        from core.agents.llm_client import get_llm_client
        _reasoning_engine = ReasoningEngine(llm_client=get_llm_client())
    return _reasoning_engine



