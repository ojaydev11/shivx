"""
Multi-Agent Debate System
==========================
Multiple AI agents debate before answering to improve accuracy
"""

import asyncio
import logging
from typing import Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Roles for different agents in the debate"""
    PROPOSER = "proposer"  # Proposes initial answer
    CRITIC = "critic"  # Critiques and finds flaws
    SYNTHESIZER = "synthesizer"  # Combines best ideas


class MultiAgentDebate:
    """
    Multi-agent debate system for improved reasoning

    Process:
    1. Agent 1 (Proposer): Generates initial answer
    2. Agent 2 (Critic): Critiques and suggests alternatives
    3. Agent 3 (Synthesizer): Combines insights into best answer

    Result: Higher accuracy through adversarial thinking
    """

    def __init__(self, llm_client):
        self.llm = llm_client
        self.debate_rounds = 1  # Can increase for more thorough debates
        logger.info("[MULTI-AGENT] Debate system initialized")

    async def debate(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """
        Run multi-agent debate on a query

        Args:
            query: User's question
            context: Additional context

        Returns:
            Dict with final_answer, debate_log, confidence, agents_used
        """
        logger.info(f"[DEBATE] Starting multi-agent debate for: {query[:50]}...")

        debate_log = []

        # Round 1: Proposer generates initial answer
        proposal = await self._agent_propose(query, context)
        debate_log.append({
            "agent": "proposer",
            "response": proposal,
            "role": "Initial answer proposal"
        })

        # Round 2: Critic challenges the proposal
        critique = await self._agent_critique(query, proposal, context)
        debate_log.append({
            "agent": "critic",
            "response": critique,
            "role": "Challenge and alternative suggestions"
        })

        # Round 3: Synthesizer combines insights
        synthesis = await self._agent_synthesize(query, proposal, critique, context)
        debate_log.append({
            "agent": "synthesizer",
            "response": synthesis["answer"],
            "role": "Final synthesis"
        })

        return {
            "final_answer": synthesis["answer"],
            "confidence": synthesis["confidence"],
            "debate_log": debate_log,
            "agents_used": 3,
            "rounds": len(debate_log)
        }

    async def _agent_propose(self, query: str, context: Dict) -> str:
        """Agent 1: Propose initial answer"""
        prompt = f"""You are Agent 1 (Proposer). Your job is to provide a clear, direct answer to the question.

Question: {query}

Provide your best answer. Be confident but acknowledge if you're uncertain about anything."""

        response = await self.llm.chat(
            prompt=prompt,
            temperature=0.7,
            max_tokens=500
        )

        logger.debug(f"[PROPOSER] Generated initial answer")
        return response

    async def _agent_critique(self, query: str, proposal: str, context: Dict) -> str:
        """Agent 2: Critique the proposal and suggest alternatives"""
        prompt = f"""You are Agent 2 (Critic). Your job is to find flaws, edge cases, and suggest better alternatives.

Question: {query}

Agent 1's Answer: {proposal}

Your task:
1. Identify any errors, oversights, or weak points in Agent 1's answer
2. Consider edge cases or alternative perspectives
3. Suggest improvements or alternative approaches

Be constructive but thorough. What did Agent 1 miss?"""

        response = await self.llm.chat(
            prompt=prompt,
            temperature=0.8,  # Higher temp for creative criticism
            max_tokens=500
        )

        logger.debug(f"[CRITIC] Generated critique")
        return response

    async def _agent_synthesize(self, query: str, proposal: str, critique: str, context: Dict) -> Dict[str, Any]:
        """Agent 3: Synthesize the best answer from debate"""
        prompt = f"""You are Agent 3 (Synthesizer). Your job is to combine the best insights from both agents.

Question: {query}

Agent 1 (Proposer) said: {proposal}

Agent 2 (Critic) said: {critique}

Your task:
1. Take the best ideas from both agents
2. Address the flaws that Agent 2 identified
3. Provide a comprehensive, accurate final answer
4. Rate your confidence (0.0-1.0)

Format your response as:
ANSWER: [your synthesized answer]
CONFIDENCE: [0.0-1.0]"""

        response = await self.llm.chat(
            prompt=prompt,
            temperature=0.6,  # Balanced temp for synthesis
            max_tokens=600
        )

        # Parse confidence
        confidence = 0.8  # Default
        answer = response

        if "CONFIDENCE:" in response:
            parts = response.split("CONFIDENCE:")
            answer = parts[0].replace("ANSWER:", "").strip()
            try:
                confidence = float(parts[1].strip())
            except:
                confidence = 0.8

        logger.debug(f"[SYNTHESIZER] Final answer synthesized (confidence: {confidence})")

        return {
            "answer": answer,
            "confidence": confidence
        }


# Singleton
_multi_agent_debate = None

def get_multi_agent_debate(llm_client):
    """Get the global Multi-Agent Debate instance"""
    global _multi_agent_debate
    if _multi_agent_debate is None:
        _multi_agent_debate = MultiAgentDebate(llm_client)
    return _multi_agent_debate
