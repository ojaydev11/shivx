"""Reasoning module for ShivX AGI"""
from .reasoning_engine import ReasoningEngine, ReasoningChain, ReasoningStep, ReasoningType, get_reasoning_engine
from .chain_of_thought import ChainOfThought, get_chain_of_thought
from .parallel_engine import ParallelReasoning, get_parallel_engine
from .multi_agent_debate import MultiAgentDebate, get_multi_agent_debate, AgentRole

__all__ = [
    'ReasoningEngine', 'ReasoningChain', 'ReasoningStep', 'ReasoningType', 'get_reasoning_engine',
    'ChainOfThought', 'get_chain_of_thought',
    'ParallelReasoning', 'get_parallel_engine',
    'MultiAgentDebate', 'get_multi_agent_debate', 'AgentRole'
]



