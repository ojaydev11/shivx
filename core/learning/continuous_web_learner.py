"""
Continuous Web Learning - Always Stay Up-to-Date
=================================================
Continuously learn from the web to keep knowledge current
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)


class ContinuousWebLearner:
    """
    Continuous Web Learning System

    Features:
    - Monitors trending topics and news
    - Learns from high-quality web sources
    - Updates knowledge base automatically
    - Tracks knowledge freshness

    Example Use Cases:
    - "What's trending in AI today?"
    - "Any new Python releases?"
    - "What happened this week in tech?"
    """

    def __init__(self, storage_path: str = "data/learning/web_knowledge.json"):
        self.storage_path = storage_path
        self.knowledge_base = {}  # {topic: {facts, last_updated, sources}}
        self.learning_topics = []  # Topics to track
        self.update_interval = timedelta(hours=24)  # Update every 24 hours
        self.is_running = False
        self._load_knowledge()
        logger.info("[WEB_LEARN] Continuous Web Learning initialized")

    def _load_knowledge(self):
        """Load stored knowledge from disk"""
        try:
            path = Path(self.storage_path)
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    self.knowledge_base = data.get("knowledge_base", {})
                    self.learning_topics = data.get("learning_topics", [])
                logger.debug(f"[WEB_LEARN] Loaded {len(self.knowledge_base)} topics")
        except Exception as e:
            logger.warning(f"[WEB_LEARN] Could not load knowledge: {e}")

    def _save_knowledge(self):
        """Save knowledge to disk"""
        try:
            path = Path(self.storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump({
                    "knowledge_base": self.knowledge_base,
                    "learning_topics": self.learning_topics,
                    "last_save": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"[WEB_LEARN] Could not save knowledge: {e}")

    def add_learning_topic(self, topic: str, priority: str = "medium"):
        """
        Add a topic to continuously learn about

        Args:
            topic: Topic to track (e.g., "artificial intelligence", "python releases")
            priority: high/medium/low - how often to update
        """
        if topic not in self.learning_topics:
            self.learning_topics.append({
                "topic": topic,
                "priority": priority,
                "added": datetime.now().isoformat()
            })
            self._save_knowledge()
            logger.info(f"[WEB_LEARN] Added learning topic: {topic} (priority: {priority})")

    async def learn_from_web(self, query: str, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Learn new knowledge from the web

        Args:
            query: What to learn about
            domain: Optional domain restriction (e.g., "technology", "finance")

        Returns:
            Dict with learned facts, sources, confidence
        """
        logger.info(f"[WEB_LEARN] Learning from web: {query}")

        try:
            # Use web search to gather information
            from core.agents.web_search_multi import search_and_summarize
            from core.agents.llm_client import get_llm_client

            llm = get_llm_client()
            search_result = await search_and_summarize(query, llm)

            if not search_result.get("success") or not search_result.get("answer"):
                logger.warning(f"[WEB_LEARN] Web search failed for: {query}")
                return {
                    "success": False,
                    "error": "Web search unavailable"
                }

            # Extract key facts from the search results
            facts = await self._extract_facts(search_result["answer"], query, llm)

            # Store in knowledge base
            topic_key = query.lower().strip()
            self.knowledge_base[topic_key] = {
                "query": query,
                "facts": facts,
                "sources": [r["url"] for r in search_result.get("results", [])[:5]],
                "last_updated": datetime.now().isoformat(),
                "domain": domain or "general",
                "confidence": 0.8  # Web-sourced knowledge is generally reliable
            }
            self._save_knowledge()

            logger.info(f"[WEB_LEARN] Learned {len(facts)} facts about: {query}")

            return {
                "success": True,
                "query": query,
                "facts": facts,
                "num_sources": len(search_result.get("results", [])),
                "confidence": 0.8
            }

        except Exception as e:
            logger.error(f"[WEB_LEARN] Learning error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _extract_facts(self, content: str, topic: str, llm) -> List[str]:
        """Extract key facts from web content"""
        prompt = f"""Extract the KEY FACTS from this information about "{topic}":

{content[:1000]}

List 3-5 most important facts. Be concise and specific.
Format each fact as: "• [fact]"
"""

        response = await llm.chat(prompt=prompt, temperature=0.3, max_tokens=400)

        # Parse facts
        facts = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                fact = line.lstrip('•-* ').strip()
                if fact and len(fact) > 10:  # Skip very short facts
                    facts.append(fact)

        return facts[:5]  # Limit to 5 facts

    async def get_fresh_knowledge(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get knowledge about a topic, refreshing if stale

        Args:
            query: What to get knowledge about

        Returns:
            Dict with facts, sources, last_updated, or None if not found
        """
        topic_key = query.lower().strip()

        # Check if we have knowledge about this topic
        if topic_key in self.knowledge_base:
            knowledge = self.knowledge_base[topic_key]

            # Check if knowledge is stale
            last_updated = datetime.fromisoformat(knowledge["last_updated"])
            if datetime.now() - last_updated > self.update_interval:
                logger.info(f"[WEB_LEARN] Knowledge stale, refreshing: {query}")
                # Refresh knowledge in background
                asyncio.create_task(self.learn_from_web(query, knowledge.get("domain")))

            return knowledge

        # Don't have knowledge yet
        return None

    async def update_all_topics(self):
        """Update knowledge for all tracked topics"""
        logger.info(f"[WEB_LEARN] Updating {len(self.learning_topics)} tracked topics")

        for topic_info in self.learning_topics:
            topic = topic_info.get("topic") if isinstance(topic_info, dict) else topic_info

            try:
                await self.learn_from_web(topic)
                # Small delay to avoid rate limiting
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"[WEB_LEARN] Failed to update topic {topic}: {e}")

        logger.info("[WEB_LEARN] Topic update complete")

    async def run_continuous_learning(self, interval_hours: int = 24):
        """
        Run continuous learning loop

        Args:
            interval_hours: How often to update knowledge (default 24 hours)
        """
        self.is_running = True
        logger.info(f"[WEB_LEARN] Starting continuous learning loop (every {interval_hours}h)")

        while self.is_running:
            try:
                await self.update_all_topics()
                # Wait before next update
                await asyncio.sleep(interval_hours * 3600)
            except Exception as e:
                logger.error(f"[WEB_LEARN] Continuous learning error: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes on error

    def stop_continuous_learning(self):
        """Stop the continuous learning loop"""
        self.is_running = False
        logger.info("[WEB_LEARN] Stopped continuous learning loop")

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        total_facts = sum(len(k.get("facts", [])) for k in self.knowledge_base.values())
        total_sources = sum(len(k.get("sources", [])) for k in self.knowledge_base.values())

        # Count stale topics
        stale_count = 0
        for knowledge in self.knowledge_base.values():
            last_updated = datetime.fromisoformat(knowledge["last_updated"])
            if datetime.now() - last_updated > self.update_interval:
                stale_count += 1

        return {
            "total_topics": len(self.knowledge_base),
            "tracked_topics": len(self.learning_topics),
            "total_facts": total_facts,
            "total_sources": total_sources,
            "stale_topics": stale_count,
            "is_running": self.is_running
        }

    async def ask_with_web_context(self, query: str) -> Dict[str, Any]:
        """
        Answer a query using web-learned knowledge

        Args:
            query: User's question

        Returns:
            Dict with answer, sources, confidence
        """
        # Check if we have relevant knowledge
        knowledge = await self.get_fresh_knowledge(query)

        if knowledge:
            # Use existing knowledge
            answer = f"**Based on recent web learning:**\n\n"
            for fact in knowledge["facts"]:
                answer += f"• {fact}\n"

            answer += f"\n*Last updated: {knowledge['last_updated'][:10]}*"

            return {
                "answer": answer,
                "sources": knowledge["sources"],
                "confidence": knowledge["confidence"],
                "from_cache": True
            }
        else:
            # Learn from web now
            result = await self.learn_from_web(query)

            if result["success"]:
                answer = f"**I just learned from the web:**\n\n"
                for fact in result["facts"]:
                    answer += f"• {fact}\n"

                return {
                    "answer": answer,
                    "sources": self.knowledge_base.get(query.lower().strip(), {}).get("sources", []),
                    "confidence": result["confidence"],
                    "from_cache": False
                }
            else:
                return {
                    "answer": "I couldn't learn about that topic from the web right now.",
                    "error": result.get("error"),
                    "confidence": 0.0
                }


# Singleton
_continuous_web_learner = None

def get_continuous_web_learner():
    """Get the global Continuous Web Learner instance"""
    global _continuous_web_learner
    if _continuous_web_learner is None:
        _continuous_web_learner = ContinuousWebLearner()
    return _continuous_web_learner
