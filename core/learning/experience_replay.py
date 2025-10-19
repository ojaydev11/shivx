"""
Experience Replay - Learn from Every Interaction
=================================================
ShivX learns from successes and failures to improve over time
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


class ExperienceReplay:
    """
    Records and learns from every interaction

    Features:
    - Tracks queries, responses, success rates
    - Identifies patterns in successful vs failed responses
    - Learns which intent detection works best
    - Suggests improvements to the system
    """

    def __init__(self, storage_path: str = "data/learning/experiences.jsonl"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory cache for fast pattern detection
        self.patterns = defaultdict(list)
        self.intent_accuracy = defaultdict(lambda: {"correct": 0, "incorrect": 0})
        self.tool_performance = defaultdict(lambda: {"success": 0, "failure": 0, "avg_time": []})

        # Load existing experiences
        self._load_experiences()

        logger.info(f"Experience Replay initialized (loaded {len(self.patterns)} patterns)")

    def record_interaction(
        self,
        query: str,
        detected_intent: str,
        actual_intent: Optional[str],
        response: Dict[str, Any],
        success: bool,
        user_feedback: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Record a single interaction for learning

        Args:
            query: User's input query
            detected_intent: What ShivX thought the intent was
            actual_intent: What the intent actually was (if corrected)
            response: Full response including answer, tools used, timing
            success: Whether the interaction was successful
            user_feedback: Optional user feedback
            metadata: Additional context
        """
        experience = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "detected_intent": detected_intent,
            "actual_intent": actual_intent or detected_intent,
            "response": response,
            "success": success,
            "user_feedback": user_feedback,
            "metadata": metadata or {},
            "response_time": response.get("response_time", 0),
            "tools_used": response.get("tools_used", []),
        }

        # Append to storage (JSONL format)
        with open(self.storage_path, "a") as f:
            f.write(json.dumps(experience) + "\n")

        # Update in-memory patterns
        self._update_patterns(experience)

        # Update intent accuracy
        intent_correct = (detected_intent == actual_intent)
        self.intent_accuracy[detected_intent]["correct" if intent_correct else "incorrect"] += 1

        # Update tool performance
        for tool in response.get("tools_used", []):
            self.tool_performance[tool]["success" if success else "failure"] += 1
            self.tool_performance[tool]["avg_time"].append(response.get("response_time", 0))

        logger.debug(f"Recorded experience: {query[:50]}... (success={success})")

    def _update_patterns(self, experience: Dict):
        """Extract and store patterns from the experience"""
        query_lower = experience["query"].lower()

        # Extract keywords
        keywords = re.findall(r'\b\w+\b', query_lower)

        # Store pattern with outcome
        pattern_key = experience["detected_intent"]
        self.patterns[pattern_key].append({
            "keywords": keywords[:5],  # Top 5 keywords
            "success": experience["success"],
            "response_time": experience["response_time"],
        })

    def analyze_patterns(self) -> Dict[str, Any]:
        """
        Analyze all experiences to find patterns

        Returns insights about:
        - Most successful intent patterns
        - Common failure modes
        - Performance bottlenecks
        - Suggested improvements
        """
        total_experiences = sum(len(v) for v in self.patterns.values())

        # Intent accuracy analysis
        intent_stats = {}
        for intent, counts in self.intent_accuracy.items():
            total = counts["correct"] + counts["incorrect"]
            accuracy = (counts["correct"] / total * 100) if total > 0 else 0
            intent_stats[intent] = {
                "accuracy": accuracy,
                "total_uses": total,
                "needs_improvement": accuracy < 70,
            }

        # Tool performance analysis
        tool_stats = {}
        for tool, perf in self.tool_performance.items():
            total = perf["success"] + perf["failure"]
            success_rate = (perf["success"] / total * 100) if total > 0 else 0
            avg_time = sum(perf["avg_time"]) / len(perf["avg_time"]) if perf["avg_time"] else 0

            tool_stats[tool] = {
                "success_rate": success_rate,
                "avg_time_ms": avg_time * 1000,
                "total_uses": total,
                "is_slow": avg_time > 5.0,  # Slower than 5 seconds
            }

        # Identify improvement opportunities
        improvements = []

        # Suggest intent detection improvements
        for intent, stats in intent_stats.items():
            if stats["needs_improvement"]:
                improvements.append({
                    "type": "intent_detection",
                    "issue": f"Low accuracy on '{intent}' intent ({stats['accuracy']:.1f}%)",
                    "suggestion": f"Add more pattern matching for '{intent}' queries",
                    "priority": "high" if stats["accuracy"] < 50 else "medium",
                })

        # Suggest tool optimizations
        for tool, stats in tool_stats.items():
            if stats["is_slow"]:
                improvements.append({
                    "type": "performance",
                    "issue": f"Tool '{tool}' is slow ({stats['avg_time_ms']:.0f}ms avg)",
                    "suggestion": f"Optimize '{tool}' or implement caching",
                    "priority": "medium",
                })

            if stats["success_rate"] < 80:
                improvements.append({
                    "type": "reliability",
                    "issue": f"Tool '{tool}' has low success rate ({stats['success_rate']:.1f}%)",
                    "suggestion": f"Add error handling or retry logic for '{tool}'",
                    "priority": "high",
                })

        return {
            "total_experiences": total_experiences,
            "intent_accuracy": intent_stats,
            "tool_performance": tool_stats,
            "improvements": sorted(improvements, key=lambda x: {"high": 0, "medium": 1, "low": 2}[x["priority"]]),
        }

    def suggest_intent_improvements(self, query: str) -> Optional[str]:
        """
        Based on learned patterns, suggest better intent for a query

        Args:
            query: The user's query

        Returns:
            Suggested intent if pattern recognized, None otherwise
        """
        query_lower = query.lower()
        keywords = set(re.findall(r'\b\w+\b', query_lower))

        # Find patterns with similar keywords
        best_match = None
        best_score = 0

        for intent, experiences in self.patterns.items():
            for exp in experiences:
                # Only consider successful experiences
                if not exp["success"]:
                    continue

                # Calculate keyword overlap
                exp_keywords = set(exp["keywords"])
                overlap = len(keywords & exp_keywords)

                if overlap > best_score:
                    best_score = overlap
                    best_match = intent

        # Return suggestion if we have strong confidence (3+ matching keywords)
        if best_score >= 3:
            logger.info(f"Learned pattern suggests intent '{best_match}' for query '{query[:50]}...'")
            return best_match

        return None

    def get_learning_report(self) -> str:
        """Generate a human-readable learning report"""
        analysis = self.analyze_patterns()

        report = [
            "="*70,
            "ShivX LEARNING REPORT",
            "="*70,
            f"\nTotal Experiences: {analysis['total_experiences']}",
            f"\n📊 Intent Detection Accuracy:",
        ]

        for intent, stats in sorted(analysis['intent_accuracy'].items(), key=lambda x: x[1]['accuracy'], reverse=True):
            emoji = "✅" if stats['accuracy'] >= 80 else "⚠️" if stats['accuracy'] >= 60 else "❌"
            report.append(f"  {emoji} {intent}: {stats['accuracy']:.1f}% ({stats['total_uses']} uses)")

        report.append(f"\n🔧 Tool Performance:")
        for tool, stats in sorted(analysis['tool_performance'].items(), key=lambda x: x[1]['success_rate'], reverse=True):
            emoji = "✅" if stats['success_rate'] >= 90 else "⚠️" if stats['success_rate'] >= 70 else "❌"
            report.append(f"  {emoji} {tool}: {stats['success_rate']:.1f}% success, {stats['avg_time_ms']:.0f}ms avg")

        if analysis['improvements']:
            report.append(f"\n💡 Suggested Improvements ({len(analysis['improvements'])}):")
            for i, imp in enumerate(analysis['improvements'][:5], 1):  # Top 5
                priority_emoji = "🔥" if imp['priority'] == 'high' else "📌"
                report.append(f"  {priority_emoji} {imp['issue']}")
                report.append(f"     → {imp['suggestion']}")

        report.append("\n" + "="*70)

        return "\n".join(report)

    def _load_experiences(self):
        """Load existing experiences from storage"""
        if not self.storage_path.exists():
            logger.info("No existing experiences found, starting fresh")
            return

        count = 0
        with open(self.storage_path) as f:
            for line in f:
                try:
                    experience = json.loads(line.strip())
                    self._update_patterns(experience)

                    # Update intent accuracy
                    detected = experience["detected_intent"]
                    actual = experience.get("actual_intent", detected)
                    intent_correct = (detected == actual)
                    self.intent_accuracy[detected]["correct" if intent_correct else "incorrect"] += 1

                    # Update tool performance
                    for tool in experience.get("response", {}).get("tools_used", []):
                        success = experience.get("success", False)
                        self.tool_performance[tool]["success" if success else "failure"] += 1
                        self.tool_performance[tool]["avg_time"].append(
                            experience.get("response_time", 0)
                        )

                    count += 1
                except json.JSONDecodeError:
                    logger.warning(f"Skipped malformed experience line")

        logger.info(f"Loaded {count} historical experiences")


# Singleton instance
_experience_replay = None

def get_experience_replay() -> ExperienceReplay:
    """Get the global ExperienceReplay instance"""
    global _experience_replay
    if _experience_replay is None:
        _experience_replay = ExperienceReplay()
    return _experience_replay
