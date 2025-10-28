"""
Reflector for self-audit and hallucination detection.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger


class ReflectionReport:
    """Report from self-reflection."""

    def __init__(self):
        self.timestamp = datetime.utcnow()
        self.outputs_checked = 0
        self.hallucinations_detected = 0
        self.low_confidence_outputs = 0
        self.issues: List[Dict] = []


class Reflector:
    """
    Self-audit system for detecting hallucinations and quality issues.

    Monitors outputs and checks for:
    - Low confidence predictions
    - Inconsistencies with memory
    - Lack of evidence
    """

    def __init__(
        self,
        hallucination_detection: bool = True,
        confidence_threshold: float = 0.8,
        evidence_retrieval: bool = True,
    ):
        """
        Initialize reflector.

        Args:
            hallucination_detection: Enable hallucination detection
            confidence_threshold: Minimum confidence for outputs
            evidence_retrieval: Retrieve evidence from memory
        """
        self.hallucination_detection = hallucination_detection
        self.confidence_threshold = confidence_threshold
        self.evidence_retrieval = evidence_retrieval

        self.reflection_history: List[ReflectionReport] = []

        logger.info(
            f"Reflector initialized: "
            f"hallucination_detection={hallucination_detection}, "
            f"confidence_threshold={confidence_threshold}"
        )

    def check_output(
        self,
        output: str,
        confidence: float,
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Check output for potential issues.

        Args:
            output: Generated output
            confidence: Confidence score [0, 1]
            context: Optional context

        Returns:
            Check results
        """
        issues = []

        # Check confidence
        if confidence < self.confidence_threshold:
            issues.append({
                "type": "low_confidence",
                "confidence": confidence,
                "threshold": self.confidence_threshold,
            })

        # Check for hallucination markers
        if self.hallucination_detection:
            hallucination_score = self._detect_hallucination(output, context)
            if hallucination_score > 0.5:
                issues.append({
                    "type": "potential_hallucination",
                    "score": hallucination_score,
                })

        # Check for evidence
        if self.evidence_retrieval and context:
            has_evidence = self._check_evidence(output, context)
            if not has_evidence:
                issues.append({
                    "type": "lacks_evidence",
                    "output": output[:100],
                })

        result = {
            "output": output,
            "confidence": confidence,
            "issues": issues,
            "safe": len(issues) == 0,
        }

        if issues:
            logger.warning(f"Output has {len(issues)} issues: {output[:50]}...")

        return result

    def _detect_hallucination(
        self, output: str, context: Optional[Dict]
    ) -> float:
        """
        Detect potential hallucination.

        Returns:
            Hallucination score [0, 1]
        """
        # Simplified detection
        # In production, would use more sophisticated checks

        score = 0.0

        # Check for uncertainty phrases
        uncertainty_phrases = [
            "i think",
            "maybe",
            "probably",
            "not sure",
            "might be",
        ]
        for phrase in uncertainty_phrases:
            if phrase in output.lower():
                score += 0.2

        # Check for specific details without context
        if context and "retrieved_facts" in context:
            # If making specific claims without retrieved facts
            if len(context["retrieved_facts"]) == 0 and len(output.split()) > 50:
                score += 0.3

        return min(1.0, score)

    def _check_evidence(self, output: str, context: Dict) -> bool:
        """Check if output is supported by evidence."""
        # Simplified check
        if "retrieved_facts" in context:
            return len(context["retrieved_facts"]) > 0
        return True

    def audit(
        self, outputs: List[Dict[str, Any]]
    ) -> ReflectionReport:
        """
        Audit a batch of outputs.

        Args:
            outputs: List of outputs with confidence scores

        Returns:
            Reflection report
        """
        report = ReflectionReport()

        for item in outputs:
            output = item.get("output", "")
            confidence = item.get("confidence", 1.0)
            context = item.get("context")

            result = self.check_output(output, confidence, context)

            report.outputs_checked += 1

            for issue in result["issues"]:
                report.issues.append({
                    "output": output[:100],
                    "issue_type": issue["type"],
                    "details": issue,
                })

                if issue["type"] == "potential_hallucination":
                    report.hallucinations_detected += 1
                elif issue["type"] == "low_confidence":
                    report.low_confidence_outputs += 1

        self.reflection_history.append(report)

        logger.info(
            f"Audit complete: checked={report.outputs_checked}, "
            f"hallucinations={report.hallucinations_detected}, "
            f"low_confidence={report.low_confidence_outputs}"
        )

        return report

    def get_stats(self) -> Dict[str, Any]:
        """Get reflection statistics."""
        if not self.reflection_history:
            return {"audits": 0}

        total_checked = sum(r.outputs_checked for r in self.reflection_history)
        total_hallucinations = sum(
            r.hallucinations_detected for r in self.reflection_history
        )

        return {
            "audits": len(self.reflection_history),
            "total_outputs_checked": total_checked,
            "total_hallucinations": total_hallucinations,
            "hallucination_rate": (
                total_hallucinations / total_checked if total_checked > 0 else 0.0
            ),
        }
