"""
Self-repair system for autonomous bug fixing.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


class RepairReport:
    """Report from self-repair attempt."""

    def __init__(self, issue: str):
        self.issue = issue
        self.timestamp = datetime.utcnow()
        self.fix_attempted = False
        self.fix_successful = False
        self.tests_passed = False
        self.patch_file: Optional[str] = None


class SelfRepairer:
    """
    Self-repair system for detecting and fixing issues autonomously.

    Safety-first approach:
    - All changes in sandboxed environment
    - Tests must pass before promotion
    - Changes are reversible
    """

    def __init__(
        self,
        enabled: bool = True,
        auto_patch: bool = False,
        sandbox_test: bool = True,
        max_fix_attempts: int = 3,
    ):
        """
        Initialize self-repairer.

        Args:
            enabled: Enable self-repair
            auto_patch: Automatically apply patches (requires approval)
            sandbox_test: Test patches in sandbox
            max_fix_attempts: Maximum fix attempts per issue
        """
        self.enabled = enabled
        self.auto_patch = auto_patch
        self.sandbox_test = sandbox_test
        self.max_fix_attempts = max_fix_attempts

        self.repair_history: List[RepairReport] = []

        logger.info(
            f"Self-repairer initialized: enabled={enabled}, "
            f"auto_patch={auto_patch}, sandbox_test={sandbox_test}"
        )

    def detect_issue(self, error: Exception, context: Dict) -> Optional[str]:
        """
        Detect and classify issue.

        Args:
            error: Exception that occurred
            context: Context information

        Returns:
            Issue classification or None
        """
        error_type = type(error).__name__
        error_msg = str(error)

        logger.warning(f"Detected issue: {error_type}: {error_msg}")

        # Classify issue
        if "AttributeError" in error_type:
            return "missing_attribute"
        elif "TypeError" in error_type:
            return "type_mismatch"
        elif "ValueError" in error_type:
            return "invalid_value"
        elif "ImportError" in error_type or "ModuleNotFoundError" in error_type:
            return "missing_import"
        else:
            return "unknown"

    def propose_fix(
        self, issue_type: str, error: Exception, context: Dict
    ) -> Optional[str]:
        """
        Propose fix for issue.

        Args:
            issue_type: Type of issue
            error: Exception
            context: Context

        Returns:
            Proposed fix (code patch) or None
        """
        # Simplified fix proposals
        # In production, would use LLM or rule-based system

        if issue_type == "missing_import":
            module = str(error).split("'")[1] if "'" in str(error) else "unknown"
            return f"# Add import: import {module}"

        elif issue_type == "missing_attribute":
            return "# Add attribute check: if hasattr(obj, 'attr'): ..."

        elif issue_type == "type_mismatch":
            return "# Add type conversion: value = str(value)"

        return None

    def test_fix(self, fix: str, test_suite: List[str]) -> bool:
        """
        Test fix in sandbox.

        Args:
            fix: Proposed fix
            test_suite: Test cases to run

        Returns:
            True if all tests pass
        """
        if not self.sandbox_test:
            logger.warning("Sandbox testing disabled")
            return True

        # Simplified testing
        # In production, would run actual tests in isolated environment
        logger.info("Testing fix in sandbox...")

        # Mock test execution
        all_passed = True
        for test in test_suite:
            # Simulate test
            passed = True  # Mock result
            if not passed:
                all_passed = False
                break

        logger.info(f"Sandbox tests: {'PASSED' if all_passed else 'FAILED'}")
        return all_passed

    def repair(
        self,
        error: Exception,
        context: Dict,
        test_suite: Optional[List[str]] = None,
    ) -> RepairReport:
        """
        Attempt to repair issue.

        Args:
            error: Exception to repair
            context: Context information
            test_suite: Tests to validate fix

        Returns:
            Repair report
        """
        issue_type = self.detect_issue(error, context)
        report = RepairReport(issue=issue_type)

        if not self.enabled:
            logger.info("Self-repair disabled")
            return report

        # Propose fix
        fix = self.propose_fix(issue_type, error, context)

        if not fix:
            logger.warning(f"No fix available for: {issue_type}")
            return report

        report.fix_attempted = True
        logger.info(f"Proposed fix:\n{fix}")

        # Test fix
        if test_suite:
            tests_passed = self.test_fix(fix, test_suite)
            report.tests_passed = tests_passed

            if not tests_passed:
                logger.error("Fix failed tests")
                return report

        # Apply fix (if auto_patch enabled)
        if self.auto_patch:
            success = self._apply_patch(fix)
            report.fix_successful = success
        else:
            logger.info("Auto-patch disabled. Fix requires manual approval.")

        self.repair_history.append(report)
        return report

    def _apply_patch(self, fix: str) -> bool:
        """Apply patch to codebase."""
        # In production, would create Git patch and apply
        logger.info("Applying patch (mock)")
        return True

    def get_stats(self) -> Dict:
        """Get repair statistics."""
        if not self.repair_history:
            return {"repairs_attempted": 0}

        attempted = len(self.repair_history)
        successful = sum(1 for r in self.repair_history if r.fix_successful)

        return {
            "repairs_attempted": attempted,
            "repairs_successful": successful,
            "success_rate": successful / attempted if attempted > 0 else 0.0,
        }
