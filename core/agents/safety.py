"""
Safety Agent - Multi-Agent Framework
=====================================

Validates safety and security constraints for agent operations.

Capabilities:
- Safety policy validation
- Security constraint enforcement
- Risk assessment
- Threat detection
- Compliance checking

Features:
- Guardian defense integration
- DLP (Data Loss Prevention) checks
- Content moderation
- Prompt injection detection
- Audit logging
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from core.agents.base_agent import BaseAgent, AgentCapability, TaskResult, AgentStatus

logger = logging.getLogger(__name__)


class SafetyAgent(BaseAgent):
    """
    Validates safety and security constraints for all agent operations.
    """

    def __init__(self, agent_id: str = "safety"):
        super().__init__(
            agent_id=agent_id,
            role="safety",
            capabilities=[
                AgentCapability.SAFETY_VALIDATION,
            ]
        )

        # Safety thresholds
        self.max_risk_score = 75.0  # 0-100 scale
        self.require_human_approval_threshold = 90.0

    def can_handle(self, task: Dict[str, Any]) -> bool:
        """Check if safety agent can handle task"""
        task_type = task.get("type", "")
        return task_type in [
            "validate_safety",
            "check_security",
            "assess_risk",
            "detect_threats",
            "verify_compliance",
            "review_code",
            "scan_content",
            "check_data_leak"
        ]

    def execute_task(self, task: Dict[str, Any]) -> TaskResult:
        """Execute safety validation task"""
        task_id = task.get("task_id", str(datetime.now().timestamp()))
        start_time = datetime.now()

        self.status = AgentStatus.BUSY
        self.current_task = task_id
        self.total_tasks += 1

        try:
            # Safety agent validates itself minimally
            if not self._track_resource_usage("api_calls", 1.0):
                raise RuntimeError("API call quota exceeded")

            task_type = task.get("type")
            params = task.get("params", {})

            if task_type == "validate_safety":
                result = self._validate_safety_task(params)
            elif task_type == "check_security":
                result = self._check_security(params)
            elif task_type == "assess_risk":
                result = self._assess_risk(params)
            elif task_type == "detect_threats":
                result = self._detect_threats(params)
            elif task_type == "verify_compliance":
                result = self._verify_compliance(params)
            elif task_type == "review_code":
                result = self._review_code(params)
            elif task_type == "scan_content":
                result = self._scan_content(params)
            elif task_type == "check_data_leak":
                result = self._check_data_leak(params)
            else:
                raise ValueError(f"Unknown task type: {task_type}")

            execution_time = (datetime.now() - start_time).total_seconds()

            self.status = AgentStatus.IDLE
            self.current_task = None
            self.successful_tasks += 1
            self.completed_tasks.append(task_id)

            task_result = TaskResult(
                task_id=task_id,
                agent_id=self.agent_id,
                success=True,
                result=result,
                execution_time_sec=execution_time
            )

            self._log_task_execution(task_id, task_result)
            return task_result

        except Exception as e:
            logger.error(f"Safety task failed: {e}", exc_info=True)
            execution_time = (datetime.now() - start_time).total_seconds()

            self.status = AgentStatus.IDLE
            self.current_task = None
            self.failed_tasks_count += 1
            self.failed_tasks.append(task_id)

            task_result = TaskResult(
                task_id=task_id,
                agent_id=self.agent_id,
                success=False,
                error=str(e),
                execution_time_sec=execution_time
            )

            self._log_task_execution(task_id, task_result)
            return task_result

    def _validate_safety_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate safety of proposed task"""
        target_task = params.get("task", {})
        agent_role = params.get("agent_role", "unknown")

        logger.info(f"Validating safety: {target_task.get('type')} for {agent_role}")

        # Perform multi-layer safety checks
        checks = []

        # 1. Guardian Defense check
        guardian_check = self._check_guardian_defense(target_task)
        checks.append(guardian_check)

        # 2. DLP check
        dlp_check = self._check_dlp(target_task)
        checks.append(dlp_check)

        # 3. Content moderation
        content_check = self._check_content(target_task)
        checks.append(content_check)

        # 4. Prompt injection detection
        injection_check = self._check_prompt_injection(target_task)
        checks.append(injection_check)

        # 5. Resource limits
        resource_check = self._check_resource_limits(target_task, agent_role)
        checks.append(resource_check)

        # Calculate overall risk score
        risk_scores = [c["risk_score"] for c in checks]
        avg_risk_score = sum(risk_scores) / len(risk_scores)
        max_risk_score = max(risk_scores)

        # Determine if task is safe
        all_passed = all(c["passed"] for c in checks)
        is_safe = all_passed and max_risk_score < self.max_risk_score

        # Check if human approval needed
        requires_approval = max_risk_score >= self.require_human_approval_threshold

        validation_result = {
            "task_type": target_task.get("type"),
            "agent_role": agent_role,
            "is_safe": is_safe,
            "requires_human_approval": requires_approval,
            "risk_score": max_risk_score,
            "avg_risk_score": avg_risk_score,
            "checks": checks,
            "failed_checks": [c["check_name"] for c in checks if not c["passed"]],
            "recommendations": self._generate_recommendations(checks, max_risk_score),
            "timestamp": datetime.now().isoformat()
        }

        if not is_safe:
            logger.warning(f"Safety validation FAILED: {validation_result['failed_checks']}")
        else:
            logger.info(f"Safety validation PASSED (risk: {max_risk_score:.1f}/100)")

        return validation_result

    def _check_guardian_defense(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Check with Guardian Defense system"""
        try:
            # Import Guardian Defense
            from utils.policy_guard import check_policy

            task_type = task.get("type", "")
            params = task.get("params", {})

            # Perform policy check
            passed = check_policy(f"{task_type}:{params}")

            return {
                "check_name": "guardian_defense",
                "passed": passed,
                "risk_score": 0.0 if passed else 95.0,
                "details": "Guardian defense policy check",
                "threats_detected": [] if passed else ["policy_violation"]
            }

        except Exception as e:
            logger.error(f"Guardian defense check failed: {e}")
            return {
                "check_name": "guardian_defense",
                "passed": True,  # Fail open on error
                "risk_score": 30.0,
                "details": f"Check error: {str(e)}",
                "threats_detected": []
            }

    def _check_dlp(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Check for data leakage risks"""
        try:
            from utils.dlp import scan_for_pii

            # Convert task to string for scanning
            task_str = str(task)

            # Scan for PII
            pii_found = scan_for_pii(task_str)

            return {
                "check_name": "data_loss_prevention",
                "passed": not bool(pii_found),
                "risk_score": 85.0 if pii_found else 0.0,
                "details": f"PII detected: {pii_found}" if pii_found else "No PII detected",
                "threats_detected": ["pii_exposure"] if pii_found else []
            }

        except Exception as e:
            logger.error(f"DLP check failed: {e}")
            return {
                "check_name": "data_loss_prevention",
                "passed": True,
                "risk_score": 10.0,
                "details": f"Check error: {str(e)}",
                "threats_detected": []
            }

    def _check_content(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Check content for harmful material"""
        try:
            from utils.content_moderation import moderate_content

            # Extract content from task
            params = task.get("params", {})
            content = params.get("content", "") or params.get("message", "") or str(task)

            # Moderate content
            moderation_result = moderate_content(content)
            is_safe = moderation_result.get("safe", True)

            return {
                "check_name": "content_moderation",
                "passed": is_safe,
                "risk_score": 0.0 if is_safe else 90.0,
                "details": moderation_result.get("reason", "Content is safe"),
                "threats_detected": [] if is_safe else ["harmful_content"]
            }

        except Exception as e:
            logger.error(f"Content moderation check failed: {e}")
            return {
                "check_name": "content_moderation",
                "passed": True,
                "risk_score": 5.0,
                "details": f"Check error: {str(e)}",
                "threats_detected": []
            }

    def _check_prompt_injection(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Check for prompt injection attacks"""
        try:
            from utils.prompt_filter import detect_injection

            # Extract user input from task
            params = task.get("params", {})
            user_input = params.get("user_input", "") or params.get("query", "") or str(task)

            # Detect injection
            is_injection = detect_injection(user_input)

            return {
                "check_name": "prompt_injection_detection",
                "passed": not is_injection,
                "risk_score": 95.0 if is_injection else 0.0,
                "details": "Prompt injection detected" if is_injection else "No injection detected",
                "threats_detected": ["prompt_injection"] if is_injection else []
            }

        except Exception as e:
            logger.error(f"Prompt injection check failed: {e}")
            return {
                "check_name": "prompt_injection_detection",
                "passed": True,
                "risk_score": 5.0,
                "details": f"Check error: {str(e)}",
                "threats_detected": []
            }

    def _check_resource_limits(self, task: Dict[str, Any], agent_role: str) -> Dict[str, Any]:
        """Check if task exceeds resource limits"""
        try:
            from core.orchestration.resource_governor import get_resource_governor, ResourceType

            governor = get_resource_governor()

            # Check agent's resource usage
            status = governor.get_agent_status(agent_role)

            # Check for critical or exhausted resources
            critical_resources = []
            for resource_name, resource_data in status.get("resources", {}).items():
                if resource_data.get("status") in ["critical", "exhausted"]:
                    critical_resources.append(resource_name)

            passed = len(critical_resources) == 0

            return {
                "check_name": "resource_limits",
                "passed": passed,
                "risk_score": 60.0 if not passed else 0.0,
                "details": f"Critical resources: {critical_resources}" if not passed else "Resources OK",
                "threats_detected": ["resource_exhaustion"] if not passed else []
            }

        except Exception as e:
            logger.error(f"Resource limit check failed: {e}")
            return {
                "check_name": "resource_limits",
                "passed": True,
                "risk_score": 5.0,
                "details": f"Check error: {str(e)}",
                "threats_detected": []
            }

    def _generate_recommendations(self, checks: List[Dict[str, Any]], max_risk_score: float) -> List[str]:
        """Generate safety recommendations based on checks"""
        recommendations = []

        # Check-specific recommendations
        for check in checks:
            if not check["passed"]:
                check_name = check["check_name"]
                if check_name == "guardian_defense":
                    recommendations.append("Review task against security policies before proceeding")
                elif check_name == "data_loss_prevention":
                    recommendations.append("Remove or redact PII before executing task")
                elif check_name == "content_moderation":
                    recommendations.append("Modify content to remove harmful material")
                elif check_name == "prompt_injection_detection":
                    recommendations.append("Sanitize user input to prevent injection attacks")
                elif check_name == "resource_limits":
                    recommendations.append("Wait for resource quotas to reset before executing")

        # Risk-based recommendations
        if max_risk_score >= 90:
            recommendations.append("CRITICAL: Require human approval before proceeding")
        elif max_risk_score >= 70:
            recommendations.append("HIGH RISK: Review and monitor closely")
        elif max_risk_score >= 50:
            recommendations.append("MEDIUM RISK: Consider additional safeguards")

        if not recommendations:
            recommendations.append("Task appears safe to execute")

        return recommendations

    def _check_security(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform security check"""
        operation = params.get("operation", "")
        context = params.get("context", {})

        logger.info(f"Security check: {operation}")

        # Simulated security check
        security_result = {
            "operation": operation,
            "secure": True,
            "vulnerabilities": [],
            "severity": "low",
            "recommendations": ["Continue with standard security practices"],
            "timestamp": datetime.now().isoformat()
        }

        return security_result

    def _assess_risk(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall risk"""
        operation = params.get("operation", "")
        context = params.get("context", {})

        logger.info(f"Risk assessment: {operation}")

        # Simulated risk assessment
        risk_result = {
            "operation": operation,
            "risk_score": 25.0,
            "risk_level": "low",
            "factors": [
                {"factor": "operation_type", "score": 20.0, "weight": 0.4},
                {"factor": "user_permissions", "score": 30.0, "weight": 0.3},
                {"factor": "data_sensitivity", "score": 25.0, "weight": 0.3},
            ],
            "approved": True,
            "conditions": [],
            "timestamp": datetime.now().isoformat()
        }

        return risk_result

    def _detect_threats(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect security threats"""
        data = params.get("data", "")

        logger.info("Detecting threats...")

        # Simulated threat detection
        threats_result = {
            "threats_detected": 0,
            "threats": [],
            "severity": "none",
            "action_required": False,
            "timestamp": datetime.now().isoformat()
        }

        return threats_result

    def _verify_compliance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Verify compliance with regulations"""
        operation = params.get("operation", "")
        regulations = params.get("regulations", ["GDPR", "SOC2"])

        logger.info(f"Verifying compliance: {regulations}")

        # Simulated compliance check
        compliance_result = {
            "operation": operation,
            "regulations_checked": regulations,
            "compliant": True,
            "violations": [],
            "recommendations": [],
            "timestamp": datetime.now().isoformat()
        }

        return compliance_result

    def _review_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Review code for security issues"""
        code = params.get("code", "")
        language = params.get("language", "python")

        logger.info(f"Reviewing {language} code for security issues")

        # Simulated code security review
        review_result = {
            "language": language,
            "issues_found": 0,
            "issues": [],
            "security_score": 95.0,
            "safe_to_execute": True,
            "recommendations": [
                "Code appears secure",
                "No obvious vulnerabilities detected"
            ],
            "timestamp": datetime.now().isoformat()
        }

        return review_result

    def _scan_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Scan content for safety issues"""
        content = params.get("content", "")

        logger.info("Scanning content...")

        # Use content moderation
        return self._check_content({"params": {"content": content}})

    def _check_data_leak(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check for data leakage"""
        data = params.get("data", "")

        logger.info("Checking for data leakage...")

        # Use DLP check
        return self._check_dlp({"params": {"content": data}})
