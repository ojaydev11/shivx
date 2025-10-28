"""
Policy-as-Code Guard for ShivX

Provides pre-flight validation for risky operations including:
- Filesystem writes outside allowlist
- Subprocess commands not in allowlist  
- Network domains not in allowlist
- High-risk actions requiring approval
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Metrics (safe no-op if prometheus_client missing)
try:
    import time
    from utils.metrics import inc_policy_decision
    from utils.metrics import POLICY_EVAL_LATENCY, POLICY_EVAL_ERRORS
except Exception:  # pragma: no cover
    def inc_policy_decision(*args, **kwargs):  # type: ignore
        pass
    class _N:
        def labels(self, **_): return self
        def observe(self, *_): pass
        def inc(self, *_): pass
    POLICY_EVAL_LATENCY = POLICY_EVAL_ERRORS = _N()


@dataclass
class PolicyDecision:
    """Result of policy evaluation"""
    decision: str  # "allow", "deny", "warn"
    reasons: List[str]
    risk_score: int
    requires_approval: bool
    metadata: Dict[str, Any]


class PolicyGuard:
    """Main policy evaluation engine"""
    
    def __init__(self, config_dir: str = "config/policy"):
        self.config_dir = Path(config_dir)
        self.policies = {}
        self.merged_rules = {}
        self.load_policies()
    
    def load_policies(self) -> None:
        """Load all YAML policy files and merge them"""
        if not self.config_dir.exists():
            logger.warning(f"Policy config directory {self.config_dir} not found")
            return
        
        # Load base policy first
        base_policy = self.config_dir / "base.yaml"
        if base_policy.exists():
            with open(base_policy, 'r') as f:
                self.policies['base'] = yaml.safe_load(f)
                self.merged_rules = self.policies['base'].copy()
        
        # Load and merge project-specific policies
        for policy_file in self.config_dir.glob("*.yaml"):
            if policy_file.name == "base.yaml":
                continue
            
            project_name = policy_file.stem
            try:
                with open(policy_file, 'r') as f:
                    project_policy = yaml.safe_load(f)
                    self.policies[project_name] = project_policy
                    self._merge_policy(project_policy)
            except Exception as e:
                logger.error(f"Failed to load policy {policy_file}: {e}")
        
        logger.info(f"Loaded {len(self.policies)} policy files")
    
    def _merge_policy(self, policy: Dict[str, Any]) -> None:
        """Merge a project policy with base rules"""
        for section, rules in policy.items():
            if section not in self.merged_rules:
                self.merged_rules[section] = rules
            elif isinstance(rules, dict) and isinstance(self.merged_rules[section], dict):
                # Deep merge for dict sections
                self._deep_merge(self.merged_rules[section], rules)
            elif isinstance(rules, list) and isinstance(self.merged_rules[section], list):
                # Extend lists
                self.merged_rules[section].extend(rules)
            else:
                # Override for other types
                self.merged_rules[section] = rules
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Deep merge two dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def evaluate(self, payload: Dict[str, Any]) -> PolicyDecision:
        """Evaluate a policy action with contextual facts and return a decision."""
        start = time.perf_counter()
        action = payload.get("action", "unknown")
        try:
            if action == "subprocess.exec":
                result = self._evaluate_subprocess(payload)
            elif action == "fs.write":
                result = self._evaluate_filesystem_write(payload)
            elif action == "net.http":
                result = self._evaluate_network_request(payload)
            elif action == "browser.automation":
                result = self._evaluate_browser_automation(payload)
            elif action == "desktop.control":
                result = self._evaluate_desktop_control(payload)
            elif action == "autodev.execution":
                result = self._evaluate_autodev_execution(payload)
            else:
                result = self._evaluate_generic_action(action, payload)
            
            inc_policy_decision(action, result.decision)
            return result
        except Exception:
            # Count and re-raise (keeps existing behavior)
            try: POLICY_EVAL_ERRORS.labels(action=action).inc()
            except Exception: pass
            raise
        finally:
            try: POLICY_EVAL_LATENCY.labels(action=action).observe(time.perf_counter() - start)
            except Exception: pass
    
    def _evaluate_subprocess(self, payload: Dict[str, Any]) -> PolicyDecision:
        """Evaluate subprocess execution"""
        cmd = payload.get("cmd", "")
        args = payload.get("args", [])
        
        # Check blocked commands
        blocked_commands = self.merged_rules.get("restrictions", {}).get("blocked_tools", [])
        if cmd in blocked_commands:
            return PolicyDecision(
                decision="deny",
                reasons=[f"Command '{cmd}' is blocked by policy"],
                risk_score=80,
                requires_approval=False,
                metadata={"cmd": cmd, "args": args}
            )
        
        # Check autodev allowed commands
        autodev_rules = self.merged_rules.get("autodev", {})
        allowed_commands = autodev_rules.get("allowed_commands", [])
        blocked_commands = autodev_rules.get("blocked_commands", [])
        
        if cmd in blocked_commands:
            return PolicyDecision(
                decision="deny",
                reasons=[f"Command '{cmd}' is blocked by autodev policy"],
                risk_score=70,
                requires_approval=False,
                metadata={"cmd": cmd, "args": args}
            )
        
        if cmd in allowed_commands:
            return PolicyDecision(
                decision="allow",
                reasons=[f"Command '{cmd}' is explicitly allowed"],
                risk_score=10,
                requires_approval=False,
                metadata={"cmd": cmd, "args": args}
            )
        
        # Default: warn for unknown commands
        return PolicyDecision(
            decision="warn",
            reasons=[f"Command '{cmd}' not in allowlist - proceed with caution"],
            risk_score=40,
            requires_approval=True,
            metadata={"cmd": cmd, "args": args}
        )
    
    def _evaluate_filesystem_write(self, payload: Dict[str, Any]) -> PolicyDecision:
        """Evaluate filesystem write operations"""
        path = payload.get("path", "")
        
        # Check blocked paths
        blocked_paths = self.merged_rules.get("restrictions", {}).get("blocked_paths", [])
        for blocked in blocked_paths:
            if path.startswith(blocked):
                return PolicyDecision(
                    decision="deny",
                    reasons=[f"Path '{path}' is blocked by policy"],
                    risk_score=90,
                    requires_approval=False,
                    metadata={"path": path}
                )
        
        # Check allowed paths
        allowed_paths = self.merged_rules.get("restrictions", {}).get("allowed_paths", [])
        for allowed in allowed_paths:
            if path.startswith(allowed):
                return PolicyDecision(
                    decision="allow",
                    reasons=[f"Path '{path}' is explicitly allowed"],
                    risk_score=20,
                    requires_approval=False,
                    metadata={"path": path}
                )
        
        # Check if outside project directory
        project_root = Path.cwd()
        try:
            path_obj = Path(path).resolve()
            if not str(path_obj).startswith(str(project_root)):
                return PolicyDecision(
                    decision="deny",
                    reasons=[f"Path '{path}' is outside project directory"],
                    risk_score=85,
                    requires_approval=False,
                    metadata={"path": path, "project_root": str(project_root)}
                )
        except Exception:
            pass
        
        # Default: warn for unknown paths
        return PolicyDecision(
            decision="warn",
            reasons=[f"Path '{path}' not in allowlist - proceed with caution"],
            risk_score=50,
            requires_approval=True,
            metadata={"path": path}
        )
    
    def _evaluate_network_request(self, payload: Dict[str, Any]) -> PolicyDecision:
        """
        Evaluate network requests using ALLOWLIST model

        CRITICAL SECURITY: Network egress is blocked by default.
        Only explicitly allowed domains can be accessed.
        """
        url = payload.get("url", "")

        try:
            parsed = urlparse(url)
            domain = parsed.netloc
        except Exception:
            return PolicyDecision(
                decision="deny",
                reasons=[f"Invalid URL format: {url}"],
                risk_score=100,
                requires_approval=False,
                metadata={"url": url}
            )

        # ALLOWLIST MODEL: Define allowed domains for network egress
        ALLOWED_DOMAINS = [
            # Solana RPC
            "api.mainnet-beta.solana.com",
            "api.devnet.solana.com",
            "api.testnet.solana.com",
            "solana.com",

            # Jupiter DEX Aggregator
            "quote-api.jup.ag",
            "api.jup.ag",
            "jupiter-swap-api.quiknode.pro",

            # AI/ML APIs
            "api.openai.com",
            "api.anthropic.com",
            "api.cohere.ai",

            # Monitoring and observability
            "*.sentry.io",
            "api.datadoghq.com",

            # Package repositories (for updates only)
            "pypi.org",
            "files.pythonhosted.org",
        ]

        # Check if domain is in allowlist
        domain_allowed = False
        matched_pattern = None

        for allowed_pattern in ALLOWED_DOMAINS:
            if self._domain_matches(domain, allowed_pattern):
                domain_allowed = True
                matched_pattern = allowed_pattern
                break

        if domain_allowed:
            logger.info(f"Network request ALLOWED: {domain} (matched: {matched_pattern})")
            return PolicyDecision(
                decision="allow",
                reasons=[f"Domain '{domain}' is in allowlist (matched: {matched_pattern})"],
                risk_score=15,
                requires_approval=False,
                metadata={"url": url, "domain": domain, "matched_pattern": matched_pattern}
            )
        else:
            # DENY by default - not in allowlist
            logger.warning(f"Network request BLOCKED: {domain} (not in allowlist)")
            return PolicyDecision(
                decision="deny",
                reasons=[
                    f"Domain '{domain}' is NOT in allowlist",
                    "Network egress is blocked by default for security",
                    "Add domain to ALLOWED_DOMAINS in policy_guard.py if needed"
                ],
                risk_score=95,
                requires_approval=False,
                metadata={"url": url, "domain": domain, "allowlist": ALLOWED_DOMAINS}
            )
    
    def _evaluate_browser_automation(self, payload: Dict[str, Any]) -> PolicyDecision:
        """Evaluate browser automation actions"""
        # Check if browser automation is high risk
        high_risk_actions = self.merged_rules.get("risk", {}).get("high_risk_actions", [])
        
        if "browser_automation" in high_risk_actions:
            return PolicyDecision(
                decision="warn",
                reasons=["Browser automation requires approval"],
                risk_score=75,
                requires_approval=True,
                metadata=payload
            )
        
        return PolicyDecision(
            decision="allow",
            reasons=["Browser automation allowed"],
            risk_score=30,
            requires_approval=False,
            metadata=payload
        )
    
    def _evaluate_desktop_control(self, payload: Dict[str, Any]) -> PolicyDecision:
        """Evaluate desktop control actions"""
        # Desktop control is always high risk
        return PolicyDecision(
            decision="warn",
            reasons=["Desktop control requires approval"],
            risk_score=85,
            requires_approval=True,
            metadata=payload
        )
    
    def _evaluate_autodev_execution(self, payload: Dict[str, Any]) -> PolicyDecision:
        """Evaluate autodev execution"""
        # Check if autodev execution is high risk
        high_risk_actions = self.merged_rules.get("risk", {}).get("high_risk_actions", [])
        
        if "autodev_execution" in high_risk_actions:
            return PolicyDecision(
                decision="warn",
                reasons=["Autodev execution requires approval"],
                risk_score=80,
                requires_approval=True,
                metadata=payload
            )
        
        return PolicyDecision(
            decision="allow",
            reasons=["Autodev execution allowed"],
            risk_score=40,
            requires_approval=False,
            metadata=payload
        )
    
    def _evaluate_generic_action(self, action: str, payload: Dict[str, Any]) -> PolicyDecision:
        """Evaluate generic actions"""
        # Check if action is high risk
        high_risk_actions = self.merged_rules.get("risk", {}).get("high_risk_actions", [])
        
        if action in high_risk_actions:
            return PolicyDecision(
                decision="warn",
                reasons=[f"Action '{action}' requires approval"],
                risk_score=70,
                requires_approval=True,
                metadata=payload
            )
        
        return PolicyDecision(
            decision="allow",
            reasons=[f"Action '{action}' allowed"],
            risk_score=20,
            requires_approval=False,
            metadata=payload
        )
    
    def _domain_matches(self, domain: str, pattern: str) -> bool:
        """Check if domain matches pattern (supports wildcards)"""
        if pattern.startswith("*."):
            return domain.endswith(pattern[1:])
        elif pattern.endswith(".*"):
            return domain.startswith(pattern[:-1])
        else:
            return domain == pattern
    
    def get_effective_rules(self) -> Dict[str, Any]:
        """Get merged policy rules for display"""
        return self.merged_rules.copy()
    
    def list_policies(self) -> Dict[str, Any]:
        """List all loaded policies"""
        return {
            "config_dir": str(self.config_dir),
            "loaded_policies": list(self.policies.keys()),
            "effective_rules": self.get_effective_rules()
        }


# Global policy guard instance
_policy_guard = None


def get_policy_guard() -> PolicyGuard:
    """Get or create global policy guard instance"""
    global _policy_guard
    if _policy_guard is None:
        _policy_guard = PolicyGuard()
    return _policy_guard


def evaluate_policy(action: str, payload: Dict[str, Any]) -> PolicyDecision:
    """Convenience function to evaluate policy"""
    guard = get_policy_guard()
    # Ensure action is in payload for the new evaluate method
    payload_with_action = payload.copy()
    payload_with_action["action"] = action
    return guard.evaluate(payload_with_action)
