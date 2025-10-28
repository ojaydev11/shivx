#!/usr/bin/env python3
"""
ShivX Environment Configuration Validator
==========================================
Validates environment configuration for production deployment security.

Usage:
    python validate_env.py [--env-file .env]
    python validate_env.py --env-file .env.production
    python validate_env.py --strict  # Fail on warnings
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse
import secrets


class Color:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class ValidationResult:
    """Represents the result of a validation check"""

    def __init__(self, passed: bool, message: str, severity: str = "error"):
        self.passed = passed
        self.message = message
        self.severity = severity  # error, warning, info

    def __str__(self):
        if self.severity == "error":
            icon = f"{Color.RED}✗{Color.END}"
        elif self.severity == "warning":
            icon = f"{Color.YELLOW}⚠{Color.END}"
        else:
            icon = f"{Color.GREEN}✓{Color.END}"

        return f"{icon} {self.message}"


class EnvValidator:
    """Validates environment configuration for security and production readiness"""

    def __init__(self, env_vars: Dict[str, str], strict: bool = False):
        self.env_vars = env_vars
        self.strict = strict
        self.errors: List[ValidationResult] = []
        self.warnings: List[ValidationResult] = []
        self.passed: List[ValidationResult] = []

    def validate_all(self) -> bool:
        """Run all validations"""
        print(f"\n{Color.BOLD}{Color.CYAN}ShivX Environment Configuration Validator{Color.END}")
        print("=" * 60)
        print()

        # Critical security checks
        self._check_environment_mode()
        self._check_debug_mode()
        self._check_skip_auth()
        self._check_secret_keys()
        self._check_cors_origins()
        self._check_trading_mode()
        self._check_database_config()
        self._check_logging_config()
        self._check_monitoring_config()
        self._check_guardrails()
        self._check_required_variables()
        self._check_placeholder_values()
        self._check_password_strength()
        self._check_ssl_config()

        # Display results
        self._display_results()

        # Determine overall pass/fail
        has_errors = len(self.errors) > 0
        has_warnings = len(self.warnings) > 0

        if has_errors:
            return False
        elif has_warnings and self.strict:
            return False
        else:
            return True

    def _check_environment_mode(self):
        """Check that environment is set to production"""
        env = self.env_vars.get("SHIVX_ENV", "").lower()

        if env != "production":
            self.errors.append(ValidationResult(
                False,
                f"SHIVX_ENV must be 'production' (found: '{env}')",
                "error"
            ))
        else:
            self.passed.append(ValidationResult(
                True,
                "SHIVX_ENV is set to 'production'",
                "info"
            ))

    def _check_debug_mode(self):
        """Check that debug mode is disabled"""
        dev_mode = self.env_vars.get("SHIVX_DEV", "true").lower()
        debug_mode = self.env_vars.get("DEBUG", "false").lower()

        if dev_mode in ("true", "1", "yes"):
            self.errors.append(ValidationResult(
                False,
                "SHIVX_DEV must be 'false' in production",
                "error"
            ))
        else:
            self.passed.append(ValidationResult(
                True,
                "SHIVX_DEV is disabled",
                "info"
            ))

        if debug_mode in ("true", "1", "yes"):
            self.errors.append(ValidationResult(
                False,
                "DEBUG must be 'false' in production",
                "error"
            ))
        else:
            self.passed.append(ValidationResult(
                True,
                "DEBUG is disabled",
                "info"
            ))

    def _check_skip_auth(self):
        """Check that authentication is not skipped"""
        skip_auth = self.env_vars.get("SKIP_AUTH", "false").lower()

        if skip_auth in ("true", "1", "yes"):
            self.errors.append(ValidationResult(
                False,
                "SKIP_AUTH must be 'false' in production - CRITICAL SECURITY ISSUE!",
                "error"
            ))
        else:
            self.passed.append(ValidationResult(
                True,
                "SKIP_AUTH is disabled",
                "info"
            ))

    def _check_secret_keys(self):
        """Check that secret keys are properly configured"""
        secret_key = self.env_vars.get("SHIVX_SECRET_KEY", "")
        jwt_secret = self.env_vars.get("SHIVX_JWT_SECRET", "")

        # Check SECRET_KEY
        if not secret_key:
            self.errors.append(ValidationResult(
                False,
                "SHIVX_SECRET_KEY is not set",
                "error"
            ))
        elif len(secret_key) < 32:
            self.errors.append(ValidationResult(
                False,
                f"SHIVX_SECRET_KEY is too short ({len(secret_key)} chars, minimum 32)",
                "error"
            ))
        elif "REPLACE" in secret_key.upper() or "CHANGEME" in secret_key.upper():
            self.errors.append(ValidationResult(
                False,
                "SHIVX_SECRET_KEY contains placeholder text - must be changed!",
                "error"
            ))
        elif secret_key == "INSECURE_SECRET_KEY_FOR_TESTING":
            self.errors.append(ValidationResult(
                False,
                "SHIVX_SECRET_KEY is using insecure test value!",
                "error"
            ))
        else:
            self.passed.append(ValidationResult(
                True,
                f"SHIVX_SECRET_KEY is properly configured ({len(secret_key)} chars)",
                "info"
            ))

        # Check JWT_SECRET
        if not jwt_secret:
            self.errors.append(ValidationResult(
                False,
                "SHIVX_JWT_SECRET is not set",
                "error"
            ))
        elif len(jwt_secret) < 32:
            self.errors.append(ValidationResult(
                False,
                f"SHIVX_JWT_SECRET is too short ({len(jwt_secret)} chars, minimum 32)",
                "error"
            ))
        elif "REPLACE" in jwt_secret.upper() or "CHANGEME" in jwt_secret.upper():
            self.errors.append(ValidationResult(
                False,
                "SHIVX_JWT_SECRET contains placeholder text - must be changed!",
                "error"
            ))
        else:
            self.passed.append(ValidationResult(
                True,
                f"SHIVX_JWT_SECRET is properly configured ({len(jwt_secret)} chars)",
                "info"
            ))

        # Check if secrets are the same (bad practice)
        if secret_key and jwt_secret and secret_key == jwt_secret:
            self.warnings.append(ValidationResult(
                False,
                "SHIVX_SECRET_KEY and SHIVX_JWT_SECRET should be different",
                "warning"
            ))

    def _check_cors_origins(self):
        """Check that CORS origins are properly restricted"""
        cors_origins = self.env_vars.get("SHIVX_CORS_ORIGINS", "")

        if not cors_origins:
            self.warnings.append(ValidationResult(
                False,
                "SHIVX_CORS_ORIGINS is not set",
                "warning"
            ))
        elif "*" in cors_origins:
            self.errors.append(ValidationResult(
                False,
                "SHIVX_CORS_ORIGINS contains wildcard '*' - must be specific domains!",
                "error"
            ))
        elif "localhost" in cors_origins.lower():
            self.warnings.append(ValidationResult(
                False,
                "SHIVX_CORS_ORIGINS contains 'localhost' - remove for production",
                "warning"
            ))
        else:
            # Check that all origins use HTTPS
            origins = [o.strip() for o in cors_origins.split(",")]
            insecure_origins = [o for o in origins if o.startswith("http://")]

            if insecure_origins:
                self.warnings.append(ValidationResult(
                    False,
                    f"SHIVX_CORS_ORIGINS contains insecure HTTP origins: {', '.join(insecure_origins)}",
                    "warning"
                ))
            else:
                self.passed.append(ValidationResult(
                    True,
                    f"SHIVX_CORS_ORIGINS is properly restricted ({len(origins)} origins)",
                    "info"
                ))

    def _check_trading_mode(self):
        """Check trading mode configuration"""
        trading_mode = self.env_vars.get("SHIVX_TRADING_MODE", "").lower()

        if not trading_mode:
            self.errors.append(ValidationResult(
                False,
                "SHIVX_TRADING_MODE is not set",
                "error"
            ))
        elif trading_mode == "live":
            self.warnings.append(ValidationResult(
                False,
                "SHIVX_TRADING_MODE is 'live' - ensure thorough testing with 'paper' mode first!",
                "warning"
            ))
        elif trading_mode == "paper":
            self.passed.append(ValidationResult(
                True,
                "SHIVX_TRADING_MODE is 'paper' (safe for initial deployment)",
                "info"
            ))
        else:
            self.errors.append(ValidationResult(
                False,
                f"SHIVX_TRADING_MODE has invalid value: '{trading_mode}' (must be 'paper' or 'live')",
                "error"
            ))

    def _check_database_config(self):
        """Check database configuration"""
        db_url = self.env_vars.get("SHIVX_DATABASE_URL", "")

        if not db_url:
            self.errors.append(ValidationResult(
                False,
                "SHIVX_DATABASE_URL is not set",
                "error"
            ))
            return

        # Check for SQLite (not suitable for production)
        if "sqlite" in db_url.lower():
            self.errors.append(ValidationResult(
                False,
                "SHIVX_DATABASE_URL uses SQLite - use PostgreSQL for production!",
                "error"
            ))
            return

        # Check for PostgreSQL with SSL
        if "postgresql" in db_url.lower():
            if "sslmode" not in db_url.lower():
                self.warnings.append(ValidationResult(
                    False,
                    "SHIVX_DATABASE_URL does not specify sslmode - add '?sslmode=require'",
                    "warning"
                ))
            elif "sslmode=disable" in db_url.lower() or "sslmode=allow" in db_url.lower():
                self.errors.append(ValidationResult(
                    False,
                    "SHIVX_DATABASE_URL has SSL disabled - must use sslmode=require or verify-full",
                    "error"
                ))
            else:
                self.passed.append(ValidationResult(
                    True,
                    "SHIVX_DATABASE_URL uses PostgreSQL with SSL",
                    "info"
                ))

            # Check for default/weak passwords
            parsed = urlparse(db_url)
            if parsed.password:
                weak_passwords = ["password", "changeme", "shivx_password", "postgres"]
                if parsed.password.lower() in weak_passwords:
                    self.errors.append(ValidationResult(
                        False,
                        f"Database password is weak: '{parsed.password}' - use strong password!",
                        "error"
                    ))

    def _check_logging_config(self):
        """Check logging configuration"""
        json_logging = self.env_vars.get("SHIVX_JSON_LOGGING", "false").lower()
        log_level = self.env_vars.get("SHIVX_LOG_LEVEL", "INFO").upper()

        if json_logging not in ("true", "1", "yes"):
            self.warnings.append(ValidationResult(
                False,
                "SHIVX_JSON_LOGGING should be 'true' for production (enables structured logging)",
                "warning"
            ))
        else:
            self.passed.append(ValidationResult(
                True,
                "SHIVX_JSON_LOGGING is enabled (structured logging)",
                "info"
            ))

        if log_level == "DEBUG":
            self.warnings.append(ValidationResult(
                False,
                "SHIVX_LOG_LEVEL is 'DEBUG' - consider 'INFO' or 'WARNING' for production",
                "warning"
            ))

    def _check_monitoring_config(self):
        """Check monitoring configuration"""
        metrics_enabled = self.env_vars.get("SHIVX_ENABLE_METRICS", "false").lower()
        otel_enabled = self.env_vars.get("OTEL_ENABLED", "false").lower()

        if metrics_enabled not in ("true", "1", "yes"):
            self.errors.append(ValidationResult(
                False,
                "SHIVX_ENABLE_METRICS must be 'true' in production",
                "error"
            ))
        else:
            self.passed.append(ValidationResult(
                True,
                "SHIVX_ENABLE_METRICS is enabled",
                "info"
            ))

        if otel_enabled not in ("true", "1", "yes"):
            self.warnings.append(ValidationResult(
                False,
                "OTEL_ENABLED is not enabled - consider enabling distributed tracing",
                "warning"
            ))

    def _check_guardrails(self):
        """Check that security guardrails are enabled"""
        guardrails = self.env_vars.get("SHIVX_FEATURE_GUARDRAILS", "false").lower()
        guardian = self.env_vars.get("SHIVX_FEATURE_GUARDIAN_DEFENSE", "false").lower()

        if guardrails not in ("true", "1", "yes"):
            self.errors.append(ValidationResult(
                False,
                "SHIVX_FEATURE_GUARDRAILS must be 'true' in production - CRITICAL!",
                "error"
            ))
        else:
            self.passed.append(ValidationResult(
                True,
                "SHIVX_FEATURE_GUARDRAILS is enabled",
                "info"
            ))

        if guardian not in ("true", "1", "yes"):
            self.errors.append(ValidationResult(
                False,
                "SHIVX_FEATURE_GUARDIAN_DEFENSE must be 'true' in production - CRITICAL!",
                "error"
            ))
        else:
            self.passed.append(ValidationResult(
                True,
                "SHIVX_FEATURE_GUARDIAN_DEFENSE is enabled",
                "info"
            ))

    def _check_required_variables(self):
        """Check that all required variables are set"""
        required_vars = [
            "SHIVX_ENV",
            "SHIVX_SECRET_KEY",
            "SHIVX_JWT_SECRET",
            "SHIVX_DATABASE_URL",
            "SHIVX_TRADING_MODE",
        ]

        missing = [var for var in required_vars if not self.env_vars.get(var)]

        if missing:
            self.errors.append(ValidationResult(
                False,
                f"Missing required variables: {', '.join(missing)}",
                "error"
            ))

    def _check_placeholder_values(self):
        """Check for placeholder values that need to be replaced"""
        placeholder_patterns = [
            "REPLACE",
            "CHANGEME",
            "YOUR_",
            "REPLACE_WITH",
            "REPLACE_IN",
            "LOAD_FROM",
        ]

        for key, value in self.env_vars.items():
            if not value:
                continue

            value_upper = value.upper()
            for pattern in placeholder_patterns:
                if pattern in value_upper:
                    self.warnings.append(ValidationResult(
                        False,
                        f"{key} contains placeholder text: '{value[:50]}...'",
                        "warning"
                    ))
                    break

    def _check_password_strength(self):
        """Check password strength for database and other services"""
        # Check for common weak passwords
        weak_password_patterns = [
            r"^password\d*$",
            r"^admin\d*$",
            r"^test\d*$",
            r"^changeme$",
            r"^123456",
        ]

        password_vars = [
            "POSTGRES_PASSWORD",
            "REDIS_PASSWORD",
            "GRAFANA_ADMIN_PASSWORD",
        ]

        for var in password_vars:
            password = self.env_vars.get(var, "")
            if not password:
                continue

            for pattern in weak_password_patterns:
                if re.match(pattern, password.lower()):
                    self.errors.append(ValidationResult(
                        False,
                        f"{var} is weak or common - use strong random password!",
                        "error"
                    ))
                    break

    def _check_ssl_config(self):
        """Check SSL/TLS configuration"""
        # Check if SSL is properly configured for various services
        ssl_vars = {
            "SSL_CERT_PATH": self.env_vars.get("SSL_CERT_PATH", ""),
            "SSL_KEY_PATH": self.env_vars.get("SSL_KEY_PATH", ""),
        }

        for var, value in ssl_vars.items():
            if value and Path(value).exists():
                self.passed.append(ValidationResult(
                    True,
                    f"{var} is set and file exists",
                    "info"
                ))
            elif value:
                self.warnings.append(ValidationResult(
                    False,
                    f"{var} is set but file does not exist: {value}",
                    "warning"
                ))

    def _display_results(self):
        """Display validation results"""
        print(f"\n{Color.BOLD}Validation Results:{Color.END}")
        print("=" * 60)

        # Show errors
        if self.errors:
            print(f"\n{Color.BOLD}{Color.RED}ERRORS ({len(self.errors)}):{Color.END}")
            for error in self.errors:
                print(f"  {error}")

        # Show warnings
        if self.warnings:
            print(f"\n{Color.BOLD}{Color.YELLOW}WARNINGS ({len(self.warnings)}):{Color.END}")
            for warning in self.warnings:
                print(f"  {warning}")

        # Show passed checks (only if no errors)
        if not self.errors and not self.warnings:
            print(f"\n{Color.BOLD}{Color.GREEN}ALL CHECKS PASSED ({len(self.passed)}):{Color.END}")
            for check in self.passed[:10]:  # Show first 10
                print(f"  {check}")
            if len(self.passed) > 10:
                print(f"  ... and {len(self.passed) - 10} more")

        # Summary
        print("\n" + "=" * 60)
        total = len(self.errors) + len(self.warnings) + len(self.passed)
        print(f"{Color.BOLD}Summary:{Color.END}")
        print(f"  Total checks: {total}")
        print(f"  {Color.GREEN}Passed: {len(self.passed)}{Color.END}")
        print(f"  {Color.YELLOW}Warnings: {len(self.warnings)}{Color.END}")
        print(f"  {Color.RED}Errors: {len(self.errors)}{Color.END}")
        print()


def load_env_file(env_file: str) -> Dict[str, str]:
    """Load environment variables from .env file"""
    env_vars = {}

    if not Path(env_file).exists():
        print(f"{Color.RED}Error: Environment file not found: {env_file}{Color.END}")
        sys.exit(1)

    with open(env_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                env_vars[key] = value

    return env_vars


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Validate ShivX environment configuration for production deployment"
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to environment file (default: .env)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on warnings (default: fail only on errors)"
    )

    args = parser.parse_args()

    # Load environment variables
    env_vars = load_env_file(args.env_file)

    # Run validation
    validator = EnvValidator(env_vars, strict=args.strict)
    passed = validator.validate_all()

    # Exit with appropriate code
    if passed:
        print(f"{Color.GREEN}{Color.BOLD}✓ Environment configuration is valid for production!{Color.END}\n")
        sys.exit(0)
    else:
        print(f"{Color.RED}{Color.BOLD}✗ Environment configuration has issues that must be fixed!{Color.END}\n")
        print(f"Fix all {Color.RED}ERRORS{Color.END} before deploying to production.")
        if args.strict:
            print(f"Running in --strict mode: also fix all {Color.YELLOW}WARNINGS{Color.END}.")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
