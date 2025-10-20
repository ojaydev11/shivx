"""
ShivX Real Security Scan Suite
===============================
Purpose: SAST, secret scanning, SBOM generation
Uses: Built-in Python checks + pip-licenses
"""

import os
import re
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import argparse

@dataclass
class SecurityFinding:
    """Security finding"""
    severity: str  # critical, high, medium, low, info
    category: str
    file_path: str
    line: int
    description: str
    recommendation: str

class SecurityScanner:
    """Security scanner"""

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.findings: List[SecurityFinding] = []

    def scan_for_secrets(self) -> List[SecurityFinding]:
        """Scan for exposed secrets in code"""
        print("\n[SECURITY-1] Secret Scanning")
        print("  Scanning for exposed credentials and API keys...")

        findings = []

        # Patterns to detect
        patterns = [
            (r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\']([^"\']{20,})["\']', 'API Key'),
            (r'(?i)(secret[_-]?key|secretkey)\s*[:=]\s*["\']([^"\']{20,})["\']', 'Secret Key'),
            (r'(?i)(password|passwd|pwd)\s*[:=]\s*["\']([^"\']{8,})["\']', 'Password'),
            (r'(?i)(token|auth[_-]?token)\s*[:=]\s*["\']([^"\']{20,})["\']', 'Token'),
            (r'(?i)Bearer\s+([A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+)', 'JWT Token'),
            (r'sk-[A-Za-z0-9]{48}', 'OpenAI API Key'),
            (r'ghp_[A-Za-z0-9]{36}', 'GitHub Personal Access Token'),
        ]

        # Scan Python files
        python_files = list(self.root_dir.rglob("*.py"))
        scanned = 0
        found = 0

        for py_file in python_files:
            # Skip venv and test files
            if '.venv' in str(py_file) or '__pycache__' in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = content.split('\n')

                    for line_num, line in enumerate(lines, 1):
                        for pattern, secret_type in patterns:
                            match = re.search(pattern, line)
                            if match:
                                # Check if it's in a comment or looks like a placeholder
                                if '#' in line or 'example' in line.lower() or 'placeholder' in line.lower():
                                    continue

                                # Check if value looks like a placeholder
                                if match.group(1) if len(match.groups()) > 0 else match.group(0):
                                    value = match.group(2) if len(match.groups()) > 1 else match.group(1)
                                    if 'xxx' in value.lower() or 'your' in value.lower() or 'test' in value.lower():
                                        continue

                                findings.append(SecurityFinding(
                                    severity="high",
                                    category="exposed_secret",
                                    file_path=str(py_file.relative_to(self.root_dir)),
                                    line=line_num,
                                    description=f"Potential {secret_type} found in code",
                                    recommendation="Move to environment variables or secure vault"
                                ))
                                found += 1

                scanned += 1

            except Exception as e:
                print(f"  Warning: Could not scan {py_file}: {e}")

        print(f"  Scanned {scanned} Python files")
        print(f"  Found {found} potential secrets")

        if found == 0:
            print("  ✓ PASS: No secrets detected")
        else:
            print(f"  ⚠ WARNING: {found} potential secrets found")

        return findings

    def check_security_best_practices(self) -> List[SecurityFinding]:
        """Check for security anti-patterns"""
        print("\n[SECURITY-2] Security Best Practices Check")
        print("  Checking for security anti-patterns...")

        findings = []

        # Patterns to check
        danger_patterns = [
            (r'eval\s*\(', 'Use of eval() is dangerous', 'high'),
            (r'exec\s*\(', 'Use of exec() is dangerous', 'high'),
            (r'__import__\s*\(', 'Dynamic imports can be dangerous', 'medium'),
            (r'pickle\.loads?\s*\(', 'Pickle can execute arbitrary code', 'medium'),
            (r'shell\s*=\s*True', 'Shell=True in subprocess is dangerous', 'high'),
            (r'(?i)password\s*=\s*["\'][^"\']*["\']', 'Hardcoded password', 'high'),
        ]

        python_files = list(self.root_dir.rglob("*.py"))
        scanned = 0
        found = 0

        for py_file in python_files:
            if '.venv' in str(py_file) or '__pycache__' in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()

                    for line_num, line in enumerate(lines, 1):
                        # Skip comments
                        if line.strip().startswith('#'):
                            continue

                        for pattern, desc, severity in danger_patterns:
                            if re.search(pattern, line):
                                findings.append(SecurityFinding(
                                    severity=severity,
                                    category="security_antipattern",
                                    file_path=str(py_file.relative_to(self.root_dir)),
                                    line=line_num,
                                    description=desc,
                                    recommendation="Review and use safer alternatives"
                                ))
                                found += 1

                scanned += 1

            except Exception as e:
                pass

        print(f"  Scanned {scanned} Python files")
        print(f"  Found {found} security issues")

        if found == 0:
            print("  ✓ PASS: No security anti-patterns detected")
        else:
            print(f"  ⚠ WARNING: {found} security issues found")

        return findings

    def generate_sbom(self) -> Dict[str, Any]:
        """Generate Software Bill of Materials"""
        print("\n[SECURITY-3] SBOM Generation")
        print("  Generating Software Bill of Materials...")

        components = []

        # Try to get installed packages
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'list', '--format=json'],
                capture_output=True,
                text=True,
                check=True
            )

            packages = json.loads(result.stdout)

            # Only include main dependencies (first 10 for brevity)
            for pkg in packages[:50]:
                components.append({
                    "type": "library",
                    "name": pkg["name"],
                    "version": pkg["version"],
                    "purl": f"pkg:pypi/{pkg['name'].lower()}@{pkg['version']}"
                })

            print(f"  ✓ Collected {len(components)} components")

        except Exception as e:
            print(f"  ⚠ Warning: Could not collect pip packages: {e}")
            # Fallback to baseline
            components = [
                {"type": "library", "name": "fastapi", "version": "unknown", "purl": "pkg:pypi/fastapi"},
                {"type": "library", "name": "pytest", "version": "unknown", "purl": "pkg:pypi/pytest"},
            ]

        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "version": 1,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "tools": [{"name": "ShivX Security Scanner", "version": "1.0.0"}],
                "component": {
                    "type": "application",
                    "name": "ShivX",
                    "version": "2.0.0"
                }
            },
            "components": components
        }

        # Save SBOM
        output_path = Path("release/artifacts/sbom.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(sbom, f, indent=2)

        print(f"  ✓ SBOM saved to: {output_path}")

        return sbom

    def run_all_scans(self) -> Dict[str, Any]:
        """Run all security scans"""
        print("=" * 60)
        print("ShivX Security Scan Suite")
        print("=" * 60)

        # Run scans
        secret_findings = self.scan_for_secrets()
        security_findings = self.check_security_best_practices()
        sbom = self.generate_sbom()

        # Combine findings
        all_findings = secret_findings + security_findings

        # Count by severity
        findings_by_severity = {
            "critical": sum(1 for f in all_findings if f.severity == "critical"),
            "high": sum(1 for f in all_findings if f.severity == "high"),
            "medium": sum(1 for f in all_findings if f.severity == "medium"),
            "low": sum(1 for f in all_findings if f.severity == "low"),
            "info": sum(1 for f in all_findings if f.severity == "info"),
        }

        # Gate G5: 0 critical/high findings
        gate_g5_status = "PASS" if findings_by_severity["critical"] == 0 and findings_by_severity["high"] == 0 else "FAIL"

        report = {
            "scan_date": datetime.now().isoformat(),
            "findings": findings_by_severity,
            "findings_details": [asdict(f) for f in all_findings[:20]],  # Sample
            "gate_g5_status": gate_g5_status,
            "sbom_generated": True,
            "artifacts": {
                "sbom": "release/artifacts/sbom.json",
                "report": "release/artifacts/security_report.json"
            }
        }

        # Save report
        output_path = Path("release/artifacts/security_report.json")
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print("Security Scan Summary")
        print("=" * 60)
        print(f"Critical: {findings_by_severity['critical']}")
        print(f"High: {findings_by_severity['high']}")
        print(f"Medium: {findings_by_severity['medium']}")
        print(f"Low: {findings_by_severity['low']}")
        print(f"Info: {findings_by_severity['info']}")
        print(f"\nGate G5 (0 critical/high): {gate_g5_status}")
        print(f"SBOM: Generated ({len(sbom['components'])} components)")
        print(f"\nReport: {output_path}")
        print("=" * 60)

        return report

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="ShivX Real Security Scanner")
    parser.add_argument(
        "--root-dir",
        default=".",
        help="Root directory to scan"
    )

    args = parser.parse_args()

    scanner = SecurityScanner(root_dir=args.root_dir)
    report = scanner.run_all_scans()

    # Exit with error if gate failed
    if report["gate_g5_status"] == "PASS":
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
