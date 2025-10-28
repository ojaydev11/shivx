#!/usr/bin/env python3
"""
Check if all acceptance criteria are met.

Returns exit code 0 if all pass, 1 if any fail.
"""

import json
import sys
from pathlib import Path

print("🎯 Checking Acceptance Criteria")
print("=" * 50)

scoreboard_path = Path("telemetry/rollups/scoreboard.json")

if not scoreboard_path.exists():
    print("❌ No scoreboard found")
    sys.exit(1)

with open(scoreboard_path) as f:
    data = json.load(f)

# Define criteria
criteria = [
    ("Memory recall p95", data.get("memory_recall_p95_ms", 0), 150, "less"),
    ("Learning improvement", data.get("learning_delta_pct", 0), 10, "greater"),
    ("Spatial success", data.get("spatial_success_pct", 0), 85, "greater"),
    ("ToM accuracy", data.get("tom_success_pct", 0), 80, "greater"),
]

passed = 0
failed = 0

for name, actual, target, compare in criteria:
    if actual == 0:
        status = "⏭️  SKIP"
        result = "not run"
    elif compare == "less":
        meets = actual < target
        status = "✅ PASS" if meets else "❌ FAIL"
        result = f"{actual:.1f} < {target}"
    else:  # greater
        meets = actual >= target
        status = "✅ PASS" if meets else "❌ FAIL"
        result = f"{actual:.1f} >= {target}"

    print(f"{status} | {name}: {result}")

    if actual > 0:
        if meets:
            passed += 1
        else:
            failed += 1

print("=" * 50)
print(f"Results: {passed} passed, {failed} failed")

if failed > 0:
    print("\n❌ Some acceptance criteria not met")
    sys.exit(1)
else:
    print("\n✅ All acceptance criteria met!")
    sys.exit(0)
