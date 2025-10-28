#!/usr/bin/env python3
"""Generate HTML scoreboard from metrics."""

import json
from datetime import datetime
from pathlib import Path

# Read scoreboard
scoreboard_path = Path("telemetry/rollups/scoreboard.json")
if not scoreboard_path.exists():
    print("No scoreboard.json found")
    exit(1)

with open(scoreboard_path) as f:
    data = json.load(f)

# Define thresholds
thresholds = {
    "memory_recall_p95_ms": {"target": 150, "compare": "less"},
    "learning_delta_pct": {"target": 10, "compare": "greater"},
    "spatial_success_pct": {"target": 85, "compare": "greater"},
    "tom_success_pct": {"target": 80, "compare": "greater"},
    "self_repair_mttr_min": {"target": 5, "compare": "less"},
    "daemon_uptime_pct": {"target": 99.9, "compare": "greater"},
}

def check_pass(value, threshold_info):
    """Check if value meets threshold."""
    if value == 0:
        return "pending"
    target = threshold_info["target"]
    compare = threshold_info["compare"]

    if compare == "less":
        return "pass" if value < target else "fail"
    else:  # greater
        return "pass" if value >= target else "fail"

# Generate HTML
html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ShivX AGI Scoreboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }}
        .meta {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .meta strong {{
            color: #666;
        }}
        table {{
            width: 100%;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-collapse: collapse;
        }}
        th {{
            background: #007bff;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        .badge {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .badge-pass {{
            background: #28a745;
            color: white;
        }}
        .badge-fail {{
            background: #dc3545;
            color: white;
        }}
        .badge-pending {{
            background: #ffc107;
            color: #333;
        }}
        .value-pass {{
            color: #28a745;
            font-weight: bold;
        }}
        .value-fail {{
            color: #dc3545;
            font-weight: bold;
        }}
        .footer {{
            margin-top: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>🚀 ShivX AGI Performance Scoreboard</h1>

    <div class="meta">
        <strong>Commit:</strong> {data.get('commit', 'N/A')}<br>
        <strong>Branch:</strong> {data.get('branch', 'N/A')}<br>
        <strong>Timestamp:</strong> {data.get('timestamp', 'N/A')}
    </div>

    <table>
        <thead>
            <tr>
                <th>Metric</th>
                <th>Actual</th>
                <th>Target</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
"""

# Add rows
metrics = [
    ("Memory Recall p95", data["memory_recall_p95_ms"], "< 150ms", "memory_recall_p95_ms"),
    ("Learning Improvement", data["learning_delta_pct"], "≥ 10%", "learning_delta_pct"),
    ("Spatial Success", data["spatial_success_pct"], "≥ 85%", "spatial_success_pct"),
    ("ToM Accuracy", data["tom_success_pct"], "≥ 80%", "tom_success_pct"),
    ("Self-Repair MTTR", data["self_repair_mttr_min"], "< 5 min", "self_repair_mttr_min"),
    ("Daemon Uptime", data["daemon_uptime_pct"], "≥ 99.9%", "daemon_uptime_pct"),
]

for name, value, target_str, key in metrics:
    status = check_pass(value, thresholds[key])
    status_class = f"badge-{status}"
    value_class = f"value-{status}" if status != "pending" else ""

    # Format value
    if "pct" in key or "uptime" in key:
        value_str = f"{value:.1f}%" if value > 0 else "N/A"
    elif "ms" in key:
        value_str = f"{value:.1f} ms" if value > 0 else "N/A"
    elif "min" in key:
        value_str = f"{value:.1f} min" if value > 0 else "N/A"
    else:
        value_str = f"{value:.1f}" if value > 0 else "N/A"

    html += f"""
            <tr>
                <td><strong>{name}</strong></td>
                <td class="{value_class}">{value_str}</td>
                <td>{target_str}</td>
                <td><span class="badge {status_class}">{status}</span></td>
            </tr>
"""

html += """
        </tbody>
    </table>

    <div class="footer">
        <p>🤖 Generated with <a href="https://claude.com/claude-code">Claude Code</a></p>
    </div>
</body>
</html>
"""

# Write HTML
output_path = Path("telemetry/rollups/scoreboard.html")
with open(output_path, "w") as f:
    f.write(html)

print(f"✅ Generated {output_path}")
