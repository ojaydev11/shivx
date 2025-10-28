# AGI Validation CI Workflow Setup

## Overview

The AGI validation pipeline includes a GitHub Actions workflow that automates validation on every PR. However, this file cannot be pushed via GitHub Apps without `workflows` permission.

## Manual Setup Required

To enable the CI workflow, manually commit the following file:

**File:** `.github/workflows/agi-validate.yml`

```yaml
name: AGI Validation Pipeline

on:
  pull_request:
    branches: [main]
  push:
    branches:
      - 'claude/shivx-agi-architecture-*'

jobs:
  validate:
    runs-on: ubuntu-latest
    timeout-minutes: 30

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run bootstrap
        run: bash ops/bootstrap.sh

      - name: Quick smoke test
        run: python quick_test.py

      - name: Generate test data (quick mode)
        run: python ops/perf_memory_generate.py --quick

      - name: Run performance benchmarks
        run: python -m pytest tests/e2e/test_performance_suite.py -v -s

      - name: Run hardening guard
        run: bash ops/hardening_guard.sh

      - name: Generate scoreboard
        run: |
          python ops/generate_scoreboard_html.py
          python ops/generate_validation_report.py

      - name: Check acceptance criteria
        run: python ops/check_acceptance.py

      - name: Upload scoreboard
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: scoreboard
          path: |
            telemetry/rollups/scoreboard.html
            telemetry/rollups/scoreboard.json
            reports/agi_validation_report.md
```

## How to Add

From your local environment with proper permissions:

```bash
# Create the workflow directory if needed
mkdir -p .github/workflows

# Create the workflow file with the content above
# Then commit and push
git add .github/workflows/agi-validate.yml
git commit -m "[CI] Add AGI validation workflow"
git push
```

## What This Enables

- **Automated Validation:** Every PR runs the full validation suite
- **Performance Regression Detection:** Catches performance degradation
- **Security Checks:** Hardening guard runs on every change
- **Acceptance Gate:** PRs must meet all acceptance criteria to merge
- **Scoreboard Artifacts:** Download HTML dashboards from CI runs

## Acceptance Criteria

The workflow enforces these non-negotiable targets:

- Memory Recall p95: < 150ms
- Learning Improvement: ≥ 10%
- Spatial Success Rate: ≥ 85%
- Theory-of-Mind Accuracy: ≥ 80%
- Daemon Uptime: ≥ 99.9%

## Local Testing

Run the same validation locally before pushing:

```bash
bash ops/run_evals.sh
```

This executes the complete pipeline and generates the scoreboard.
