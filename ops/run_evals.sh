#!/bin/bash
# ============================================================================
# ShivX AGI Evaluation Runner
# ============================================================================
# Runs complete evaluation suite and generates scoreboard
# ============================================================================

set -e

echo "=============================================="
echo "ShivX AGI Evaluation Suite"
echo "=============================================="
echo ""

# Create telemetry directories
mkdir -p telemetry/rollups
mkdir -p reports

# Get commit info
COMMIT=$(git rev-parse --short HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "Branch: $BRANCH"
echo "Commit: $COMMIT"
echo "Time: $TIMESTAMP"
echo ""

# Step 1: Generate performance test data if needed
if [ ! -f "data/memory/perf_test.db" ]; then
    echo "📊 Generating performance test data (quick mode)..."
    python ops/perf_memory_generate.py --quick || true
    echo ""
fi

# Step 2: Run performance tests
echo "🧪 Running performance test suite..."
python -m pytest tests/e2e/test_performance_suite.py -v -s --tb=short || true
echo ""

# Step 3: Collect metrics from telemetry
echo "📈 Collecting metrics..."

# Default values
MEMORY_P95=0
LEARNING_DELTA=0
SPATIAL_SUCCESS=0
TOM_SUCCESS=0
REFLECTION_RATE=0

# Read from telemetry files
if [ -f "telemetry/rollups/memory_perf.json" ]; then
    MEMORY_P95=$(python -c "import json; print(json.load(open('telemetry/rollups/memory_perf.json'))['memory_recall_p95_ms'])")
fi

if [ -f "telemetry/rollups/learning_perf.json" ]; then
    LEARNING_DELTA=$(python -c "import json; print(json.load(open('telemetry/rollups/learning_perf.json'))['learning_delta_pct'])")
fi

if [ -f "telemetry/rollups/spatial_perf.json" ]; then
    SPATIAL_SUCCESS=$(python -c "import json; print(json.load(open('telemetry/rollups/spatial_perf.json'))['spatial_success_pct'])")
fi

if [ -f "telemetry/rollups/tom_perf.json" ]; then
    TOM_SUCCESS=$(python -c "import json; print(json.load(open('telemetry/rollups/tom_perf.json'))['tom_success_pct'])")
fi

if [ -f "telemetry/rollups/reflection_perf.json" ]; then
    REFLECTION_RATE=$(python -c "import json; print(json.load(open('telemetry/rollups/reflection_perf.json'))['hallucination_rate_pct'])")
fi

# Step 4: Generate scoreboard.json
echo "📋 Generating scoreboard..."

cat > telemetry/rollups/scoreboard.json <<EOF
{
  "memory_recall_p95_ms": $MEMORY_P95,
  "learning_delta_pct": $LEARNING_DELTA,
  "spatial_success_pct": $SPATIAL_SUCCESS,
  "tom_success_pct": $TOM_SUCCESS,
  "self_repair_mttr_min": 0,
  "daemon_uptime_pct": 0,
  "hallucination_rate_pct": $REFLECTION_RATE,
  "commit": "$COMMIT",
  "branch": "$BRANCH",
  "timestamp": "$TIMESTAMP"
}
EOF

# Step 5: Generate HTML scoreboard
python ops/generate_scoreboard_html.py

echo ""
echo "=============================================="
echo "✅ Evaluation Complete!"
echo "=============================================="
echo ""
echo "Results:"
echo "  Memory recall p95: ${MEMORY_P95}ms (target: <150ms)"
echo "  Learning improvement: ${LEARNING_DELTA}% (target: >=10%)"
echo "  Spatial success: ${SPATIAL_SUCCESS}% (target: >=85%)"
echo "  ToM accuracy: ${TOM_SUCCESS}% (target: >=80%)"
echo ""
echo "Artifacts:"
echo "  📊 telemetry/rollups/scoreboard.json"
echo "  🌐 telemetry/rollups/scoreboard.html"
echo ""
