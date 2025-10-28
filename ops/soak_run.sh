#!/bin/bash
# ============================================================================
# ShivX AGI Soak Test Runner
# ============================================================================
# Runs daemons for extended period with health monitoring
# Full soak: 24 hours
# Quick soak: 5 minutes (for CI)
# ============================================================================

DURATION_HOURS=${1:-24}
DURATION_SECONDS=$((DURATION_HOURS * 3600))

if [ "$DURATION_HOURS" -lt 1 ]; then
    # Quick mode (5 minutes)
    DURATION_SECONDS=300
    echo "🚀 Quick soak mode: 5 minutes"
else
    echo "🚀 Full soak test: $DURATION_HOURS hours"
fi

echo "=============================================="
echo ""

mkdir -p telemetry/soak
LOG_FILE="telemetry/soak/soak_$(date +%Y%m%d_%H%M%S).log"

echo "📝 Logging to: $LOG_FILE"
echo "⏱️  Duration: $DURATION_SECONDS seconds"
echo ""

# Start timestamp
START_TIME=$(date +%s)

# Health check function
check_health() {
    # Simple health check - verify quick_test passes
    python quick_test.py > /dev/null 2>&1
    return $?
}

# Main soak loop
echo "🏃 Starting soak test..." | tee -a "$LOG_FILE"
CHECKS=0
FAILURES=0

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    if [ $ELAPSED -ge $DURATION_SECONDS ]; then
        break
    fi

    # Run health check
    CHECKS=$((CHECKS + 1))
    if check_health; then
        STATUS="✅ PASS"
    else
        STATUS="❌ FAIL"
        FAILURES=$((FAILURES + 1))
    fi

    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "$TIMESTAMP | Check #$CHECKS | $STATUS | Elapsed: ${ELAPSED}s" | tee -a "$LOG_FILE"

    # Sleep between checks (30 seconds)
    sleep 30
done

# Calculate uptime
UPTIME_PCT=$(python -c "print(((${CHECKS} - ${FAILURES}) / ${CHECKS} * 100) if ${CHECKS} > 0 else 0)")

echo "" | tee -a "$LOG_FILE"
echo "=============================================="  | tee -a "$LOG_FILE"
echo "✅ Soak test complete!" | tee -a "$LOG_FILE"
echo "=============================================="  | tee -a "$LOG_FILE"
echo "Duration: ${ELAPSED}s" | tee -a "$LOG_FILE"
echo "Health checks: $CHECKS" | tee -a "$LOG_FILE"
echo "Failures: $FAILURES" | tee -a "$LOG_FILE"
echo "Uptime: ${UPTIME_PCT}%" | tee -a "$LOG_FILE"
echo "=============================================="  | tee -a "$LOG_FILE"

# Write metrics
cat > telemetry/rollups/soak_results.json <<EOF
{
  "daemon_uptime_pct": $UPTIME_PCT,
  "duration_seconds": $ELAPSED,
  "health_checks": $CHECKS,
  "failures": $FAILURES,
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo ""
echo "📊 Results: telemetry/rollups/soak_results.json"

# Check acceptance
if (( $(echo "$UPTIME_PCT >= 99.9" | bc -l) )); then
    echo "✅ Uptime ${UPTIME_PCT}% meets 99.9% target"
    exit 0
else
    echo "❌ Uptime ${UPTIME_PCT}% below 99.9% target"
    exit 1
fi
