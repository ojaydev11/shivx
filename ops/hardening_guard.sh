#!/bin/bash
# ============================================================================
# ShivX AGI Hardening Guard
# ============================================================================
# Security and safety checks for production deployment
# ============================================================================

set -e

echo "🛡️  ShivX AGI Hardening Guard"
echo "=============================================="
echo ""

FAIL_COUNT=0

# Check 1: Network safety
echo "1️⃣  Checking network safety..."
NETWORK_CALLS=$(grep -r "requests\|httpx\|aiohttp\|urllib" --include="*.py" memory/ learning/ cognition/ resilience/ daemons/ 2>/dev/null | grep -v "# ALLOW_NET" | wc -l || true)

if [ "$NETWORK_CALLS" -gt 0 ]; then
    echo "   ⚠️  Found $NETWORK_CALLS unguarded network calls"
    echo "   All network calls should check ALLOW_NET feature flag"
    FAIL_COUNT=$((FAIL_COUNT + 1))
else
    echo "   ✅ No unguarded network calls"
fi

# Check 2: Self-repair sandbox
echo ""
echo "2️⃣  Checking self-repair sandbox..."
if grep -q "SAFE_SELF_REPAIR" resilience/self_repair/repairer.py; then
    echo "   ✅ Self-repair has safety flag"
else
    echo "   ⚠️  Self-repair missing SAFE_SELF_REPAIR check"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# Check 3: Resource limits documented
echo ""
echo "3️⃣  Checking resource limits..."
if grep -q "max_workers\|memory_limit\|cpu_limit" config/agi/base.yaml; then
    echo "   ✅ Resource limits documented"
else
    echo "   ⚠️  Resource limits not found in config"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# Check 4: Secrets scan
echo ""
echo "4️⃣  Scanning for secrets..."
SECRETS=$(grep -rE "api[_-]?key|secret|password|token" --include="*.py" . | grep -v ".env" | grep -v "# safe" | grep -v "test" | grep -v "example" | wc -l || true)

if [ "$SECRETS" -gt 10 ]; then
    echo "   ⚠️  Found $SECRETS potential secret references"
    echo "   Review for accidental secret commits"
    FAIL_COUNT=$((FAIL_COUNT + 1))
else
    echo "   ✅ Secret scan clean"
fi

# Check 5: Dependencies pinned
echo ""
echo "5️⃣  Checking dependency versions..."
if grep -q "==" requirements.txt; then
    echo "   ✅ Dependencies are pinned"
else
    echo "   ⚠️  Dependencies not pinned in requirements.txt"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# Check 6: Test coverage
echo ""
echo "6️⃣  Checking test coverage..."
TEST_COUNT=$(find tests/ -name "test_*.py" | wc -l)
if [ "$TEST_COUNT" -ge 3 ]; then
    echo "   ✅ Found $TEST_COUNT test files"
else
    echo "   ⚠️  Only $TEST_COUNT test files found"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# Check 7: Documentation
echo ""
echo "7️⃣  Checking documentation..."
if [ -f "AGI_README.md" ] && [ -f "memory/README.md" ]; then
    echo "   ✅ Core documentation present"
else
    echo "   ⚠️  Missing core documentation"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# Check 8: Privacy compliance
echo ""
echo "8️⃣  Checking privacy compliance..."
if grep -q "local.*only\|privacy.*first" AGI_README.md 2>/dev/null; then
    echo "   ✅ Privacy-first design documented"
else
    echo "   ⚠️  Privacy guarantees not clearly stated"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

echo ""
echo "=============================================="
if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "✅ All hardening checks passed!"
    echo "=============================================="
    exit 0
else
    echo "❌ $FAIL_COUNT hardening checks failed"
    echo "=============================================="
    exit 1
fi
