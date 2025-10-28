#!/bin/bash
# ============================================================================
# Fault Injection for MTTR Testing
# ============================================================================
# Introduces a safe, reversible fault to test self-repair
# ============================================================================

echo "💉 Injecting fault for MTTR test..."

# Create a temporary faulty module
FAULT_FILE="memory/test_fault_temp.py"

cat > "$FAULT_FILE" <<'EOF'
"""Temporary faulty module for MTTR testing."""

def buggy_function():
    # This will cause an AttributeError
    obj = None
    return obj.value  # AttributeError: 'NoneType' object has no attribute 'value'
EOF

echo "   ✅ Fault injected: $FAULT_FILE"
echo "   Type: AttributeError (missing attribute)"
echo ""
echo "To remove fault, run:"
echo "   rm $FAULT_FILE"
