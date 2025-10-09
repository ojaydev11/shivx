#!/usr/bin/env bash
# ShivX Development Environment Bootstrap Script (Linux/macOS)
# ==============================================================
# Purpose: Zero-to-running setup in ≤10 minutes (Gate G7)
# Usage: ./scripts/dev_bootstrap.sh

set -e  # Exit on error

echo "🚀 ShivX Development Bootstrap"
echo "================================"
echo ""

# ============================================================================
# Step 1: Check Python 3.10+ installed
# ============================================================================
echo "[1/10] Checking Python version..."

if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "❌ Python not found in PATH"
    echo "   Please install Python 3.10+ from https://www.python.org/"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "✅ Found: Python $PYTHON_VERSION"

# Extract major.minor version
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; }; then
    echo "❌ Python 3.10+ required, found $PYTHON_MAJOR.$PYTHON_MINOR"
    echo "   Please install Python 3.10 or later from https://www.python.org/"
    exit 1
fi

# ============================================================================
# Step 2: Create virtual environment
# ============================================================================
echo "[2/10] Creating virtual environment (.venv)..."

if [ -d ".venv" ]; then
    echo "⚠️  .venv already exists, skipping creation"
else
    $PYTHON_CMD -m venv .venv
    echo "✅ Created .venv"
fi

# ============================================================================
# Step 3: Activate virtual environment
# ============================================================================
echo "[3/10] Activating virtual environment..."

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Activated .venv"
else
    echo "❌ Failed to find activation script at .venv/bin/activate"
    exit 1
fi

# ============================================================================
# Step 4: Upgrade pip/setuptools
# ============================================================================
echo "[4/10] Upgrading pip and setuptools..."

python -m pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo "✅ Upgraded pip and setuptools"

# ============================================================================
# Step 5: Install dependencies from requirements.txt
# ============================================================================
echo "[5/10] Installing Python dependencies (this may take 3-5 minutes)..."

if [ -f "requirements.txt" ]; then
    python -m pip install -r requirements.txt
    echo "✅ Installed dependencies from requirements.txt"
else
    echo "❌ requirements.txt not found"
    exit 1
fi

# ============================================================================
# Step 6: Check Node.js installed (for Playwright)
# ============================================================================
echo "[6/10] Checking Node.js..."

if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo "✅ Found: Node $NODE_VERSION"
else
    echo "⚠️  Node.js not found (optional, needed for Playwright browsers)"
    echo "   You can install it later from https://nodejs.org/"
fi

# ============================================================================
# Step 7: Install Playwright browsers
# ============================================================================
echo "[7/10] Installing Playwright browsers (chromium)..."

if python -m playwright install chromium > /dev/null 2>&1; then
    echo "✅ Installed Playwright chromium"
else
    echo "⚠️  Failed to install Playwright browsers (non-fatal)"
fi

# ============================================================================
# Step 8: Validate .env exists
# ============================================================================
echo "[8/10] Checking .env configuration..."

if [ -f ".env" ]; then
    echo "✅ .env file exists"
    
    # Validate critical env vars
    WARNINGS=()
    
    if grep -q "ADMIN_TOKEN=changeme" .env; then
        WARNINGS+=("⚠️  ADMIN_TOKEN is still set to 'changeme'")
    fi
    
    if grep -q "JWT_SECRET=changeme" .env; then
        WARNINGS+=("⚠️  JWT_SECRET is still set to 'changeme'")
    fi
    
    if grep -q "ENCRYPTION_KEY=dGVzdC1lbmNyeXB0aW9uLWtleS1tdXN0LWJlLTMyLWNoYXJzIQ==" .env; then
        WARNINGS+=("⚠️  ENCRYPTION_KEY is still set to example value")
    fi
    
    if [ ${#WARNINGS[@]} -gt 0 ]; then
        echo "⚠️  WARNING: Please update these values in .env before running in production:"
        for warning in "${WARNINGS[@]}"; do
            echo "   $warning"
        done
    fi
    
else
    echo "⚠️  .env not found, copying from env.example..."
    
    if [ -f "env.example" ]; then
        cp env.example .env
        echo "✅ Created .env from env.example"
        echo "⚠️  IMPORTANT: Edit .env and set ADMIN_TOKEN, JWT_SECRET, and ENCRYPTION_KEY"
    else
        echo "❌ env.example not found, cannot create .env"
        echo "   Please create .env manually with required variables"
        exit 1
    fi
fi

# ============================================================================
# Step 9: Create required directories
# ============================================================================
echo "[9/10] Creating required directories..."

REQUIRED_DIRS=(
    "var"
    "var/runtime"
    "var/queues"
    "var/runs"
    "var/security"
    "logs"
    "data"
    "memory"
    "memory/episodic"
    "memory/chroma_db"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    mkdir -p "$dir"
done

echo "✅ Created required directories"

# ============================================================================
# Step 10: Run readiness check
# ============================================================================
echo "[10/10] Running readiness check..."

if python shivx_runner.py --mode readiness > /dev/null 2>&1; then
    echo "✅ Readiness check passed"
else
    READINESS_EXIT=$?
    echo "⚠️  Readiness check completed with warnings (exit code: $READINESS_EXIT)"
    echo "   This is OK for development; fix warnings before production deployment"
fi

# ============================================================================
# Success!
# ============================================================================
echo ""
echo "🎉 Bootstrap Complete!"
echo "======================"
echo ""
echo "Next steps:"
echo "  1. Review and update .env with secure values (ADMIN_TOKEN, JWT_SECRET, ENCRYPTION_KEY)"
echo "  2. Run the GUI dashboard:"
echo "     python shivx_runner.py --mode gui"
echo "     Then open: http://127.0.0.1:8051/dashboard"
echo "  3. OR run the main API server:"
echo "     python shivx_runner.py --mode prod"
echo "     API at: http://127.0.0.1:5099"
echo "  4. Run tests:"
echo "     pytest"
echo "  5. Install pre-commit hooks (optional):"
echo "     pre-commit install"
echo ""
echo "Documentation:"
echo "  - README.md"
echo "  - release/artifacts/wirecheck_report.md"
echo "  - release/STATUS.md"
echo ""
echo "Happy coding! 🚀"

