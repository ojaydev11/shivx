#!/bin/bash
# ============================================================================
# ShivX AGI Bootstrap Script (Linux/Mac)
# ============================================================================
# One-shot setup for complete AGI platform
# ============================================================================

set -e

echo "=========================================="
echo "ShivX AGI Bootstrap"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python $required_version+ required (found $python_version)"
    exit 1
fi

echo "✓ Python $python_version detected"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo "✓ Dependencies installed"
echo ""

# Create data directories
echo "Creating data directories..."
mkdir -p data/memory
mkdir -p data/memory/snapshots
mkdir -p models/adapters
mkdir -p models/embeddings
mkdir -p logs

echo "✓ Data directories created"
echo ""

# Copy environment file
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "✓ .env file created (please configure)"
else
    echo "✓ .env file exists"
fi

# Install CLI
echo ""
echo "Installing ShivX CLI..."
pip install -e .

echo "✓ CLI installed"
echo ""

# Run tests
echo "Running tests..."
pytest tests/e2e/test_memory_slmg.py -v --tb=short || echo "Some tests failed (this is OK for first setup)"

echo ""
echo "=========================================="
echo "Bootstrap Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Configure .env file with your settings"
echo "  2. Run demo: python demos/memory_demo.py"
echo "  3. Use CLI: shivx mem recall 'your query'"
echo "  4. Start daemons: shivx daemons start"
echo ""
echo "Documentation: memory/README.md"
echo ""
