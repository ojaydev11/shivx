#!/bin/bash
# ============================================================================
# ShivX Integrations Setup Script
# ============================================================================
# Sets up all external integration dependencies and configurations
#
# Usage:
#   ./scripts/setup_integrations.sh
#
# Prerequisites:
#   - Python 3.10+ installed
#   - pip installed
#   - Virtual environment activated (recommended)
# ============================================================================

set -e  # Exit on error

echo "=========================================="
echo "ShivX Integrations Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    print_error "Python 3.10+ required. Found: $PYTHON_VERSION"
    exit 1
fi
print_status "Python version OK: $PYTHON_VERSION"

# Install Python dependencies
print_status "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_status "Python dependencies installed"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Install Playwright browsers
print_status "Installing Playwright browsers..."
if command -v playwright &> /dev/null; then
    playwright install chromium
    print_status "Playwright Chromium browser installed"
else
    print_error "Playwright not found. Install with: pip install playwright"
    exit 1
fi

# Create necessary directories
print_status "Creating integration directories..."
mkdir -p var/audit
mkdir -p var/tokens
mkdir -p var/logs
mkdir -p var/screenshots
mkdir -p config

print_status "Directories created"

# Check for environment variables
print_status "Checking environment configuration..."

ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
    print_warning ".env file not found. Creating from template..."
    cp .env.example .env
    print_warning "Please configure .env file with your API keys"
fi

# Required environment variables
REQUIRED_VARS=(
    "GITHUB_ACCESS_TOKEN"
    "GOOGLE_CREDENTIALS_PATH"
    "TELEGRAM_BOT_TOKEN"
    "TELEGRAM_ALLOWED_USER_IDS"
    "ANTHROPIC_API_KEY"
    "OPENAI_API_KEY"
    "BROWSER_URL_ALLOWLIST"
)

MISSING_VARS=()
for var in "${REQUIRED_VARS[@]}"; do
    if ! grep -q "^$var=" "$ENV_FILE" 2>/dev/null || grep -q "^$var=$" "$ENV_FILE" 2>/dev/null || grep -q "^$var=your_" "$ENV_FILE" 2>/dev/null; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    print_warning "Missing or incomplete environment variables:"
    for var in "${MISSING_VARS[@]}"; do
        echo "  - $var"
    done
    print_warning "Please configure these in .env file"
fi

# Setup Google OAuth2
print_status "Checking Google OAuth2 setup..."
if [ ! -f "config/google_credentials.json" ]; then
    print_warning "Google OAuth2 credentials not found at config/google_credentials.json"
    print_warning "Download from Google Cloud Console: https://console.cloud.google.com/apis/credentials"
else
    print_status "Google OAuth2 credentials found"
fi

# Test integrations (optional)
print_status "Testing integration imports..."
python3 -c "
import sys
try:
    from integrations import (
        get_github_client,
        get_google_client,
        get_telegram_bot,
        get_browser_automation,
        get_llm_client
    )
    print('✓ All integrations loaded successfully')
except Exception as e:
    print(f'✗ Integration import failed: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    print_status "Integration imports OK"
else
    print_error "Integration imports failed"
    exit 1
fi

# Run integration tests (optional)
echo ""
read -p "Run integration tests? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Running integration tests..."
    pytest tests/test_github_integration.py -v
    pytest tests/test_google_integration.py -v
    print_status "Tests completed"
fi

# Print summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
print_status "Integration dependencies installed"
print_status "Directory structure created"
print_status "Playwright browsers installed"
echo ""
echo "Next steps:"
echo "  1. Configure .env file with API keys"
echo "  2. Download Google OAuth2 credentials (if using Gmail/Calendar)"
echo "  3. Test integrations with: pytest tests/"
echo "  4. Start using integrations in your agents"
echo ""
echo "Documentation: docs/integrations.md"
echo ""
