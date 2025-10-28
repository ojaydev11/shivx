#!/bin/bash
# ShivX Development Environment Validation Script
# Validates that the development environment is properly configured
# Usage: ./scripts/validate_dev_env.sh [--fix]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
FIX_ISSUES="${FIX_ISSUES:-false}"
VERBOSE="${VERBOSE:-false}"

# Counters
CHECKS_TOTAL=0
CHECKS_PASSED=0
CHECKS_FAILED=0
CHECKS_WARNING=0

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
    ((CHECKS_PASSED++))
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
    ((CHECKS_WARNING++))
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
    ((CHECKS_FAILED++))
}

check() {
    ((CHECKS_TOTAL++))
}

show_help() {
    echo "ShivX Development Environment Validation Script"
    echo ""
    echo "Usage: ./scripts/validate_dev_env.sh [options]"
    echo ""
    echo "Options:"
    echo "  --fix        Attempt to fix issues automatically"
    echo "  --verbose    Show detailed output"
    echo "  --help       Show this help message"
    echo ""
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fix)
            FIX_ISSUES="true"
            shift
            ;;
        --verbose)
            VERBOSE="true"
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

log_info "========================================"
log_info "ShivX Environment Validation"
log_info "========================================"
echo ""

if [ "$FIX_ISSUES" = "true" ]; then
    log_warning "Auto-fix mode enabled - will attempt to fix issues"
    echo ""
fi

# ============================================================================
# 1. Python Environment
# ============================================================================

log_info "Checking Python environment..."
echo ""

# Check Python installation
check
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 10 ] && [ "$PYTHON_MINOR" -le 12 ]; then
        log_success "Python version: $PYTHON_VERSION"
    else
        log_error "Python version $PYTHON_VERSION not supported (need 3.10-3.12)"
        log_info "  Install Python 3.10, 3.11, or 3.12"
    fi
else
    log_error "Python 3 not found"
    if [ "$FIX_ISSUES" = "true" ]; then
        log_info "  Please install Python 3.10+ manually"
    fi
fi

# Check pip
check
if command -v pip3 &> /dev/null || command -v pip &> /dev/null; then
    PIP_VERSION=$(pip3 --version 2>/dev/null || pip --version | awk '{print $2}')
    log_success "pip version: $PIP_VERSION"
else
    log_error "pip not found"
    if [ "$FIX_ISSUES" = "true" ]; then
        log_info "  Installing pip..."
        python3 -m ensurepip --upgrade || log_error "Failed to install pip"
    fi
fi

# Check virtual environment
check
if [ -d "venv" ]; then
    log_success "Virtual environment exists at: venv/"

    # Check if activated
    if [ -n "$VIRTUAL_ENV" ]; then
        log_success "Virtual environment is activated"
    else
        log_warning "Virtual environment not activated"
        log_info "  Activate with: source venv/bin/activate"
    fi

    # Check installed packages
    check
    if [ -f "venv/bin/python" ]; then
        INSTALLED_COUNT=$(venv/bin/pip list 2>/dev/null | wc -l)
        if [ "$INSTALLED_COUNT" -gt 10 ]; then
            log_success "Dependencies installed ($INSTALLED_COUNT packages)"
        else
            log_warning "Few dependencies installed ($INSTALLED_COUNT packages)"
            log_info "  Run: pip install -r requirements.txt"
        fi
    fi
else
    log_error "Virtual environment not found"
    if [ "$FIX_ISSUES" = "true" ]; then
        log_info "  Creating virtual environment..."
        python3 -m venv venv
        log_success "Virtual environment created"
        log_info "  Installing dependencies..."
        venv/bin/pip install --upgrade pip
        venv/bin/pip install -r requirements.txt
        venv/bin/pip install -r requirements-dev.txt
        log_success "Dependencies installed"
    else
        log_info "  Create with: make setup"
    fi
fi

echo ""

# ============================================================================
# 2. System Dependencies
# ============================================================================

log_info "Checking system dependencies..."
echo ""

# Check Git
check
if command -v git &> /dev/null; then
    GIT_VERSION=$(git --version | awk '{print $3}')
    log_success "Git version: $GIT_VERSION"
else
    log_error "Git not found"
    log_info "  Install: sudo apt-get install git (Ubuntu/Debian)"
fi

# Check Docker
check
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | awk '{print $3}' | tr -d ',')
    log_success "Docker version: $DOCKER_VERSION"

    # Check if Docker daemon is running
    check
    if docker ps &> /dev/null; then
        log_success "Docker daemon is running"
    else
        log_error "Docker daemon not running"
        log_info "  Start with: sudo systemctl start docker"
    fi
else
    log_warning "Docker not found (optional but recommended)"
    log_info "  Install: https://docs.docker.com/get-docker/"
fi

# Check Docker Compose
check
if command -v docker-compose &> /dev/null || command -v docker &> /dev/null && docker compose version &> /dev/null; then
    if command -v docker-compose &> /dev/null; then
        COMPOSE_VERSION=$(docker-compose --version | awk '{print $3}' | tr -d ',')
    else
        COMPOSE_VERSION=$(docker compose version --short)
    fi
    log_success "Docker Compose version: $COMPOSE_VERSION"
else
    log_warning "Docker Compose not found (optional)"
fi

# Check Make
check
if command -v make &> /dev/null; then
    MAKE_VERSION=$(make --version | head -1 | awk '{print $3}')
    log_success "Make version: $MAKE_VERSION"
else
    log_warning "Make not found (optional but convenient)"
    log_info "  Install: sudo apt-get install build-essential (Ubuntu/Debian)"
fi

# Check curl
check
if command -v curl &> /dev/null; then
    CURL_VERSION=$(curl --version | head -1 | awk '{print $2}')
    log_success "curl version: $CURL_VERSION"
else
    log_error "curl not found"
    log_info "  Install: sudo apt-get install curl (Ubuntu/Debian)"
fi

# Check jq (for JSON processing)
check
if command -v jq &> /dev/null; then
    JQ_VERSION=$(jq --version | tr -d 'jq-')
    log_success "jq version: $JQ_VERSION"
else
    log_warning "jq not found (recommended for JSON processing)"
    log_info "  Install: sudo apt-get install jq (Ubuntu/Debian)"
fi

echo ""

# ============================================================================
# 3. Configuration Files
# ============================================================================

log_info "Checking configuration files..."
echo ""

# Check .env file
check
if [ -f ".env" ]; then
    log_success ".env file exists"

    # Check for critical variables
    REQUIRED_VARS="DATABASE_URL REDIS_URL SECRET_KEY"
    for VAR in $REQUIRED_VARS; do
        check
        if grep -q "^$VAR=" .env; then
            VALUE=$(grep "^$VAR=" .env | cut -d= -f2)
            if [ -n "$VALUE" ] && [ "$VALUE" != "your_" ] && [ "$VALUE" != "changeme" ]; then
                log_success "$VAR is configured"
            else
                log_warning "$VAR needs to be set in .env"
            fi
        else
            log_warning "$VAR not found in .env"
        fi
    done
else
    log_error ".env file not found"
    if [ "$FIX_ISSUES" = "true" ]; then
        log_info "  Creating .env from template..."
        cp .env.example .env
        log_success ".env created from .env.example"
        log_warning "  Please update .env with your configuration"
    else
        log_info "  Create with: cp .env.example .env"
    fi
fi

# Check pyproject.toml
check
if [ -f "pyproject.toml" ]; then
    log_success "pyproject.toml exists"
else
    log_error "pyproject.toml not found (critical)"
fi

# Check requirements.txt
check
if [ -f "requirements.txt" ]; then
    REQ_COUNT=$(grep -v '^#' requirements.txt | grep -v '^$' | wc -l)
    log_success "requirements.txt exists ($REQ_COUNT dependencies)"
else
    log_error "requirements.txt not found (critical)"
fi

# Check requirements-dev.txt
check
if [ -f "requirements-dev.txt" ]; then
    log_success "requirements-dev.txt exists"
else
    log_warning "requirements-dev.txt not found"
fi

echo ""

# ============================================================================
# 4. Directory Structure
# ============================================================================

log_info "Checking directory structure..."
echo ""

REQUIRED_DIRS="app core utils config tests"
for DIR in $REQUIRED_DIRS; do
    check
    if [ -d "$DIR" ]; then
        log_success "Directory exists: $DIR/"
    else
        log_error "Directory missing: $DIR/"
    fi
done

OPTIONAL_DIRS="logs data var/resilience models/checkpoints release/artifacts"
for DIR in $OPTIONAL_DIRS; do
    check
    if [ -d "$DIR" ]; then
        log_success "Directory exists: $DIR/"
    else
        log_warning "Directory missing: $DIR/"
        if [ "$FIX_ISSUES" = "true" ]; then
            mkdir -p "$DIR"
            log_success "Created: $DIR/"
        else
            log_info "  Create with: mkdir -p $DIR"
        fi
    fi
done

echo ""

# ============================================================================
# 5. Database Connectivity
# ============================================================================

log_info "Checking database connectivity..."
echo ""

if [ -f ".env" ]; then
    # Check PostgreSQL (if configured)
    check
    if grep -q "postgresql://" .env; then
        log_info "PostgreSQL configured"
        if command -v psql &> /dev/null; then
            log_success "psql client available"
        else
            log_warning "psql client not found (optional)"
        fi
    fi

    # Check Redis (if configured)
    check
    if grep -q "redis://" .env; then
        log_info "Redis configured"
        if command -v redis-cli &> /dev/null; then
            REDIS_URL=$(grep "^REDIS_URL=" .env | cut -d= -f2 | cut -d/ -f3)
            REDIS_HOST=$(echo "$REDIS_URL" | cut -d: -f1)
            REDIS_PORT=$(echo "$REDIS_URL" | cut -d: -f2)

            if redis-cli -h "${REDIS_HOST:-localhost}" -p "${REDIS_PORT:-6379}" ping &> /dev/null; then
                log_success "Redis is reachable"
            else
                log_warning "Redis not reachable (may not be running)"
            fi
        else
            log_warning "redis-cli not found (optional)"
        fi
    fi
fi

echo ""

# ============================================================================
# 6. Development Tools
# ============================================================================

log_info "Checking development tools..."
echo ""

DEV_TOOLS="black flake8 pytest mypy bandit isort"
for TOOL in $DEV_TOOLS; do
    check
    if [ -d "venv" ] && [ -f "venv/bin/$TOOL" ]; then
        log_success "$TOOL is installed"
    elif command -v $TOOL &> /dev/null; then
        log_success "$TOOL is installed (system)"
    else
        log_warning "$TOOL not found"
        if [ "$FIX_ISSUES" = "true" ] && [ -d "venv" ]; then
            log_info "  Installing $TOOL..."
            venv/bin/pip install $TOOL -q
            log_success "$TOOL installed"
        else
            log_info "  Install with: pip install -r requirements-dev.txt"
        fi
    fi
done

echo ""

# ============================================================================
# 7. Port Availability
# ============================================================================

log_info "Checking port availability..."
echo ""

PORTS="8000 5432 6379 9090 3000"
for PORT in $PORTS; do
    check
    if ! lsof -i ":$PORT" &> /dev/null && ! netstat -an 2>/dev/null | grep -q ":$PORT "; then
        log_success "Port $PORT is available"
    else
        log_warning "Port $PORT is in use"
        if [ "$VERBOSE" = "true" ]; then
            lsof -i ":$PORT" 2>/dev/null || netstat -an | grep ":$PORT"
        fi
    fi
done

echo ""

# ============================================================================
# 8. Git Configuration
# ============================================================================

log_info "Checking Git configuration..."
echo ""

# Check if in git repo
check
if git rev-parse --git-dir > /dev/null 2>&1; then
    log_success "Git repository initialized"

    # Check git user config
    check
    GIT_NAME=$(git config user.name || echo "")
    if [ -n "$GIT_NAME" ]; then
        log_success "Git user.name configured: $GIT_NAME"
    else
        log_warning "Git user.name not configured"
        log_info "  Set with: git config user.name 'Your Name'"
    fi

    check
    GIT_EMAIL=$(git config user.email || echo "")
    if [ -n "$GIT_EMAIL" ]; then
        log_success "Git user.email configured: $GIT_EMAIL"
    else
        log_warning "Git user.email not configured"
        log_info "  Set with: git config user.email 'your.email@example.com'"
    fi

    # Check for remote
    check
    if git remote -v | grep -q origin; then
        ORIGIN_URL=$(git remote get-url origin)
        log_success "Git remote configured: $ORIGIN_URL"
    else
        log_warning "No git remote configured"
    fi
else
    log_error "Not a git repository"
fi

echo ""

# ============================================================================
# Summary Report
# ============================================================================

log_info "========================================"
log_info "VALIDATION SUMMARY"
log_info "========================================"
echo ""

echo "Total checks:     $CHECKS_TOTAL"
echo -e "${GREEN}Passed:          $CHECKS_PASSED${NC}"
echo -e "${RED}Failed:          $CHECKS_FAILED${NC}"
echo -e "${YELLOW}Warnings:        $CHECKS_WARNING${NC}"

echo ""

# Calculate score
SCORE=$((CHECKS_PASSED * 100 / CHECKS_TOTAL))

if [ $CHECKS_FAILED -eq 0 ]; then
    if [ $CHECKS_WARNING -eq 0 ]; then
        log_success "Environment is fully configured! Score: ${SCORE}%"
        exit 0
    else
        log_warning "Environment is mostly configured but has warnings. Score: ${SCORE}%"
        log_info "Review warnings above for optional improvements"
        exit 0
    fi
else
    log_error "Environment has critical issues. Score: ${SCORE}%"
    log_error "Please fix the errors above before proceeding"
    echo ""
    log_info "Quick fix:"
    log_info "  ./scripts/validate_dev_env.sh --fix"
    log_info "Or step by step:"
    log_info "  make setup"
    exit 1
fi
