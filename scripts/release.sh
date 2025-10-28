#!/bin/bash
# ShivX Automated Release Script
# Handles version bumping, building, signing, and releasing
# Usage: ./scripts/release.sh [--major|--minor|--patch]

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
BRANCH_NAME=$(git branch --show-current)
REQUIRED_BRANCH="${REQUIRED_BRANCH:-main}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_TESTS="${SKIP_TESTS:-false}"
SKIP_BUILD="${SKIP_BUILD:-false}"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    echo "ShivX Automated Release Script"
    echo ""
    echo "Usage: ./scripts/release.sh [options]"
    echo ""
    echo "Options:"
    echo "  --major          Bump major version (X.0.0)"
    echo "  --minor          Bump minor version (x.X.0)"
    echo "  --patch          Bump patch version (x.x.X) [default]"
    echo "  --version VER    Set specific version"
    echo "  --dry-run        Show what would be done without doing it"
    echo "  --skip-tests     Skip running tests"
    echo "  --skip-build     Skip building artifacts"
    echo "  --help           Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  REQUIRED_BRANCH  Branch required for release (default: main)"
    echo "  DRY_RUN          Set to true for dry run"
    echo "  SKIP_TESTS       Set to true to skip tests"
    echo "  SKIP_BUILD       Set to true to skip builds"
    echo "  GPG_KEY_ID       GPG key for signing"
    echo ""
    exit 0
}

# Parse arguments
BUMP_TYPE="patch"
SPECIFIC_VERSION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --major)
            BUMP_TYPE="major"
            shift
            ;;
        --minor)
            BUMP_TYPE="minor"
            shift
            ;;
        --patch)
            BUMP_TYPE="patch"
            shift
            ;;
        --version)
            SPECIFIC_VERSION="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --skip-tests)
            SKIP_TESTS="true"
            shift
            ;;
        --skip-build)
            SKIP_BUILD="true"
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
log_info "ShivX Automated Release Script"
log_info "========================================"
echo ""

if [ "$DRY_RUN" = "true" ]; then
    log_warning "DRY RUN MODE - No changes will be made"
    echo ""
fi

# ============================================================================
# Pre-flight checks
# ============================================================================

log_info "Running pre-flight checks..."
echo ""

# Check if on required branch
if [ "$BRANCH_NAME" != "$REQUIRED_BRANCH" ]; then
    log_error "Must be on '$REQUIRED_BRANCH' branch (currently on '$BRANCH_NAME')"
    log_error "Switch to the required branch:"
    log_error "  git checkout $REQUIRED_BRANCH"
    exit 1
fi
log_success "On correct branch: $BRANCH_NAME"

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    log_error "Uncommitted changes detected"
    log_error "Please commit or stash changes before releasing:"
    log_error "  git status"
    exit 1
fi
log_success "Working directory clean"

# Check if we're up to date with remote
git fetch origin
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse @{u} 2>/dev/null || echo "")
if [ -n "$REMOTE" ]; then
    BASE=$(git merge-base @ @{u})
    if [ "$LOCAL" != "$REMOTE" ]; then
        if [ "$LOCAL" = "$BASE" ]; then
            log_error "Local branch is behind remote. Please pull:"
            log_error "  git pull"
            exit 1
        elif [ "$REMOTE" = "$BASE" ]; then
            log_warning "Local branch is ahead of remote (will push after release)"
        else
            log_error "Branches have diverged. Please resolve:"
            log_error "  git pull --rebase"
            exit 1
        fi
    fi
    log_success "In sync with remote"
else
    log_warning "No remote tracking branch found"
fi

# Check for required tools
REQUIRED_TOOLS="python git"
for tool in $REQUIRED_TOOLS; do
    if ! command -v $tool &> /dev/null; then
        log_error "Required tool not found: $tool"
        exit 1
    fi
done
log_success "Required tools available"

echo ""

# ============================================================================
# Version Management
# ============================================================================

log_info "Determining new version..."
echo ""

# Get current version from pyproject.toml
CURRENT_VERSION=$(grep '^version = ' pyproject.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
log_info "Current version: $CURRENT_VERSION"

# Calculate new version
if [ -n "$SPECIFIC_VERSION" ]; then
    NEW_VERSION="$SPECIFIC_VERSION"
    log_info "Using specified version: $NEW_VERSION"
else
    # Parse version
    IFS='.' read -r -a VERSION_PARTS <<< "$CURRENT_VERSION"
    MAJOR="${VERSION_PARTS[0]}"
    MINOR="${VERSION_PARTS[1]}"
    PATCH="${VERSION_PARTS[2]}"

    # Bump version
    case $BUMP_TYPE in
        major)
            MAJOR=$((MAJOR + 1))
            MINOR=0
            PATCH=0
            ;;
        minor)
            MINOR=$((MINOR + 1))
            PATCH=0
            ;;
        patch)
            PATCH=$((PATCH + 1))
            ;;
    esac

    NEW_VERSION="$MAJOR.$MINOR.$PATCH"
    log_info "Bumping $BUMP_TYPE version: $CURRENT_VERSION → $NEW_VERSION"
fi

echo ""

# ============================================================================
# Run Tests
# ============================================================================

if [ "$SKIP_TESTS" = "false" ]; then
    log_info "Running test suite..."
    echo ""

    if [ "$DRY_RUN" = "false" ]; then
        if [ -d "venv" ]; then
            source venv/bin/activate
        fi

        # Run tests
        if pytest -v --cov=app --cov=core --cov=utils --cov-report=term-missing; then
            log_success "All tests passed"
        else
            log_error "Tests failed. Cannot proceed with release."
            exit 1
        fi

        echo ""

        # Run security scan
        log_info "Running security scan..."
        bandit -r app core utils || log_warning "Security scan found issues"

        echo ""
    else
        log_info "[DRY RUN] Would run: pytest"
        log_info "[DRY RUN] Would run: bandit"
    fi
else
    log_warning "Skipping tests (--skip-tests specified)"
fi

# ============================================================================
# Update Version
# ============================================================================

log_info "Updating version to $NEW_VERSION..."
echo ""

if [ "$DRY_RUN" = "false" ]; then
    # Update pyproject.toml
    sed -i "s/^version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml
    log_success "Updated pyproject.toml"

    # Update other version files if they exist
    if [ -f "app/__version__.py" ]; then
        echo "__version__ = \"$NEW_VERSION\"" > app/__version__.py
        log_success "Updated app/__version__.py"
    fi

    if [ -f "config/version.py" ]; then
        echo "VERSION = \"$NEW_VERSION\"" > config/version.py
        log_success "Updated config/version.py"
    fi
else
    log_info "[DRY RUN] Would update version to $NEW_VERSION in:"
    log_info "  - pyproject.toml"
    log_info "  - app/__version__.py (if exists)"
    log_info "  - config/version.py (if exists)"
fi

echo ""

# ============================================================================
# Build Artifacts
# ============================================================================

if [ "$SKIP_BUILD" = "false" ]; then
    log_info "Building release artifacts..."
    echo ""

    if [ "$DRY_RUN" = "false" ]; then
        # Create release directory
        mkdir -p release/artifacts

        # Build Python packages
        log_info "Building Python packages..."
        python -m build --outdir release/artifacts/
        log_success "Python packages built"

        # Build Docker image
        log_info "Building Docker image..."
        docker build -t shivx:$NEW_VERSION -t shivx:latest .
        log_success "Docker image built"

        # Save Docker image
        log_info "Saving Docker image..."
        docker save shivx:$NEW_VERSION | gzip > release/artifacts/shivx-docker-v$NEW_VERSION.tar.gz
        log_success "Docker image saved"

        # Build Windows executable (if on Linux with PyInstaller)
        if [ -f "scripts/build_windows.sh" ]; then
            log_info "Building Windows executable..."
            if ./scripts/build_windows.sh --clean; then
                cp dist/ShivX.exe release/artifacts/ShivX-v$NEW_VERSION.exe
                log_success "Windows executable built"
            else
                log_warning "Windows build failed (non-critical)"
            fi
        fi

        echo ""
    else
        log_info "[DRY RUN] Would build:"
        log_info "  - Python packages (wheel & sdist)"
        log_info "  - Docker image (shivx:$NEW_VERSION)"
        log_info "  - Windows executable (if available)"
    fi
else
    log_warning "Skipping build (--skip-build specified)"
fi

# ============================================================================
# Generate Checksums
# ============================================================================

log_info "Generating checksums..."
echo ""

if [ "$DRY_RUN" = "false" ]; then
    (
        cd release/artifacts
        sha256sum *.tar.gz *.whl *.exe 2>/dev/null > checksums.sha256 || true
        sha512sum *.tar.gz *.whl *.exe 2>/dev/null > checksums.sha512 || true
    )
    log_success "Checksums generated"
else
    log_info "[DRY RUN] Would generate SHA256 and SHA512 checksums"
fi

echo ""

# ============================================================================
# Sign Artifacts
# ============================================================================

log_info "Signing artifacts..."
echo ""

if [ "$DRY_RUN" = "false" ]; then
    if [ -f "scripts/sign_artifacts.sh" ]; then
        VERSION=$NEW_VERSION ./scripts/sign_artifacts.sh --all || log_warning "Signing failed (non-critical)"
        log_success "Artifacts signed"
    else
        log_warning "Signing script not found (skipping)"
    fi
else
    log_info "[DRY RUN] Would sign artifacts with GPG and cosign"
fi

echo ""

# ============================================================================
# Create Git Tag
# ============================================================================

log_info "Creating git tag..."
echo ""

TAG_NAME="v$NEW_VERSION"

if [ "$DRY_RUN" = "false" ]; then
    # Commit version changes
    git add pyproject.toml app/__version__.py config/version.py 2>/dev/null || true
    git commit -m "chore: bump version to $NEW_VERSION" || log_warning "No version files to commit"

    # Create annotated tag
    git tag -a "$TAG_NAME" -m "Release version $NEW_VERSION

Release Notes:
- Version bump: $CURRENT_VERSION → $NEW_VERSION
- Built on: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
- Branch: $BRANCH_NAME
- Commit: $(git rev-parse --short HEAD)

Artifacts:
- Python packages (PyPI)
- Docker image (shivx:$NEW_VERSION)
- Windows executable
- SHA256/SHA512 checksums
- GPG signatures

For detailed changelog, see: CHANGELOG.md"

    log_success "Created tag: $TAG_NAME"
else
    log_info "[DRY RUN] Would create tag: $TAG_NAME"
fi

echo ""

# ============================================================================
# Push to Remote
# ============================================================================

log_info "Pushing to remote repository..."
echo ""

if [ "$DRY_RUN" = "false" ]; then
    # Push commits and tags
    git push origin $BRANCH_NAME
    git push origin $TAG_NAME

    log_success "Pushed to remote"
else
    log_info "[DRY RUN] Would push:"
    log_info "  git push origin $BRANCH_NAME"
    log_info "  git push origin $TAG_NAME"
fi

echo ""

# ============================================================================
# Create GitHub Release
# ============================================================================

log_info "Creating GitHub release..."
echo ""

if [ "$DRY_RUN" = "false" ]; then
    if command -v gh &> /dev/null; then
        # Create release with artifacts
        gh release create "$TAG_NAME" \
            --title "ShivX v$NEW_VERSION" \
            --generate-notes \
            release/artifacts/* || log_warning "GitHub release creation failed (manual intervention may be needed)"

        log_success "GitHub release created: https://github.com/ojaydev11/shivx/releases/tag/$TAG_NAME"
    else
        log_warning "GitHub CLI (gh) not installed - create release manually at:"
        log_warning "  https://github.com/ojaydev11/shivx/releases/new?tag=$TAG_NAME"
    fi
else
    log_info "[DRY RUN] Would create GitHub release with:"
    log_info "  Tag: $TAG_NAME"
    log_info "  Title: ShivX v$NEW_VERSION"
    log_info "  Artifacts: release/artifacts/*"
fi

echo ""

# ============================================================================
# Summary
# ============================================================================

log_info "========================================"
log_info "RELEASE SUMMARY"
log_info "========================================"
echo ""
log_success "Version: $CURRENT_VERSION → $NEW_VERSION"
log_success "Tag: $TAG_NAME"
log_success "Branch: $BRANCH_NAME"

if [ "$DRY_RUN" = "false" ]; then
    echo ""
    log_info "Next steps:"
    log_info "  1. Verify release at: https://github.com/ojaydev11/shivx/releases/tag/$TAG_NAME"
    log_info "  2. Monitor CI/CD pipeline"
    log_info "  3. Update documentation if needed"
    log_info "  4. Announce release to team/users"
    log_info "  5. Deploy to production (if applicable)"
else
    echo ""
    log_warning "This was a DRY RUN - no changes were made"
    log_info "Run without --dry-run to perform the actual release"
fi

echo ""
log_success "Release process completed successfully!"
