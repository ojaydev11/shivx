#!/bin/bash
# ShivX Artifact Signature Verification Script
# Verifies signatures for Docker images, Python packages, and Windows executables
# Usage: ./scripts/verify_signatures.sh [options]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
VERSION="${VERSION:-2.0.0}"
GPG_PUBLIC_KEY="${GPG_PUBLIC_KEY:-release/artifacts/gpg_public_key.asc}"
COSIGN_PUBLIC_KEY="${COSIGN_PUBLIC_KEY:-cosign.pub}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-dist}"

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
SKIPPED_CHECKS=0

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
    ((PASSED_CHECKS++))
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
    ((SKIPPED_CHECKS++))
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
    ((FAILED_CHECKS++))
}

# Parse arguments
VERIFY_DOCKER=false
VERIFY_PYTHON=false
VERIFY_WINDOWS=false
VERIFY_HASHES=false
VERIFY_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --docker)
            VERIFY_DOCKER=true
            shift
            ;;
        --python)
            VERIFY_PYTHON=true
            shift
            ;;
        --windows)
            VERIFY_WINDOWS=true
            shift
            ;;
        --hashes)
            VERIFY_HASHES=true
            shift
            ;;
        --all)
            VERIFY_ALL=true
            VERIFY_DOCKER=true
            VERIFY_PYTHON=true
            VERIFY_WINDOWS=true
            VERIFY_HASHES=true
            shift
            ;;
        --help)
            echo "ShivX Artifact Signature Verification Script"
            echo ""
            echo "Usage: ./scripts/verify_signatures.sh [options]"
            echo ""
            echo "Options:"
            echo "  --docker         Verify Docker image signatures"
            echo "  --python         Verify Python package signatures"
            echo "  --windows        Verify Windows executable signatures"
            echo "  --hashes         Verify SHA256 hashes"
            echo "  --all            Verify all artifacts (default)"
            echo "  --help           Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  VERSION              Version to verify (default: 2.0.0)"
            echo "  GPG_PUBLIC_KEY       Path to GPG public key"
            echo "  COSIGN_PUBLIC_KEY    Path to cosign public key"
            echo "  ARTIFACTS_DIR        Directory containing artifacts"
            echo ""
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default to all if nothing specified
if [ "$VERIFY_DOCKER" = false ] && [ "$VERIFY_PYTHON" = false ] && \
   [ "$VERIFY_WINDOWS" = false ] && [ "$VERIFY_HASHES" = false ]; then
    VERIFY_ALL=true
    VERIFY_DOCKER=true
    VERIFY_PYTHON=true
    VERIFY_WINDOWS=true
    VERIFY_HASHES=true
fi

log_info "========================================"
log_info "ShivX Signature Verification v$VERSION"
log_info "========================================"
echo ""

# ============================================================================
# 1. Verify Docker Image Signatures
# ============================================================================

if [ "$VERIFY_DOCKER" = true ]; then
    log_info "Verifying Docker image signatures..."
    echo ""

    # Check if cosign is installed
    if ! command -v cosign &> /dev/null; then
        log_warning "cosign not found - Docker verification skipped"
        log_warning "Install from: https://docs.sigstore.dev/cosign/installation/"
        echo ""
    else
        # Check if public key exists
        if [ ! -f "$COSIGN_PUBLIC_KEY" ]; then
            log_warning "Cosign public key not found at $COSIGN_PUBLIC_KEY"
            log_warning "Docker verification skipped"
            echo ""
        else
            DOCKER_IMAGES=(
                "shivx:latest"
                "shivx:$VERSION"
                "ghcr.io/yourusername/shivx:latest"
                "ghcr.io/yourusername/shivx:$VERSION"
            )

            for IMAGE in "${DOCKER_IMAGES[@]}"; do
                ((TOTAL_CHECKS++))

                if docker image inspect "$IMAGE" &> /dev/null; then
                    log_info "Verifying: $IMAGE"

                    # Verify signature
                    if cosign verify --key "$COSIGN_PUBLIC_KEY" "$IMAGE" &> /dev/null; then
                        log_success "Signature valid: $IMAGE"
                    else
                        log_error "Signature verification failed: $IMAGE"
                    fi

                    # Verify SBOM attestation (if available)
                    if cosign verify-attestation --key "$COSIGN_PUBLIC_KEY" "$IMAGE" &> /dev/null; then
                        log_success "SBOM attestation valid: $IMAGE"
                    else
                        log_warning "SBOM attestation not found or invalid: $IMAGE"
                    fi
                else
                    log_warning "Image not found locally: $IMAGE"
                fi

                echo ""
            done
        fi
    fi
fi

# ============================================================================
# 2. Verify Python Package Signatures
# ============================================================================

if [ "$VERIFY_PYTHON" = true ]; then
    log_info "Verifying Python package signatures..."
    echo ""

    # Check if GPG is installed
    if ! command -v gpg &> /dev/null; then
        log_warning "GPG not found - Python package verification skipped"
        log_warning "Install with: sudo apt-get install gnupg (Ubuntu/Debian)"
        echo ""
    else
        # Import public key if available
        if [ -f "$GPG_PUBLIC_KEY" ]; then
            log_info "Importing GPG public key..."
            gpg --import "$GPG_PUBLIC_KEY" &> /dev/null || log_warning "Key already imported or import failed"
        else
            log_warning "GPG public key not found at $GPG_PUBLIC_KEY"
            log_warning "Attempting verification with existing keyring..."
        fi

        PYTHON_PACKAGES=(
            "$ARTIFACTS_DIR/shivx-$VERSION.tar.gz"
            "$ARTIFACTS_DIR/shivx-$VERSION-py3-none-any.whl"
        )

        for PACKAGE in "${PYTHON_PACKAGES[@]}"; do
            ((TOTAL_CHECKS++))

            if [ -f "$PACKAGE" ]; then
                if [ -f "$PACKAGE.asc" ]; then
                    log_info "Verifying: $(basename $PACKAGE)"

                    # Verify GPG signature
                    if gpg --verify "$PACKAGE.asc" "$PACKAGE" 2>&1 | grep -q "Good signature"; then
                        log_success "Signature valid: $(basename $PACKAGE)"
                    else
                        log_error "Signature verification failed: $(basename $PACKAGE)"
                    fi
                else
                    log_warning "Signature file not found: $PACKAGE.asc"
                fi
            else
                log_warning "Package not found: $(basename $PACKAGE)"
            fi

            echo ""
        done
    fi
fi

# ============================================================================
# 3. Verify Windows Executable Signatures
# ============================================================================

if [ "$VERIFY_WINDOWS" = true ]; then
    log_info "Verifying Windows executable signatures..."
    echo ""

    WINDOWS_EXES=(
        "$ARTIFACTS_DIR/ShivX.exe"
    )

    # Check for verification tool
    if command -v osslsigncode &> /dev/null; then
        for EXE in "${WINDOWS_EXES[@]}"; do
            ((TOTAL_CHECKS++))

            if [ -f "$EXE" ]; then
                log_info "Verifying: $(basename $EXE)"

                # Verify signature
                if osslsigncode verify "$EXE" 2>&1 | grep -q "Signature verification: ok"; then
                    log_success "Signature valid: $(basename $EXE)"
                else
                    log_error "Signature verification failed: $(basename $EXE)"
                fi
            else
                log_warning "Executable not found: $(basename $EXE)"
            fi

            echo ""
        done

    elif command -v signtool.exe &> /dev/null; then
        for EXE in "${WINDOWS_EXES[@]}"; do
            ((TOTAL_CHECKS++))

            if [ -f "$EXE" ]; then
                log_info "Verifying: $(basename $EXE)"

                # Verify signature
                if signtool.exe verify /pa "$EXE" &> /dev/null; then
                    log_success "Signature valid: $(basename $EXE)"
                else
                    log_error "Signature verification failed: $(basename $EXE)"
                fi
            else
                log_warning "Executable not found: $(basename $EXE)"
            fi

            echo ""
        done

    else
        log_warning "No verification tool found for Windows executables"
        log_warning "Install osslsigncode (Linux) or use signtool.exe (Windows)"
        echo ""
    fi
fi

# ============================================================================
# 4. Verify SHA256 Hashes
# ============================================================================

if [ "$VERIFY_HASHES" = true ]; then
    log_info "Verifying SHA256 hashes..."
    echo ""

    HASH_FILES=(
        "release/artifacts/hashes.sha256"
        "$ARTIFACTS_DIR/ShivX.exe.sha256"
    )

    for HASH_FILE in "${HASH_FILES[@]}"; do
        if [ -f "$HASH_FILE" ]; then
            log_info "Checking hashes from: $HASH_FILE"

            # Change to the directory containing the hash file
            HASH_DIR=$(dirname "$HASH_FILE")
            HASH_FILENAME=$(basename "$HASH_FILE")

            (
                cd "$HASH_DIR"

                # Verify hashes
                while IFS= read -r line; do
                    # Skip empty lines and comments
                    [[ -z "$line" || "$line" =~ ^# ]] && continue

                    ((TOTAL_CHECKS++))

                    # Extract hash and filename
                    EXPECTED_HASH=$(echo "$line" | awk '{print $1}')
                    FILENAME=$(echo "$line" | awk '{print $2}')

                    if [ -f "$FILENAME" ]; then
                        # Calculate actual hash
                        ACTUAL_HASH=$(sha256sum "$FILENAME" | awk '{print $1}')

                        if [ "$EXPECTED_HASH" = "$ACTUAL_HASH" ]; then
                            log_success "Hash match: $FILENAME"
                        else
                            log_error "Hash mismatch: $FILENAME"
                            log_error "  Expected: $EXPECTED_HASH"
                            log_error "  Actual:   $ACTUAL_HASH"
                        fi
                    else
                        log_warning "File not found for hash check: $FILENAME"
                    fi
                done < "$HASH_FILENAME"
            )

            echo ""
        else
            log_warning "Hash file not found: $HASH_FILE"
        fi
    done
fi

# ============================================================================
# Summary Report
# ============================================================================

echo ""
log_info "========================================"
log_info "VERIFICATION SUMMARY"
log_info "========================================"
echo ""

echo "Total checks:   $TOTAL_CHECKS"
echo -e "${GREEN}Passed:         $PASSED_CHECKS${NC}"
echo -e "${RED}Failed:         $FAILED_CHECKS${NC}"
echo -e "${YELLOW}Skipped:        $SKIPPED_CHECKS${NC}"

echo ""

if [ $FAILED_CHECKS -eq 0 ]; then
    if [ $PASSED_CHECKS -gt 0 ]; then
        log_success "All signature verifications passed!"
        exit 0
    else
        log_warning "No verifications were performed"
        exit 0
    fi
else
    log_error "Some signature verifications failed!"
    log_error "DO NOT use artifacts that fail verification!"
    exit 1
fi
