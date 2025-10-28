#!/bin/bash
# ShivX Artifact Signing Script
# Signs Docker images, Python packages, and Windows executables
# Usage: ./scripts/sign_artifacts.sh [options]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
VERSION="${VERSION:-2.0.0}"
GPG_KEY_ID="${GPG_KEY_ID:-}"
COSIGN_KEY="${COSIGN_KEY:-cosign.key}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-dist}"

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

# Parse arguments
SIGN_DOCKER=false
SIGN_PYTHON=false
SIGN_WINDOWS=false
SIGN_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --docker)
            SIGN_DOCKER=true
            shift
            ;;
        --python)
            SIGN_PYTHON=true
            shift
            ;;
        --windows)
            SIGN_WINDOWS=true
            shift
            ;;
        --all)
            SIGN_ALL=true
            SIGN_DOCKER=true
            SIGN_PYTHON=true
            SIGN_WINDOWS=true
            shift
            ;;
        --gpg-key)
            GPG_KEY_ID="$2"
            shift 2
            ;;
        --help)
            echo "ShivX Artifact Signing Script"
            echo ""
            echo "Usage: ./scripts/sign_artifacts.sh [options]"
            echo ""
            echo "Options:"
            echo "  --docker         Sign Docker images"
            echo "  --python         Sign Python packages"
            echo "  --windows        Sign Windows executables"
            echo "  --all            Sign all artifacts"
            echo "  --gpg-key ID     Specify GPG key ID"
            echo "  --help           Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  VERSION          Version to sign (default: 2.0.0)"
            echo "  GPG_KEY_ID       GPG key ID for signing"
            echo "  COSIGN_KEY       Path to cosign private key"
            echo "  ARTIFACTS_DIR    Directory containing artifacts (default: dist)"
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
if [ "$SIGN_DOCKER" = false ] && [ "$SIGN_PYTHON" = false ] && [ "$SIGN_WINDOWS" = false ]; then
    SIGN_ALL=true
    SIGN_DOCKER=true
    SIGN_PYTHON=true
    SIGN_WINDOWS=true
fi

log_info "========================================"
log_info "ShivX Artifact Signing Script v$VERSION"
log_info "========================================"
echo ""

# Create signatures directory
mkdir -p release/artifacts/signatures

# ============================================================================
# 1. Sign Docker Images with cosign
# ============================================================================

if [ "$SIGN_DOCKER" = true ]; then
    log_info "Signing Docker images..."

    # Check if cosign is installed
    if ! command -v cosign &> /dev/null; then
        log_warning "cosign not found. Installing..."

        # Install cosign
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            wget https://github.com/sigstore/cosign/releases/latest/download/cosign-linux-amd64
            sudo mv cosign-linux-amd64 /usr/local/bin/cosign
            sudo chmod +x /usr/local/bin/cosign
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            brew install cosign
        else
            log_error "Unsupported OS for automatic cosign installation"
            log_error "Install manually: https://docs.sigstore.dev/cosign/installation/"
            exit 1
        fi

        log_success "cosign installed"
    fi

    # Check if cosign key exists
    if [ ! -f "$COSIGN_KEY" ]; then
        log_warning "Cosign key not found at $COSIGN_KEY"
        log_info "Generating new cosign key pair..."

        cosign generate-key-pair

        log_success "Cosign key pair generated"
        log_warning "IMPORTANT: Store cosign.key securely and back it up!"
        log_warning "IMPORTANT: Distribute cosign.pub for signature verification!"
    fi

    # Sign Docker images
    DOCKER_IMAGES=(
        "shivx:latest"
        "shivx:$VERSION"
        "ghcr.io/yourusername/shivx:latest"
        "ghcr.io/yourusername/shivx:$VERSION"
    )

    for IMAGE in "${DOCKER_IMAGES[@]}"; do
        if docker image inspect "$IMAGE" &> /dev/null; then
            log_info "Signing Docker image: $IMAGE"

            # Sign with cosign
            cosign sign --key "$COSIGN_KEY" "$IMAGE" || log_warning "Failed to sign $IMAGE (may not exist locally)"

            # Generate SBOM and sign it
            if command -v syft &> /dev/null; then
                log_info "Generating and signing SBOM for $IMAGE"
                syft "$IMAGE" -o spdx-json > "release/artifacts/sbom-${IMAGE//[:\/]/-}.json"
                cosign attest --key "$COSIGN_KEY" --predicate "release/artifacts/sbom-${IMAGE//[:\/]/-}.json" "$IMAGE" || log_warning "SBOM attestation failed"
            fi

            log_success "Signed: $IMAGE"
        else
            log_warning "Docker image not found: $IMAGE"
        fi
    done

    # Export signatures
    log_info "Exporting Docker image signatures..."
    for IMAGE in "${DOCKER_IMAGES[@]}"; do
        if docker image inspect "$IMAGE" &> /dev/null; then
            cosign verify --key cosign.pub "$IMAGE" > "release/artifacts/signatures/${IMAGE//[:\/]/-}.sig.json" 2>/dev/null || log_warning "No signature found for $IMAGE"
        fi
    done

    log_success "Docker image signing completed"
fi

# ============================================================================
# 2. Sign Python Packages with GPG
# ============================================================================

if [ "$SIGN_PYTHON" = true ]; then
    log_info "Signing Python packages..."

    # Check if GPG is installed
    if ! command -v gpg &> /dev/null; then
        log_error "GPG not found. Install with:"
        log_error "  Ubuntu/Debian: sudo apt-get install gnupg"
        log_error "  macOS: brew install gnupg"
        exit 1
    fi

    # Check if GPG key is specified
    if [ -z "$GPG_KEY_ID" ]; then
        log_warning "No GPG key ID specified"

        # List available keys
        log_info "Available GPG keys:"
        gpg --list-secret-keys --keyid-format LONG

        # Check if any keys exist
        if ! gpg --list-secret-keys | grep -q "sec"; then
            log_warning "No GPG keys found. Generating new key..."
            log_info "Follow the prompts to create a GPG key:"

            gpg --full-generate-key

            log_success "GPG key generated"
            log_info "Available keys:"
            gpg --list-secret-keys --keyid-format LONG
        fi

        # Try to auto-detect key
        GPG_KEY_ID=$(gpg --list-secret-keys --keyid-format LONG | grep "sec" | head -1 | awk '{print $2}' | cut -d'/' -f2)

        if [ -z "$GPG_KEY_ID" ]; then
            log_error "Could not determine GPG key ID"
            log_error "Please specify with --gpg-key option"
            exit 1
        fi

        log_info "Using GPG key: $GPG_KEY_ID"
    fi

    # Export public key
    log_info "Exporting public GPG key..."
    gpg --armor --export "$GPG_KEY_ID" > release/artifacts/gpg_public_key.asc
    log_success "Public key exported to release/artifacts/gpg_public_key.asc"

    # Sign Python packages
    PYTHON_PACKAGES=(
        "$ARTIFACTS_DIR/shivx-$VERSION.tar.gz"
        "$ARTIFACTS_DIR/shivx-$VERSION-py3-none-any.whl"
    )

    for PACKAGE in "${PYTHON_PACKAGES[@]}"; do
        if [ -f "$PACKAGE" ]; then
            log_info "Signing: $PACKAGE"

            # Create detached signature
            gpg --detach-sign --armor --local-user "$GPG_KEY_ID" "$PACKAGE"

            log_success "Signed: $PACKAGE.asc"
        else
            log_warning "Package not found: $PACKAGE"
        fi
    done

    log_success "Python package signing completed"
fi

# ============================================================================
# 3. Sign Windows Executables
# ============================================================================

if [ "$SIGN_WINDOWS" = true ]; then
    log_info "Signing Windows executables..."

    WINDOWS_EXES=(
        "$ARTIFACTS_DIR/ShivX.exe"
    )

    # Check for osslsigncode (Linux) or signtool (Windows)
    if command -v osslsigncode &> /dev/null; then
        log_info "Using osslsigncode for Windows executable signing"

        # Check for certificate
        CERT_FILE="${CERT_FILE:-certificates/codesign.pfx}"
        CERT_PASSWORD_FILE="${CERT_PASSWORD_FILE:-certificates/codesign.password}"

        if [ ! -f "$CERT_FILE" ]; then
            log_warning "Code signing certificate not found at $CERT_FILE"
            log_warning "Windows executable signing skipped"
            log_warning ""
            log_warning "To sign Windows executables:"
            log_warning "  1. Obtain a code signing certificate (.pfx or .p12)"
            log_warning "  2. Place it at: $CERT_FILE"
            log_warning "  3. Create password file: $CERT_PASSWORD_FILE"
            log_warning "  4. Run this script again"
        else
            # Read certificate password
            if [ -f "$CERT_PASSWORD_FILE" ]; then
                CERT_PASSWORD=$(cat "$CERT_PASSWORD_FILE")
            else
                log_warning "Certificate password file not found"
                read -s -p "Enter certificate password: " CERT_PASSWORD
                echo ""
            fi

            for EXE in "${WINDOWS_EXES[@]}"; do
                if [ -f "$EXE" ]; then
                    log_info "Signing: $EXE"

                    # Sign with timestamp
                    osslsigncode sign \
                        -pkcs12 "$CERT_FILE" \
                        -pass "$CERT_PASSWORD" \
                        -t http://timestamp.digicert.com \
                        -h sha256 \
                        -n "ShivX AI Trading System" \
                        -i https://shivx.ai \
                        -in "$EXE" \
                        -out "${EXE}.signed"

                    # Replace original with signed version
                    mv "${EXE}.signed" "$EXE"

                    log_success "Signed: $EXE"

                    # Verify signature
                    log_info "Verifying signature..."
                    osslsigncode verify "$EXE"

                else
                    log_warning "Executable not found: $EXE"
                fi
            done
        fi

    elif command -v signtool.exe &> /dev/null; then
        log_info "Using signtool.exe for Windows executable signing"

        # Windows signtool is available
        for EXE in "${WINDOWS_EXES[@]}"; do
            if [ -f "$EXE" ]; then
                log_info "Signing: $EXE"

                # Sign with timestamp
                signtool.exe sign \
                    /tr http://timestamp.digicert.com \
                    /td sha256 \
                    /fd sha256 \
                    /n "ShivX Team" \
                    "$EXE"

                log_success "Signed: $EXE"
            else
                log_warning "Executable not found: $EXE"
            fi
        done

    else
        log_warning "No signing tool found for Windows executables"
        log_warning "Install osslsigncode (Linux): sudo apt-get install osslsigncode"
        log_warning "Or use signtool.exe on Windows"
        log_warning "Windows executable signing skipped"
    fi

    log_success "Windows executable signing completed"
fi

# ============================================================================
# Generate Signature Manifest
# ============================================================================

log_info "Generating signature manifest..."

cat > release/artifacts/SIGNATURES.md << EOF
# ShivX v$VERSION - Signature Verification

This file contains information for verifying the authenticity of ShivX artifacts.

## Signing Keys

### GPG Public Key

\`\`\`
$(cat release/artifacts/gpg_public_key.asc 2>/dev/null || echo "Not available")
\`\`\`

Import with:
\`\`\`bash
gpg --import gpg_public_key.asc
\`\`\`

### Cosign Public Key

\`\`\`
$(cat cosign.pub 2>/dev/null || echo "Not available")
\`\`\`

## Verification Instructions

### Docker Images

\`\`\`bash
# Verify Docker image signature
cosign verify --key cosign.pub shivx:$VERSION

# Verify SBOM attestation
cosign verify-attestation --key cosign.pub shivx:$VERSION
\`\`\`

### Python Packages

\`\`\`bash
# Import public key
gpg --import gpg_public_key.asc

# Verify package signature
gpg --verify shivx-$VERSION.tar.gz.asc shivx-$VERSION.tar.gz
gpg --verify shivx-$VERSION-py3-none-any.whl.asc shivx-$VERSION-py3-none-any.whl
\`\`\`

### Windows Executables

\`\`\`bash
# On Linux
osslsigncode verify ShivX.exe

# On Windows
signtool verify /pa ShivX.exe
\`\`\`

## SHA256 Hashes

\`\`\`
$(cd release/artifacts && sha256sum *.tar.gz *.whl *.exe 2>/dev/null || echo "Run generate_hashes.sh first")
\`\`\`

---

**Generated:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Version:** $VERSION
EOF

log_success "Signature manifest saved to release/artifacts/SIGNATURES.md"

# Summary
echo ""
log_info "========================================"
log_info "SIGNING SUMMARY"
log_info "========================================"

if [ "$SIGN_DOCKER" = true ]; then
    log_success "Docker images signed with cosign"
fi

if [ "$SIGN_PYTHON" = true ]; then
    log_success "Python packages signed with GPG"
    log_success "Public key: release/artifacts/gpg_public_key.asc"
fi

if [ "$SIGN_WINDOWS" = true ]; then
    log_success "Windows executables signed (if certificate available)"
fi

log_success "Signature manifest: release/artifacts/SIGNATURES.md"

echo ""
log_info "Next steps:"
log_info "  1. Verify signatures: ./scripts/verify_signatures.sh"
log_info "  2. Distribute public keys with release"
log_info "  3. Document verification process for users"

echo ""
log_success "Signing completed successfully!"
