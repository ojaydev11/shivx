#!/bin/bash
# ShivX Windows Build Script for Linux
# Builds a Windows .exe using PyInstaller (optionally with Wine)
# Usage: ./scripts/build_windows.sh [options]
# Requirements: Python 3.10+, pip, (optional: wine for testing)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VERSION="${VERSION:-2.0.0}"
CLEAN="${CLEAN:-false}"
SIGN="${SIGN:-false}"
TEST="${TEST:-true}"

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
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN="true"
            shift
            ;;
        --sign)
            SIGN="true"
            shift
            ;;
        --no-test)
            TEST="false"
            shift
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --help)
            echo "ShivX Windows Build Script"
            echo ""
            echo "Usage: ./scripts/build_windows.sh [options]"
            echo ""
            echo "Options:"
            echo "  --clean       Clean previous build artifacts"
            echo "  --sign        Sign the executable (requires certificate)"
            echo "  --no-test     Skip testing the executable"
            echo "  --version VER Set version number (default: 2.0.0)"
            echo "  --help        Show this help message"
            echo ""
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_info "========================================"
log_info "ShivX Windows Build Script v$VERSION"
log_info "========================================"
echo ""

# Check Python installation
log_info "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 not found. Please install Python 3.10 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MAJOR" -eq 3 -a "$PYTHON_MINOR" -lt 10 ]; then
    log_error "Python 3.10 or higher required. Found: $PYTHON_VERSION"
    exit 1
fi

log_success "Found Python $PYTHON_VERSION"

# Clean previous build artifacts
if [ "$CLEAN" = "true" ]; then
    log_info "Cleaning previous build artifacts..."
    rm -rf build dist __pycache__
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
    log_success "Cleaned all build artifacts"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    log_info "Creating virtual environment..."
    python3 -m venv venv
    log_success "Virtual environment created"
fi

# Activate virtual environment
log_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
log_info "Installing dependencies..."
pip install -r requirements.txt --quiet
log_success "Dependencies installed"

# Install PyInstaller
log_info "Installing PyInstaller..."
pip install pyinstaller --quiet
log_success "PyInstaller installed"

# Check for UPX compression tool
log_info "Checking for UPX compression tool..."
if command -v upx &> /dev/null; then
    log_success "UPX found: $(upx --version | head -1)"
else
    log_warning "UPX not found. Install for smaller executable size:"
    log_warning "  Ubuntu/Debian: sudo apt-get install upx"
    log_warning "  Fedora: sudo dnf install upx"
    log_warning "  macOS: brew install upx"
fi

# Create version info file
log_info "Creating version information file..."
cat > version_info.txt << EOF
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=($(echo $VERSION | tr '.' ', '), 0),
    prodvers=($(echo $VERSION | tr '.' ', '), 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          u'040904B0',
          [StringStruct(u'CompanyName', u'ShivX Team'),
           StringStruct(u'FileDescription', u'ShivX AI Trading System'),
           StringStruct(u'FileVersion', u'$VERSION'),
           StringStruct(u'InternalName', u'ShivX'),
           StringStruct(u'LegalCopyright', u'Copyright (C) 2025 ShivX Team'),
           StringStruct(u'OriginalFilename', u'ShivX.exe'),
           StringStruct(u'ProductName', u'ShivX AI Trading System'),
           StringStruct(u'ProductVersion', u'$VERSION')])
      ]
    ),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
EOF
log_success "Version info created"

# Build with PyInstaller
log_info "Building Windows executable..."
log_info "This may take 5-10 minutes depending on your system..."
BUILD_START=$(date +%s)

if pyinstaller pyinstaller.spec --clean --noconfirm; then
    BUILD_END=$(date +%s)
    BUILD_DURATION=$((BUILD_END - BUILD_START))
    log_success "Build completed in $BUILD_DURATION seconds"
else
    log_error "Build failed"
    exit 1
fi

# Check if executable was created
EXE_PATH="dist/ShivX.exe"
if [ ! -f "$EXE_PATH" ]; then
    log_error "Executable not found at $EXE_PATH"
    exit 1
fi

# Get executable size
EXE_SIZE=$(du -h "$EXE_PATH" | cut -f1)
EXE_SIZE_BYTES=$(stat -f%z "$EXE_PATH" 2>/dev/null || stat -c%s "$EXE_PATH")
EXE_SIZE_MB=$(echo "scale=2; $EXE_SIZE_BYTES / 1024 / 1024" | bc)
log_success "Executable created: $EXE_PATH ($EXE_SIZE_MB MB)"

# Test executable (if Wine is available)
if [ "$TEST" = "true" ]; then
    log_info "Testing executable..."

    if command -v wine &> /dev/null; then
        log_info "Wine found, testing executable..."
        if wine "$EXE_PATH" --help &> /dev/null; then
            log_success "Executable runs successfully under Wine"
        else
            log_warning "Executable may have issues under Wine. Test on Windows."
        fi
    else
        log_warning "Wine not installed. Cannot test Windows executable on Linux."
        log_warning "Install Wine to test: sudo apt-get install wine (Ubuntu/Debian)"
        log_warning "Please test the executable on a Windows machine."
    fi
fi

# Code signing (placeholder)
if [ "$SIGN" = "true" ]; then
    log_info "Code signing requested..."
    log_warning "Code signing on Linux for Windows executables requires:"
    log_warning "  1. osslsigncode tool: sudo apt-get install osslsigncode"
    log_warning "  2. Code signing certificate (.pfx/.p12 file)"
    log_warning ""

    if command -v osslsigncode &> /dev/null; then
        log_info "osslsigncode found"

        # Check for certificate
        if [ -f "certificates/codesign.pfx" ]; then
            log_info "Signing executable..."
            osslsigncode sign \
                -pkcs12 certificates/codesign.pfx \
                -pass "$(cat certificates/codesign.password 2>/dev/null || echo '')" \
                -t http://timestamp.digicert.com \
                -in "$EXE_PATH" \
                -out "${EXE_PATH}.signed"

            if [ -f "${EXE_PATH}.signed" ]; then
                mv "${EXE_PATH}.signed" "$EXE_PATH"
                log_success "Executable signed successfully"
            else
                log_error "Signing failed"
            fi
        else
            log_warning "Code signing certificate not found at certificates/codesign.pfx"
            log_warning "Skipping code signing"
        fi
    else
        log_warning "osslsigncode not installed. Skipping code signing."
    fi
fi

# Generate build report
log_info "Generating build report..."
cat > dist/build_report.json << EOF
{
  "version": "$VERSION",
  "build_date": "$(date -u +"%Y-%m-%d %H:%M:%S UTC")",
  "executable_path": "$EXE_PATH",
  "executable_size": "$EXE_SIZE_MB MB",
  "build_duration": "$BUILD_DURATION seconds",
  "signed": $SIGN,
  "python_version": "$PYTHON_VERSION",
  "platform": "Linux (cross-compiled for Windows)"
}
EOF
log_success "Build report saved to dist/build_report.json"

# Calculate SHA256 hash
log_info "Calculating SHA256 hash..."
HASH=$(sha256sum "$EXE_PATH" | awk '{print $1}')
echo "$HASH  ShivX.exe" > dist/ShivX.exe.sha256
log_success "SHA256: $HASH"

# Summary
echo ""
log_info "========================================"
log_info "BUILD SUMMARY"
log_info "========================================"
log_success "Executable: $EXE_PATH"
log_success "Size: $EXE_SIZE_MB MB"
log_success "Build time: $BUILD_DURATION seconds"
log_success "SHA256 hash saved to: dist/ShivX.exe.sha256"

if [ "$SIGN" = "true" ]; then
    if [ -f "certificates/codesign.pfx" ]; then
        log_success "Code signing: Completed"
    else
        log_warning "Code signing: Certificate not found"
    fi
fi

echo ""
log_info "Next steps:"
log_info "  1. Test the executable on Windows"
log_info "  2. Verify hash: sha256sum dist/ShivX.exe"
log_info "  3. Distribute the executable"

if [ "$SIGN" != "true" ]; then
    log_info "  4. Consider code signing for production: ./scripts/build_windows.sh --sign"
fi

echo ""
log_success "Build completed successfully!"
