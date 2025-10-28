# ShivX Windows Build Script
# Builds a single-file Windows .exe using PyInstaller
# Usage: .\scripts\build_windows.ps1
# Requirements: Python 3.10+, pip

[CmdletBinding()]
param(
    [Parameter()]
    [switch]$Clean = $false,

    [Parameter()]
    [switch]$Sign = $false,

    [Parameter()]
    [switch]$CreateInstaller = $false,

    [Parameter()]
    [string]$Version = "2.0.0"
)

# Script configuration
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Colors for output
function Write-ColorOutput($ForegroundColor, $Message) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    Write-Output $Message
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Info($Message) {
    Write-ColorOutput "Cyan" "[INFO] $Message"
}

function Write-Success($Message) {
    Write-ColorOutput "Green" "[SUCCESS] $Message"
}

function Write-Error($Message) {
    Write-ColorOutput "Red" "[ERROR] $Message"
}

function Write-Warning($Message) {
    Write-ColorOutput "Yellow" "[WARNING] $Message"
}

Write-Info "========================================"
Write-Info "ShivX Windows Build Script v$Version"
Write-Info "========================================"
Write-Info ""

# Check Python installation
Write-Info "Checking Python installation..."
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python (\d+)\.(\d+)") {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
            Write-Error "Python 3.10 or higher required. Found: $pythonVersion"
            exit 1
        }
        Write-Success "Found $pythonVersion"
    }
} catch {
    Write-Error "Python not found. Please install Python 3.10 or higher."
    exit 1
}

# Clean previous build artifacts
if ($Clean) {
    Write-Info "Cleaning previous build artifacts..."
    $dirsToClean = @("build", "dist", "__pycache__")
    foreach ($dir in $dirsToClean) {
        if (Test-Path $dir) {
            Remove-Item -Recurse -Force $dir
            Write-Success "Removed $dir/"
        }
    }

    # Clean .spec file cache
    Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force
    Write-Success "Cleaned all build artifacts"
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Info "Creating virtual environment..."
    python -m venv venv
    Write-Success "Virtual environment created"
}

# Activate virtual environment
Write-Info "Activating virtual environment..."
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Info "Upgrading pip..."
python -m pip install --upgrade pip --quiet

# Install dependencies
Write-Info "Installing dependencies..."
pip install -r requirements.txt --quiet
Write-Success "Dependencies installed"

# Install PyInstaller
Write-Info "Installing PyInstaller..."
pip install pyinstaller --quiet
Write-Success "PyInstaller installed"

# Install UPX for compression (optional but recommended)
Write-Info "Checking for UPX compression tool..."
$upxPath = Get-Command upx -ErrorAction SilentlyContinue
if (-not $upxPath) {
    Write-Warning "UPX not found. Install UPX for smaller executable size:"
    Write-Warning "  Download from: https://upx.github.io/"
    Write-Warning "  Or: choco install upx (if using Chocolatey)"
}

# Create version info file
Write-Info "Creating version information file..."
$versionInfo = @"
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=($($Version.Replace('.', ', ')), 0),
    prodvers=($($Version.Replace('.', ', ')), 0),
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
           StringStruct(u'FileVersion', u'$Version'),
           StringStruct(u'InternalName', u'ShivX'),
           StringStruct(u'LegalCopyright', u'Copyright (C) 2025 ShivX Team'),
           StringStruct(u'OriginalFilename', u'ShivX.exe'),
           StringStruct(u'ProductName', u'ShivX AI Trading System'),
           StringStruct(u'ProductVersion', u'$Version')])
      ]
    ),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"@
Set-Content -Path "version_info.txt" -Value $versionInfo
Write-Success "Version info created"

# Build with PyInstaller
Write-Info "Building Windows executable..."
Write-Info "This may take 5-10 minutes depending on your system..."
$buildStart = Get-Date

try {
    pyinstaller pyinstaller.spec --clean --noconfirm
    $buildEnd = Get-Date
    $buildDuration = ($buildEnd - $buildStart).TotalSeconds
    Write-Success "Build completed in $([math]::Round($buildDuration, 2)) seconds"
} catch {
    Write-Error "Build failed: $_"
    exit 1
}

# Check if executable was created
$exePath = "dist\ShivX.exe"
if (-not (Test-Path $exePath)) {
    Write-Error "Executable not found at $exePath"
    exit 1
}

# Get executable size
$exeSize = (Get-Item $exePath).Length
$exeSizeMB = [math]::Round($exeSize / 1MB, 2)
Write-Success "Executable created: $exePath ($exeSizeMB MB)"

# Test executable
Write-Info "Testing executable..."
try {
    # Run a quick version check
    $testOutput = & $exePath --help 2>&1
    if ($LASTEXITCODE -eq 0 -or $testOutput) {
        Write-Success "Executable runs successfully"
    } else {
        Write-Warning "Executable may have issues. Please test manually."
    }
} catch {
    Write-Warning "Could not test executable automatically. Please test manually."
}

# Code signing (if requested and certificate available)
if ($Sign) {
    Write-Info "Code signing requested..."

    # Check for code signing certificate
    $cert = Get-ChildItem -Path Cert:\CurrentUser\My -CodeSigningCert | Select-Object -First 1

    if ($cert) {
        Write-Info "Found code signing certificate: $($cert.Subject)"
        try {
            # Sign the executable
            Set-AuthenticodeSignature -FilePath $exePath -Certificate $cert -TimestampServer "http://timestamp.digicert.com"
            Write-Success "Executable signed successfully"
        } catch {
            Write-Error "Failed to sign executable: $_"
        }
    } else {
        Write-Warning "No code signing certificate found in Certificate Store"
        Write-Warning "To sign the executable:"
        Write-Warning "  1. Obtain a code signing certificate"
        Write-Warning "  2. Import it to your Certificate Store"
        Write-Warning "  3. Run this script with -Sign parameter"
    }
}

# Create installer (if requested)
if ($CreateInstaller) {
    Write-Info "Creating installer..."

    # Check for Inno Setup
    $innoSetup = Get-Command iscc -ErrorAction SilentlyContinue

    if ($innoSetup) {
        Write-Info "Using Inno Setup to create installer..."
        # TODO: Create Inno Setup script and run it
        Write-Warning "Inno Setup script not yet implemented"
        Write-Warning "Create scripts/installer.iss for Inno Setup configuration"
    } else {
        Write-Warning "Inno Setup not found. Installer creation skipped."
        Write-Warning "Install Inno Setup from: https://jrsoftware.org/isinfo.php"
    }
}

# Generate build report
Write-Info "Generating build report..."
$buildReport = @{
    Version = $Version
    BuildDate = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    ExecutablePath = $exePath
    ExecutableSize = "$exeSizeMB MB"
    BuildDuration = "$([math]::Round($buildDuration, 2)) seconds"
    Signed = $Sign
    PythonVersion = $pythonVersion
    Platform = "Windows"
}

$buildReportJson = $buildReport | ConvertTo-Json -Depth 10
Set-Content -Path "dist\build_report.json" -Value $buildReportJson
Write-Success "Build report saved to dist\build_report.json"

# Calculate SHA256 hash
Write-Info "Calculating SHA256 hash..."
$hash = Get-FileHash -Path $exePath -Algorithm SHA256
$hashValue = $hash.Hash
Set-Content -Path "dist\ShivX.exe.sha256" -Value "$hashValue  ShivX.exe"
Write-Success "SHA256: $hashValue"

# Summary
Write-Info ""
Write-Info "========================================"
Write-Info "BUILD SUMMARY"
Write-Info "========================================"
Write-Success "Executable: $exePath"
Write-Success "Size: $exeSizeMB MB"
Write-Success "Build time: $([math]::Round($buildDuration, 2)) seconds"
Write-Success "SHA256 hash saved to: dist\ShivX.exe.sha256"

if ($Sign) {
    if ($cert) {
        Write-Success "Code signing: Completed"
    } else {
        Write-Warning "Code signing: Certificate not found"
    }
}

Write-Info ""
Write-Info "Next steps:"
Write-Info "  1. Test the executable: .\dist\ShivX.exe"
Write-Info "  2. Verify hash: Get-FileHash dist\ShivX.exe -Algorithm SHA256"
Write-Info "  3. Distribute the executable"

if (-not $Sign) {
    Write-Info "  4. Consider code signing for production: .\scripts\build_windows.ps1 -Sign"
}

Write-Info ""
Write-Success "Build completed successfully!"
