# ShivX Development Environment Bootstrap Script (Windows PowerShell)
# =====================================================================
# Purpose: Zero-to-running setup in ≤10 minutes (Gate G7)
# Usage: .\scripts\dev_bootstrap.ps1

$ErrorActionPreference = "Stop"

Write-Host "🚀 ShivX Development Bootstrap" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# Step 1: Check Python 3.10+ installed
# ============================================================================
Write-Host "[1/10] Checking Python version..." -ForegroundColor Yellow

try {
    $pythonVersion = python --version 2>&1 | Out-String
    Write-Host "✅ Found: $pythonVersion" -ForegroundColor Green
    
    # Extract version number
    if ($pythonVersion -match "Python (\d+)\.(\d+)") {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
            Write-Host "❌ Python 3.10+ required, found $major.$minor" -ForegroundColor Red
            Write-Host "   Please install Python 3.10 or later from https://www.python.org/" -ForegroundColor Red
            exit 1
        }
    }
} catch {
    Write-Host "❌ Python not found in PATH" -ForegroundColor Red
    Write-Host "   Please install Python 3.10+ from https://www.python.org/" -ForegroundColor Red
    exit 1
}

# ============================================================================
# Step 2: Create virtual environment
# ============================================================================
Write-Host "[2/10] Creating virtual environment (.venv)..." -ForegroundColor Yellow

if (Test-Path ".venv") {
    Write-Host "⚠️  .venv already exists, skipping creation" -ForegroundColor Yellow
} else {
    python -m venv .venv
    Write-Host "✅ Created .venv" -ForegroundColor Green
}

# ============================================================================
# Step 3: Activate virtual environment
# ============================================================================
Write-Host "[3/10] Activating virtual environment..." -ForegroundColor Yellow

$activateScript = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "✅ Activated .venv" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to find activation script at $activateScript" -ForegroundColor Red
    exit 1
}

# ============================================================================
# Step 4: Upgrade pip/setuptools
# ============================================================================
Write-Host "[4/10] Upgrading pip and setuptools..." -ForegroundColor Yellow

python -m pip install --upgrade pip setuptools wheel | Out-Null
Write-Host "✅ Upgraded pip and setuptools" -ForegroundColor Green

# ============================================================================
# Step 5: Install dependencies from requirements.txt
# ============================================================================
Write-Host "[5/10] Installing Python dependencies (this may take 3-5 minutes)..." -ForegroundColor Yellow

if (Test-Path "requirements.txt") {
    python -m pip install -r requirements.txt
    Write-Host "✅ Installed dependencies from requirements.txt" -ForegroundColor Green
} else {
    Write-Host "❌ requirements.txt not found" -ForegroundColor Red
    exit 1
}

# ============================================================================
# Step 6: Check Node.js installed (for Playwright)
# ============================================================================
Write-Host "[6/10] Checking Node.js..." -ForegroundColor Yellow

try {
    $nodeVersion = node --version 2>&1 | Out-String
    Write-Host "✅ Found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Node.js not found (optional, needed for Playwright browsers)" -ForegroundColor Yellow
    Write-Host "   You can install it later from https://nodejs.org/" -ForegroundColor Yellow
}

# ============================================================================
# Step 7: Install Playwright browsers
# ============================================================================
Write-Host "[7/10] Installing Playwright browsers (chromium)..." -ForegroundColor Yellow

try {
    python -m playwright install chromium 2>&1 | Out-Null
    Write-Host "✅ Installed Playwright chromium" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Failed to install Playwright browsers (non-fatal)" -ForegroundColor Yellow
}

# ============================================================================
# Step 8: Validate .env exists
# ============================================================================
Write-Host "[8/10] Checking .env configuration..." -ForegroundColor Yellow

if (Test-Path ".env") {
    Write-Host "✅ .env file exists" -ForegroundColor Green
    
    # Validate critical env vars
    $envContent = Get-Content ".env" -Raw
    
    $warnings = @()
    if ($envContent -match "ADMIN_TOKEN=changeme") {
        $warnings += "⚠️  ADMIN_TOKEN is still set to 'changeme'"
    }
    if ($envContent -match "JWT_SECRET=changeme") {
        $warnings += "⚠️  JWT_SECRET is still set to 'changeme'"
    }
    if ($envContent -match "ENCRYPTION_KEY=dGVzdC1lbmNyeXB0aW9uLWtleS1tdXN0LWJlLTMyLWNoYXJzIQ==") {
        $warnings += "⚠️  ENCRYPTION_KEY is still set to example value"
    }
    
    if ($warnings.Count -gt 0) {
        Write-Host "⚠️  WARNING: Please update these values in .env before running in production:" -ForegroundColor Yellow
        foreach ($warning in $warnings) {
            Write-Host "   $warning" -ForegroundColor Yellow
        }
    }
    
} else {
    Write-Host "⚠️  .env not found, copying from env.example..." -ForegroundColor Yellow
    
    if (Test-Path "env.example") {
        Copy-Item "env.example" ".env"
        Write-Host "✅ Created .env from env.example" -ForegroundColor Green
        Write-Host "⚠️  IMPORTANT: Edit .env and set ADMIN_TOKEN, JWT_SECRET, and ENCRYPTION_KEY" -ForegroundColor Yellow
    } else {
        Write-Host "❌ env.example not found, cannot create .env" -ForegroundColor Red
        Write-Host "   Please create .env manually with required variables" -ForegroundColor Red
        exit 1
    }
}

# ============================================================================
# Step 9: Create required directories
# ============================================================================
Write-Host "[9/10] Creating required directories..." -ForegroundColor Yellow

$requiredDirs = @(
    "var",
    "var/runtime",
    "var/queues",
    "var/runs",
    "var/security",
    "logs",
    "data",
    "memory",
    "memory/episodic",
    "memory/chroma_db"
)

foreach ($dir in $requiredDirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

Write-Host "✅ Created required directories" -ForegroundColor Green

# ============================================================================
# Step 10: Run readiness check
# ============================================================================
Write-Host "[10/10] Running readiness check..." -ForegroundColor Yellow

try {
    python shivx_runner.py --mode readiness 2>&1 | Out-Null
    $readinessExitCode = $LASTEXITCODE
    
    if ($readinessExitCode -eq 0) {
        Write-Host "✅ Readiness check passed" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Readiness check completed with warnings (exit code: $readinessExitCode)" -ForegroundColor Yellow
        Write-Host "   This is OK for development; fix warnings before production deployment" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Readiness check failed (non-fatal)" -ForegroundColor Yellow
    Write-Host "   You can run it manually: python shivx_runner.py --mode readiness" -ForegroundColor Yellow
}

# ============================================================================
# Success!
# ============================================================================
Write-Host ""
Write-Host "🎉 Bootstrap Complete!" -ForegroundColor Green
Write-Host "======================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Review and update .env with secure values (ADMIN_TOKEN, JWT_SECRET, ENCRYPTION_KEY)" -ForegroundColor White
Write-Host "  2. Run the GUI dashboard:" -ForegroundColor White
Write-Host "     python shivx_runner.py --mode gui" -ForegroundColor Gray
Write-Host "     Then open: http://127.0.0.1:8051/dashboard" -ForegroundColor Gray
Write-Host "  3. OR run the main API server:" -ForegroundColor White
Write-Host "     python shivx_runner.py --mode prod" -ForegroundColor Gray
Write-Host "     API at: http://127.0.0.1:5099" -ForegroundColor Gray
Write-Host "  4. Run tests:" -ForegroundColor White
Write-Host "     pytest" -ForegroundColor Gray
Write-Host "  5. Install pre-commit hooks (optional):" -ForegroundColor White
Write-Host "     pre-commit install" -ForegroundColor Gray
Write-Host ""
Write-Host "Documentation:" -ForegroundColor Cyan
Write-Host "  - README.md" -ForegroundColor White
Write-Host "  - release/artifacts/wirecheck_report.md" -ForegroundColor White
Write-Host "  - release/STATUS.md" -ForegroundColor White
Write-Host ""
Write-Host "Happy coding! 🚀" -ForegroundColor Green

