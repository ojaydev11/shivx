# ============================================================================
# ShivX AGI Bootstrap Script (Windows PowerShell)
# ============================================================================

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "ShivX AGI Bootstrap" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$python_version = (python --version 2>&1) -replace "Python ", ""
$required_version = [Version]"3.10"
$current_version = [Version]$python_version

if ($current_version -lt $required_version) {
    Write-Host "Error: Python 3.10+ required (found $python_version)" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Python $python_version detected" -ForegroundColor Green
Write-Host ""

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
if (-Not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✓ Virtual environment exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip wheel setuptools

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "✓ Dependencies installed" -ForegroundColor Green
Write-Host ""

# Create data directories
Write-Host "Creating data directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data/memory" | Out-Null
New-Item -ItemType Directory -Force -Path "data/memory/snapshots" | Out-Null
New-Item -ItemType Directory -Force -Path "models/adapters" | Out-Null
New-Item -ItemType Directory -Force -Path "models/embeddings" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

Write-Host "✓ Data directories created" -ForegroundColor Green
Write-Host ""

# Copy environment file
if (-Not (Test-Path ".env")) {
    Write-Host "Creating .env file..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "✓ .env file created (please configure)" -ForegroundColor Green
} else {
    Write-Host "✓ .env file exists" -ForegroundColor Green
}

# Install CLI
Write-Host ""
Write-Host "Installing ShivX CLI..." -ForegroundColor Yellow
pip install -e .

Write-Host "✓ CLI installed" -ForegroundColor Green
Write-Host ""

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Bootstrap Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Configure .env file with your settings"
Write-Host "  2. Run demo: python demos/memory_demo.py"
Write-Host "  3. Use CLI: shivx mem recall 'your query'"
Write-Host "  4. Start daemons: shivx daemons start"
Write-Host ""
Write-Host "Documentation: memory/README.md"
Write-Host ""
