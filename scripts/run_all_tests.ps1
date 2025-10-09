# ShivX Master Test Runner
# =========================
# Purpose: Execute full test battery for production readiness
# Usage: .\scripts\run_all_tests.ps1

param(
    [switch]$SkipSlow,
    [switch]$SkipIntegration,
    [switch]$SkipE2E,
    [switch]$Parallel,
    [int]$Workers = 4
)

$ErrorActionPreference = "Stop"

Write-Host "🧪 ShivX Master Test Runner" -ForegroundColor Cyan
Write-Host "============================" -ForegroundColor Cyan
Write-Host ""

# Ensure we're in the right directory
if (-not (Test-Path "pytest.ini")) {
    Write-Host "❌ Must run from project root (pytest.ini not found)" -ForegroundColor Red
    exit 1
}

# Build test command
$testCmd = "pytest"
$testArgs = @(
    "--cov"
    "--cov-report=html:release/artifacts/coverage_report.html"
    "--cov-report=term-missing"
    "--cov-report=json:release/artifacts/coverage.json"
    "-v"
    "--tb=short"
)

# Apply filters
$markers = @()
if ($SkipSlow) {
    $markers += "not slow"
}
if ($SkipIntegration) {
    $markers += "not integration"
}
if ($SkipE2E) {
    $markers += "not e2e"
}

if ($markers.Count -gt 0) {
    $markerExpr = $markers -join " and "
    $testArgs += "-m"
    $testArgs += $markerExpr
}

# Parallel execution
if ($Parallel) {
    $testArgs += "-n"
    $testArgs += $Workers.ToString()
}

Write-Host "[1/3] Running test suite..." -ForegroundColor Yellow
Write-Host "Command: $testCmd $($testArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

# Run tests
& $testCmd @testArgs
$exitCode = $LASTEXITCODE

Write-Host ""
Write-Host "[2/3] Checking coverage thresholds..." -ForegroundColor Yellow

if (Test-Path "release/artifacts/coverage.json") {
    $coverage = Get-Content "release/artifacts/coverage.json" | ConvertFrom-Json
    $totalCoverage = [math]::Round($coverage.totals.percent_covered, 2)
    
    Write-Host "Total coverage: $totalCoverage%" -ForegroundColor White
    
    # Gate G1 thresholds (will be enforced once baseline is established)
    # Critical path: 90%, Overall: 75%
    
    if ($totalCoverage -ge 75) {
        Write-Host "✅ Coverage meets overall target (≥75%)" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Coverage below target: $totalCoverage% < 75%" -ForegroundColor Yellow
        Write-Host "   (Target will be enforced once baseline is established)" -ForegroundColor Gray
    }
} else {
    Write-Host "⚠️  Coverage report not found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[3/3] Test Results Summary" -ForegroundColor Yellow

if ($exitCode -eq 0) {
    Write-Host "✅ All tests passed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Reports generated:" -ForegroundColor Cyan
    Write-Host "  - HTML: release/artifacts/coverage_report.html" -ForegroundColor White
    Write-Host "  - JSON: release/artifacts/coverage.json" -ForegroundColor White
    Write-Host ""
    Write-Host "View coverage: start release\artifacts\coverage_report.html\index.html" -ForegroundColor Gray
} else {
    Write-Host "❌ Tests failed (exit code: $exitCode)" -ForegroundColor Red
    Write-Host "   Review output above for details" -ForegroundColor Yellow
}

Write-Host ""
exit $exitCode

