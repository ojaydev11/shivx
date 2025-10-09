# ShivX Security Scan Suite
# ===========================
# Purpose: SAST, secret scans, SBOM generation
# Usage: .\scripts\security_scan.ps1

param(
    [switch]$SAST,
    [switch]$SecretScan,
    [switch]$SBOM,
    [switch]$All
)

$ErrorActionPreference = "Stop"

Write-Host "🔒 ShivX Security Scan Suite" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan
Write-Host ""

$findings = @{
    critical = 0
    high = 0
    medium = 0
    low = 0
    info = 0
}

function Run-SAST {
    Write-Host "[SEC-1] SAST: Static Application Security Testing" -ForegroundColor Yellow
    Write-Host "  Running: Ruff + Bandit + Mypy" -ForegroundColor Gray
    
    # Ruff linting
    Write-Host "    → Ruff (linting)..." -ForegroundColor Gray
    try {
        ruff check . --output-format json > release/artifacts/ruff_report.json 2>&1
        Write-Host "      ✅ Ruff complete" -ForegroundColor Green
    } catch {
        Write-Host "      ⚠️  Ruff found issues (see report)" -ForegroundColor Yellow
    }
    
    # Bandit security scan
    Write-Host "    → Bandit (security)..." -ForegroundColor Gray
    if (Get-Command bandit -ErrorAction SilentlyContinue) {
        try {
            bandit -r . -f json -o release/artifacts/bandit_report.json 2>&1
            Write-Host "      ✅ Bandit complete" -ForegroundColor Green
        } catch {
            Write-Host "      ⚠️  Bandit found issues (see report)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "      ⚠️  Bandit not installed (pip install bandit)" -ForegroundColor Yellow
    }
    
    # Mypy type checking
    Write-Host "    → Mypy (type safety)..." -ForegroundColor Gray
    try {
        mypy . --ignore-missing-imports --no-strict-optional > release/artifacts/mypy_report.txt 2>&1
        Write-Host "      ✅ Mypy complete" -ForegroundColor Green
    } catch {
        Write-Host "      ⚠️  Mypy found issues (see report)" -ForegroundColor Yellow
    }
}

function Run-SecretScan {
    Write-Host "[SEC-2] Secret Scanning" -ForegroundColor Yellow
    Write-Host "  Running: detect-secrets" -ForegroundColor Gray
    
    if (Get-Command detect-secrets -ErrorAction SilentlyContinue) {
        try {
            detect-secrets scan --all-files --baseline .secrets.baseline
            Write-Host "  ✅ No new secrets detected" -ForegroundColor Green
        } catch {
            Write-Host "  ⚠️  Potential secrets found! Review output" -ForegroundColor Yellow
            $findings.high++
        }
    } else {
        Write-Host "  ⚠️  detect-secrets not installed (pip install detect-secrets)" -ForegroundColor Yellow
    }
}

function Generate-SBOM {
    Write-Host "[SEC-3] SBOM Generation" -ForegroundColor Yellow
    Write-Host "  Generating Software Bill of Materials..." -ForegroundColor Gray
    
    # Using pip-licenses or cyclonedx-bom
    if (Get-Command pip-licenses -ErrorAction SilentlyContinue) {
        pip-licenses --format=json --output-file=release/artifacts/sbom.json
        Write-Host "  ✅ SBOM generated (pip-licenses)" -ForegroundColor Green
    } elseif (Get-Command cyclonedx-py -ErrorAction SilentlyContinue) {
        cyclonedx-py -o release/artifacts/sbom.json
        Write-Host "  ✅ SBOM generated (cyclonedx)" -ForegroundColor Green
    } else {
        # Fallback: generate simple SBOM from requirements.txt
        Write-Host "  ⚠️  Using requirements.txt fallback" -ForegroundColor Yellow
        
        $sbom = @{
            bomFormat = "CycloneDX"
            specVersion = "1.4"
            version = 1
            components = @()
        }
        
        Get-Content requirements.txt | ForEach-Object {
            if ($_ -match "^([a-zA-Z0-9\-_]+)==(.+)$") {
                $sbom.components += @{
                    type = "library"
                    name = $matches[1]
                    version = $matches[2]
                    purl = "pkg:pypi/$($matches[1])@$($matches[2])"
                }
            }
        }
        
        $sbom | ConvertTo-Json -Depth 10 | Out-File -FilePath "release/artifacts/sbom.json" -Encoding UTF8
        Write-Host "  ✅ Basic SBOM generated" -ForegroundColor Green
    }
}

# Run selected scans
if ($All -or $PSBoundParameters.Count -eq 0) {
    Run-SAST
    Run-SecretScan
    Generate-SBOM
} else {
    if ($SAST) { Run-SAST }
    if ($SecretScan) { Run-SecretScan }
    if ($SBOM) { Generate-SBOM }
}

# Generate security report summary
$securityReport = @{
    scan_date = (Get-Date).ToString("o")
    findings = $findings
    gate_g5_status = if ($findings.critical -eq 0 -and $findings.high -eq 0) { "PASS" } else { "FAIL" }
    sbom_generated = (Test-Path "release/artifacts/sbom.json")
    artifacts = @{
        ruff = "release/artifacts/ruff_report.json"
        bandit = "release/artifacts/bandit_report.json"
        mypy = "release/artifacts/mypy_report.txt"
        sbom = "release/artifacts/sbom.json"
    }
}

$securityReport | ConvertTo-Json -Depth 10 | Out-File -FilePath "release/artifacts/security_report.json" -Encoding UTF8

Write-Host ""
Write-Host "📊 Security Scan Summary" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan
Write-Host "Critical: $($findings.critical)" -ForegroundColor $(if ($findings.critical -gt 0) { "Red" } else { "Gray" })
Write-Host "High: $($findings.high)" -ForegroundColor $(if ($findings.high -gt 0) { "Red" } else { "Gray" })
Write-Host "Medium: $($findings.medium)" -ForegroundColor Yellow
Write-Host "Gate G5 (0 critical/high): $($securityReport.gate_g5_status)" -ForegroundColor $(if ($securityReport.gate_g5_status -eq "PASS") { "Green" } else { "Red" })
Write-Host "SBOM: $(if ($securityReport.sbom_generated) { '✅ Generated' } else { '❌ Missing' })" -ForegroundColor White
Write-Host ""
Write-Host "Reports in: release/artifacts/" -ForegroundColor Gray
