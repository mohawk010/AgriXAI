# ── AgriXAI Web UI — Startup Script ──────────────────────────────────────────
# Run this from the project root:  powershell -File start_app.ps1

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Activate virtual environment
$venvActivate = Join-Path $root ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    Write-Host "Activating .venv..." -ForegroundColor Cyan
    . $venvActivate
} else {
    Write-Warning ".venv not found. Using system Python. Consider running: python -m venv .venv"
}

# Install any missing dependencies
Write-Host "Checking dependencies..." -ForegroundColor Cyan
pip install -q -r requirements.txt

Write-Host ""
Write-Host "===================================================" -ForegroundColor Green
Write-Host "  [ AgriXAI Web UI starting... ]" -ForegroundColor Green
Write-Host "  Open -> http://localhost:8000" -ForegroundColor Yellow
Write-Host "  Stop -> Ctrl+C" -ForegroundColor Gray
Write-Host "===================================================" -ForegroundColor Green
Write-Host ""

# Start FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

