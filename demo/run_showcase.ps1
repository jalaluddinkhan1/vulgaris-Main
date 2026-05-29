# VULGARIS company showcase — Windows
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot) | Out-Null
Write-Host "Running VULGARIS company showcase..." -ForegroundColor Cyan
python demo/company_showcase.py @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host ""
Write-Host "Open leave-behind:" -ForegroundColor Green
Write-Host "  demo\output\showcase_report.md"
