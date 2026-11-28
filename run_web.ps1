# Script khoi dong Web App Streamlit
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   KHOI DONG HE THONG CHAN DOAN TIEU DUONG STREAMLIT WEB APP (ICTU)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

Set-Location -Path $PSScriptRoot

$pythonExe = "$PSScriptRoot\.venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "C:\Users\main1\AppData\Roaming\uv\python\cpython-3.11.17-windows-x86_64-none\python.exe"
}

Write-Host "[1/2] Dang mo trinh duyet tai: http://localhost:8501 ..." -ForegroundColor Green
Start-Process "http://localhost:8501"

Write-Host "[2/2] Dang khoi dong may chu Streamlit..." -ForegroundColor Green
Write-Host "Giu cua so nay de web hoat dong. Nhan Ctrl+C de dung." -ForegroundColor Yellow
Write-Host ""

& $pythonExe -m streamlit run app/app.py --server.headless=true --server.port=8501 --server.showEmailPrompt=false --browser.gatherUsageStats=false
