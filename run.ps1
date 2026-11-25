Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  CHẠY PIPELINE HỌC MÁY - BÀI KIỂM TRA SỐ 1" -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""

$pythonExe = "D:\miniconda\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = ".venv\Scripts\python.exe"
}

& $pythonExe main.py @args
