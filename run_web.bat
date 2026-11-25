@echo off
cd /d "%~dp0"
title HE THONG CHAN DOAN TIEU DUONG STREAMLIT (ICTU)
echo ======================================================================
echo   KHOI DONG HE THONG CHAN DOAN TIEU DUONG STREAMLIT WEB APP (ICTU)
echo ======================================================================
echo.

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=C:\Users\main1\AppData\Roaming\uv\python\cpython-3.11.17-windows-x86_64-none\python.exe"
)

echo [1/2] Dang mo trinh duyet tai dia chi: http://localhost:8501 ...
start http://localhost:8501

echo [2/2] Dang khoi dong may chu Web Streamlit...
echo (Luu y: Giu nguyen cua so nay de web hoat dong)
echo.

"%PYTHON_EXE%" -m streamlit run app/app.py --server.headless=true --server.port=8501 --server.showEmailPrompt=false --browser.gatherUsageStats=false

echo.
pause
