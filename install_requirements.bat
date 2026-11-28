@echo off
chcp 65001 > nul
echo ===================================================
echo   CÀI ĐẶT THƯ VIỆN CẦN THIẾT CHO DỰ ÁN HỌC MÁY
echo ===================================================
echo.
set PYTHON_EXE=.venv\Scripts\python.exe

if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=C:\Users\main1\AppData\Roaming\uv\python\cpython-3.11.17-windows-x86_64-none\python.exe
)

if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=D:\miniconda\python.exe
)

echo Dang cai dat thu vien vao Python: %PYTHON_EXE%
"%PYTHON_EXE%" -m pip install -r requirements.txt
echo.
echo ===================================================
echo Cai dat hoan tat!
echo - Chay pipeline CLI:       run.bat
echo - Chay Web App Streamlit:  run_web.bat
echo ===================================================
pause
