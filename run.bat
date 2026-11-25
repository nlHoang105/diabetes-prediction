@echo off
chcp 65001 > nul
echo ===================================================
echo   CHẠY PIPELINE HỌC MÁY - BÀI KIỂM TRA SỐ 1
echo ===================================================
echo.
set PYTHON_EXE=D:\miniconda\python.exe

if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=.venv\Scripts\python.exe
)

"%PYTHON_EXE%" main.py %*
echo.
pause
