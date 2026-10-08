@echo off
title TrustShield AI

cd /d "%~dp0"

echo ==========================================
echo        TRUSTSHIELD AI
echo ==========================================
echo.
echo Starting TrustShield AI server...
echo.

if not exist "backend\app.py" (
    echo ERROR: backend\app.py was not found.
    echo.
    echo Current folder:
    cd
    echo.
    pause
    exit /b 1
)

if exist "venv\Scripts\python.exe" (
    set "PYTHON=venv\Scripts\python.exe"
) else if exist "%USERPROFILE%\venv\Scripts\python.exe" (
    set "PYTHON=%USERPROFILE%\venv\Scripts\python.exe"
) else (
    echo ERROR: Python virtual environment was not found.
    echo Expected:
    echo %USERPROFILE%\venv
    echo.
    pause
    exit /b 1
)

echo Using Python:
echo %PYTHON%
echo.

"%PYTHON%" backend\app.py

echo.
echo ==========================================
echo TrustShield AI server stopped.
echo ==========================================
pause