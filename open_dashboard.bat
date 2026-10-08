@echo off
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (echo Run setup_windows.bat first.&pause&exit /b 1)
start "TrustShieldAI Server" /min cmd /c ""%~dp0venv\Scripts\python.exe" "%~dp0backend\app.py""
timeout /t 4 /nobreak >nul
start "" "http://127.0.0.1:5000"
