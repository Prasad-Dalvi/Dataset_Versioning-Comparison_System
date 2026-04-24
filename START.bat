@echo off
title DataVault V2 - Starting...
color 0A

echo.
echo  ██████╗  █████╗ ████████╗ █████╗ ██╗   ██╗ █████╗ ██╗  ████████╗
echo  ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██║   ██║██╔══██╗██║  ╚══██╔══╝
echo  ██║  ██║███████║   ██║   ███████║██║   ██║███████║██║     ██║
echo  ██║  ██║██╔══██║   ██║   ██╔══██║╚██╗ ██╔╝██╔══██║██║     ██║
echo  ██████╔╝██║  ██║   ██║   ██║  ██║ ╚████╔╝ ██║  ██║███████╗██║
echo  ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝╚═╝
echo.
echo  DataVault V2 - Dataset Versioning and ML Platform
echo  ===================================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Install Python 3.9+ from python.org
    pause
    exit /b 1
)

echo [1/3] Checking Python packages...
cd backend

REM Install requirements silently
pip install -r requirements.txt -q --no-warn-script-location
if %errorlevel% neq 0 (
    echo [WARNING] Some packages may have failed. Trying to continue...
)

echo [2/3] Creating data directories...
if not exist "..\data\storage" mkdir "..\data\storage"

echo [3/3] Starting DataVault V2 backend...
echo.
echo  ====================================
echo   Backend:  http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo   Admin:    admin / Admin@123
echo  ====================================
echo.
echo  Open frontend/index.html in your browser after startup.
echo  Press Ctrl+C to stop the server.
echo.

start "" "http://localhost:8000"
timeout /t 2 /nobreak >nul

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
