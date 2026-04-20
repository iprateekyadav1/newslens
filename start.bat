@echo off
title NewsLens Bias Detector
cd /d %~dp0

echo.
echo  ================================================
echo   NewsLens Media Bias Detector
echo  ================================================
echo.

:: Check if .venv exists; if not, create it
if not exist ".venv\" (
    echo [1/3] Creating virtual environment...
    python -m venv .venv
    echo.
)

:: Activate
call .venv\Scripts\activate.bat

:: Install dependencies if not done
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo [2/3] Installing dependencies...
    pip install -r requirements.txt
    echo.
)

echo [3/3] Starting server...
echo.
echo  Dashboard : http://localhost:5001
echo  API Docs  : http://localhost:5001/api/docs
echo  Health    : http://localhost:5001/api/health
echo.
echo  Press CTRL+C to stop.
echo.

python -m uvicorn app.main:app --host 0.0.0.0 --port 5001 --reload

pause
