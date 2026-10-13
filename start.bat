@echo off
setlocal
cd /d "%~dp0"
title Compression-Robust Deepfake Detector

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    echo [*] First-time setup detected: creating virtual environment (.venv)...
    python -m venv .venv
    if exist ".venv\Scripts\python.exe" (
        set "PY=.venv\Scripts\python.exe"
        echo [*] Installing dependencies from requirements.txt...
        "%PY%" -m pip install --upgrade pip
        "%PY%" -m pip install -r requirements.txt
    ) else (
        echo [!] Warning: Could not create .venv. Falling back to system python.
        set "PY=python"
    )
)

echo.
echo [*] Starting Compression-Robust Deepfake Detection System...
"%PY%" start.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo System exited with code %ERRORLEVEL%.
    pause
)
