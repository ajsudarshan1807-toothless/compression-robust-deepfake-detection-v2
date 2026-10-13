@echo off
setlocal
cd /d "%~dp0"
title Compression-Robust Deepfake Detector

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

echo Starting Compression-Robust Deepfake Detection System...
"%PY%" start.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo System exited with code %ERRORLEVEL%.
    pause
)
