# PowerShell one-click launcher for Compression-Robust Deepfake Detection

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvPy = Join-Path $ScriptDir ".venv\Scripts\python.exe"

Set-Location $ScriptDir

if (-not (Test-Path $VenvPy)) {
    Write-Host "[*] First-time setup: Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
    if (Test-Path $VenvPy) {
        Write-Host "[*] Installing project dependencies from requirements.txt..." -ForegroundColor Yellow
        & $VenvPy -m pip install --upgrade pip
        & $VenvPy -m pip install -r requirements.txt
        $PyExe = $VenvPy
    } else {
        $PyExe = "python"
    }
} else {
    $PyExe = $VenvPy
}

Write-Host "Starting Compression-Robust Deepfake Detector..." -ForegroundColor Cyan
& $PyExe "start.py" $args
