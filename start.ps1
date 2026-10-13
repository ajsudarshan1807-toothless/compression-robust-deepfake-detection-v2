# PowerShell one-click launcher for Compression-Robust Deepfake Detection

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvPy = Join-Path $ScriptDir ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPy)) {
    $PyExe = "python"
} else {
    $PyExe = $VenvPy
}

Set-Location $ScriptDir
Write-Host "Starting Compression-Robust Deepfake Detector..." -ForegroundColor Cyan
& $PyExe "start.py" $args
