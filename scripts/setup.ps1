param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo crear el entorno. Especifica -Python con Python 3.11 o 3.12 instalado.' }
}
& '.\.venv\Scripts\python.exe' -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw 'No se pudo instalar PyTorch CUDA.' }
& '.\.venv\Scripts\python.exe' -m pip install -e '.[dev]'
if ($LASTEXITCODE -ne 0) { throw 'No se pudo instalar sketchlab.' }
