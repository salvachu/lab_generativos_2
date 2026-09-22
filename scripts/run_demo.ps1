param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot 'src'
& (Join-Path $projectRoot '.venv\Scripts\python.exe') -m uvicorn app.server:app --host 127.0.0.1 --port $Port
