$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "src\football_quant\cli.py"))) {
    $Root = $PSScriptRoot
}
Set-Location $Root
$env:PYTHONPATH = Join-Path $Root "src"
$env:PYTHONUNBUFFERED = "1"

$running = Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match "football_quant.cli live-corner-scan" }
if ($running) {
    exit 0
}

$logDir = Join-Path $Root "data\forward"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "live_watch.log"
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    $pythonCmd = Get-Command py -ErrorAction SilentlyContinue
}
$python = if ($pythonCmd) { $pythonCmd.Source } else { $null }
if (-not $python) {
    Add-Content -Path $log -Value "$(Get-Date -Format o) python not found"
    exit 1
}

Add-Content -Path $log -Value "$(Get-Date -Format o) starting live match watch (goals, cards, corners, odds)"
& $python -u -m football_quant.cli live-corner-scan --poll-seconds 180 --idle-seconds 600 --max-cycles 0 *>> $log
