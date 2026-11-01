$Root = Split-Path -Parent $PSScriptRoot
$script = Join-Path $Root "scripts\start-live-corner-watch.ps1"
$taskName = "FootballQuantLiveCornerWatch"
$action = "powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`""
schtasks /Create /TN $taskName /SC ONLOGON /RL LIMITED /F /TR $action | Out-Host
schtasks /Query /TN $taskName /FO LIST /V | Out-Host
