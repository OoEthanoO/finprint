<#
    update.ps1 - deploy a new version. This replaces the GitHub Actions
    "push to main -> Cloud Run" pipeline.

        .\scripts\selfhost\update.ps1

    Run as Administrator on the hosting laptop.

    The pipeline it replaces did four things: pull the code, install deps, roll
    the service, and refuse to call it a success unless the new commit was
    actually being served. This keeps all four - the last one especially, since
    "the deploy said OK but the old build is still live" is the failure that
    made the CI check worth having.

    Rollback is git: check out the previous commit and run this again.
#>

[CmdletBinding()]
param(
    [switch]$SkipPull,
    [string]$Domain = "finprint.ethanyanxu.com",
    [int]$Port = 8000
)

# Continue, not Stop. Under "Stop", Windows PowerShell 5.1 turns anything a
# native executable writes to stderr into a terminating error - and `git pull`
# reports "From https://github.com/..." on stderr on a perfectly successful
# pull, as pip does for deprecation notices. Failure is caught explicitly below
# through $LASTEXITCODE and throw.
$ErrorActionPreference = "Continue"

function Step($t) { Write-Host ""; Write-Host "==> $t" -ForegroundColor Cyan }
function Info($m) { Write-Host "    $m" }
function Good($m) { Write-Host "    $m" -ForegroundColor Green }
function Note($m) { Write-Host "    $m" -ForegroundColor Yellow }

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw "Run this in an Administrator PowerShell (it restarts SYSTEM scheduled tasks)." }

# Free the port the app has to bind, and confirm it actually came free.
#
# Identifying the old process by command line ("python.exe running uvicorn") is
# not reliable enough: it depends on being able to read another SYSTEM process's
# command line, and any process it fails to match still owns the socket. The
# replacement then exits because it cannot bind, the health check finds the OLD
# build answering, and the deploy looks fine while shipping nothing. Whatever is
# listening on the port is the thing in the way, so target that.
function Stop-AppOnPort($p) {
    $owners = @(Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique)
    foreach ($procId in $owners) {
        # PIDs 0 and 4 are System/Idle and are never ours.
        if ($procId -and $procId -gt 4) {
            $n = (Get-Process -Id $procId -ErrorAction SilentlyContinue).ProcessName
            Info "stopping pid $procId ($n) holding port $p"
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        }
    }
    for ($i = 1; $i -le 15; $i++) {
        if (-not (Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction SilentlyContinue)) { return $true }
        Start-Sleep -Seconds 1
    }
    return $false
}

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) { throw "No .venv at $Root - run setup.ps1 first." }

$before = (& git -C $Root rev-parse HEAD).Trim()
Info "current commit: $($before.Substring(0,7))"

# --- 1. pull --------------------------------------------------------------
if (-not $SkipPull) {
    Step "Pulling"
    $dirty = & git -C $Root status --porcelain
    if ($dirty) {
        Note "working tree has local changes:"
        $dirty | ForEach-Object { Note "  $_" }
        throw "Refusing to pull over local changes. Commit, stash, or discard them first."
    }
    & git -C $Root pull --ff-only
    if ($LASTEXITCODE -ne 0) { throw "git pull failed (not a fast-forward?)" }
}
$after = (& git -C $Root rev-parse HEAD).Trim()
if ($after -eq $before -and -not $SkipPull) { Info "already up to date at $($before.Substring(0,7))" }
else { Good "now at $($after.Substring(0,7))" }

# --- 2. dependencies ------------------------------------------------------
# Cheap when nothing changed (pip no-ops), and skipping it is how a deploy ends
# up importing a module the new code needs and the venv does not have.
Step "Dependencies"
& $VenvPy -m pip install -r (Join-Path $Root "requirements-serve.txt") --quiet
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }
Good "up to date"

Step "Precompiling bytecode"
& $VenvPy -m compileall -q -j 0 $Root *> $null
Good "done"

# --- 3. roll the service --------------------------------------------------
Step "Restarting"
Stop-ScheduledTask -TaskName "finprint-app" -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
if (-not (Stop-AppOnPort $Port)) {
    throw "Port $Port is still held after trying to free it. Find the owner with: netstat -ano -p tcp | findstr $Port"
}
Start-ScheduledTask -TaskName "finprint-app"
Good "finprint-app restarted"

# --- 4. verify the new commit is the one serving --------------------------
# Not just "something answers": traffic still served by the old process would
# pass a plain health check while running the previous build.
Step "Verifying"
Info "waiting for warm-up (the port opens only once it can predict) ..."
$ok = $false
for ($i = 1; $i -le 40; $i++) {
    Start-Sleep -Seconds 5
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 10
        if ($h.version -eq $after) { Good "serving $($after.Substring(0,7))"; $ok = $true; break }
        Info "  answering, but reports $($h.version.Substring(0,7)) - waiting for the new process"
    } catch { }
    if ($i % 4 -eq 0) { Info "  still starting ... ($($i * 5)s)" }
}
if (-not $ok) {
    throw "The service never reported $($after.Substring(0,7)). Run scripts\selfhost\run-app.cmd in a console to see why. Roll back with: git -C `"$Root`" checkout $before ; .\scripts\selfhost\update.ps1 -SkipPull"
}

try {
    $p = Invoke-RestMethod -Uri "https://$Domain/api/health" -TimeoutSec 60
    if ($p.version -eq $after) { Good "https://$Domain is serving $($after.Substring(0,7))" }
    else { Note "public endpoint reports $($p.version) - DNS or Caddy may need a moment" }
} catch {
    Note "could not reach https://$Domain : $($_.Exception.Message)"
    Note "the app is healthy locally, so this is DNS, the router forward, or Caddy - run verify.ps1"
}

Write-Host ""
Write-Host "Deploy complete: $($before.Substring(0,7)) -> $($after.Substring(0,7))" -ForegroundColor Green
