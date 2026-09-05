<#
    setup.ps1 - install finprint as an always-on service on a Windows laptop.

    Run once, as Administrator, from the repo root:

        .\scripts\selfhost\setup.ps1 -AcmeEmail you@example.com

    What it builds:

        internet -> router (80/443 forwarded) -> this laptop
                      caddy.exe          :80/:443   TLS, auto Let's Encrypt cert
                        -> reverse proxy 127.0.0.1:8000
                             uvicorn app.main:app   (.venv, Python 3.11)

    uvicorn binds 127.0.0.1, not 0.0.0.0: the only way in is through Caddy, so
    the app is never reachable in plaintext, not even from the LAN.

    Both processes run as scheduled tasks under SYSTEM with an "at startup"
    trigger, so the service survives a reboot and needs nobody logged in - the
    property that made Cloud Run useful and that a console window would lose.

    Re-running the script is safe: every step is idempotent.
#>

[CmdletBinding()]
param(
    # Certificate expiry notices from Let's Encrypt go here. Required: a wrong
    # address means no warning when renewal starts failing.
    [Parameter(Mandatory = $true)][string]$AcmeEmail,
    [string]$Domain = "finprint.ethanyanxu.com",
    [int]$Port = 8000,
    # A laptop that suspends is a server that is down. Skip only if you have
    # already configured the power plan yourself.
    [switch]$SkipPowerConfig,
    # Warm-up runs one throwaway prediction to JIT the librosa/numba kernels and
    # build the matplotlib font cache (~50 s once, instead of inside a visitor's
    # first request). Skipping makes setup faster and the first request slow.
    [switch]$SkipWarmup
)

# Continue, not Stop, and deliberately so. In Windows PowerShell 5.1 anything a
# native executable writes to stderr is wrapped in a NativeCommandError, which
# under "Stop" becomes a *terminating* error. That is fatal for this script:
# `py -3.11 --version` announcing "No suitable Python runtime found" is how the
# probe reports "not installed", and pip prints deprecation notices to stderr on
# a perfectly good install. Under "Stop" both abort the run.
#
# So failure is detected explicitly instead: every native call that must succeed
# is followed by a $LASTEXITCODE check that throws, and cmdlets that must not
# fail silently carry -ErrorAction Stop.
$ErrorActionPreference = "Continue"

function Step($t) { Write-Host ""; Write-Host "==> $t" -ForegroundColor Cyan }
function Info($m) { Write-Host "    $m" }
function Good($m) { Write-Host "    $m" -ForegroundColor Green }
function Note($m) { Write-Host "    $m" -ForegroundColor Yellow }

# --- 0. preconditions -----------------------------------------------------
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    throw "Run this in an Administrator PowerShell. It registers SYSTEM scheduled tasks and firewall rules."
}

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Here = $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "app\main.py"))) {
    throw "Could not find app\main.py under '$Root' - run this script from inside the finprint repo."
}
Info "repo root: $Root"

$Venv = Join-Path $Root ".venv"
$VenvPy = Join-Path $Venv "Scripts\python.exe"
$Caches = Join-Path $Root ".caches"

# --- 1. dependencies ------------------------------------------------------
Step "Dependencies (winget)"
function Ensure-Tool($exe, $wingetId, $why) {
    if (Get-Command $exe -ErrorAction SilentlyContinue) { Good "$exe already installed"; return }
    Info "installing $exe ($why) ..."
    winget install --id $wingetId --exact --silent --accept-package-agreements --accept-source-agreements --disable-interactivity
    # winget returns non-zero for "already installed" and for "no upgrade
    # found", neither of which is a failure here; the Get-Command below is the
    # real test.
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
    if (Get-Command $exe -ErrorAction SilentlyContinue) { Good "$exe installed" }
    else { Note "$exe still not on PATH - you may need to reopen PowerShell and re-run" }
}
Ensure-Tool "git"    "Git.Git"           "pulls updates, stamps the build SHA"
Ensure-Tool "caddy"  "CaddyServer.Caddy" "HTTPS + automatic certificates"
Ensure-Tool "ffmpeg" "Gyan.FFmpeg"       "decodes mp3/m4a/webm uploads"

# winget installs Caddy and ffmpeg *into the user profile*
# (%LOCALAPPDATA%\Microsoft\WinGet\...) and advertises them on the USER PATH.
# The services below run as SYSTEM, whose PATH contains none of that. Trusting
# PATH at run time therefore fails in two different ways, neither of them loud:
#
#   * `caddy run` in the task exits immediately with "not recognized", so the
#     task shows Ready instead of Running and nothing is listening on 443;
#   * the app cannot find ffmpeg, so wav/flac/ogg keep working (soundfile reads
#     those directly) while every mp3/m4a/webm upload fails to decode.
#
# Resolve the real locations now and bake them into the launchers.
Step "Resolving tool paths for the SYSTEM services"
function Resolve-Exe($name) {
    $c = Get-Command $name -ErrorAction SilentlyContinue
    if ($c -and $c.Source) { return $c.Source }
    return $null
}
$CaddyExe = Resolve-Exe "caddy"
$FfmpegExe = Resolve-Exe "ffmpeg"
$GitExe = Resolve-Exe "git"
if (-not $CaddyExe) { throw "caddy.exe not found even after install - reopen PowerShell and re-run." }
Good "caddy : $CaddyExe"
if ($FfmpegExe) { Good "ffmpeg: $FfmpegExe" }
else { Note "ffmpeg NOT found - mp3/m4a/webm uploads will fail (wav/flac/ogg still work)" }
if ($GitExe) { Good "git   : $GitExe" }
else { Note "git NOT found - the served build will report version 'unknown'" }

$FfmpegDir = if ($FfmpegExe) { Split-Path $FfmpegExe } else { "" }
$GitDir = if ($GitExe) { Split-Path $GitExe } else { "" }

# Python 3.11 specifically: torch and torchaudio publish CPU wheels for it, and
# the Dockerfile pinned the same version so training and serving cannot drift.
Step "Python 3.11"
# Kept as (executable, leading-args) rather than one array: the `py` launcher
# needs a "-3.11" selector in front of every command, a direct python.exe needs
# nothing, and splatting an empty array handles both without special-casing.
$PyExe = $null
$PyArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.11 --version *> $null
    if ($LASTEXITCODE -eq 0) { $PyExe = "py"; $PyArgs = @("-3.11") }
}
if (-not $PyExe) {
    foreach ($c in "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe", "C:\Python311\python.exe") {
        if (Test-Path $c) { $PyExe = $c; $PyArgs = @(); break }
    }
}
if (-not $PyExe) {
    Info "installing Python 3.11 ..."
    winget install --id Python.Python.3.11 --exact --silent --accept-package-agreements --accept-source-agreements --disable-interactivity
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 --version *> $null
        if ($LASTEXITCODE -eq 0) { $PyExe = "py"; $PyArgs = @("-3.11") }
    }
    if (-not $PyExe -and (Test-Path "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe")) {
        $PyExe = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"; $PyArgs = @()
    }
    if (-not $PyExe) { throw "Python 3.11 install did not take. Close and reopen PowerShell, then re-run." }
}
Good "using: $PyExe $($PyArgs -join ' ')"

# --- 2. virtualenv --------------------------------------------------------
Step "Virtualenv (.venv)"
if (Test-Path $VenvPy) {
    Good ".venv already exists"
} else {
    & $PyExe @PyArgs -m venv $Venv
    if (-not (Test-Path $VenvPy)) { throw "venv creation failed at $Venv" }
    Good "created $Venv"
}

Step "Python packages (this is the slow part - torch is ~200 MB)"
& $VenvPy -m pip install --upgrade pip --quiet
# Same split as the Dockerfile: CPU-only torch from PyTorch's own index (the
# default PyPI wheel drags in ~5 GB of CUDA that this app never touches),
# then the serving-only requirement set.
& $VenvPy -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }
& $VenvPy -m pip install -r (Join-Path $Root "requirements-serve.txt")
if ($LASTEXITCODE -ne 0) { throw "requirements-serve.txt install failed" }
Good "packages installed"

# Bytecode for the dependencies, for the same reason the Dockerfile does it:
# librosa imports submodules lazily, so without cached .pyc every cold start
# recompiles them inside the first request.
Step "Precompiling bytecode"
& $VenvPy -m compileall -q -j 0 (Join-Path $Venv "Lib\site-packages") $Root *> $null
Good "bytecode cached"

# --- 3. launchers ---------------------------------------------------------
# Scheduled tasks get a .cmd rather than a long argument string: the env vars
# live somewhere readable, and you can double-click either file to debug the
# exact command the service runs.
Step "Launchers"
New-Item -ItemType Directory -Force -Path (Join-Path $Caches "mpl"), (Join-Path $Caches "numba"), (Join-Path $Caches "caddy"), (Join-Path $Root "logs") | Out-Null

$runApp = Join-Path $Here "run-app.cmd"
@"
@echo off
REM Generated by setup.ps1 - runs the finprint API. Do not edit by hand; re-run setup.ps1.
cd /d "$Root"

REM This task runs as SYSTEM, which does not inherit the user PATH that winget
REM put ffmpeg on. Without this line soundfile still reads wav/flac/ogg, but
REM every mp3/m4a/webm upload fails to decode, because librosa's audioread path
REM shells out to ffmpeg by name.
set PATH=$FfmpegDir;$GitDir;%PATH%

REM Which commit is serving. In the container this had to be injected at deploy
REM time because .gcloudignore stripped .git; here the repo is right there, so
REM the answer is always correct and never stale.
set FINPRINT_GIT_SHA=unknown
for /f "delims=" %%i in ('"$GitExe" rev-parse HEAD 2^>nul') do set FINPRINT_GIT_SHA=%%i

REM Keep the matplotlib font cache and numba JIT cache on disk and out of the
REM user profile, so a restart does not pay ~50 s of rebuild.
set MPLCONFIGDIR=$Caches\mpl
set NUMBA_CACHE_DIR=$Caches\numba
set PYTHONUNBUFFERED=1

REM 127.0.0.1, not 0.0.0.0: Caddy is the only thing that may reach the app, so
REM nothing on the LAN can talk to it in plaintext.
"$VenvPy" -m uvicorn app.main:app --host 127.0.0.1 --port $Port
"@ | Set-Content -Path $runApp -Encoding ASCII -ErrorAction Stop

$runCaddy = Join-Path $Here "run-caddy.cmd"
@"
@echo off
REM Generated by setup.ps1 - terminates HTTPS and proxies to the API.
cd /d "$Here"

REM Certificates and ACME account keys land here instead of the SYSTEM account
REM profile, where they would be effectively unfindable.
set XDG_DATA_HOME=$Caches\caddy
set XDG_CONFIG_HOME=$Caches\caddy

REM Absolute path, not "caddy": winget installed it under the user profile and
REM this task runs as SYSTEM, which does not have that directory on its PATH.
"$CaddyExe" run --config "$Here\Caddyfile"
"@ | Set-Content -Path $runCaddy -Encoding ASCII -ErrorAction Stop
Good "wrote run-app.cmd and run-caddy.cmd"

# --- 4. Caddyfile ---------------------------------------------------------
Step "Caddyfile"
@"
# Generated by setup.ps1 for $Domain. Re-run setup.ps1 to regenerate.
{
	email $AcmeEmail
}

$Domain {
	# Automatic HTTPS: Caddy obtains and renews a Let's Encrypt certificate on
	# its own. That only works once $Domain resolves to this connection and the
	# router forwards 80/443 here - Let's Encrypt validates by connecting back
	# from the public internet.
	encode gzip zstd

	# The API accepts uploads up to 25 MB (config.MAX_UPLOAD_BYTES). Caddy has
	# to allow at least that or it would reject large clips before the app can
	# return its own 413 with a useful message.
	request_body {
		max_size 26MB
	}

	reverse_proxy 127.0.0.1:$Port {
		# A cold process runs warm-up before binding the port, but a prediction
		# on a long clip is still genuinely slow. Cloud Run allowed 300 s; match
		# it so behaviour does not change with the move.
		transport http {
			response_header_timeout 300s
		}
	}

	log {
		output file "$Root\logs\access.log" {
			roll_size 10MB
			roll_keep 5
		}
	}
}
"@ | Set-Content -Path (Join-Path $Here "Caddyfile") -Encoding UTF8 -ErrorAction Stop
Good "wrote Caddyfile for $Domain"

& caddy fmt --overwrite (Join-Path $Here "Caddyfile") *> $null
& caddy validate --config (Join-Path $Here "Caddyfile") *> $null
if ($LASTEXITCODE -ne 0) { Note "caddy validate reported a problem - check $Here\Caddyfile" }
else { Good "Caddyfile validates" }

# --- 5. firewall ----------------------------------------------------------
# The router forward gets packets to the machine; Windows Firewall still has to
# let them into the process. Missing this looks exactly like a broken forward.
Step "Windows Firewall"
foreach ($r in @(@{ n = "finprint HTTP (80)"; p = 80 }, @{ n = "finprint HTTPS (443)"; p = 443 })) {
    if (Get-NetFirewallRule -DisplayName $r.n -ErrorAction SilentlyContinue) {
        Good "$($r.n) rule already present"
    } else {
        New-NetFirewallRule -DisplayName $r.n -Direction Inbound -Action Allow `
            -Protocol TCP -LocalPort $r.p -Profile Any -ErrorAction Stop | Out-Null
        Good "opened inbound TCP $($r.p)"
    }
}

# --- 6. power -------------------------------------------------------------
if (-not $SkipPowerConfig) {
    Step "Power (a laptop that sleeps is a server that is down)"
    powercfg /change standby-timeout-ac 0
    powercfg /change hibernate-timeout-ac 0
    powercfg /change monitor-timeout-ac 10
    Good "on AC power: never sleep, never hibernate, screen off after 10 min"
    Note "battery settings untouched - keep this laptop plugged in"
    Note "closing the lid may still suspend it: Control Panel > Power Options > Choose what closing the lid does > Do nothing (plugged in)"
}

# --- 7. warm-up -----------------------------------------------------------
if (-not $SkipWarmup) {
    Step "Warm-up (one throwaway prediction; ~1 min, populates the JIT/font caches)"
    Push-Location $Root
    $env:MPLCONFIGDIR = Join-Path $Caches "mpl"
    $env:NUMBA_CACHE_DIR = Join-Path $Caches "numba"
    & $VenvPy -m scripts.warmup
    $wok = ($LASTEXITCODE -eq 0)
    Pop-Location
    if ($wok) { Good "warm-up succeeded - the app can predict on this machine" }
    else { Note "warm-up failed; the app may still work, but check ffmpeg and the model checkpoint" }
}

# --- 8. services ----------------------------------------------------------
# SYSTEM + "at startup" is what replaces Cloud Run's always-on property: no
# login required, survives reboot, restarts on crash.
Step "Registering services (scheduled tasks)"
function Register-FinprintTask($name, $cmd, $desc) {
    $action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$cmd`"" -WorkingDirectory $Root
    $trigger = New-ScheduledTaskTrigger -AtStartup
    $principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
    # ExecutionTimeLimit 0 = never kill it for running too long (the default 3
    # days would silently stop the server). Battery flags matter on a laptop:
    # the defaults refuse to start, and stop, on battery.
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
        -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
        -MultipleInstances IgnoreNew -StartWhenAvailable
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $name -Confirm:$false
    }
    Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger `
        -Principal $principal -Settings $settings -Description $desc -ErrorAction Stop | Out-Null
    Good "registered task '$name'"
}
Register-FinprintTask "finprint-app"   $runApp   "finprint API (uvicorn on 127.0.0.1:$Port)"
Register-FinprintTask "finprint-caddy" $runCaddy "finprint HTTPS reverse proxy (Caddy on 80/443)"

Step "Starting"
# Re-registering a task does not kill the process it previously started, and a
# leftover uvicorn keeps port $Port. The new instance would then fail to bind
# and the health check below would pass happily against the OLD build. Clearing
# it out first is what makes re-running setup.ps1 genuinely idempotent.
Stop-ScheduledTask -TaskName "finprint-app" -ErrorAction SilentlyContinue
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*uvicorn*app.main*" } |
    ForEach-Object {
        Info "stopping leftover uvicorn pid $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
Start-Sleep -Seconds 2
Start-ScheduledTask -TaskName "finprint-app"

# Which commit *should* be serving. Checking the version rather than just
# "something answered" is what catches a stale process still holding the port.
$HeadSha = ""
if ($GitExe) {
    $HeadSha = & $GitExe -C $Root rev-parse HEAD 2>$null
    if ($HeadSha) { $HeadSha = $HeadSha.Trim() }
}

Info "waiting for the API to answer (warm-up runs before the port opens) ..."
$ok = $false
for ($i = 1; $i -le 40; $i++) {
    Start-Sleep -Seconds 5
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 10
        if ($h.status -eq "ok" -and (-not $HeadSha -or $h.version -eq $HeadSha)) {
            Good "API healthy - version $($h.version), model_available=$($h.model_available)"
            $ok = $true; break
        }
        if ($h.status -eq "ok") {
            Info "  answering, but reports $($h.version.Substring(0,7)) - waiting for the new process"
        }
    } catch { }
    if ($i % 4 -eq 0) { Info "  still starting ... ($($i * 5)s)" }
}
if (-not $ok) {
    Note "API did not answer on 127.0.0.1:$Port within 200s."
    Note "Run $runApp in a console to see the error."
} else {
    Start-ScheduledTask -TaskName "finprint-caddy"
    # "Started the task" is not "the server is up". A task whose process exits
    # immediately drops straight back to Ready and reports no error anywhere
    # obvious -- which is precisely what a missing executable looks like. The
    # only honest check is whether something is now listening on 443.
    $caddyUp = $false
    for ($i = 1; $i -le 10; $i++) {
        Start-Sleep -Seconds 3
        if (Get-NetTCPConnection -State Listen -LocalPort 443 -ErrorAction SilentlyContinue) { $caddyUp = $true; break }
    }
    if ($caddyUp) {
        Good "Caddy is listening on 443"
    } else {
        $st = (Get-ScheduledTask -TaskName "finprint-caddy").State
        Note "Caddy is NOT listening on 443 (task state: $st)."
        Note "Run $runCaddy in a console to see the error."
    }
}

# --- 9. what is left for a human -----------------------------------------
$pub = "<unknown>"
try { $pub = (Invoke-RestMethod -Uri "https://api.ipify.org" -TimeoutSec 12).ToString().Trim() } catch { }
$lan = "<unknown>"
$cfg = Get-NetIPConfiguration | Where-Object { $_.IPv4Address -and $_.IPv4DefaultGateway } | Select-Object -First 1
if ($cfg) { $lan = $cfg.IPv4Address.IPAddress; $gw = $cfg.IPv4DefaultGateway.NextHop }

Write-Host ""
Write-Host "=========================================================" -ForegroundColor White
Write-Host " finprint is running locally. Two manual steps remain." -ForegroundColor White
Write-Host "=========================================================" -ForegroundColor White
Write-Host ""
Write-Host " 1. ROUTER - forward these to this laptop:" -ForegroundColor Cyan
Write-Host "      TCP 80  -> $lan : 80"
Write-Host "      TCP 443 -> $lan : 443"
Write-Host "    Router admin is usually http://$gw"
Write-Host "    Also reserve $lan for this laptop (DHCP reservation), or the"
Write-Host "    forward breaks the next time the address changes."
Write-Host ""
Write-Host " 2. DNS (Vercel dashboard -> ethanyanxu.com -> DNS):" -ForegroundColor Cyan
Write-Host "      DELETE  CNAME  finprint -> ghs.googlehosted.com"
Write-Host "      CREATE  A      finprint -> $pub    (TTL 60)"
Write-Host ""
Write-Host "    Caddy gets the certificate automatically once both are done -"
Write-Host "    a working https://$Domain is the proof the forward is open,"
Write-Host "    because Let's Encrypt validates from the public internet."
Write-Host ""
Write-Host " Then verify:  .\scripts\selfhost\verify.ps1" -ForegroundColor Cyan
Write-Host " Keep DNS current: .\scripts\selfhost\update-dns.ps1 -Install  (see DEPLOY.md)"
Write-Host ""
