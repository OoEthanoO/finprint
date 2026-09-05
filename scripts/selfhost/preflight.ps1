<#
    preflight.ps1 - check whether this machine can serve finprint on a public
    domain, BEFORE anything is installed or any DNS is repointed.

    Read-only: it changes nothing. Run it on the laptop that will host finprint.

    The port-forward design has four hard requirements, and each one fails in a
    different, confusing way at a different time. Checking them up front turns
    "Caddy cannot get a certificate and I do not know why" into a named cause:

      1. a real public IPv4 (not CGNAT) - otherwise no port forward can exist
      2. ports 80/443 free on this machine  - otherwise Caddy cannot bind
      3. Python 3.11                        - torch/torchaudio wheels need it
      4. DNS pointing here                  - Let's Encrypt validates over it
#>

[CmdletBinding()]
param(
    [string]$Domain = "finprint.ethanyanxu.com"
)

$ErrorActionPreference = "Continue"
$script:Fail = 0
$script:Warn = 0

function Section($t) { Write-Host ""; Write-Host "== $t ==" -ForegroundColor Cyan }
function Pass($m)    { Write-Host "  [ok]   $m" -ForegroundColor Green }
function Warn($m)    { Write-Host "  [warn] $m" -ForegroundColor Yellow; $script:Warn++ }
function Fail($m)    { Write-Host "  [FAIL] $m" -ForegroundColor Red;    $script:Fail++ }

Write-Host "finprint self-host preflight  (domain: $Domain)" -ForegroundColor White

# --- 1. public IP ---------------------------------------------------------
# A carrier-grade-NAT address means the ISP shares one public IP across many
# customers; there is no router setting that can forward a port to you, so the
# whole port-forward plan is dead and the fix is a tunnel instead.
Section "Public IP / CGNAT"
$pub = $null
foreach ($svc in "https://api.ipify.org", "https://ifconfig.me/ip", "https://icanhazip.com") {
    try { $pub = (Invoke-RestMethod -Uri $svc -TimeoutSec 12).ToString().Trim(); break } catch { }
}
if (-not $pub) {
    Fail "could not determine the public IP (no internet?)"
} else {
    Write-Host "  public IP: $pub"
    $o = $pub.Split('.')
    if ($o.Count -ne 4) {
        Warn "public address is not IPv4 ($pub) - an IPv6-only uplink cannot be port-forwarded the usual way"
    } else {
        $a = [int]$o[0]; $b = [int]$o[1]
        if ($a -eq 100 -and $b -ge 64 -and $b -le 127) {
            Fail "CGNAT range 100.64.0.0/10 - port forwarding is impossible on this connection"
        } elseif ($a -eq 10 -or ($a -eq 192 -and $b -eq 168) -or ($a -eq 172 -and $b -ge 16 -and $b -le 31)) {
            Fail "the 'public' IP is private ($pub) - you are behind a second NAT (double-NAT) or a VPN"
        } else {
            Pass "real public IPv4"
        }
    }
}

# --- 2. this machine on the LAN ------------------------------------------
# The router needs a fixed target for the forward. DHCP will eventually hand
# this machine a different address and silently break the forward, so the
# address below has to be reserved in the router before it is trusted.
Section "LAN address (the port-forward target)"
$cfg = Get-NetIPConfiguration | Where-Object { $_.IPv4Address -and $_.IPv4DefaultGateway } | Select-Object -First 1
if (-not $cfg) {
    Fail "no interface with an IPv4 address and a default gateway"
} else {
    $lan = $cfg.IPv4Address.IPAddress
    $gw = $cfg.IPv4DefaultGateway.NextHop
    Pass "interface '$($cfg.InterfaceAlias)'  ip=$lan  gateway=$gw"
    Write-Host "         forward router ports 80 and 443 -> $lan"
    Write-Host "         router admin page is usually http://$gw"
    $origin = (Get-NetIPAddress -IPAddress $lan -ErrorAction SilentlyContinue).PrefixOrigin
    if ($origin -eq 'Dhcp') {
        Warn "this address came from DHCP - reserve it (DHCP reservation / static lease) or the forward breaks when it changes"
    }
    if ($cfg.InterfaceAlias -like '*Wi-Fi*') {
        Warn "hosting over Wi-Fi works, but Ethernet is steadier for an always-on service"
    }
}

# --- 3. ports free --------------------------------------------------------
Section "Ports 80 / 443 / 8000 free on this machine"
foreach ($p in 80, 443, 8000) {
    $used = Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction SilentlyContinue
    if ($used) {
        $procs = ($used | ForEach-Object { (Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue).ProcessName } | Sort-Object -Unique) -join ', '
        if ($p -eq 8000) { Warn "port $p already in use by: $procs (the app will need it)" }
        else { Fail "port $p already in use by: $procs (Caddy cannot bind it)" }
    } else {
        Pass "port $p is free"
    }
}

# --- 4. Python 3.11 -------------------------------------------------------
# torch/torchaudio CPU wheels are what pin the version; the Dockerfile pinned
# 3.11 for exactly this reason and a venv here has the same constraint.
Section "Python 3.11"
$py = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.11 --version *> $null
    if ($LASTEXITCODE -eq 0) { $py = "py -3.11" }
}
if (-not $py) {
    foreach ($cand in "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe", "C:\Python311\python.exe", "$env:ProgramFiles\Python311\python.exe") {
        if (Test-Path $cand) { $py = $cand; break }
    }
}
if ($py) { Pass "found: $py" }
else { Warn "Python 3.11 not found - setup.ps1 will install it (winget install Python.Python.3.11)" }

# --- 5. supporting tools --------------------------------------------------
Section "Supporting tools"
$tools = @(
    @{ n = 'caddy';  w = 'CaddyServer.Caddy'; why = 'terminates HTTPS and gets the certificate' },
    @{ n = 'ffmpeg'; w = 'Gyan.FFmpeg';       why = 'decodes mp3/m4a/webm uploads' },
    @{ n = 'git';    w = 'Git.Git';           why = 'pulls updates and stamps the build SHA' },
    @{ n = 'winget'; w = '';                  why = 'installs the three above' }
)
foreach ($t in $tools) {
    if (Get-Command $t.n -ErrorAction SilentlyContinue) { Pass "$($t.n) present" }
    elseif ($t.n -eq 'winget') { Fail "winget missing - install the others by hand" }
    else { Warn "$($t.n) missing ($($t.why)) - setup.ps1 installs it: winget install $($t.w)" }
}

# --- 6. DNS ---------------------------------------------------------------
# Let's Encrypt resolves this name from the public internet and connects back,
# so the record has to point at THIS connection public IP before a certificate
# can be issued.
Section "DNS for $Domain"
try {
    $rr = Resolve-DnsName -Name $Domain -Server 1.1.1.1 -ErrorAction Stop
    $a = ($rr | Where-Object { $_.Type -eq 'A' } | Select-Object -First 1).IPAddress
    $cn = ($rr | Where-Object { $_.Type -eq 'CNAME' } | Select-Object -First 1).NameHost
    if ($cn) { Write-Host "  CNAME -> $cn" }
    if ($a) { Write-Host "  A     -> $a" }
    if ($a -and $pub -and $a -eq $pub) {
        Pass "DNS already points at this connection - Caddy can be issued a certificate"
    } elseif ($cn -like '*googlehosted*') {
        Warn "still pointing at Cloud Run - expected until you cut over; replace it with an A record -> $pub"
    } else {
        Warn "does not point here yet (want A -> $pub) - Caddy cannot get a certificate until it does"
    }
} catch {
    Warn "could not resolve $Domain : $($_.Exception.Message)"
}

# --- verdict --------------------------------------------------------------
Write-Host ""
if ($script:Fail -gt 0) {
    Write-Host "PREFLIGHT FAILED: $($script:Fail) blocking problem(s), $($script:Warn) warning(s)." -ForegroundColor Red
    Write-Host "A CGNAT or double-NAT failure cannot be fixed on this machine - use a tunnel (Cloudflare Tunnel) instead of port forwarding." -ForegroundColor Red
    exit 1
}
Write-Host "PREFLIGHT OK: no blocking problems, $($script:Warn) warning(s)." -ForegroundColor Green
Write-Host "Next: .\scripts\selfhost\setup.ps1 -AcmeEmail you@example.com   (run as Administrator)"
exit 0
