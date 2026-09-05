<#
    verify.ps1 - prove the self-hosted deployment actually works, end to end.

    Run on the hosting laptop after the router forward and the DNS change.
    Read-only.

    It checks the chain in the order it breaks:

        services running -> app answers locally -> DNS points here
        -> certificate issued -> HTTPS answers -> serving the current commit

    The certificate check is the load-bearing one. Let's Encrypt validates by
    connecting back from the public internet, so a valid certificate is direct
    evidence that the port forward is open - something this machine cannot
    otherwise test about itself.
#>

[CmdletBinding()]
param(
    [string]$Domain = "finprint.ethanyanxu.com",
    [int]$Port = 8000
)

$ErrorActionPreference = "Continue"
$script:Fail = 0

function Section($t) { Write-Host ""; Write-Host "== $t ==" -ForegroundColor Cyan }
function Pass($m) { Write-Host "  [ok]   $m" -ForegroundColor Green }
function Warn($m) { Write-Host "  [warn] $m" -ForegroundColor Yellow }
function Fail($m) { Write-Host "  [FAIL] $m" -ForegroundColor Red; $script:Fail++ }

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

# --- 1. services ----------------------------------------------------------
Section "Services"
foreach ($t in "finprint-app", "finprint-caddy") {
    $task = Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue
    if (-not $task) { Fail "$t is not registered (run setup.ps1)" ; continue }
    $info = Get-ScheduledTaskInfo -TaskName $t
    if ($task.State -eq 'Running') { Pass "$t running" }
    else { Fail "$t is '$($task.State)' (last result: $($info.LastTaskResult)) - start it: Start-ScheduledTask -TaskName $t" }
}
$dnsTask = Get-ScheduledTask -TaskName "finprint-dns" -ErrorAction SilentlyContinue
if ($dnsTask) { Pass "finprint-dns registered (dynamic IP will be tracked)" }
else { Warn "finprint-dns not installed - a change of public IP will take the site down until you fix DNS by hand" }

# The launchers hard-code absolute paths to caddy and ffmpeg because winget puts
# both under the user profile while these services run as SYSTEM. If a re-install
# moved either one, the failure is silent: Caddy simply never listens, and
# mp3/m4a/webm uploads fail to decode while wav/flac/ogg keep working.
Section "Tool paths baked into the launchers"
foreach ($f in @("run-app.cmd", "run-caddy.cmd")) {
    $p = Join-Path $PSScriptRoot $f
    if (-not (Test-Path $p)) { Fail "$f missing - run setup.ps1" ; continue }
    $missing = @()
    foreach ($m in [regex]::Matches((Get-Content $p -Raw), '"([A-Za-z]:\\[^"]+\.exe)"')) {
        $exe = $m.Groups[1].Value
        if (-not (Test-Path $exe)) { $missing += $exe }
    }
    if ($missing) { Fail "$f references missing executables: $($missing -join ', ') - re-run setup.ps1" }
    else { Pass "$f executable paths all resolve" }
}
if (Get-NetTCPConnection -State Listen -LocalPort 443 -ErrorAction SilentlyContinue) {
    Pass "something is listening on 443"
} else {
    Fail "nothing is listening on 443 - Caddy is not actually running"
}

# --- 2. app on localhost --------------------------------------------------
Section "App on 127.0.0.1:$Port"
$local = $null
try {
    $local = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 30
    Pass "health: status=$($local.status) model_available=$($local.model_available)"
    if (-not $local.model_available) {
        Fail "the trained checkpoint is missing - species prediction is disabled (models\species_cnn.pt)"
    }
} catch {
    Fail "app not answering locally: $($_.Exception.Message)"
}

# --- 3. DNS ---------------------------------------------------------------
Section "DNS"
$pub = $null
try { $pub = (Invoke-RestMethod -Uri "https://api.ipify.org" -TimeoutSec 15).ToString().Trim() } catch { }
$a = $null
try {
    $rr = Resolve-DnsName -Name $Domain -Server 1.1.1.1 -ErrorAction Stop
    $a = ($rr | Where-Object { $_.Type -eq 'A' } | Select-Object -First 1).IPAddress
    $cn = ($rr | Where-Object { $_.Type -eq 'CNAME' } | Select-Object -First 1).NameHost
    if ($cn) { Fail "$Domain is still a CNAME -> $cn (cutover not done)" }
    elseif (-not $a) { Fail "$Domain has no A record" }
    elseif ($pub -and $a -ne $pub) { Fail "$Domain -> $a but this connection is $pub (stale record; run update-dns.ps1)" }
    else { Pass "$Domain -> $a (matches this connection)" }
} catch {
    Fail "cannot resolve $Domain : $($_.Exception.Message)"
}

# --- 4. certificate -------------------------------------------------------
Section "TLS certificate"
try {
    $tcp = New-Object System.Net.Sockets.TcpClient
    $tcp.ReceiveTimeout = 15000; $tcp.SendTimeout = 15000
    $tcp.Connect($Domain, 443)
    $ssl = New-Object System.Net.Security.SslStream($tcp.GetStream(), $false, { $true })
    $ssl.AuthenticateAsClient($Domain)
    $cert = New-Object System.Security.Cryptography.X509Certificates.X509Certificate2($ssl.RemoteCertificate)
    $days = [int]($cert.NotAfter - (Get-Date)).TotalDays
    Pass "issuer: $($cert.Issuer)"
    Pass "subject: $($cert.Subject), expires $($cert.NotAfter.ToString('yyyy-MM-dd')) ($days days)"
    if ($cert.Issuer -match "Caddy Local|localhost") {
        Fail "this is Caddy internal CA certificate, not a public one - Let's Encrypt could not reach this machine (port forward closed?)"
        Warn "why it failed is in logs\caddy.log on the host"
    } else {
        Pass "publicly-trusted certificate - Let's Encrypt reached this machine, so the port forward is open"
    }
    if ($days -lt 14) { Warn "renewal window is close; Caddy renews at ~30 days left - check it can still reach port 80/443" }
    $ssl.Dispose(); $tcp.Close()
} catch {
    Fail "TLS handshake with $Domain failed: $($_.Exception.Message)"
}

# --- 5. public HTTPS ------------------------------------------------------
Section "Public HTTPS"
try {
    $pubh = Invoke-RestMethod -Uri "https://$Domain/api/health" -TimeoutSec 60
    Pass "https://$Domain/api/health -> status=$($pubh.status) version=$($pubh.version)"

    # Is the public endpoint really this machine, or a leftover cache/proxy?
    if ($local -and $local.version -ne $pubh.version) {
        Warn "local version $($local.version) != public $($pubh.version) - something else may still be answering"
    }

    $head = (& git -C $Root rev-parse HEAD 2>$null)
    if ($LASTEXITCODE -eq 0 -and $head) {
        $head = $head.Trim()
        if ($pubh.version -eq $head) { Pass "serving the current commit ($($head.Substring(0,7)))" }
        elseif ($pubh.version -eq "unknown") { Warn "version reports 'unknown' - git was not on PATH for the service; re-run setup.ps1" }
        else { Warn "serving $($pubh.version.Substring(0,7)) but local HEAD is $($head.Substring(0,7)) - run update.ps1" }
    }
} catch {
    Fail "https://$Domain/api/health failed: $($_.Exception.Message)"
}

Section "HTTP redirect"
try {
    $r = Invoke-WebRequest -Uri "http://$Domain/api/health" -MaximumRedirection 0 -TimeoutSec 30 -ErrorAction Stop
    Warn "http:// returned $($r.StatusCode) without redirecting"
} catch {
    $code = $_.Exception.Response.StatusCode.value__
    if ($code -eq 301 -or $code -eq 302 -or $code -eq 308) { Pass "http:// redirects to https:// ($code)" }
    else { Warn "unexpected response on http:// : $($_.Exception.Message)" }
}

Write-Host ""
if ($script:Fail -gt 0) {
    Write-Host "VERIFY FAILED: $($script:Fail) problem(s)." -ForegroundColor Red
    Write-Host "Logs: $Root\logs\access.log, and 'Get-ScheduledTaskInfo -TaskName finprint-app'" -ForegroundColor Red
    exit 1
}
Write-Host "VERIFY OK - $Domain is served by this laptop over HTTPS." -ForegroundColor Green
exit 0
