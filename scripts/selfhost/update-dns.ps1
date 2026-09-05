<#
    update-dns.ps1 - keep the Vercel A record for finprint pointing at this
    connection, so a residential IP change does not silently take the site down.

    Cloud Run never needed this: the CNAME pointed at Google and Google owned a
    stable address. Self-hosting on a home line moves that problem here - the
    ISP can hand out a new IP at any lease renewal, and the only symptom is that
    the domain stops resolving to you.

    One-off run (also the first cutover, replacing the Cloud Run CNAME):

        .\scripts\selfhost\update-dns.ps1 -Token <vercel-token> -Force

    Install as a background check every 5 minutes:

        .\scripts\selfhost\update-dns.ps1 -Token <vercel-token> -Install

    The token: Vercel -> Account Settings -> Tokens. Vercel tokens are
    account-wide - there is no per-domain scope - so treat it as a real secret.
    -Install stores it under .caches\ with an ACL that admits only SYSTEM and
    Administrators, and .gitignore keeps it out of the repo.
#>

[CmdletBinding()]
param(
    [string]$Token,
    [string]$Domain = "ethanyanxu.com",
    [string]$Subdomain = "finprint",
    [int]$Ttl = 60,
    # Replace a record that is not already an A record pointing somewhere else
    # (i.e. the Cloud Run CNAME). Required for the initial cutover, so that a
    # routine IP check can never quietly rewrite unrelated DNS.
    [switch]$Force,
    [switch]$Install,
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$TokenFile = Join-Path $Root ".caches\vercel-token.txt"
$TaskName = "finprint-dns"
$Fqdn = "$Subdomain.$Domain"

function Info($m) { Write-Host "    $m" }
function Good($m) { Write-Host "    $m" -ForegroundColor Green }
function Note($m) { Write-Host "    $m" -ForegroundColor Yellow }

# --- uninstall ------------------------------------------------------------
if ($Uninstall) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Good "removed scheduled task '$TaskName'"
    } else { Info "no scheduled task to remove" }
    if (Test-Path $TokenFile) { Remove-Item $TokenFile -Force; Good "removed stored token" }
    exit 0
}

# --- token ----------------------------------------------------------------
if (-not $Token -and (Test-Path $TokenFile)) {
    $Token = (Get-Content $TokenFile -Raw).Trim()
}
if (-not $Token) {
    throw "No Vercel token. Pass -Token <token>, or run with -Install once to store it."
}

# --- install --------------------------------------------------------------
if ($Install) {
    New-Item -ItemType Directory -Force -Path (Split-Path $TokenFile) | Out-Null
    Set-Content -Path $TokenFile -Value $Token -Encoding ASCII -NoNewline

    # Lock the file down: inheritance off, then only SYSTEM and Administrators.
    # A plaintext account-wide token readable by every local user would be a
    # worse problem than the one this script solves.
    $acl = Get-Acl $TokenFile
    $acl.SetAccessRuleProtection($true, $false)
    foreach ($id in "SYSTEM", "Administrators") {
        $acl.AddAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule(
            $id, "FullControl", "Allow")))
    }
    Set-Acl -Path $TokenFile -AclObject $acl
    Good "stored token at $TokenFile (SYSTEM + Administrators only)"

    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Domain $Domain -Subdomain $Subdomain -Ttl $Ttl" `
        -WorkingDirectory $Root
    $trigger = New-ScheduledTaskTrigger -AtStartup
    $trigger.Delay = "PT2M"
    $repeat = New-ScheduledTaskTrigger -Once -At (Get-Date) `
        -RepetitionInterval (New-TimeSpan -Minutes 5)
    $principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -MultipleInstances IgnoreNew -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger @($trigger, $repeat) `
        -Principal $principal -Settings $settings `
        -Description "Keep $Fqdn pointing at this connection public IP" | Out-Null
    Good "registered task '$TaskName' (every 5 minutes)"
    Info "running one update now ..."
}

# --- current public IP ----------------------------------------------------
$pub = $null
foreach ($svc in "https://api.ipify.org", "https://ifconfig.me/ip", "https://icanhazip.com") {
    try { $pub = (Invoke-RestMethod -Uri $svc -TimeoutSec 15).ToString().Trim(); break } catch { }
}
if (-not $pub) { throw "Could not determine the public IP." }
if ($pub -notmatch '^\d{1,3}(\.\d{1,3}){3}$') { throw "Public address '$pub' is not IPv4." }
Info "public IP: $pub"

# --- Vercel API -----------------------------------------------------------
$headers = @{ Authorization = "Bearer $Token" }

function Invoke-Vercel($method, $url, $body) {
    # Not named $args: that is an automatic variable inside a function, and
    # shadowing it is a trap for whoever edits this next.
    $req = @{ Uri = $url; Method = $method; Headers = $headers; TimeoutSec = 30 }
    if ($body) {
        $req.Body = ($body | ConvertTo-Json -Compress)
        $req.ContentType = "application/json"
    }
    return Invoke-RestMethod @req
}

try {
    $resp = Invoke-Vercel GET "https://api.vercel.com/v4/domains/$Domain/records?limit=100" $null
} catch {
    throw "Vercel API rejected the request (is the token valid, and does it cover $Domain?): $($_.Exception.Message)"
}

$mine = $resp.records | Where-Object { $_.name -eq $Subdomain }
$aRec = $mine | Where-Object { $_.type -eq 'A' } | Select-Object -First 1
$other = $mine | Where-Object { $_.type -ne 'A' }

# --- decide ---------------------------------------------------------------
if ($aRec -and $aRec.value -eq $pub) {
    Good "$Fqdn already points at $pub - nothing to do"
    exit 0
}

if ($other -and -not $aRec -and -not $Force) {
    Note "$Fqdn currently has: $(($other | ForEach-Object { "$($_.type) -> $($_.value)" }) -join ', ')"
    Note "Refusing to replace a non-A record without -Force. This is the Cloud Run"
    Note "CNAME; re-run with -Force to complete the cutover."
    exit 2
}

# Remove anything at this name that is not the A record we are about to manage.
# A stale CNAME alongside an A record is not merely untidy - it is invalid, and
# resolvers behave unpredictably.
foreach ($r in $other) {
    Info "deleting $($r.type) record $Fqdn -> $($r.value)"
    Invoke-Vercel DELETE "https://api.vercel.com/v2/domains/$Domain/records/$($r.id)" $null | Out-Null
}

if ($aRec) {
    Info "updating A record: $($aRec.value) -> $pub"
    try {
        Invoke-Vercel PATCH "https://api.vercel.com/v1/domains/records/$($aRec.id)" @{
            name = $Subdomain; type = "A"; value = $pub; ttl = $Ttl
        } | Out-Null
    } catch {
        # Older/changed API surface: fall back to delete-then-create. Ordered so
        # the window with no record is as short as possible.
        Note "PATCH failed ($($_.Exception.Message)); falling back to delete + create"
        Invoke-Vercel DELETE "https://api.vercel.com/v2/domains/$Domain/records/$($aRec.id)" $null | Out-Null
        Invoke-Vercel POST "https://api.vercel.com/v2/domains/$Domain/records" @{
            name = $Subdomain; type = "A"; value = $pub; ttl = $Ttl
        } | Out-Null
    }
} else {
    Info "creating A record $Fqdn -> $pub"
    Invoke-Vercel POST "https://api.vercel.com/v2/domains/$Domain/records" @{
        name = $Subdomain; type = "A"; value = $pub; ttl = $Ttl
    } | Out-Null
}

Good "$Fqdn now points at $pub"
Info "propagation takes up to the old TTL; Caddy picks up certificates on its own."
