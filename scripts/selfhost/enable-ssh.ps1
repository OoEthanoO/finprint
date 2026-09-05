<#
    enable-ssh.ps1 - one-time bootstrap so the finprint host laptop can be
    administered remotely, instead of by sitting in front of it.

    Run ONCE, on the laptop that will host finprint, in an Administrator
    PowerShell. This is the only step that genuinely requires being at that
    machine; everything afterwards (setup.ps1, update.ps1, verify.ps1) can be
    driven over SSH from elsewhere.

        .\scripts\selfhost\enable-ssh.ps1

    What it turns on, and the limits it puts on it:

      * OpenSSH Server - ships with Windows, disabled by default.
      * Key authentication only for the admin path. The public key below is
        the one half of a keypair; the private half never leaves the machine
        that generated it.
      * Firewall opens port 22 to the LOCAL SUBNET ONLY. Port 22 is never
        exposed to the internet - unlike 80/443, it is not in the router
        forward, and it must not be added to one.

    To undo everything this does:

        .\scripts\selfhost\enable-ssh.ps1 -Uninstall
#>

[CmdletBinding()]
param(
    # The public key allowed to log in. Default is the key generated on the
    # machine that will be driving this one; pass -PublicKey to use another.
    [string]$PublicKey = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDYNwUoiffXwAQGatrU2nqVCDXpAiCeNSZh5P0qBfK7Y claude-code-finprint-deploy",
    # Allow SSH from anywhere on the LAN rather than only this subnet. Still
    # never the internet.
    [switch]$AnyLocalNetwork,
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

function Step($t) { Write-Host ""; Write-Host "==> $t" -ForegroundColor Cyan }
function Info($m) { Write-Host "    $m" }
function Good($m) { Write-Host "    $m" -ForegroundColor Green }
function Note($m) { Write-Host "    $m" -ForegroundColor Yellow }

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw "Run this in an Administrator PowerShell." }

$AuthKeys = "$env:ProgramData\ssh\administrators_authorized_keys"
$FwRule = "finprint SSH (LAN only)"

# --- uninstall ------------------------------------------------------------
if ($Uninstall) {
    Step "Reverting"
    Get-Service sshd -ErrorAction SilentlyContinue | Stop-Service -Force -ErrorAction SilentlyContinue
    Set-Service sshd -StartupType Disabled -ErrorAction SilentlyContinue
    Good "sshd stopped and disabled"
    if (Get-NetFirewallRule -DisplayName $FwRule -ErrorAction SilentlyContinue) {
        Remove-NetFirewallRule -DisplayName $FwRule; Good "firewall rule removed"
    }
    if (Test-Path $AuthKeys) {
        # Remove only our key, so a key you added yourself survives.
        $kept = Get-Content $AuthKeys | Where-Object { $_ -notmatch "claude-code-finprint-deploy" }
        if ($kept) { Set-Content -Path $AuthKeys -Value $kept -Encoding ASCII }
        else { Remove-Item $AuthKeys -Force }
        Good "deploy key removed from administrators_authorized_keys"
    }
    Note "The OpenSSH Server feature itself is left installed (harmless while disabled)."
    Note "To remove it entirely: Remove-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0"
    exit 0
}

# --- 1. install the server ------------------------------------------------
Step "OpenSSH Server"
$cap = Get-WindowsCapability -Online -Name "OpenSSH.Server*" | Select-Object -First 1
if (-not $cap) { throw "OpenSSH.Server capability not offered by this Windows build." }
if ($cap.State -eq "Installed") {
    Good "already installed"
} else {
    Info "installing (needs internet; this is a Windows optional feature) ..."
    Add-WindowsCapability -Online -Name $cap.Name | Out-Null
    Good "installed"
}

# --- 2. run it, and keep running it --------------------------------------
Step "sshd service"
Set-Service -Name sshd -StartupType Automatic
Start-Service sshd
$svc = Get-Service sshd
Good "sshd is $($svc.Status), startup=Automatic"

# The default shell for an SSH session is cmd.exe. Everything in this repo is
# PowerShell, so switch it - otherwise every remote command needs a wrapper.
Step "Default shell"
$psPath = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
New-Item -Path "HKLM:\SOFTWARE\OpenSSH" -Force | Out-Null
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell -Value $psPath `
    -PropertyType String -Force | Out-Null
Good "SSH sessions will start in PowerShell"

# --- 3. authorise the key -------------------------------------------------
# The gotcha that costs people an afternoon: for a user in the Administrators
# group, Windows OpenSSH ignores ~/.ssh/authorized_keys entirely and reads
# ProgramData\ssh\administrators_authorized_keys instead - and it refuses the
# file unless its ACL grants only Administrators and SYSTEM.
Step "Authorising the deploy key"
if (-not (Test-Path (Split-Path $AuthKeys))) { New-Item -ItemType Directory -Path (Split-Path $AuthKeys) -Force | Out-Null }
$existing = @()
if (Test-Path $AuthKeys) { $existing = @(Get-Content $AuthKeys | Where-Object { $_.Trim() }) }
if ($existing -contains $PublicKey.Trim()) {
    Good "key already authorised"
} else {
    $existing += $PublicKey.Trim()
    Set-Content -Path $AuthKeys -Value $existing -Encoding ASCII
    Good "key added to administrators_authorized_keys"
}
icacls $AuthKeys /inheritance:r /grant "Administrators:F" /grant "SYSTEM:F" | Out-Null
Good "permissions locked to Administrators + SYSTEM (sshd rejects the file otherwise)"

# --- 4. firewall, LAN only -----------------------------------------------
Step "Firewall (port 22, local network only)"
if (Get-NetFirewallRule -DisplayName $FwRule -ErrorAction SilentlyContinue) {
    Remove-NetFirewallRule -DisplayName $FwRule
}
# Windows' own "OpenSSH-Server-In-TCP" rule is Any-scoped. Disable it and use a
# scoped rule, so enabling SSH here cannot become an internet-facing service by
# accident if someone later forwards port 22.
Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue |
    Set-NetFirewallRule -Enabled False -ErrorAction SilentlyContinue
$scope = "LocalSubnet"
if ($AnyLocalNetwork) { $scope = @("192.168.0.0/16", "10.0.0.0/8", "172.16.0.0/12") }
New-NetFirewallRule -DisplayName $FwRule -Direction Inbound -Action Allow `
    -Protocol TCP -LocalPort 22 -RemoteAddress $scope -Profile Any | Out-Null
Good "inbound TCP 22 allowed from: $($scope -join ', ')"
Note "Do NOT add port 22 to the router port-forward. Only 80 and 443 belong there."

# --- 5. tell the operator how to connect ---------------------------------
Step "Connection details"
$ips = Get-NetIPConfiguration | Where-Object { $_.IPv4Address -and $_.IPv4DefaultGateway } |
    ForEach-Object { $_.IPv4Address.IPAddress }
$me = "$env:USERNAME"
$host_ = "$env:COMPUTERNAME"

Write-Host ""
Write-Host "=========================================================" -ForegroundColor White
Write-Host " SSH is ready. Give these to the machine driving this one:" -ForegroundColor White
Write-Host "=========================================================" -ForegroundColor White
Write-Host ""
Write-Host "   hostname : $host_"
Write-Host "   user     : $me"
foreach ($ip in $ips) { Write-Host "   address  : $ip" }
Write-Host ""
Write-Host "   test from the other machine:" -ForegroundColor Cyan
foreach ($ip in $ips) { Write-Host "     ssh -i `$env:USERPROFILE\.ssh\finprint_host $me@$ip `"hostname`"" }
Write-Host ""
Note "If the two machines are on different networks, this address is not"
Note "reachable and SSH will time out. Put both on the same LAN, or join them"
Note "with a mesh VPN (Tailscale) and use that address instead."
Write-Host ""
