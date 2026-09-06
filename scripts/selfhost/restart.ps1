<# Restart one finprint task and confirm that its local endpoint returns. #>
[CmdletBinding()]
param(
    [ValidateSet('app','caddy')][string]$Service = 'app',
    [int]$Port = 8000
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'service-control.ps1')
$rootPath = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$taskName = "finprint-$Service"
Stop-FinprintService -TaskName $taskName -Root $rootPath -Port $Port
Start-ScheduledTask -TaskName $taskName
$deadline = (Get-Date).AddSeconds(240)
do {
    Start-Sleep -Seconds 2
    if ((Get-ScheduledTask -TaskName $taskName).State -ne 'Running') { continue }
    if ($Service -eq 'caddy') {
        if (Get-NetTCPConnection -State Listen -LocalPort 443 -ErrorAction SilentlyContinue) {
            Write-Output 'Caddy task is running and listening on 443. Certificate issuance still needs DNS and public access.'
            exit 0
        }
    } else {
        try {
            $health = Invoke-RestMethod "http://127.0.0.1:$Port/api/health" -TimeoutSec 5
            if ($health.status -eq 'ok' -and $health.model_available) {
                Write-Output "App restarted; serving $($health.version)"
                exit 0
            }
        } catch { }
    }
} while ((Get-Date) -lt $deadline)
throw "$taskName did not become ready within 240 seconds."
