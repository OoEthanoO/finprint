# Shared by setup, update and restart. Dot-source this file; it changes nothing
# until Stop-FinprintService is called.
function Stop-FinprintService {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][ValidateSet('finprint-app','finprint-caddy')][string]$TaskName,
        [Parameter(Mandatory)][string]$Root,
        [int]$Port = 8000
    )

    $Root = (Resolve-Path -LiteralPath $Root -ErrorAction Stop).Path
    $kind = if ($TaskName -eq 'finprint-app') { 'app' } else { 'caddy' }
    $launcher = Join-Path $Root "scripts\selfhost\run-$kind.cmd"
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($task -and @($task.Actions | Where-Object {
        $_.Arguments -and $_.Arguments.IndexOf($launcher, [StringComparison]::OrdinalIgnoreCase) -ge 0
    }).Count -ne 1) {
        throw "Task $TaskName does not use $launcher; refusing to stop another deployment."
    }

    # Capture ownership BEFORE stopping the task: Windows venv Python starts a
    # child interpreter outside .venv. Follow that process tree instead of
    # killing whichever unrelated application happens to own a port.
    $processes = @(Get-CimInstance Win32_Process -ErrorAction Stop)
    $owned = @{}
    $venv = Join-Path $Root '.venv\Scripts\python.exe'
    $config = Join-Path $Root 'scripts\selfhost\Caddyfile'
    foreach ($proc in $processes) {
        $line = [string]$proc.CommandLine
        $isLauncher = $proc.Name -ieq 'cmd.exe' -and $line.IndexOf($launcher, [StringComparison]::OrdinalIgnoreCase) -ge 0
        $isApp = $kind -eq 'app' -and $proc.ExecutablePath -ieq $venv -and $line -match '\buvicorn\s+app\.main:app\b'
        $isCaddy = $kind -eq 'caddy' -and $proc.Name -ieq 'caddy.exe' -and $line.IndexOf($config, [StringComparison]::OrdinalIgnoreCase) -ge 0
        if ($isLauncher -or $isApp -or $isCaddy) { $owned[[int]$proc.ProcessId] = $proc }
    }
    do {
        $added = $false
        foreach ($proc in $processes) {
            $parent = $owned[[int]$proc.ParentProcessId]
            if ($parent -and -not $owned.ContainsKey([int]$proc.ProcessId) -and $proc.CreationDate -ge $parent.CreationDate) {
                $owned[[int]$proc.ProcessId] = $proc
                $added = $true
            }
        }
    } while ($added)

    if ($task) { Stop-ScheduledTask -TaskName $TaskName -ErrorAction Stop }
    foreach ($proc in @($owned.Values | Sort-Object CreationDate -Descending)) {
        $current = Get-CimInstance Win32_Process -Filter "ProcessId=$($proc.ProcessId)" -ErrorAction Stop
        # PID reuse must never make this kill a new, unrelated process.
        if ($current -and $current.CreationDate -eq $proc.CreationDate -and $current.ExecutablePath -eq $proc.ExecutablePath) {
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
        }
    }
    $ports = if ($kind -eq 'app') { @($Port) } else { @(80,443,2019) }
    $deadline = (Get-Date).AddSeconds(15)
    do {
        $listeners = @(Get-NetTCPConnection -State Listen -ErrorAction Stop | Where-Object LocalPort -in $ports)
        if (-not $listeners) { return }
        Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $deadline)
    $details = ($listeners | ForEach-Object { "$($_.LocalPort) (PID $($_.OwningProcess))" }) -join ', '
    throw "Ports still occupied after stopping $TaskName : $details. No unrelated processes were terminated."
}
