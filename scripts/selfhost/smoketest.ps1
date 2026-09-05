<#
    smoketest.ps1 - prove the deployment can actually predict, not merely answer.

        powershell -ExecutionPolicy Bypass -File .\scripts\selfhost\smoketest.ps1

    /api/health only reports that the process is up and a checkpoint exists. It
    says nothing about whether an upload decodes, so the deployment can look
    perfectly healthy while every real request fails.

    The compressed formats are the point. soundfile reads wav/flac/ogg in-process,
    so those pass even on a badly broken host; mp3/m4a/webm go through librosa's
    audioread path, which shells out to **ffmpeg**. The services run as SYSTEM,
    which does not inherit the user PATH that winget put ffmpeg on - so if the
    launcher ever loses its baked-in PATH, wav keeps working and everything else
    stops. That failure is invisible to every other check in this repo.

    Exits non-zero if any format fails, so it can gate a deploy.
#>

[CmdletBinding()]
param(
    # Point at https://finprint.ethanyanxu.com to exercise the full public path
    # (Caddy included) once DNS has been cut over.
    [string]$BaseUrl = "http://127.0.0.1:8000",
    # Skip certificate validation, for testing an internal name or a host whose
    # certificate does not match the address being dialled.
    [switch]$Insecure,
    [switch]$KeepFiles
)

$ErrorActionPreference = "Continue"

function Section($t) { Write-Host ""; Write-Host "== $t ==" -ForegroundColor Cyan }
function Pass($m) { Write-Host "  [ok]   $m" -ForegroundColor Green }
function Warn($m) { Write-Host "  [warn] $m" -ForegroundColor Yellow }
function Fail($m) { Write-Host "  [FAIL] $m" -ForegroundColor Red; $script:Failed++ }

$script:Failed = 0
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) { throw "No .venv at $Root - run setup.ps1 first." }

$work = Join-Path ([System.IO.Path]::GetTempPath()) ("finprint-smoke-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Path $work -Force | Out-Null

try {
    # --- 1. a clip with something in it to find ---------------------------
    # A frequency-modulated tone, the same shape finprint.warmup uses: voiced
    # enough to drive the pitch tracker, so the DSP layer returns real numbers
    # rather than the "no call to identify" path.
    Section "Generating test audio"
    $gen = Join-Path $work "gen.py"
    @'
import numpy as np, soundfile as sf, sys
sr = 32000
t = np.arange(int(sr * 2.0)) / sr
f0 = 900.0 + 250.0 * np.sin(2 * np.pi * 2.0 * t)
phase = 2 * np.pi * np.cumsum(f0) / sr
sf.write(sys.argv[1], (0.5 * np.sin(phase)).astype("float32"), sr)
print("wrote", sys.argv[1])
'@ | Set-Content -Path $gen -Encoding ASCII

    $wav = Join-Path $work "clip.wav"
    & $VenvPy $gen $wav | Out-Null
    if (-not (Test-Path $wav)) { throw "could not generate the test wav" }
    Pass "wav generated ($([int]((Get-Item $wav).Length / 1KB)) KB)"

    # --- 2. transcode to the formats that need ffmpeg ---------------------
    $cases = @(@{ name = "wav"; path = $wav; needsFfmpeg = $false })
    $ffmpeg = (Get-Command ffmpeg -ErrorAction SilentlyContinue).Source
    if (-not $ffmpeg) {
        Warn "ffmpeg not on PATH in THIS shell - skipping compressed formats."
        Warn "That is not the same PATH the service uses; this test then proves less."
    } else {
        foreach ($f in @("mp3", "m4a", "webm")) {
            $out = Join-Path $work "clip.$f"
            & $ffmpeg -y -loglevel error -i $wav $out 2>&1 | Out-Null
            if (Test-Path $out) {
                $cases += @{ name = $f; path = $out; needsFfmpeg = $true }
                Pass "$f transcoded"
            } else {
                Warn "could not transcode to $f - skipping that case"
            }
        }
    }

    # --- 3. push each through the real endpoint ---------------------------
    Section "POST $BaseUrl/api/predict"
    foreach ($c in $cases) {
        $bodyFile = Join-Path $work "resp-$($c.name).json"
        $curlArgs = @("-s", "-o", $bodyFile, "-w", "%{http_code}", "--max-time", "180",
                      "-X", "POST", "-F", "file=@$($c.path)", "$BaseUrl/api/predict")
        if ($Insecure) { $curlArgs = @("-k") + $curlArgs }
        $code = (& curl.exe @curlArgs)

        if ($code -ne "200") {
            $detail = ""
            if (Test-Path $bodyFile) { $detail = (Get-Content $bodyFile -Raw).Trim() }
            if ($detail.Length -gt 300) { $detail = $detail.Substring(0, 300) + "..." }
            Fail "$($c.name): HTTP $code $detail"
            if ($c.needsFfmpeg) {
                Fail "  a compressed format failing while wav passes means the service cannot reach ffmpeg"
            }
            continue
        }

        try { $r = Get-Content $bodyFile -Raw | ConvertFrom-Json }
        catch { Fail "$($c.name): response was not JSON"; continue }

        if (-not $r.model_available) { Fail "$($c.name): model_available=false - the checkpoint did not load" ; continue }
        if (-not $r.species -or $r.species.Count -lt 1) { Fail "$($c.name): no species in the response"; continue }
        if (-not $r.features) { Fail "$($c.name): no acoustic features in the response"; continue }

        $top = $r.species[0]
        $ct = $r.call_type.label
        $dom = [math]::Round([double]$r.features.dominant_freq_hz)
        Pass ("{0,-4} -> {1} ({2}) group={3} call={4} dominant={5} Hz" -f `
              $c.name, $top.species, $top.confidence, $r.group.label, $ct, $dom)

        # The generated tone sweeps 650-1150 Hz; a dominant frequency far outside
        # that means the audio was decoded wrongly (wrong sample rate or garbage),
        # which a 200 response alone would not reveal.
        if ($dom -lt 400 -or $dom -gt 1600) {
            Fail "  dominant frequency $dom Hz is outside the generated 650-1150 Hz sweep - decoded wrongly?"
        }
    }
} finally {
    if ($KeepFiles) { Write-Host ""; Write-Host "test files kept in $work" }
    else { Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue }
}

Write-Host ""
if ($script:Failed -gt 0) {
    Write-Host "SMOKE TEST FAILED: $($script:Failed) problem(s)." -ForegroundColor Red
    exit 1
}
Write-Host "SMOKE TEST OK - the deployment decodes every format and predicts." -ForegroundColor Green
exit 0
