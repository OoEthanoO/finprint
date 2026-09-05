<#
    teardown-gcp.ps1 - remove every finprint resource from Google Cloud.

        .\scripts\selfhost\teardown-gcp.ps1            # dry run: list, delete nothing
        .\scripts\selfhost\teardown-gcp.ps1 -Confirm   # actually delete

    Run this only AFTER https://finprint.ethanyanxu.com is served by the laptop.
    Without -SkipDnsCheck the script refuses to run while DNS still points at
    Cloud Run, because deleting the service first would take the site down.

    Deliberately scoped, not a project delete: project 'study-autopilot' is
    shared with the study-autopilot app. Everything below was verified to belong
    to finprint alone - in particular the Artifact Registry repo, whose only
    package is 'finprint'.

    What goes:

      1. Cloud Run domain mapping  finprint.ethanyanxu.com
      2. Cloud Run service         finprint (us-central1)
      3. Artifact Registry repo    cloud-run-source-deploy   <- the actual bill
      4. GCS bucket                run-sources-study-autopilot-us-central1
      5. Service account           github-deploy@... + its project IAM bindings
      6. Workload Identity pool    github (+ its provider)

    What stays: the project, enabled APIs (free), and everything belonging to
    study-autopilot.

    Not deletable from here - do it by hand:
      gh variable delete GCP_WIF_PROVIDER --repo OoEthanoO/finprint
      gh variable delete GCP_DEPLOY_SA    --repo OoEthanoO/finprint
#>

[CmdletBinding()]
param(
    [string]$Project = "study-autopilot",
    [string]$Region = "us-central1",
    [string]$Service = "finprint",
    [string]$Domain = "finprint.ethanyanxu.com",
    [string]$Repo = "cloud-run-source-deploy",
    [string]$Bucket = "run-sources-study-autopilot-us-central1",
    [string]$ServiceAccount = "github-deploy",
    [string]$WifPool = "github",
    [string]$WifProvider = "github",
    # Perform the deletions. Without it the script only reports.
    [switch]$Confirm,
    # Skip the "is the domain already off Cloud Run?" guard.
    [switch]$SkipDnsCheck
)

$ErrorActionPreference = "Continue"

function Step($t) { Write-Host ""; Write-Host "==> $t" -ForegroundColor Cyan }
function Info($m) { Write-Host "    $m" }
function Good($m) { Write-Host "    $m" -ForegroundColor Green }
function Note($m) { Write-Host "    $m" -ForegroundColor Yellow }
function Bad($m) { Write-Host "    $m" -ForegroundColor Red }

# gcloud is often installed without being on PATH.
$sdkBin = "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin"
if ((Test-Path $sdkBin) -and -not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    $env:Path = "$sdkBin;$env:Path"
}
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) { throw "gcloud not found." }

$SaEmail = "$ServiceAccount@$Project.iam.gserviceaccount.com"

if (-not $Confirm) {
    Write-Host ""
    Write-Host "DRY RUN - nothing will be deleted. Re-run with -Confirm to execute." -ForegroundColor Yellow
}

# --- guard: is the site already moved? ------------------------------------
Step "Safety check: is $Domain still served by Cloud Run?"
if ($SkipDnsCheck) {
    Note "skipped (-SkipDnsCheck)"
} else {
    $stillGoogle = $false
    try {
        $rr = Resolve-DnsName -Name $Domain -Server 1.1.1.1 -ErrorAction Stop
        $cn = ($rr | Where-Object { $_.Type -eq 'CNAME' } | Select-Object -First 1).NameHost
        if ($cn -like "*googlehosted*") { $stillGoogle = $true; Bad "$Domain is still a CNAME -> $cn" }
        else {
            $a = ($rr | Where-Object { $_.Type -eq 'A' } | Select-Object -First 1).IPAddress
            Good "$Domain -> $a (no longer Cloud Run)"
        }
    } catch { Note "could not resolve $Domain : $($_.Exception.Message)" }

    if ($stillGoogle) {
        Bad ""
        Bad "Refusing to proceed: deleting the Cloud Run service now would take"
        Bad "$Domain offline. Finish the cutover first (setup.ps1, router"
        Bad "forward, DNS A record), confirm with verify.ps1, then re-run."
        Bad "Override with -SkipDnsCheck only if you accept the outage."
        exit 1
    }

    # Also confirm the new host is genuinely answering before removing the old one.
    try {
        $h = Invoke-RestMethod -Uri "https://$Domain/api/health" -TimeoutSec 45
        Good "https://$Domain is healthy (version $($h.version)) - safe to tear down"
    } catch {
        Note "https://$Domain did not answer: $($_.Exception.Message)"
        if (-not $Confirm) { Note "(dry run - continuing to list resources)" }
        else {
            Bad "Refusing to delete the fallback while the replacement is down."
            Bad "Fix the laptop deployment first, or pass -SkipDnsCheck."
            exit 1
        }
    }
}

function Do-Step($desc, $scriptblock) {
    Info $desc
    if (-not $Confirm) { Note "  (dry run - skipped)"; return }
    & $scriptblock
    if ($LASTEXITCODE -eq 0) { Good "  deleted" } else { Note "  gcloud returned $LASTEXITCODE (already gone?)" }
}

# --- 1. domain mapping ----------------------------------------------------
# Must go before the service: a mapping pointing at a deleted service is an
# orphan that still holds the hostname. The GA gcloud surface for this is
# region-less and inconsistent across versions, so call the API directly.
Step "1/6  Cloud Run domain mapping: $Domain"
$token = (& gcloud auth print-access-token).Trim()
$mapUrl = "https://$Region-run.googleapis.com/apis/domains.cloudrun.com/v1/namespaces/$Project/domainmappings/$Domain"
try {
    Invoke-RestMethod -Uri $mapUrl -Headers @{ Authorization = "Bearer $token" } -TimeoutSec 30 | Out-Null
    Info "mapping exists"
    if ($Confirm) {
        Invoke-RestMethod -Uri $mapUrl -Method DELETE -Headers @{ Authorization = "Bearer $token" } -TimeoutSec 60 | Out-Null
        Good "  deleted"
    } else { Note "  (dry run - skipped)" }
} catch {
    Note "no mapping found (already deleted?)"
}

# --- 2. Cloud Run service -------------------------------------------------
Step "2/6  Cloud Run service: $Service"
$svc = & gcloud run services describe $Service --region $Region --project $Project --format="value(status.url)" 2>$null
if ($svc) {
    Info "service exists at $svc"
    Do-Step "deleting service $Service" { & gcloud run services delete $Service --region $Region --project $Project --quiet 2>&1 | Out-Null }
} else { Note "service not found (already deleted?)" }

# --- 3. Artifact Registry -------------------------------------------------
# The bill. 22 x ~1.5 GB images against a 0.5 GB free tier; every push to main
# added one and nothing ever removed them.
Step "3/6  Artifact Registry repo: $Repo"
$pkgs = & gcloud artifacts packages list --repository=$Repo --location=$Region --project $Project --format="value(name)" 2>$null
if ($LASTEXITCODE -eq 0) {
    $pkgs = @($pkgs | Where-Object { $_ })
    Info "packages in repo: $($pkgs -join ', ')"
    $foreign = $pkgs | Where-Object { $_ -notlike "*finprint*" }
    if ($foreign) {
        Bad "repo also contains non-finprint packages: $($foreign -join ', ')"
        Bad "NOT deleting the repo. Delete only the finprint package instead:"
        Bad "  gcloud artifacts packages delete finprint --repository=$Repo --location=$Region --project $Project"
    } else {
        Do-Step "deleting repo $Repo (only finprint inside)" { & gcloud artifacts repositories delete $Repo --location=$Region --project $Project --quiet 2>&1 | Out-Null }
    }
} else { Note "repo not found (already deleted?)" }

# --- 4. build source staging ---------------------------------------------
Step "4/6  GCS staging bucket: $Bucket"
$b = & gcloud storage buckets describe "gs://$Bucket" --project $Project --format="value(name)" 2>$null
if ($b) {
    Do-Step "deleting bucket gs://$Bucket and its contents" { & gcloud storage rm --recursive "gs://$Bucket" --project $Project --quiet 2>&1 | Out-Null }
} else { Note "bucket not found (already deleted?)" }

# --- 5. deploy service account -------------------------------------------
# Removing the project-level role bindings first: deleting the account alone
# leaves the bindings behind as dangling 'deleted:' principals in the policy.
Step "5/6  Deploy service account: $SaEmail"
$sa = & gcloud iam service-accounts describe $SaEmail --project $Project --format="value(email)" 2>$null
if ($sa) {
    $roles = @("roles/run.admin", "roles/cloudbuild.builds.editor", "roles/artifactregistry.admin",
               "roles/storage.admin", "roles/iam.serviceAccountUser", "roles/logging.viewer")
    foreach ($r in $roles) {
        Do-Step "removing $r" { & gcloud projects remove-iam-policy-binding $Project --member="serviceAccount:$SaEmail" --role=$r --condition=None --quiet 2>&1 | Out-Null }
    }
    Do-Step "deleting service account $SaEmail" { & gcloud iam service-accounts delete $SaEmail --project $Project --quiet 2>&1 | Out-Null }
} else { Note "service account not found (already deleted?)" }

# --- 6. workload identity federation -------------------------------------
# This is what let GitHub Actions deploy without a stored key. With the repo no
# longer deploying, leaving it would be a standing trust relationship with
# nothing on the other end.
Step "6/6  Workload Identity pool/provider: $WifPool / $WifProvider"
$prov = & gcloud iam workload-identity-pools providers describe $WifProvider --location=global --workload-identity-pool=$WifPool --project $Project --format="value(attributeCondition)" 2>$null
if ($prov) {
    Info "provider condition: $prov"
    if ($prov -notlike "*finprint*") {
        Bad "this provider is not pinned to the finprint repo - it may serve something else."
        Bad "NOT deleting. Review it by hand."
    } else {
        Do-Step "deleting provider $WifProvider" { & gcloud iam workload-identity-pools providers delete $WifProvider --location=global --workload-identity-pool=$WifPool --project $Project --quiet 2>&1 | Out-Null }
        Do-Step "deleting pool $WifPool" { & gcloud iam workload-identity-pools delete $WifPool --location=global --project $Project --quiet 2>&1 | Out-Null }
        Note "pools and providers are soft-deleted; Google purges them after 30 days."
    }
} else { Note "provider not found (already deleted?)" }

# --- report ---------------------------------------------------------------
Write-Host ""
if (-not $Confirm) {
    Write-Host "DRY RUN complete - nothing was deleted. Re-run with -Confirm." -ForegroundColor Yellow
    exit 0
}
Step "Remaining finprint resources in $Project (should be empty)"
& gcloud run services list --project $Project --format="table(metadata.name,region)" 2>&1 | Out-String | Write-Host
& gcloud artifacts repositories list --project $Project --format="table(name,sizeBytes)" 2>&1 | Out-String | Write-Host
& gcloud storage buckets list --project $Project --format="table(name)" 2>&1 | Out-String | Write-Host

Write-Host ""
Write-Host "Teardown complete." -ForegroundColor Green
Write-Host "Still to do by hand (needs the GitHub CLI):" -ForegroundColor Yellow
Write-Host "  gh variable delete GCP_WIF_PROVIDER --repo OoEthanoO/finprint"
Write-Host "  gh variable delete GCP_DEPLOY_SA    --repo OoEthanoO/finprint"
Write-Host "Then confirm the bill stops: https://console.cloud.google.com/billing"
