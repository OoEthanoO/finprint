# Deploying finprint

finprint ships as a single container (`Dockerfile`): a FastAPI app serving the
UI and `/api/predict`. Every host below builds the image **remotely**, so you do
not need Docker installed locally.

## Google Cloud Run — the live deployment

Deployed as service **`finprint`** in project **`study-autopilot`**, region
**`us-central1`**, serving at:

- https://finprint.ethanyanxu.com (custom domain mapping)
- https://finprint-999841164638.us-central1.run.app (direct)

Free-tier friendly: 2 GB RAM, scale-to-zero, and demo traffic stays inside the
always-free quota. Cloud Build compiles the `Dockerfile` remotely (no local
Docker; `.gcloudignore` trims the upload).

```bash
# redeploy after changes
gcloud run deploy finprint --source . --memory 2Gi --cpu 1 \
  --region us-central1 --project study-autopilot --allow-unauthenticated
```

### Continuous deployment

Every push to `main` redeploys automatically, from the `deploy` job in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml). It `needs: test`, so a
failing suite blocks the deploy rather than shipping a red build; pull requests
never reach it. After deploying it polls `/api/health` and fails the run if the
new revision never answers.

Auth is **Workload Identity Federation** — GitHub mints a short-lived OIDC token
and impersonates a deploy service account. There is no service-account key in the
repo to leak or rotate, and the provider is pinned to this one repository, so no
other repo can impersonate it. The two GitHub *variables* it reads
(`GCP_WIF_PROVIDER`, `GCP_DEPLOY_SA`) are non-secret identifiers, not credentials.

<details>
<summary>One-time setup (already done — kept for rebuilding the project)</summary>

```bash
PROJECT=study-autopilot
REPO=OoEthanoO/finprint
SA=github-deploy
PROJECT_NUM=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
SA_EMAIL="$SA@$PROJECT.iam.gserviceaccount.com"

gcloud services enable iamcredentials.googleapis.com run.googleapis.com \
  cloudbuild.googleapis.com artifactregistry.googleapis.com --project "$PROJECT"

gcloud iam service-accounts create "$SA" \
  --display-name="GitHub Actions deployer" --project "$PROJECT"

# What `gcloud run deploy --source` actually touches: it submits a Cloud Build,
# pushes the image to Artifact Registry, stages the source in GCS, and acts as
# the runtime service account.
for role in roles/run.admin roles/cloudbuild.builds.editor \
            roles/artifactregistry.admin roles/storage.admin \
            roles/iam.serviceAccountUser roles/logging.viewer; do
  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member="serviceAccount:$SA_EMAIL" --role="$role" --condition=None
done

gcloud iam workload-identity-pools create github \
  --location=global --project "$PROJECT"

# The attribute-condition is the security boundary: without it, any GitHub repo
# on the internet could mint a token for this provider.
gcloud iam workload-identity-pools providers create-oidc github \
  --location=global --workload-identity-pool=github --project "$PROJECT" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='$REPO'"

gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" --project "$PROJECT" \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUM/locations/global/workloadIdentityPools/github/attribute.repository/$REPO"

gh variable set GCP_DEPLOY_SA --repo "$REPO" --body "$SA_EMAIL"
gh variable set GCP_WIF_PROVIDER --repo "$REPO" \
  --body "projects/$PROJECT_NUM/locations/global/workloadIdentityPools/github/providers/github"
```

</details>

Every push to `main` costs a full remote image build (a few minutes; the torch
layer is cached). If doc-only commits start feeling wasteful, add a
`paths-ignore` for `**.md` and `reports/**` to the workflow's `push` trigger —
nothing under those paths is served.

The custom domain was wired once and needs no maintenance:

```bash
gcloud beta run domain-mappings create --service finprint \
  --domain finprint.ethanyanxu.com --region us-central1 --project study-autopilot
vercel dns add ethanyanxu.com finprint CNAME ghs.googlehosted.com
```

## Alternatives (config kept in-repo, not currently used)

> Hugging Face Spaces was ruled out: Docker Spaces moved behind a paid plan
> (mid-2026) and Spaces never supported custom domains. The `README.md`
> frontmatter for it is harmless and kept in case that changes.

### Fly.io  (`fly.toml` included) — not free

```bash
# one-time
brew install flyctl        # or: curl -L https://fly.io/install.sh | sh
fly auth login
fly launch --copy-config --no-deploy   # accept a unique app name when prompted

# every release
fly deploy
```

Serves at `https://<app-name>.fly.dev`. The config runs a 2 GB shared-cpu VM
(torch OOMs on the 256 MB default) and health-checks `/api/health`.

### Render  (`render.yaml` Blueprint included) — 2 GB tier is paid

1. Push this repo to GitHub.
2. Render Dashboard → **New → Blueprint** → select the repo.
3. Render reads `render.yaml`, builds the Dockerfile, and deploys on the
   **Standard** plan (2 GB — Free/Starter's 512 MB OOMs on torch).

### Local Docker (to test the exact image)

```bash
docker build -t finprint .
docker run -p 8000:8000 finprint
# → http://localhost:8000
```

---

## Notes

- **The trained model ships in the repo.** `models/species_cnn.pt` (~2.3 MB) is
  committed — `.gitignore` tracks that one checkpoint while ignoring any other
  `*.pt` — so `COPY . .` bakes it into the image and species prediction works on
  a fresh clone with no retraining. It's loaded lazily on the first request. If
  the checkpoint is ever missing, `/api/predict` still returns call type,
  acoustic features, and the spectrogram, with `model_available: false`. To
  refresh it, retrain (`python -m scripts.prepare_data && python -m
  finprint.train`) and commit the new file.
- **Cold starts (~50 s, and why).** The service scales to zero. A fresh *process*
  spends ~50 s before it can answer, which `predict`'s stage timings attribute as:

  ```
  cold:  decode 33.0s  features 15.4s  species 1.6s  spectrogram 1.8s
  warm:  decode  0.001s features  2.0s species 0.1s  spectrogram 0.3s
  ```

  `decode` dropping to a millisecond when warm shows it is not decoding at all:
  it is librosa materialising its lazily-imported submodules, and numba
  JIT-compiling the pitch kernels, on first use. Two mitigations are in place:

  * `compileall` in the Dockerfile bakes `.pyc` for the dependencies (and
    `PYTHONDONTWRITEBYTECODE` is deliberately *not* set). This more than halved
    warm requests, 5.4 s → 2.4 s.
  * `finprint.warmup` runs from the app's lifespan hook. uvicorn completes it
    before binding the port, so an instance never reports ready until it can
    actually predict — instances started ahead of traffic (deploy rollouts,
    autoscaling) serve their first request warm.

  What did *not* work, so it is not retried: baking numba's on-disk cache yields
  only 3 entries (most librosa kernels are not cacheable), and the work is not
  CPU-bound — `--cpu 2` measured 56.5 s against 52.1 s at `--cpu 1`. The
  remaining cost is serial, per-process import/JIT time. Removing it means
  either paying for a warm instance (`--min-instances 1`, ~$10–12/month) or
  dropping librosa for direct `soundfile` + numpy/scipy DSP.
- **Memory.** torch + librosa need ~1–2 GB resident; both configs above provision
  2 GB. Dropping below that risks OOM on model/library load.
- **CPU-only.** The image installs CPU torch wheels; inference runs on CPU.
- **Image size** is ~1.5 GB (torch/torchaudio dominate). First remote build
  takes a few minutes; subsequent builds reuse the cached dependency layer.
