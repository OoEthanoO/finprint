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
