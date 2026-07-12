# Deploying finprint

finprint ships as a single container (`Dockerfile`): a FastAPI app serving the
UI and `/api/predict`. Every host below builds the image **remotely**, so you do
not need Docker installed locally.

Pick one:

## Hugging Face Spaces — free, recommended

Free tier is 2 vCPU / 16 GB RAM with Docker support and no credit card — plenty
for torch. The `README.md` frontmatter (`sdk: docker`, `app_port: 8000`) already
marks this repo as a Docker Space; the `Dockerfile` is used as-is.

```bash
pip install -U "huggingface_hub[cli]"
hf auth login                                   # paste a token from hf.co/settings/tokens
hf repo create finprint --repo-type space --space_sdk docker
git remote add space https://huggingface.co/spaces/<your-username>/finprint
git push space main                             # builds & deploys automatically
```

Serves at `https://<your-username>-finprint.hf.space`. Free Spaces sleep after
~48 h idle and wake on the next visit.

## Fly.io  (`fly.toml` included) — not free

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

## Render  (`render.yaml` Blueprint included)

1. Push this repo to GitHub.
2. Render Dashboard → **New → Blueprint** → select the repo.
3. Render reads `render.yaml`, builds the Dockerfile, and deploys on the
   **Standard** plan (2 GB — Free/Starter's 512 MB OOMs on torch).

## Local Docker (to test the exact image)

```bash
docker build -t finprint .
docker run -p 8000:8000 finprint
# → http://localhost:8000
```

---

## Notes

- **Species prediction is disabled until you add the model.** The trained
  checkpoint `models/species_cnn.pt` is gitignored and not in the repo, so
  `/api/predict` returns `model_available: false` — call type, acoustic
  features, and the spectrogram still work. To enable species classification,
  produce the checkpoint (`python -m scripts.prepare_data && python -m
  finprint.train`) and ship it in the image — either commit it, or add a build
  step / volume that places it at `models/species_cnn.pt`. It's loaded lazily on
  the first request.
- **Memory.** torch + librosa need ~1–2 GB resident; both configs above provision
  2 GB. Dropping below that risks OOM on model/library load.
- **CPU-only.** The image installs CPU torch wheels; inference runs on CPU.
- **Image size** is ~1.5 GB (torch/torchaudio dominate). First remote build
  takes a few minutes; subsequent builds reuse the cached dependency layer.
