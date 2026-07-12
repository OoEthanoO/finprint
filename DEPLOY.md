# Deploying finprint

finprint ships as a single container (`Dockerfile`): a FastAPI app serving the
UI and `/api/predict`. Both hosts below build the image **remotely**, so you do
not need Docker installed locally.

Pick one:

## Fly.io  (`fly.toml` included)

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
