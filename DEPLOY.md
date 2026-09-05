# Deploying finprint

finprint runs on a **self-hosted Windows laptop**, behind Caddy, serving
<https://finprint.ethanyanxu.com>. It used to run on Google Cloud Run; that
deployment was removed (see [Migration off Google Cloud](#migration-off-google-cloud)).

```
internet
   |
   v  ports 80 + 443 forwarded
router (192.168.x.1)
   |
   v
Windows laptop
   caddy.exe        :80 / :443   TLS, automatic Let's Encrypt certificate
      |
      v  reverse proxy
   uvicorn app:main  127.0.0.1:8000   (.venv, Python 3.11)
```

uvicorn binds `127.0.0.1`, not `0.0.0.0`: Caddy is the only thing that can reach
it, so the app is never exposed in plaintext, not even to the LAN.

Both processes run as **scheduled tasks under SYSTEM with an at-startup
trigger**. That is what replaces the one property Cloud Run supplied for free —
the service comes back after a reboot, with nobody logged in.

Everything below lives in [`scripts/selfhost/`](scripts/selfhost/).

---

## What this design requires

Four things have to be true. [`preflight.ps1`](scripts/selfhost/preflight.ps1)
checks all four before anything is installed, because each one otherwise fails
later, in a different way, and looks like the others:

| Requirement | Why | If it is false |
|---|---|---|
| A real public IPv4 | A port forward has to have somewhere to land | **Fatal.** On CGNAT (`100.64.0.0/10`) no router setting can help — use a tunnel (Cloudflare Tunnel) instead |
| Router can forward 80 + 443 | Let's Encrypt validates by connecting back | Some ISPs block 80; Caddy can still use TLS-ALPN on 443 |
| The laptop stays awake and online | It *is* the server now | `setup.ps1` disables sleep on AC; set the lid action to "Do nothing" by hand |
| Python 3.11 | torch/torchaudio CPU wheels | `setup.ps1` installs it |

## First-time setup

Run all of this **on the hosting laptop**, in an **Administrator PowerShell**,
from the repo root.

```powershell
git clone https://github.com/OoEthanoO/finprint.git
cd finprint

# 1. Check the connection can host at all. Changes nothing.
powershell -ExecutionPolicy Bypass -File .\scripts\selfhost\preflight.ps1

# 2. Install everything: Python 3.11, ffmpeg, Caddy, the venv, the services.
#    Slow once (torch is ~200 MB). Idempotent — safe to re-run.
powershell -ExecutionPolicy Bypass -File .\scripts\selfhost\setup.ps1 -AcmeEmail you@example.com
```

### Why `-ExecutionPolicy Bypass`

A default Windows install refuses to run unsigned `.ps1` files at all:

```
... cannot be loaded because running scripts is disabled on this system.
```

Launching a child `powershell` with `-ExecutionPolicy Bypass` relaxes that for
one invocation and changes nothing permanent — preferable to
`Set-ExecutionPolicy`, which is a lasting change to the machine and is not
needed here. Elevation is inherited, so an Administrator prompt stays
Administrator. The scheduled tasks are unaffected either way: they already
invoke PowerShell with the flag, and `run-app.cmd` / `run-caddy.cmd` are batch
files, which the policy does not govern.

The same prefix applies to every script below (`verify.ps1`, `update.ps1`,
`enable-ssh.ps1`, `teardown-gcp.ps1`); it is omitted from later examples for
readability.

`setup.ps1` prints the two steps it cannot do for you:

**3. Forward the ports.** In the router admin page (usually `http://<gateway>`),
forward TCP **80** and TCP **443** to the laptop's LAN address. Also give that
address a **DHCP reservation** — otherwise the lease eventually changes and the
forward silently points at nothing.

**4. Point the DNS record at your connection.** In the Vercel dashboard
(`ethanyanxu.com` → DNS): delete the `CNAME finprint → ghs.googlehosted.com`
left over from Cloud Run, and add `A finprint → <your public IP>`, TTL 60.

Or do step 4 from the command line, which is also how the record stays correct
afterwards:

```powershell
# -Force is required the first time: it authorises replacing the Cloud Run CNAME
.\scripts\selfhost\update-dns.ps1 -Token <vercel-api-token> -Force
```

**5. Verify.**

```powershell
.\scripts\selfhost\verify.ps1
```

This walks the whole chain — services running, app answering locally, DNS
resolving here, certificate valid, HTTPS answering, and the served commit
matching `git HEAD`.

The certificate check is the one that matters most: Let's Encrypt validates by
connecting back **from the public internet**, so a publicly-trusted certificate
is direct proof that the port forward is open. This machine has no other way to
learn that about itself — a self-signed Caddy internal certificate means the
forward is closed.

## Dynamic IP

Cloud Run never needed this; a home line does. When the ISP hands out a new IP,
the A record goes stale and the site simply stops resolving to you.

```powershell
# check every 5 minutes, update Vercel when the IP moves
.\scripts\selfhost\update-dns.ps1 -Token <vercel-api-token> -Install
```

The token is stored at `.caches\vercel-token.txt` with an ACL admitting only
SYSTEM and Administrators, and `.gitignore` keeps it out of the repo. Note that
**Vercel API tokens are account-wide** — there is no per-domain scope — so it is
a real secret on a machine exposed to the internet. If that trade is not worth
it, the alternative is to point `finprint` at a dynamic-DNS hostname with a
`CNAME` (DuckDNS and similar), which keeps the changing part outside Vercel and
turns the Vercel record into a constant.

## Deploying a change

This replaces the old "push to `main` → GitHub Actions → Cloud Run" pipeline.
A GitHub runner has no route to a laptop behind a home router, so deploys happen
on the host:

```powershell
.\scripts\selfhost\update.ps1
```

It pulls, installs dependencies, restarts the service, and — the part worth
keeping from the CI job — **refuses to report success until `/api/health`
reports the new commit**. "The deploy said OK but the old build is still live"
was the failure the old check existed to catch, and it is still possible here.

Rollback is git:

```powershell
git checkout <previous-sha>
.\scripts\selfhost\update.ps1 -SkipPull
```

CI still runs on every push: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
runs the test suite. It just no longer deploys.

### Which commit is live?

```powershell
(Invoke-RestMethod https://finprint.ethanyanxu.com/api/health).version
git rev-parse HEAD
```

The page prints the short form in its footer (`· build 3bfd2e7`). This got
*more* reliable with the move: in the container, `.gcloudignore` stripped `.git`,
so the SHA had to be injected at deploy time and a deploy that forgot the flag
reported `unknown`. On the laptop the repository is right there, and
`run-app.cmd` reads `git rev-parse HEAD` at every start.

## Operations

| Task | Command |
|---|---|
| Restart the app | `Restart-ScheduledTask -TaskName finprint-app` |
| Restart Caddy | `Restart-ScheduledTask -TaskName finprint-caddy` |
| Service state | `Get-ScheduledTask finprint-app, finprint-caddy` |
| Access log | `Get-Content logs\access.log -Tail 50 -Wait` |
| Debug the app in a console | run `scripts\selfhost\run-app.cmd` (stop the task first) |
| Certificate store | `.caches\caddy\caddy\certificates\` |

Certificates renew automatically at ~30 days remaining, provided ports 80/443
are still reachable. `verify.ps1` warns when expiry is close, which is the
signal that renewal has been failing quietly.

### Administering the host remotely

The host is a laptop, and sitting in front of it to run `update.ps1` gets old.
[`enable-ssh.ps1`](scripts/selfhost/enable-ssh.ps1) turns on the OpenSSH server
that Windows ships but disables, once, from that machine:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\selfhost\enable-ssh.ps1
```

Two things it does that are easy to get wrong by hand:

* **Port 22 is scoped to the local subnet**, and Windows' own Any-scoped
  `OpenSSH-Server-In-TCP` rule is disabled. SSH must not become internet-facing
  — only 80 and 443 belong in the router forward.
* **The key goes in `%ProgramData%\ssh\administrators_authorized_keys`**, with
  its ACL reduced to Administrators and SYSTEM. For an account in the
  Administrators group, Windows OpenSSH ignores `~/.ssh/authorized_keys`
  entirely, and rejects the file outright if its permissions are looser than
  that. Neither failure produces a useful error.

`enable-ssh.ps1 -Uninstall` reverses all of it, removing only the deploy key so
a key you added yourself survives.

## What changed, honestly

**Better than Cloud Run:**

* **No cold start.** The old service scaled to zero and a fresh process spent
  ~50 s before it could answer (librosa materialising lazily-imported
  submodules, numba JIT-compiling the pitch kernels). The process here just
  stays up. Requests are warm-path — roughly 2 s — from the first visitor after
  a restart onwards, and `setup.ps1` runs the warm-up before the port ever opens.
* **No bill**, and no build minutes.
* **Honest build identity**, as above.

**Worse than Cloud Run:**

* **Availability is now yours.** Power cuts, ISP outages, Windows Update
  reboots, and a closed lid all take the site down. The at-startup tasks
  handle reboots; nothing handles the rest.
* **One machine, one uplink.** Residential upload bandwidth bounds how many
  people can fetch a spectrogram at once, and there is no autoscaling.
* **Your home IP is public.** `finprint.ethanyanxu.com` now resolves to your
  house. That is inherent to port forwarding, and is the main reason a tunnel
  (Cloudflare Tunnel) is the usual recommendation for this shape of hosting —
  it also hides the origin.
* **You are the security boundary.** Caddy only exposes the app, and the app
  caps uploads at 25 MB (`config.MAX_UPLOAD_BYTES`, mirrored in the Caddyfile),
  but this is a public endpoint on a personal machine. Keep Windows updated,
  and do not forward anything beyond 80 and 443.

## Migration off Google Cloud

finprint ran as Cloud Run service `finprint` in project `study-autopilot`
(`us-central1`). The service itself was inside the always-free tier; **the cost
was Artifact Registry**. `gcloud run deploy --source` pushes a new ~1.5 GB image
on every deploy and nothing ever removes the old ones — 22 images had
accumulated against a 0.5 GB free tier.

[`teardown-gcp.ps1`](scripts/selfhost/teardown-gcp.ps1) removes the rest:

```powershell
.\scripts\selfhost\teardown-gcp.ps1            # dry run — lists, deletes nothing
.\scripts\selfhost\teardown-gcp.ps1 -Confirm   # execute
```

It deletes, in order: the domain mapping, the Cloud Run service, the
`cloud-run-source-deploy` Artifact Registry repo, the `run-sources-…` build
staging bucket, the `github-deploy` service account with its six project IAM
bindings, and the `github` Workload Identity pool and provider.

Two safeguards, both deliberate:

* **It is not a project delete.** `study-autopilot` is shared with the
  study-autopilot app. The script only touches resources verified to belong to
  finprint — and it re-checks at run time, refusing to delete the Artifact
  Registry repo if a non-finprint package has appeared in it.
* **It refuses to run while DNS still points at Cloud Run**, and while the new
  host is not answering. Deleting the service before the cutover would take the
  site down. `-SkipDnsCheck` overrides this if you accept that.

Two GitHub repository variables outlive the teardown and need the GitHub CLI:

```bash
gh variable delete GCP_WIF_PROVIDER --repo OoEthanoO/finprint
gh variable delete GCP_DEPLOY_SA --repo OoEthanoO/finprint
```

They are non-secret identifiers, not credentials, and with the pool deleted they
point at nothing — but leaving them is misleading.

## Alternatives (config kept in-repo, not currently used)

The repo still ships a `Dockerfile`, so any container host remains one command
away if self-hosting stops being worth the trouble.

> Hugging Face Spaces was ruled out: Docker Spaces moved behind a paid plan
> (mid-2026) and Spaces never supported custom domains. The `README.md`
> frontmatter for it is harmless and kept in case that changes.

### Fly.io (`fly.toml` included) — not free

```bash
fly auth login
fly launch --copy-config --no-deploy   # accept a unique app name when prompted
fly deploy
```

The config runs a 2 GB shared-cpu VM (torch OOMs on the 256 MB default) and
health-checks `/api/health`.

### Render (`render.yaml` Blueprint included) — 2 GB tier is paid

Render Dashboard → **New → Blueprint** → select the repo. It reads
`render.yaml`, builds the Dockerfile, and deploys on the Standard plan (2 GB —
Free/Starter's 512 MB OOMs on torch).

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
  `*.pt` — so a fresh clone predicts species with no retraining. It is loaded
  lazily on the first request. If the checkpoint is ever missing, `/api/predict`
  still returns call type, acoustic features, and the spectrogram, with
  `model_available: false`. To refresh it, retrain (`python -m
  scripts.prepare_data && python -m finprint.train`) and commit the new file.
- **Warm-up.** `finprint.warmup` runs one throwaway prediction from the app's
  lifespan hook, and uvicorn completes it before binding the port — so the
  process never accepts a request it cannot serve quickly. `setup.ps1` also runs
  it once at install time to populate the on-disk matplotlib and numba caches
  (`.caches\mpl`, `.caches\numba`), which is why a restart is fast.
- **Bytecode is precompiled** (`compileall`, in both `setup.ps1` and
  `update.ps1`) for the same reason the Dockerfile did it: librosa imports its
  submodules lazily, so without cached `.pyc` every fresh process recompiles
  them inside the first request. This more than halved warm requests, 5.4 s → 2.4 s.
- **Memory.** torch + librosa need ~1–2 GB resident. Any laptop with 8 GB is
  comfortable; 4 GB is tight.
- **CPU-only.** The venv installs CPU torch wheels from PyTorch's own index;
  the default PyPI wheel would drag in ~5 GB of CUDA this app never touches.
- **ffmpeg** is a real dependency, not a nicety: `soundfile` handles wav/flac/ogg,
  but mp3/m4a/webm uploads decode through librosa's audioread path, which shells
  out to ffmpeg. The container got it from `apt`; `setup.ps1` installs it via winget.
