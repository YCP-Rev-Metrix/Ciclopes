# Running the Ciclopes API

## .env Setup

Copy `.env.example` to `.env` and fill in the values below.

```dotenv
# RevMetrix API credentials
API_BASE=https://api.revmetrix.io
API_USERNAME=your_username
API_PASSWORD=your_password

# Request controls (leave as-is unless you know why)
VERIFY_API=true
VERIFY_PRESIGNED=true
PRESIGN_TTL_SECONDS=3600

# HuggingFace token (required to download model weights on first run)
HF_TOKEN=hf_...

# Inference batch sizes — tune to your GPU VRAM
# Home dev machine (RTX 5070 Ti, 16 GB single GPU): 48 / 48
# School prod machine (2x RTX 2080, 8 GB each):    16 / 2
LANE_BALL_BATCH_SIZE=48
SAM3D_BODY_BATCH_SIZE=48

# Which compose files to merge — change only when switching machines
# Home dev:   docker-compose.yaml:docker-compose.ngrok.yaml
# School prod: docker-compose.prod.yaml:docker-compose.ngrok.yaml
COMPOSE_FILE=docker-compose.yaml:docker-compose.ngrok.yaml

# Set to `ngrok` to start the tunnel sidecar alongside the API.
# Leave blank to run local only (no public URL).
COMPOSE_PROFILES=ngrok

# Multi-GPU: false on dev (single GPU), true on school prod (2 GPUs)
MULTI_GPU=false
```

> **Note on the "ngrok" naming:** `COMPOSE_PROFILES=ngrok` and `docker-compose.ngrok.yaml`
> still say "ngrok" in their names, but the tunnel is actually **Cloudflare Tunnel**
> (`cloudflare/cloudflared`). The names were not changed to avoid breaking existing `.env`
> files. Do not set any `NGROK_*` variables — they are no longer used.

---

## Running the Dev Server (Home Machine, Single GPU)

```dotenv
# .env settings for this mode
COMPOSE_FILE=docker-compose.yaml:docker-compose.ngrok.yaml
COMPOSE_PROFILES=ngrok
MULTI_GPU=false
LANE_BALL_BATCH_SIZE=48
SAM3D_BODY_BATCH_SIZE=48
```

```bash
docker compose up --build
```

First run downloads model weights from HuggingFace — this takes a few minutes.
Subsequent runs skip the download (weights are cached in `~/.cache/huggingface`).

---

## Running the Prod Server (School Machine, 2x RTX 2080)

```dotenv
# .env settings for this mode
COMPOSE_FILE=docker-compose.prod.yaml:docker-compose.ngrok.yaml
COMPOSE_PROFILES=ngrok
MULTI_GPU=true
LANE_BALL_BATCH_SIZE=16
SAM3D_BODY_BATCH_SIZE=2
```

```bash
docker compose up --build
```

Same single command — the compose file swap handles the rest.

---

## Getting the Public URL (for the Client .env)

After the stack is up, run:

```bash
bash scripts/get-ngrok-url.sh
```

This prints the current Cloudflare tunnel URL, e.g.:

```
https://some-random-words.trycloudflare.com
```

Copy that URL into the client-side `.env`:

```dotenv
API_BASE=https://some-random-words.trycloudflare.com
```

> The URL changes every time the tunnel container restarts. Re-run the script and
> update the client `.env` whenever you bring the stack down and back up.

---

## Quick Reference

| Task | Command |
|---|---|
| Start stack (with tunnel) | `docker compose up --build` |
| Start stack (local only) | `COMPOSE_PROFILES= docker compose up --build` |
| Get public URL | `bash scripts/get-ngrok-url.sh` |
| Stop stack | `docker compose down` |
| View API logs | `docker compose logs -f ciclopes-api` |
| View tunnel logs | `docker compose logs -f cloudflared` |
| Local API (no tunnel) | `http://localhost:8000` |
| Local API docs | `http://localhost:8000/docs` |
