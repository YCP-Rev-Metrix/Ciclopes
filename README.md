# Ciclopes

Ciclopes is the computer-vision backend for RevMetrix bowling analysis. It combines YOLO instance segmentation of the ball, lane, and pins; geometric reconstruction of a ball path on a regulation lane; per-quarter kinematics; Meta SAM 3D Body pose estimation; and a FastAPI service that retrieves and returns shot analysis.

> **Use `pipeline_v2`.** `pipeline_v1` is deprecated and retained as a historical example of how to overcomplicate this work. It wraps Ultralytics in Hydra, registries, builders, runners, and HPO machinery. Do not start new work there.

## Repository map

| Path | Purpose | Status |
|---|---|---|
| [`pipeline_v2/`](pipeline_v2/) | Current YOLO training, evaluation, and standalone video/post-processing tools | Active |
| [`Ciclopes-API/`](Ciclopes-API/) | FastAPI inference, production post-processing, SAM 3D Body, data access, mock DB | Active |
| [`processing/`](processing/) | Earlier modular preprocessing/post-processing prototype and tests | Reference/test harness |
| [`isaac-sim-data-scripts/`](isaac-sim-data-scripts/) | Isaac Sim data generation and YOLO conversion | Optional tooling |
| [`pipeline_v1/`](pipeline_v1/) | Original Hydra experiment framework | Deprecated |
| [`docker/`](docker/) | Container environment for the deprecated root/v1 pipeline | Deprecated with v1 |
| `weights/` | Legacy/local YOLO checkpoints | Use an explicit known-good checkpoint |
| `data/`, `outputs/` | Local datasets and artifacts | Git-ignored |

```text
Roboflow dataset -> pipeline_v2 training -> YOLO segmentation weights
                                                |
RevMetrix video + sensor JSON -> Ciclopes API ---+
                                 |              +-> lane homography -> ball path -> kinematics
                                 +-> SAM 3D Body -> smoothed skeleton
```

## Prerequisites

- Git and Docker Compose v2 (`docker compose`).
- An NVIDIA GPU, recent driver, and NVIDIA Container Toolkit. On Windows, use WSL2 with Docker Desktop GPU support.
- RevMetrix API credentials for API workflows.
- A Hugging Face account, read token, and approved access to the gated Meta model.
- A W&B account only if training metrics should be uploaded.
- A Roboflow export of the ball/lane/pins dataset for training; data is not committed.

Verify the machine before building:

```bash
nvidia-smi
docker version
docker compose version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## First API build

### 1. Obtain SAM 3D Body permission

The API defaults to the gated Hugging Face repository `facebook/sam-3d-body-dinov3`. A token alone is not enough: while signed in to Hugging Face, request/accept Meta/Facebook access to that model and wait for approval. Then create a read token.

The first API startup downloads the model snapshot and DINO backbone. Compose mounts the host Hugging Face cache so later starts can reuse it. Never commit the token, cache, or gated weights.

### 2. Configure environment variables

```bash
cd Ciclopes-API
cp .env.example .env
```

PowerShell: `Copy-Item .env.example .env`.

Edit `.env`:

```dotenv
API_BASE=https://api.revmetrix.io
API_USERNAME=your_username
API_PASSWORD=your_password
VERIFY_API=true
VERIFY_PRESIGNED=true
PRESIGN_TTL_SECONDS=3600

HF_TOKEN=hf_your_read_token
SAM3D_BODY_HF_REPO_ID=facebook/sam-3d-body-dinov3

LANE_BALL_BATCH_SIZE=48
SAM3D_BODY_BATCH_SIZE=48
MAX_VIDEO_FRAMES=600
MAX_VIDEO_DIMENSION=1024
LANE_BALL_MAX_VIDEO_DIMENSION=1280
FORCE_LANE_BALL_START_FRAME=false

COMPOSE_FILE=docker-compose.yaml:docker-compose.ngrok.yaml
COMPOSE_PROFILES=
MULTI_GPU=false
```

- `API_*` credentials authorize requests for RevMetrix video/sensor objects.
- Keep TLS verification flags `true`; disable only to diagnose a known certificate issue.
- `HF_TOKEN` authorizes the gated model. The Python loader also accepts `HUGGING_FACE_HUB_TOKEN`, but Compose passes `HF_TOKEN`.
- Frame and dimension caps are OOM protection. Lower values reduce memory and latency but discard detail.
- `FORCE_LANE_BALL_START_FRAME=false` lets sensor data determine ball contact. Set it to `true` only when intentionally forcing the demo start behavior.
- `.env` is ignored. Do not put secrets in source, YAML, documentation, or shell history.

### 3. Select a hardware profile

Home/development (one RTX 5070 Ti, 16 GB):

```dotenv
COMPOSE_FILE=docker-compose.yaml:docker-compose.ngrok.yaml
MULTI_GPU=false
LANE_BALL_BATCH_SIZE=48
SAM3D_BODY_BATCH_SIZE=48
```

School server (two RTX 2080 GPUs, 8 GB each):

```dotenv
COMPOSE_FILE=docker-compose.prod.yaml:docker-compose.ngrok.yaml
MULTI_GPU=true
LANE_BALL_BATCH_SIZE=16
SAM3D_BODY_BATCH_SIZE=2
```

Build and start from `Ciclopes-API/`:

```bash
docker compose build
docker compose up
```

Two-GPU mode assigns YOLO lane/ball inference to `cuda:0` and SAM 3D Body to `cuda:1`. One-GPU mode shares the device. `MULTI_GPU=true` falls back if fewer than two GPUs are visible, but batch sizes still need to fit. Check <http://localhost:8000/docs> and <http://localhost:8000/test/health>.

## GPU and VRAM guidance

These are starting points from checked-in configurations, not guaranteed peak measurements. CUDA/PyTorch versions, resolution, mask count, and concurrent requests affect memory.

| Workflow | Practical target | Starting configuration |
|---|---:|---|
| YOLO segmentation inference only | 6-8 GB | 1024-1280 px; batch 8-16 |
| API, both models on one GPU | 16 GB | Dev values 48/48; reduce SAM3D first on OOM |
| API, school split | 2 x 8 GB | lane/ball 16 on GPU 0; SAM3D 2 on GPU 1 |
| Train YOLO26n-seg at 1024 px | 12-16 GB | batch 16, AMP; lower to 8/4 as needed |
| Train YOLO26m-seg at 1024 px | 16 GB minimum; more preferred | batch 16, AMP, multi-scale; lower batch first |
| CPU inference | Not practical for normal use | Fallback exists, but SAM 3D Body is very slow |

For OOM errors, stop concurrent runs, lower the relevant batch, then lower the video dimension cap or training image size. Watch `nvidia-smi` and `/test/engine_status`. Multiple GPUs do not pool VRAM; each model must fit on its assigned GPU.

## Train the current CV model

### Roboflow dataset

Export the project from Roboflow in **YOLO segmentation** format. Required class IDs are:

```text
0 ball
1 lane
2 pins
```

Extract the export so `pipeline_v2/data/data.yaml` exists, alongside `train/images`, `train/labels`, `valid/images`, `valid/labels`, and optional test folders. Roboflow commonly names validation `valid`; trust the exported YAML and verify its class ordering. The dataset is ignored because it is large and separately controlled.

### W&B

Weights & Biases records losses, segmentation metrics, hyperparameters, plots, and artifacts. It is experiment tracking, not the dataset or model-weight source.

```bash
wandb login
# or export WANDB_API_KEY=...
```

Experiment YAML supplies the project/entity/run name to Ultralytics. Replace the checked-in personal entity before a team run. For local-only tracking, set `WANDB_MODE=offline`; to disable it, set `WANDB_DISABLED=true`.

### Build and run v2

```bash
cd pipeline_v2
docker compose build
docker compose up -d
docker compose exec ciclopes python -m src.entrypoints.train_yolo_seg --exp yolo26n-seg
```

Medium model:

```bash
docker compose exec ciclopes python -m src.entrypoints.train_yolo_seg --exp yolo26m-seg
```

Outputs go to each experiment's `training.output_path`. See [`pipeline_v2/README.md`](pipeline_v2/README.md) for config behavior, evaluation, and utilities.

## External services at a glance

| Service | Purpose | Access |
|---|---|---|
| Hugging Face | Gated SAM 3D Body assets | `HF_TOKEN` plus explicit model approval |
| Weights & Biases | YOLO experiment tracking | `WANDB_API_KEY` / `wandb login` |
| Roboflow | Labeled ball/lane/pins dataset export | Project/account access |
| RevMetrix | Source video/sensor objects and result integration | `API_USERNAME`, `API_PASSWORD` |

## Public proxy/tunnel

`docker-compose.ngrok.yaml` and profile `ngrok` are legacy names. The actual proxy is an ephemeral **Cloudflare Quick Tunnel** forwarding public HTTPS to `http://ciclopes-api:8000`.

```dotenv
COMPOSE_PROFILES=ngrok
```

```bash
docker compose up --build
bash scripts/get-ngrok-url.sh
```

Put the printed `https://...trycloudflare.com` address in the client configuration. It changes whenever the tunnel restarts and is for development/demo use, not stable production. Do not set `NGROK_*`; ngrok is not used.

If the school network requires an outbound HTTP proxy, pass `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY` to Docker/Compose according to school IT policy. Include `ciclopes-api,localhost,127.0.0.1` in `NO_PROXY`. No institutional proxy host or credentials belong in this repository.

## Post-processing algorithm

Production code lives in `Ciclopes-API/core/LaneBalls/`; the standalone v2 overlay carries a closely related implementation.

1. YOLO emits ball, lane, and pins masks per frame.
2. Lane masks are morphologically closed/opened and insignificant components are removed.
3. Dense scan-line boundaries are robustly fit as lane edges; polygon approximation and Hough lines are fallbacks.
4. Trapezoids are scored for mask coverage/purity, perspective taper, span, symmetry, vanishing point, and pins/lane support.
5. Candidates are associated across frames by horizontal centroid. An EMA smooths corners and rejects large jumps.
6. Tracks are ranked using geometry, support, persistence, area, and ball evidence; nearby observations stabilize the final quadrilateral.
7. A homography maps it to a regulation lane (`1.0541 m x 18.288 m`). The bottom contact point of each ball mask is projected into meters.
8. Pre-contact, out-of-bounds, backward, stalled, and detector-latch samples are trimmed. Missing frames are interpolated with monotonic forward-distance constraints.
9. A conservative linear/quadratic curve blend may append one pin-deck departure point while preserving hook direction and capping lateral extrapolation.
10. Finite differences produce speed and acceleration summarized over four equal lane quarters.

See [`Ciclopes-API/README.md`](Ciclopes-API/README.md) and [`processing/README.md`](processing/README.md).

## Common problems

- **Hugging Face 401/403:** the token user must also be approved for `facebook/sam-3d-body-dinov3`.
- **DINO download fails:** SAM also loads DINOv3 through `torch.hub`; check outbound GitHub access and cache permissions.
- **RTX 5070 Ti CUDA/NVRTC error:** use the development Dockerfile, which installs nightly cu130 PyTorch and CUDA 13 NVRTC pieces.
- **RTX 2080 build issue:** use `docker-compose.prod.yaml`/`Dockerfile.prod`; do not force the Blackwell nightly stack.
- **Exit 137:** likely memory pressure. Reduce frame/dimension caps and inference batches; prod Compose caps host RAM at 24 GB.
- **Dataset YAML missing:** place the Roboflow export at `pipeline_v2/data/data.yaml`.
- **W&B wrong account:** update `wandb.entity` in the experiment YAML or runtime environment.
- **Tunnel URL dead:** retrieve a new URL after each tunnel restart.

## Checks and handoff rules

```bash
python -m pytest processing -q
```

The API initializes models during FastAPI lifespan startup. Inspect startup logs and `/test/engine_status`; a listening web process is not proof that both engines loaded.

- Never commit `.env`, tokens, RevMetrix credentials, datasets, videos, caches, or outputs.
- Do not redistribute gated Meta files; each developer needs authorized access.
- Record the Roboflow dataset version/export and exact checkpoint used; `best.pt` alone is ambiguous.
- Prefer a small explicit v2 experiment YAML over another abstraction layer.
