# Ciclopes API

The FastAPI service runs the production bowling-analysis workflows:

- `/laneballs/run`: YOLO segmentation, lane geometry, ball trajectory, and kinematics;
- `/fourdbody/run`: SAM 3D Body inference and skeleton smoothing;
- `/agg/run`: both workflows for one shot;
- corresponding `/query` routes and `/query/names`: retrieve saved/mock results; and
- `/test/*`: health, engine status, and development diagnostics.

Interactive schemas and request models are always available at `/docs` after startup.

## File map

| Path | Purpose |
|---|---|
| `main.py` | Creates FastAPI app and initializes settings/inference engine during lifespan |
| `src/settings.py` | Parses `.env` and runtime controls |
| `src/modules/` | Route handlers and Pydantic API models |
| `core/InferenceEngine/` | Loads YOLO and SAM3D, assigns GPUs, batches work, exposes health |
| `core/LaneBalls/` | Production mask geometry, homography, trajectory cleanup/interpolation/extrapolation, kinematics |
| `core/4DBody/` | Skeleton models and EMA smoothing |
| `core/sam_3d_body/` | Vendored/integrated Meta SAM 3D Body architecture and loader |
| `core/SensorData/` | Parses shot sensor JSON and estimates contact/start frame |
| `core/VideoUtil/` | RevMetrix object requests, temporary video download, and frame splitting |
| `core/DB/` | Database/API helper code |
| `core/MockDB/` | Checked-in sample results and local read/write helpers |
| `core/weights/best_v2_26n.pt` | YOLO segmentation checkpoint loaded by the API |
| `Dockerfile` | RTX 5070 Ti/Blackwell development image |
| `Dockerfile.prod` | RTX 2080/Turing school image |
| `docker-compose*.yaml` | Single-GPU, two-GPU, and Cloudflare tunnel overlays |
| `.env.example` | Complete non-secret configuration template |
| `RUNNING.md` | Short operations cheat sheet |

## Required access

The SAM model defaults to `facebook/sam-3d-body-dinov3`, a gated Hugging Face repository. Each operator must:

1. sign in to Hugging Face;
2. request/accept Meta/Facebook's terms for that model and receive access;
3. create a read token; and
4. set `HF_TOKEN` in `.env`.

Do not share or commit downloaded model assets. The loader first checks optional local checkpoint paths in `Sam3DBodyInference.py`; otherwise it downloads the Hugging Face snapshot. The DINOv3 backbone may also use `torch.hub`, so first startup needs outbound Hugging Face and GitHub access.

## Configure `.env`

```bash
cp .env.example .env
```

| Variable | Meaning | Normal/default guidance |
|---|---|---|
| `API_BASE` | RevMetrix API origin | `https://api.revmetrix.io` |
| `API_USERNAME`, `API_PASSWORD` | Basic credentials used to request shot objects | Required for `/run` routes |
| `VERIFY_API` | Verify RevMetrix TLS | Keep `true` |
| `VERIFY_PRESIGNED` | Verify presigned-object TLS | Keep `true` |
| `PRESIGN_TTL_SECONDS` | Requested URL lifetime | `3600` |
| `HF_TOKEN` | Hugging Face read token for gated SAM3D | Required on first download |
| `SAM3D_BODY_HF_REPO_ID` | Alternate SAM3D repository | Normally leave default |
| `LANE_BALL_BATCH_SIZE` | YOLO frames per inference batch | 48 on 16 GB dev; 16 on 8 GB school GPU |
| `SAM3D_BODY_BATCH_SIZE` | SAM3D frames per batch | 48 on 16 GB dev; 2 on 8 GB school GPU |
| `MAX_VIDEO_FRAMES` | Hard extraction cap | `600`; lower for OOM |
| `MAX_VIDEO_DIMENSION` | SAM/body longest-edge cap | `1024` |
| `LANE_BALL_MAX_VIDEO_DIMENSION` | YOLO lane/ball longest-edge cap | `1280` |
| `FORCE_LANE_BALL_START_FRAME` | Ignore sensor-derived contact and force demo start behavior | Normally `false` |
| `MULTI_GPU` | Assign YOLO and SAM3D to separate GPUs | `true` only on school 2-GPU host |
| `COMPOSE_FILE` | Base hardware Compose plus tunnel overlay | See below |
| `COMPOSE_PROFILES` | Set `ngrok` to enable the Cloudflare sidecar | Blank for local only |

Lowercase credential aliases remain accepted by `settings.py` for legacy environments, but new files should use uppercase names.

## Start by hardware profile

### One RTX 5070 Ti (16 GB)

```dotenv
COMPOSE_FILE=docker-compose.yaml:docker-compose.ngrok.yaml
COMPOSE_PROFILES=
MULTI_GPU=false
LANE_BALL_BATCH_SIZE=48
SAM3D_BODY_BATCH_SIZE=48
```

The development Dockerfile replaces stable PyTorch with nightly cu130 wheels for Blackwell `sm_120` and installs CUDA 13 NVRTC runtime packages. Its explicit `LD_LIBRARY_PATH` is intentional.

### School: two RTX 2080 GPUs (8 GB each)

```dotenv
COMPOSE_FILE=docker-compose.prod.yaml:docker-compose.ngrok.yaml
COMPOSE_PROFILES=
MULTI_GPU=true
LANE_BALL_BATCH_SIZE=16
SAM3D_BODY_BATCH_SIZE=2
```

The production Dockerfile keeps stable PyTorch 2.5.1/CUDA 12.4 for Turing `sm_75`. The engine places lane/ball on `cuda:0` and SAM3D on `cuda:1`. Compose caps system memory/swap at 24 GB to fail more predictably instead of destabilizing the host.

### Start and verify

```bash
docker compose build
docker compose up
docker compose logs -f ciclopes-api
```

```bash
curl http://localhost:8000/test/health
curl http://localhost:8000/test/engine_status
```

The first startup can take several minutes. FastAPI lifespan constructs both models before the engine is ready. A fallback to CPU is useful for diagnostics but not an acceptable production configuration.

## Concurrency and memory

The route handlers dispatch GPU work through the inference engine's executors so async request handling does not directly block the event loop. This does not make GPU memory unlimited: concurrent requests and retained video frames can increase both VRAM and host RAM.

On OOM or exit 137:

1. make sure only one stack/model copy is running;
2. lower `SAM3D_BODY_BATCH_SIZE` first for SAM/body failures;
3. lower `LANE_BALL_BATCH_SIZE` for YOLO failures;
4. lower `MAX_VIDEO_DIMENSION`, `LANE_BALL_MAX_VIDEO_DIMENSION`, or `MAX_VIDEO_FRAMES`;
5. inspect `nvidia-smi`, container memory, logs, and `/test/engine_status`.

Two 8 GB GPUs are not equivalent to one 16 GB allocation. The models run independently and each must fit its own device.

## Lane/ball algorithm

### 1. Inference and mask extraction

`LaneBallInference` loads `core/weights/best_v2_26n.pt`, runs Ultralytics segmentation in batches, and returns per-frame masks. The postprocessor separates classes by semantic name. Pins help choose lane geometry even though only lane and ball become trajectory outputs.

### 2. Lane quadrilateral

For each lane mask, morphological close fills gaps and open removes spurs. Significant connected components are kept so an occluding bowler does not split away valid lane pixels. The implementation samples left/right mask boundaries across image rows, fits `x = ay + b` with median-absolute-deviation rejection, and constructs a quadrilateral. Adaptive convex-hull approximation and Hough lines are fallback paths.

Candidates are evaluated using:

- coverage and purity against the segmentation mask;
- top/bottom width ratio and perspective taper;
- vertical span and bottom width;
- center-line symmetry and vanishing-point plausibility;
- support in a lane band; and
- proximity/support from pins boxes.

Nested partial-lane candidates are penalized.

### 3. Temporal lane selection

`TemporalSmoother` matches candidates to tracks by x-centroid, rejects excessive corner jumps, and uses an exponential moving average for accepted updates. Tracks accumulate mask area, geometry/support scores, persistence, and ball-in-polygon votes. The selector ranks tracks, gathers compatible observations, aggregates robust corner/pins estimates, and corrects top corners when pins evidence helps.

### 4. Homography and ball contact

The chosen image quadrilateral maps to a regulation rectangle: width `1.0541 m`, length `18.288 m`. The ball's bottom contact point is preferred over its centroid. If multiple balls/masks exist, candidates are filtered and scored against the selected lane and plausible motion. Projected points outside a tolerant lane envelope are rejected. Health output includes detection counts, lane coverage, homography determinant, and condition number.

### 5. Trajectory cleanup

`Extrapolation.trim_raw_detections` identifies the strongest in-bounds forward-motion interval and removes pre-contact artifacts, large gaps, backward jumps, lateral spikes, stalls, and tail clusters where the detector may have latched onto pins/debris.

`Interprolation.interpolate_ball_positions` (filename kept for compatibility) sorts and deduplicates detections, enforces non-decreasing down-lane distance with isotonic regression, anchors endpoints, spreads artificial flat segments, and uses splines/linear interpolation to fill missing frames.

`append_departure_point` may add one point at the pin deck. It estimates forward rate, fits tail-weighted linear and quadratic lateral curves, blends them conservatively, refuses to reverse an established hook, caps departure lateral movement, and clamps to lane bounds.

### 6. Kinematics

`compute_kinematics_per_quarter` takes finite differences of metric positions and timestamps. Euclidean displacement divided by time gives speed; successive speed differences give acceleration. Samples are averaged into four equal `18.288 / 4 m` lane sections. Sparse or empty quarters return zeros and a sample count so callers can distinguish missing evidence.

## SAM 3D Body flow

`Sam3DBodyInference` loads the gated checkpoint/config/MHR assets, constructs `SAM3DBodyEstimator`, prepares frame batches, moves nested tensors to its assigned device, and returns skeleton predictions. Route-level code converts predictions into response models and applies EMA smoothing across frames to reduce jitter while retaining the original temporal sequence.

The integrated `core/sam_3d_body/` tree contains Meta-derived model architecture, transforms, geometry, renderer, and checkpoint helpers. Treat it as third-party model integration: avoid casual refactors, preserve copyright headers, and check upstream licensing/terms before redistribution.

## RevMetrix video and sensor flow

Run routes require API credentials. `SpacesApiClient` requests a presigned object URL from RevMetrix and streams the video to a temporary file. `FrameSplit` decodes/caps/downscales frames. Sensor JSON can identify the ball-contact frame so expensive lane/ball analysis skips irrelevant lead-in; `FORCE_LANE_BALL_START_FRAME=true` bypasses that signal.

Temporary video files are scoped to the request. The checked-in MockDB is for demos/tests, not a production database.

## Cloudflare public tunnel

Despite the legacy filenames/profile, this uses Cloudflare, not ngrok:

```dotenv
COMPOSE_PROFILES=ngrok
```

```bash
docker compose up --build
bash scripts/get-ngrok-url.sh
docker compose logs -f cloudflared
```

The Quick Tunnel URL changes on restart. It is a temporary public reverse proxy and should not be treated as a stable or authenticated production ingress. Keep application credentials and authorization enabled.

For a school outbound proxy, configure Docker with IT-provided `HTTP_PROXY`/`HTTPS_PROXY` values and set `NO_PROXY=ciclopes-api,localhost,127.0.0.1`. Never commit proxy credentials.

## Development cautions

- `core/LaneBalls/Postprocessing.py` and `pipeline_v2/detection_seg_overlay_video_gen.py` share concepts but are separate copies. Apply intended fixes to both or explicitly document divergence.
- `Interprolation.py` is misspelled on disk; rename only as a coordinated import migration.
- Preserve class names/order when changing the YOLO checkpoint.
- Validate geometry changes against multiple lanes, camera angles, lofts, hooks, and failure cases—not one attractive overlay.
- Use `/docs` for authoritative request/response fields; Pydantic models are the source of truth.
