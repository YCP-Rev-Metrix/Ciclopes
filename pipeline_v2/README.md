# Pipeline v2: current YOLO segmentation workflow

This is the supported training pipeline. It deliberately keeps configuration and execution close to Ultralytics instead of reproducing a general ML framework.

## Files

| Path | Role |
|---|---|
| `configs/base/common.yaml` | Defaults merged into every experiment |
| `configs/exp/yolo26n-seg.yaml` | Nano segmentation training recipe |
| `configs/exp/yolo26m-seg.yaml` | Medium segmentation training recipe |
| `configs/data/ball_lane_pins_dataset.yaml` | Example dataset metadata; experiment files currently use `data/data.yaml` |
| `src/core/config.py` | Recursive YAML merge and `${section.key}` interpolation |
| `src/core/registry.py` | Tiny trainer factory registry |
| `src/builders/trainer.py` | Resolves model/data paths and calls `ultralytics.YOLO.train()` |
| `src/entrypoints/train_yolo_seg.py` | Training CLI |
| `src/entrypoints/eval_yolo_seg.py` | Empty placeholder; do not invoke it |
| `detection_seg_overlay_video_gen.py` | Standalone inference CLI plus full overlay, lane selection, homography, trajectory, interpolation, and kinematics |
| `poster_seg_overlay.py` | Generates a presentation/poster segmentation image |
| `avi_to_mp4.py` | Small AVI-to-MP4 conversion utility |
| `Dockerfile`, `docker-compose.yaml` | Persistent GPU development container |

## Dataset

Download/export the labeled dataset from Roboflow in YOLO segmentation format and extract it under `data/`:

```text
data/
  data.yaml
  train/images/   train/labels/
  valid/images/   valid/labels/
  test/images/    test/labels/    # optional
```

Required class order:

```yaml
names:
  0: ball
  1: lane
  2: pins
```

Do not assume a numeric class ID from a different dataset. Post-processing normalizes model class names and expects these semantic names. Roboflow exports often use `valid`, whereas Ultralytics examples often use `val`; use whatever directory is declared in `data/data.yaml`.

The dataset is intentionally ignored by Git. Record the Roboflow project/version, export format, preprocessing, and augmentation settings with each important training run.

## Docker setup

The image is built for a development RTX 5070 Ti/Blackwell machine. It starts from CUDA 12.4 and force-installs nightly PyTorch wheels. This is a moving dependency and should be rebuilt/tested when the driver or PyTorch index changes.

```bash
cd pipeline_v2
docker compose build
docker compose up -d
docker compose exec ciclopes nvidia-smi
docker compose exec ciclopes python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
```

The container is intentionally kept alive with `tail -f /dev/null`; run commands with `docker compose exec`. Mounted paths keep data, outputs, runs, and Hugging Face cache outside the image.

`CUDA_VISIBLE_DEVICES=0` means v2 training uses one GPU by default. To use another device, change Compose and the experiment's `training.device`. Do not expect a list of GPUs to pool memory automatically.

## W&B

Authenticate on the host (`wandb login`) or export `WANDB_API_KEY` before Compose starts. Compose forwards the key. The trainer maps experiment values to `WANDB_PROJECT`, `WANDB_ENTITY`, and `WANDB_NAME` before calling Ultralytics.

The checked-in entity is a personal account and should be changed for team runs:

```yaml
wandb:
  project: ciclopes-yolo-seg
  entity: your-team-or-user
  run_name: ${run.name}
```

Use `WANDB_MODE=offline` for deferred sync or `WANDB_DISABLED=true` to disable tracking. Local Ultralytics artifacts are still written to the configured output directory.

## Train

```bash
docker compose exec ciclopes python -m src.entrypoints.train_yolo_seg --exp yolo26n-seg
docker compose exec ciclopes python -m src.entrypoints.train_yolo_seg --exp yolo26m-seg
```

Configuration loading is straightforward:

1. load `configs/base/common.yaml`;
2. recursively merge `configs/exp/<name>.yaml`, with experiment values winning;
3. resolve references such as `${run.name}` and `${data.workers}`;
4. pass `training` keys directly to `YOLO.train()`;
5. map `training.output_path` to Ultralytics `project` and `run.name` to `name`.

Model references are resolved against `pipeline_v2/`, then repository root, then treated as an Ultralytics alias that may download automatically. Dataset YAML is resolved against `pipeline_v2/` first.

## VRAM tuning

The nano and medium recipes both use 1024 px and batch 16. Nano generally targets 12-16 GB; medium should be treated as a 16 GB minimum, especially because its recipe enables multi-scale training. These are starting points, not measured guarantees.

OOM tuning order:

1. reduce `training.batch` from 16 to 8, then 4;
2. disable `multi_scale` for the medium recipe;
3. reduce `imgsz` from 1024 to 896/768;
4. reduce workers if host/shared memory is the constraint;
5. keep `amp: true` unless debugging numerical behavior.

Changing image size changes the apparent size of the ball and may affect small-object accuracy. Compare validation metrics instead of treating a successful run as sufficient.

## Evaluate a video

Use the standalone development tool, not the empty `src/entrypoints/eval_yolo_seg.py` placeholder:

```bash
docker compose exec ciclopes python detection_seg_overlay_video_gen.py \
  --model_path /workspace/outputs/yolo26n-seg-BLP/yolo26n-seg/weights/best.pt \
  --video_path /workspace/path/to/shot.mp4 \
  --output_path /workspace/outputs/shot-overlay.mp4 \
  --api_service_mode
```

Run `python detection_seg_overlay_video_gen.py --help` for current options. The tool reads/transcodes video, performs YOLO inference, selects a lane, projects ball points into meters, interpolates gaps, calculates kinematics, and writes diagnostic overlays. `--api_service_mode` applies the API's default frame cap, dimension cap, and ball start frame for closer comparisons.

## Post-processing details

`detection_seg_overlay_video_gen.py` is large because it is an inspectable end-to-end algorithm:

- masks are cleaned with morphological close/open and connected-component filters;
- dense left/right boundaries are sampled by row and fit with MAD outlier rejection;
- adaptive polygon approximation, Hough-line intersection, and image-edge refinement provide fallbacks;
- geometry scoring rewards lane-like taper, vertical span, bottom width, symmetry, and a plausible vanishing point;
- mask coverage/purity and pins support reject adjacent-lane fragments;
- `TemporalSmoother` associates candidates by centroid, applies corner EMA, and rejects abrupt corner jumps;
- the best time-consistent track is aggregated and mapped to the regulation rectangle with a homography;
- the bottom center/contact point of the ball mask is projected, because it approximates contact with the lane better than mask centroid;
- motion-interval trimming removes artifacts, isotonic processing enforces forward progress, splines fill frame gaps, and a bounded curve fit estimates one departure point;
- speed and acceleration use timestamped finite differences and are grouped into lane quarters.

The API contains the maintained production variant under `Ciclopes-API/core/LaneBalls/`. If changing the standalone algorithm, check whether the API version needs the same fix; do not assume the two copies remain identical.

## Utilities

```bash
python avi_to_mp4.py input.avi output.mp4
python poster_seg_overlay.py --help
```

Use `--help` as the authority for utility arguments. Generated videos, weights, data, runs, and outputs are ignored by Git.

## Adding an experiment

Copy an existing file under `configs/exp/`, give `run.name` and `training.output_path` unique values, point `data.path` at a valid YAML, update the W&B entity/tags, and smoke-test with a short epoch count before launching a long run. Avoid adding Hydra or a second trainer framework; v2's value is its small surface area.
