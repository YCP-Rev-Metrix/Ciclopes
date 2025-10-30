# Pipeline V4 Usage

## Prerequisites
- Docker with NVIDIA GPU support (Linux: nvidia-container-toolkit; Windows: WSL2 + Docker Desktop + GPU enabled).
- Python 3.11+ for local runs (optional; Docker handles deps).

## Build Image
Build once for Docker usage:
```bash
# Linux/macOS
BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime docker compose -f docker/docker-compose.yaml build --no-cache

# Windows PowerShell
$env:BASE_IMAGE="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"; docker compose -f docker/docker-compose.yaml build --no-cache
```
Omit `BASE_IMAGE` for default.

## Run Training (Local or Container)
Outputs go to `outputs/${exp.name}/${modes.mode}/${now:%Y%m%d_%H%M%S}/reports/` (config, metrics.jsonl, checkpoints, etc.).

### Local (Host)
```bash
# Standard CIFAR-10 SL training
python -m scripts.train exp=cifar10_train

# Quick debug
python -m scripts.train exp=quick_debug_sl
```

### Container (Recommended for GPU/Isolation)
From project root:
```bash
cd docker
docker-compose up -d trainer  # Runs python scripts/train.py (default exp)
```
Override exp:
```bash
cd docker
docker-compose run --rm trainer python -m scripts.train exp=cifar10_train
```

## Evaluate
```bash
# Local
python -m scripts.eval exp=cifar10_eval

# Container
cd docker && docker-compose run --rm eval  # Runs python scripts/eval.py
```

## Hyperparameter Search (Optuna)
```bash
# Local multirun
python -m scripts.search -m +hpo=optuna +hpo_space=space_sl_basic exp=cifar10_train hydra.sweeper.n_trials=50

# Container
cd docker && docker-compose run --rm search  # Example: +hpo=optuna +hpo_space=space_sl_basic
```
Results in per-trial run dirs with topk.json summary.

## Interactive Shell
GPU-enabled shell with mounted source:
```bash
cd docker && docker-compose run --rm trainer bash
```
Inside: `python -m scripts.train exp=quick_debug_sl` or `pytest`.

## Monitor Logs
- Local: Watch terminal (progress bars + metrics).
- Container: `docker-compose logs -f trainer` (from docker dir).

## Troubleshooting
- GPU not available? Run `docker-compose run --rm trainer nvidia-smi` to check.
- Interpolation errors? Ensure Hydra overrides are correct (e.g., `exp=...`).
- Data missing? CIFAR-10 downloads to `./data` on first run (no copying to outputs).
- Rebuild if deps change: `docker compose build --no-cache`.

For custom exps, edit `config/exp/*.yaml` and rerun.