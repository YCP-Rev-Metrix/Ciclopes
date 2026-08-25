# Deprecated pipeline v1 Docker environment

This directory builds the root requirements and starts `pipeline_v1`-style `scripts/train.py`, `scripts/eval.py`, and `scripts/search.py` commands. It is retained for reproducing legacy runs. New CV training uses [`../pipeline_v2/docker-compose.yaml`](../pipeline_v2/docker-compose.yaml); API inference uses [`../Ciclopes-API/`](../Ciclopes-API/).

## Files

- `Dockerfile` uses `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime`, installs root `requirements.txt`, and copies the repository to `/app`.
- `docker-compose.yaml` defines legacy `trainer`, `eval`, and profile-gated `search` services with all GPUs visible and 8 GB shared memory.

Historical use:

```bash
docker compose -f docker/docker-compose.yaml build
docker compose -f docker/docker-compose.yaml run --rm trainer
```

The Compose file declares a `BASE_IMAGE` build argument, but the Dockerfile currently hard-codes its base image and does not consume `ARG BASE_IMAGE`; setting that variable has no effect. Do not copy this setup into v2 without fixing and testing that mismatch.

See [`../pipeline_v1/README.md`](../pipeline_v1/README.md) for the deprecation rationale.
