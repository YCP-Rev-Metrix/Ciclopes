# Pipeline v1 (deprecated)

> **Do not use this pipeline for new training or evaluation. Use [`../pipeline_v2/`](../pipeline_v2/) instead.**

Pipeline v1 is retained for history and for understanding earlier experiment outputs. It is also a useful example of how a small YOLO training problem became overcomplicated.

## Why it was deprecated

The implementation builds a general-purpose framework around Hydra configuration composition, registries, builders, abstract trainer/evaluator classes, custom run directories, logger routing, checkpoints, progress reporting, seed/speed layers, Optuna search, CIFAR-10 examples, and a separate YOLO wrapper. Most of that machinery is either already provided by Ultralytics or irrelevant to the bowling segmentation workflow.

Consequences included:

- configuration spread across many files and override groups;
- indirection between an entrypoint and the code that actually trains;
- two different problem types (CIFAR classification and YOLO segmentation) in one framework;
- more failure modes for path resolution, interpolation, registration, and dependency compatibility;
- harder onboarding and slower iteration for changes that should be one experiment YAML edit.

Pipeline v2 keeps a small YAML merge, a minimal trainer factory, and a direct `YOLO.train()` call.

## Historical file map

| Path | Historical purpose |
|---|---|
| `scripts/train.py`, `eval.py`, `search.py` | Hydra entrypoints |
| `config/defaults.yaml` | Top-level Hydra defaults composition |
| `config/exp/` | Experiment selections |
| `config/model/`, `optimizer/`, `trainer/`, `evaluator/` | Component config groups |
| `config/hpo/` | Optuna sweeper and search space |
| `src/core/registry.py`, `builder.py` | Dynamic component registration/building |
| `src/core/runner.py`, `bootstrap.py`, `run_io.py` | Run orchestration and outputs |
| `src/core/trainer_base.py`, `evaluator_base.py` | Abstract execution loops |
| `src/components/` | Dataset/model/optimizer/evaluator adapters |
| `src/trainers/yolo_seg_trainer.py` | Ultralytics wrapper |
| `src/trainers/yolo_seg_configs/` | Additional YOLO-specific YAML nested inside the Hydra system |

## If reproducing an old result

Use the root `requirements.txt` and `docker/` environment, locate the exact experiment and nested YOLO configuration from the old run output, and treat any result as legacy. Old documentation referred to the root as “Pipeline V4”; that label did not correspond to the now-supported `pipeline_v2` directory.

Typical historical commands were:

```bash
python -m scripts.train exp=yolo_seg_debug
python -m scripts.train exp=yolo_seg_train
python -m scripts.eval exp=cifar10_eval
```

They are shown only for archaeology. Migrate useful hyperparameters into a v2 experiment YAML rather than extending v1.
