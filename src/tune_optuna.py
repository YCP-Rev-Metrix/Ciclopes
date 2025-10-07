from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import optuna
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from src.train_lightning import run_training


def _set_by_path(cfg: DictConfig, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = cfg
    with open_dict(node):
        for p in parts[:-1]:
            if p not in node or node[p] is None:
                node[p] = {}
            node = node[p]
        node[parts[-1]] = value


def _apply_minimal_logging(cfg: DictConfig) -> None:
    with open_dict(cfg):
        if "logging" not in cfg:
            cfg.logging = {}
        cfg.logging["enable_tb"] = False
        if "checkpoint" not in cfg:
            cfg.checkpoint = {}
        cfg.checkpoint["enable"] = False
        if "trainer" not in cfg:
            cfg.trainer = {}
        cfg.trainer["lr_monitor"] = False
        cfg.trainer["enable_progress_bar"] = False


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_cfg = getattr(cfg, "run", {})
    direction = str(getattr(run_cfg, "direction", "maximize"))
    metric_name = str(getattr(run_cfg, "metric", "val_acc"))
    n_trials = int(getattr(run_cfg, "n_trials", 6))
    timeout = getattr(run_cfg, "timeout", None)
    seed = int(getattr(run_cfg, "seed", 42))
    top_k = int(getattr(run_cfg, "top_k", 3))
    trial_max_epochs = int(getattr(run_cfg, "trial_max_epochs", cfg.trainer.max_epochs))
    minimal_logging = bool(getattr(run_cfg, "minimal_logging", True))

    sampler_name = str(getattr(run_cfg, "sampler", "tpe")).lower()
    if sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(direction=direction, sampler=sampler)

    base_dir = Path(HydraConfig.get().runtime.output_dir)
    search_dir = base_dir / "optuna_search"
    trials_dir = search_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    param_specs = list(getattr(run_cfg, "params", []))

    trials_jsonl = (search_dir / "trials.jsonl").open("w", encoding="utf-8")

    def objective(trial: optuna.trial.Trial) -> float:
        # Fresh resolved copy
        trial_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        # Small-batch quick test to fit GPU memory and be fast
        _set_by_path(trial_cfg, "io.batch_size", 128)
        # One GPU accelerator auto
        _set_by_path(trial_cfg, "trainer.devices", 1)
        _set_by_path(trial_cfg, "trainer.accelerator", "auto")
        # Limit epochs per trial
        _set_by_path(trial_cfg, "trainer.max_epochs", trial_max_epochs)

        # Apply sampled params
        for spec in param_specs:
            key = str(spec["key"])
            typ = str(spec["type"]).lower()
            if typ == "loguniform":
                val = trial.suggest_float(key, float(spec["low"]), float(spec["high"]), log=True)
            elif typ == "uniform":
                val = trial.suggest_float(key, float(spec["low"]), float(spec["high"]))
            elif typ == "int":
                val = trial.suggest_int(key, int(spec["low"]), int(spec["high"]))
            elif typ == "categorical":
                val = trial.suggest_categorical(key, list(spec["choices"]))
            else:
                raise ValueError(f"Unknown param type: {typ}")
            _set_by_path(trial_cfg, key, val)

        # Reduce logging for speed
        if minimal_logging:
            _apply_minimal_logging(trial_cfg)

        # Route artifacts
        trial_dir = trials_dir / f"trial_{trial.number:04d}"
        artifacts = not minimal_logging
        summary = run_training(trial_cfg, work_dir=trial_dir if artifacts else search_dir, artifacts=artifacts)

        score = summary.get("best_score")
        if score is None:
            score = summary.get("metrics", {}).get(metric_name)
        if score is None:
            raise RuntimeError("No optimization metric produced by training.")

        # Write a row to JSONL for this trial
        row = {
            "number": trial.number,
            "value": float(score),
            "params": dict(trial.params),
        }
        trials_jsonl.write(json.dumps(row) + "\n")
        trials_jsonl.flush()
        return float(score)

    try:
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
    finally:
        try:
            trials_jsonl.close()
        except Exception:
            pass

    # Save top-k
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trials_sorted = sorted(trials, key=lambda t: t.value, reverse=(direction == "maximize"))
    top = trials_sorted[:top_k]

    export = []
    for t in top:
        params = dict(t.params)
        overrides = [f"{k}={v}" for k, v in params.items()]
        export.append({
            "number": t.number,
            "value": t.value,
            "params": params,
            "overrides": overrides,
        })

    (search_dir / "topk.json").write_text(json.dumps(export, indent=2))
    (search_dir / "topk.yaml").write_text(OmegaConf.to_yaml({"top_k": export}))

    study_summary = {
        "best_value": study.best_value if study.best_trial is not None else None,
        "best_params": dict(study.best_params),
        "direction": direction,
        "metric": metric_name,
        "n_trials_complete": len(trials),
    }
    (search_dir / "summary.json").write_text(json.dumps(study_summary, indent=2))


if __name__ == "__main__":
    main()


