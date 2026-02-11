from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from core.registry import register
from ultralytics import YOLO


class YoloSegTrainer:
    def __init__(
        self,
        model_name: str,
        data_yaml: str,
        train_args: Dict[str, Any] | None = None,
        wandb_config: Dict[str, Any] | None = None,
    ):
        self.model_name = self._resolve_model_ref(model_name)
        self.data_yaml = self._resolve_data_yaml(data_yaml)
        self.train_args = dict(train_args or {})
        self.wandb_config = dict(wandb_config or {})

        self.model = YOLO(self.model_name)

    @staticmethod
    def _resolve_model_ref(model_ref: str) -> str:
        path = Path(model_ref).expanduser()
        if path.is_absolute() and path.exists():
            return str(path)

        project_root = Path(__file__).resolve().parents[2]
        repo_root = project_root.parent
        for base in (project_root, repo_root):
            candidate = base / path
            if candidate.exists():
                return str(candidate)

        # Fall back to Ultralytics model aliases (e.g. yolo11n-seg.pt).
        return model_ref

    @staticmethod
    def _resolve_data_yaml(data_yaml: str) -> str:
        path = Path(data_yaml).expanduser()
        if path.is_absolute():
            return str(path)

        # Resolve relative to pipeline_v2 root first for stable CLI behavior.
        project_root = Path(__file__).resolve().parents[2]
        project_relative = project_root / path
        if project_relative.exists():
            return str(project_relative)

        return str(path)

    def train(self):
        if not Path(self.data_yaml).exists():
            raise FileNotFoundError(f"Data YAML not found: {self.data_yaml}")

        # Configure W&B integration if provided
        import os
        if self.wandb_config:
            # Set W&B environment variables for Ultralytics integration
            if self.wandb_config.get("project"):
                os.environ["WANDB_PROJECT"] = self.wandb_config["project"]
            if self.wandb_config.get("entity"):
                os.environ["WANDB_ENTITY"] = self.wandb_config["entity"]
            if self.wandb_config.get("run_name"):
                os.environ["WANDB_NAME"] = self.wandb_config["run_name"]

            print(f"[W&B] Logging enabled:")
            print(f"  - Project: {self.wandb_config.get('project')}")
            print(f"  - Entity: {self.wandb_config.get('entity')}")
            print(f"  - Run: {self.wandb_config.get('run_name')}")

        return self.model.train(data=self.data_yaml, **self.train_args)


@register("trainer", "yolo_seg")
def build_yolo_seg_trainer(
    run: Dict[str, Any],
    model: Dict[str, Any],
    data: Dict[str, Any],
    training: Dict[str, Any],
    wandb: Dict[str, Any] | None = None,
    **kwargs,
) -> YoloSegTrainer:
    model_name = model.get("name")
    if not model_name:
        raise ValueError("Missing required config: model.name")

    data_yaml = data.get("path")
    if not data_yaml:
        raise ValueError("Missing required config: data.path")

    train_args = dict(training)

    output_path = train_args.pop("output_path", None)
    if output_path:
        train_args.setdefault("project", output_path)

    run_name = run.get("name")
    if run_name:
        train_args.setdefault("name", run_name)

    return YoloSegTrainer(
        model_name=model_name,
        data_yaml=data_yaml,
        train_args=train_args,
        wandb_config=wandb,
    )
