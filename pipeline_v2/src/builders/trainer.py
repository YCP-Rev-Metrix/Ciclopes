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
    ):
        self.model_name = self._resolve_model_ref(model_name)
        self.data_yaml = self._resolve_data_yaml(data_yaml)
        self.train_args = dict(train_args or {})

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

    # Keep wandb accepted by signature for future integration, but do not
    # overload YOLO's 'project' output directory with wandb project names.
    _ = wandb

    return YoloSegTrainer(
        model_name=model_name,
        data_yaml=data_yaml,
        train_args=train_args,
    )
