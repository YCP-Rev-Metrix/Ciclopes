from __future__ import annotations
import csv
import yaml
from pathlib import Path
from typing import Any, Dict
import matplotlib.pyplot as plt
from src.core.trainer_base import BaseTrainer
from src.core.registry import register
from ultralytics import YOLO

@register("trainer", "yolo_seg")
class YOLOSegTrainer(BaseTrainer):
    """Self-sufficient YOLO segmentation trainer.

    All runtime configuration (batch size, epochs, hyperparameters, etc.) is managed
    via YAML files colocated with this trainer. Hydra is used only to select the
    trainer and to provide run directories/seed management.
    """

    required_components: list[str] = []

    def __init__(self, cfg, **components):
        super().__init__(cfg, **components)

        self.mode = self._resolve_mode(cfg)
        self.assets_dir = Path(__file__).with_name("yolo_seg_configs")
        self.config = self._load_local_config(self.mode)

        # YOLO model weights (ultralytics downloads automatically if needed)
        model_ref = str(self.config.get("model", "yolo11s-seg.pt"))
        self._model = YOLO(model_ref)

        # Training arguments (directly forwarded to YOLO.train)
        self.train_args: Dict[str, Any] = dict(self.config.get("train_args", {}))
        self.data_yaml = self.config.get("data_yaml", "data/yolo_dataset_v1/data.yaml")
        self.seed_override = self.config.get("seed", None)

        # Sensible defaults if not explicitly set
        self.train_args.setdefault("epochs", 100)
        self.train_args.setdefault("batch", 8)
        self.train_args.setdefault("imgsz", 2048)
        self.train_args.setdefault("workers", 4)
        self.train_args.setdefault("val", True)
        self.train_args.setdefault("save", True)
        self.train_args.setdefault("plots", True)
        self.train_args.setdefault("cos_lr", True)
        self.train_args.setdefault("overlap_mask", True)
        self.train_args.setdefault("mask_ratio", 4)
        self.train_args.setdefault("pretrained", True)
        self.train_args.setdefault("exist_ok", True)
        self.train_args.setdefault("amp", True)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _resolve_mode(self, cfg) -> str:
        """Determine the training mode (debug/train/...) from Hydra config."""
        try:
            mode = getattr(cfg.trainer, "mode", None)
            if mode:
                return str(mode).lower()
        except Exception:
            pass

        try:
            exp_trainer = getattr(getattr(cfg, "exp", None), "trainer", None)
            if exp_trainer is not None and getattr(exp_trainer, "mode", None):
                return str(exp_trainer.mode).lower()
        except Exception:
            pass

        try:
            exp_name = getattr(getattr(cfg, "exp", None), "name", None)
            if exp_name and "debug" in str(exp_name).lower():
                return "debug"
        except Exception:
            pass

        return "train"

    def _load_local_config(self, mode: str) -> Dict[str, Any]:
        """Load YAML configuration for the requested mode."""
        candidates = [
            self.assets_dir / f"{mode}.yaml",
            self.assets_dir / f"{mode}.yml",
        ]
        for path in candidates:
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    return yaml.safe_load(handle) or {}
        if mode != "train":
            return self._load_local_config("train")
        raise FileNotFoundError(
            f"No YOLO segmentation configuration found for mode='{mode}'."
            f" Expected one of: {', '.join(str(p) for p in candidates)}"
        )

    def _extract_exp_name(self) -> str:
        """Best-effort retrieval of the experiment name for logging/output."""
        try:
            exp_node = getattr(self.cfg, "exp", None)
            if exp_node is not None:
                if hasattr(exp_node, "name") and exp_node.name:
                    return str(exp_node.name)
                inner_exp = getattr(exp_node, "exp", None)
                if inner_exp is not None and hasattr(inner_exp, "name") and inner_exp.name:
                    return str(inner_exp.name)
        except Exception:
            pass
        return f"yolo_seg_{self.mode}"

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    def fit(self):
        """Run YOLO training using the locally managed configuration."""
        from hydra.utils import get_original_cwd

        # Resolve project root (Hydra overrides cwd)
        try:
            project_root = Path(get_original_cwd())
        except Exception:
            project_root = Path.cwd()

        data_yaml_path = project_root / self.data_yaml
        if not data_yaml_path.exists():
            raise FileNotFoundError(
                f"YOLO data YAML not found at '{data_yaml_path}'. Update "
                f"src/trainers/yolo_seg_configs/<mode>.yaml if the dataset moved."
            )

        # Compose YOLO.train arguments
        train_args = dict(self.train_args)
        epochs = int(train_args.get("epochs", 1))
        batch_size = int(train_args.get("batch", 4))
        imgsz = int(train_args.get("imgsz", 640))
        workers = int(train_args.get("workers", 4))
        cache = train_args.get("cache", "ram")

        # Device selection (single GPU assumed; can be extended easily)
        if self.device.type == "cuda":
            train_args["device"] = 0 if self.device.index is None else self.device.index
        else:
            train_args["device"] = "cpu"

        # Guarantee correct output location & naming
        exp_name = self.config.get("run_name") or self._extract_exp_name()
        project_dir = self.reports_dir.parent
        train_args.update(
            {
                "data": str(data_yaml_path),
                "project": str(project_dir),
                "name": exp_name,
                "workers": workers,
                "cache": cache,
            }
        )

        # Seed handling
        seed_val = self._resolve_seed()
        if seed_val is not None:
            train_args["seed"] = seed_val
        train_args.setdefault("deterministic", True)

        # Training header for quick reference
        print(f"\n{'=' * 60}")
        print("Starting YOLO Segmentation Training")
        print(f"{'=' * 60}")
        print(f"Mode: {self.mode}")
        print(f"Model: {self.config.get('model', 'yolo11s-seg.pt')}")
        print(f"Dataset: {data_yaml_path}")
        print(f"Epochs: {epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"Image Size: {imgsz}")
        print(f"Workers: {workers}")
        print(f"Device: {self.device}")
        print(f"Output: {project_dir}")
        print(f"{'=' * 60}\n")

        try:
            results = self._model.train(**train_args)

            print(f"\n{'=' * 60}")
            print("Training Complete!")
            print(f"{'=' * 60}")

            if hasattr(results, "results_dict"):
                print("\nFinal Metrics:")
                for key, value in results.results_dict.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")

            self.logger.log(
                {
                    "status": "training_complete",
                    "mode": self.mode,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "imgsz": imgsz,
                }
            )

            self._extract_and_plot_history(project_dir / exp_name)

        except Exception as exc:  # pragma: no cover - runtime diagnostics
            print(f"\n{'!' * 60}")
            print(f"Training Error: {exc}")
            print(f"{'!' * 60}\n")
            raise
        finally:
            try:
                self.logger.finish()
            except Exception:
                pass

    def _resolve_seed(self) -> int | None:
        """Prefer local seed override, otherwise fall back to Hydra config."""
        if self.seed_override is not None:
            try:
                return int(self.seed_override)
            except Exception:
                pass
        try:
            if hasattr(self.cfg, "seed"):
                seed_obj = self.cfg.seed
                if hasattr(seed_obj, "seed"):
                    return int(seed_obj.seed)
                return int(seed_obj)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Metrics & plots
    # ------------------------------------------------------------------
    def _extract_and_plot_history(self, train_dir: Path) -> None:
        """Extract YOLO metrics from results.csv and create summary plots."""
        results_csv = train_dir / "results.csv"
        if not results_csv.exists():
            print(f"Results CSV not found at {results_csv}")
            return

        epochs: list[int] = []
        train_box_loss: list[float] = []
        train_cls_loss: list[float] = []
        train_dfl_loss: list[float] = []
        val_box_loss: list[float] = []
        val_cls_loss: list[float] = []
        val_dfl_loss: list[float] = []
        metrics_map50: list[float] = []
        metrics_map50_95: list[float] = []

        with results_csv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
            for row in reader:
                epochs.append(int(row["epoch"].strip()))
                train_box_loss.append(float(row["train/box_loss"].strip()))
                train_cls_loss.append(float(row["train/cls_loss"].strip()))
                train_dfl_loss.append(float(row["train/dfl_loss"].strip()))
                val_box_loss.append(float(row["val/box_loss"].strip()))
                val_cls_loss.append(float(row["val/cls_loss"].strip()))
                val_dfl_loss.append(float(row["val/dfl_loss"].strip()))
                metrics_map50.append(float(row["metrics/mAP50(B)"].strip()))
                metrics_map50_95.append(float(row["metrics/mAP50-95(B)"].strip()))

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("YOLO Training History", fontsize=16, fontweight="bold")

        axes[0, 0].plot(epochs, train_box_loss, label="Train Box Loss", marker="o", markersize=3)
        axes[0, 0].plot(epochs, val_box_loss, label="Val Box Loss", marker="s", markersize=3)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Box Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, train_cls_loss, label="Train Cls Loss", marker="o", markersize=3)
        axes[0, 1].plot(epochs, val_cls_loss, label="Val Cls Loss", marker="s", markersize=3)
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].set_title("Classification Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(epochs, train_dfl_loss, label="Train DFL Loss", marker="o", markersize=3)
        axes[1, 0].plot(epochs, val_dfl_loss, label="Val DFL Loss", marker="s", markersize=3)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].set_title("DFL Loss")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(epochs, metrics_map50, label="mAP50", marker="o", markersize=3)
        axes[1, 1].plot(epochs, metrics_map50_95, label="mAP50-95", marker="s", markersize=3)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("mAP")
        axes[1, 1].set_title("Mean Average Precision")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = self.reports_dir / "training_history.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"\nTraining history plot saved to: {plot_path}")

        print(f"\n{'=' * 60}")
        print("Training Summary:")
        print(f"{'=' * 60}")
        print(f"Final Train Box Loss: {train_box_loss[-1]:.4f}")
        print(f"Final Val Box Loss: {val_box_loss[-1]:.4f}")
        print(f"Final Train Cls Loss: {train_cls_loss[-1]:.4f}")
        print(f"Final Val Cls Loss: {val_cls_loss[-1]:.4f}")
        print(f"Final mAP50: {metrics_map50[-1]:.4f}")
        print(f"Final mAP50-95: {metrics_map50_95[-1]:.4f}")
        print(
            f"Best mAP50: {max(metrics_map50):.4f} "
            f"(Epoch {epochs[metrics_map50.index(max(metrics_map50))] + 1})"
        )
        print(
            f"Best mAP50-95: {max(metrics_map50_95):.4f} "
            f"(Epoch {epochs[metrics_map50_95.index(max(metrics_map50_95))] + 1})"
        )
        print(f"{'=' * 60}\n")

        plt.close()

