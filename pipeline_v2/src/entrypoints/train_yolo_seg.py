import argparse

from builders import trainer as _trainer_builders
from core.config import Config
from core.registry import build


def main(exp_name: str):
    print(f"[Train Yolo Seg] Starting training for experiment: {exp_name}")
    cfg = Config.from_experiment(exp_name)

    print("[Train Yolo Seg] Config:")
    print(f"  - Run: {cfg.run.get('name')}")
    print(f"  - Trainer: {cfg.trainer.get('type')}")
    print(f"  - Model: {cfg.model.get('name')}")
    print(f"  - Data: {cfg.data.get('name')}")
    print(f"  - Data YAML: {cfg.data.get('path')}")
    print(f"  - Outputs: {cfg.training.get('output_path')}")
    print(f"  - WandB Project: {cfg.wandb.get('project')}")
    print(f"  - WandB Entity: {cfg.wandb.get('entity')}")
    print(f"  - WandB Run: {cfg.wandb.get('run_name')}")
    print()

    trainer = build(
        "trainer",
        type=cfg.trainer.get("type"),
        run=cfg.run,
        model=cfg.model,
        data=cfg.data,
        training=cfg.training,
        wandb=cfg.wandb,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CV Model using Ultralytics setup")
    parser.add_argument("--exp", type=str, required=True, help="Experiment name")

    args = parser.parse_args()
    main(exp_name=args.exp)
