import argparse
from core.config import Config
from core.registry import build

def main(exp_name: str):
    print(f"[Train Yolo Seg] Starting training for experiment: {exp_name}")
    cfg = Config.from_experiment(exp_name)

    print(f"[Train Yolo Seg] Config:")
    print(f"  - Run: {cfg.run['name']}")
    print(f"  - Model: {cfg.model['name']}")
    print(f"  - Data: {cfg.data['name']}")
    print(f"  - Outputs: {cfg.training['output_path']}")
    print(f"  - WandB: {cfg.wandb['project']}")
    print(f"  - WandB: {cfg.wandb['entity']}")
    print(f"  - WandB: {cfg.wandb['run_name']}")
    print()

    trainer = build("trainer", )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CV Model using Ultralytics setup")
    parser.add_argument("--exp", type=str, required=True, help="Experiment name")

    args = parser.parse_args()
    main(exp_name=args.exp)