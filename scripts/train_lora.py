from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.training import train_lora


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--instance_token", type=str, default="<sks>")
    parser.add_argument("--output_dir", type=str, default="lora_out")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=800)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    train_lora(
        data_dir=args.data_dir,
        instance_token=args.instance_token,
        output_dir=args.output_dir,
        rank=args.rank,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        resolution=args.resolution,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()