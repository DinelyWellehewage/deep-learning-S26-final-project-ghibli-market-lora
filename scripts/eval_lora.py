from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.inference import generate_samples


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a busy market, in <sks> style")
    parser.add_argument("--outdir", type=str, default="samples")
    parser.add_argument("--num_images", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    generate_samples(
        weights=args.weights,
        prompt=args.prompt,
        outdir=args.outdir,
        num_images=args.num_images,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()