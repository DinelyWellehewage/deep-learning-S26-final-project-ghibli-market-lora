import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import generate_samples

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate images using trained UNet LoRA, "
            "text-encoder LoRA, and learned <sks> "
            "embedding weights."
        )
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help=(
            "Path to "
            "pytorch_lora_weights.safetensors."
        ),
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt used for image generation.",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="samples",
        help="Directory used to save generated images.",
    )

    parser.add_argument(
        "--num_images",
        type=int,
        default=3,
        help="Number of images to generate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )

    parser.add_argument(
        "--instance_token",
        type=str,
        default="<sks>",
        help="Custom style token.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help=(
            "Inference device. Automatically selected "
            "when omitted."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    generate_samples(
        weights=args.weights,
        prompt=args.prompt,
        outdir=args.outdir,
        num_images=args.num_images,
        device=args.device,
        seed=args.seed,
        instance_token=args.instance_token,
    )


if __name__ == "__main__":
    main()