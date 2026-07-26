import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training import train_lora


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune Stable Diffusion v1.5 "
            "using UNet LoRA, text-encoder LoRA, "
            "and a learned style-token embedding."
        )
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the training images.",
    )

    parser.add_argument(
        "--instance_token",
        type=str,
        default="<sks>",
        help="Custom style token.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora_out",
        help="Directory used to save the checkpoint.",
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help="LoRA rank.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=800,
        help="Maximum number of training steps.",
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Training image resolution.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training batch size.",
    )

    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help="Save a checkpoint every N steps.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help=(
            "Training device. Automatically selected "
            "when omitted."
        ),
    )

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
        checkpointing_steps=(
            args.checkpointing_steps
        ),
        overwrite=args.overwrite,
        device=args.device,
    )


if __name__ == "__main__":
    main()