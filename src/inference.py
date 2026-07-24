from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

from src.model import MODEL_NAME, add_style_token


def generate_samples(
    weights,
    prompt,
    outdir="samples",
    num_images=3,
    device=None,
    seed=42,
    instance_token="<sks>",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load the same Stable Diffusion 1.5 base model used for training.
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # Recreate the same custom token used during training.
    add_style_token(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        instance_token=instance_token,
        initializer_token="style",
    )

    # Load both the trained UNet and text-encoder LoRA weights.
    pipe.load_lora_weights(weights)

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(desc="Generating")

    # Use a separate deterministic seed for each generated image.
    for i in range(num_images):
        generator = torch.Generator(device=device).manual_seed(seed + i)

        image = pipe(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]

        output_path = outdir / f"sample_{i + 1}.png"
        image.save(output_path)

        print(
            f"Saved: {output_path} "
            f"(seed={seed + i})"
        )