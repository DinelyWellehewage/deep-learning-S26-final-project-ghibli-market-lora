from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


MODEL_NAME = "runwayml/stable-diffusion-v1-5"


def generate_samples(
    weights,
    prompt,
    outdir="samples",
    num_images=3,
    device=None,
    seed=42,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    pipe.load_lora_weights(weights)
    pipe = pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(seed)

    for i in range(num_images):
        image = pipe(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]

        output_path = outdir / f"sample_{i + 1}.png"
        image.save(output_path)
        print(f"Saved: {output_path}")