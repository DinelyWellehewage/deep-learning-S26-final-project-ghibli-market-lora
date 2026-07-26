from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file

from src.model import (
    MODEL_NAME,
    TOKEN_EMBEDDING_KEY,
    add_style_token,
)


def load_combined_checkpoint(
    pipe,
    weights,
    instance_token_id,
):
    """
    Load the learned custom-token embedding and LoRA tensors from the
    same SafeTensors file.

    The custom embedding key is removed before the remaining LoRA state
    dictionary is passed to Diffusers.
    """

    weights = Path(weights)

    combined_state_dict = load_file(
        str(weights),
        device="cpu",
    )

    if (
        TOKEN_EMBEDDING_KEY
        not in combined_state_dict
    ):
        raise KeyError(
            f"The checkpoint does not contain "
            f"'{TOKEN_EMBEDDING_KEY}'. "
            "Retrain the model with learned style-token "
            "embedding support enabled."
        )

    learned_embedding = (
        combined_state_dict.pop(
            TOKEN_EMBEDDING_KEY
        )
    )

    embedding_weight = (
        pipe.text_encoder
        .get_input_embeddings()
        .weight
    )

    expected_shape = (
        embedding_weight[
            instance_token_id
        ].shape
    )

    if learned_embedding.shape != expected_shape:
        raise ValueError(
            "Style-token embedding shape mismatch. "
            f"Expected {tuple(expected_shape)}, "
            f"received "
            f"{tuple(learned_embedding.shape)}."
        )

    # Replace the initializer embedding with the learned embedding.
    with torch.no_grad():
        embedding_weight[
            instance_token_id
        ].copy_(
            learned_embedding.to(
                device=embedding_weight.device,
                dtype=embedding_weight.dtype,
            )
        )

    if not combined_state_dict:
        raise ValueError(
            "The checkpoint contains the style-token "
            "embedding but no LoRA tensors."
        )

    # Load only the remaining UNet and text-encoder LoRA tensors.
    pipe.load_lora_weights(
        combined_state_dict
    )

    print(
        f"Loaded learned style embedding from: "
        f"{weights}"
    )

    print(
        f"Loaded UNet and text-encoder LoRA "
        f"weights from: {weights}"
    )


def generate_samples(
    weights,
    prompt,
    outdir="samples",
    num_images=3,
    device=None,
    seed=42,
    instance_token="<sks>",
):
    weights = Path(weights)

    if not weights.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {weights}"
        )

    if not weights.is_file():
        raise ValueError(
            "--weights must point to a "
            f"SafeTensors file: {weights}"
        )

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print("Device:", device)

    outdir = Path(outdir)

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=(
            torch.float16
            if device == "cuda"
            else torch.float32
        ),
    )

    # Add the token and resize the embedding table before loading the
    # learned embedding.
    instance_token_id = add_style_token(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        instance_token=instance_token,
        initializer_token="style",
    )

    # Load the token embedding and both LoRA adapters from one file.
    load_combined_checkpoint(
        pipe=pipe,
        weights=weights,
        instance_token_id=instance_token_id,
    )

    pipe = pipe.to(device)

    pipe.set_progress_bar_config(
        desc="Generating"
    )

    for index in range(num_images):
        image_seed = seed + index

        generator = torch.Generator(
            device=device
        ).manual_seed(image_seed)

        image = pipe(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]

        output_path = (
            outdir
            / f"sample_{index + 1}.png"
        )

        image.save(output_path)

        print(
            f"Saved: {output_path} "
            f"(seed={image_seed})"
        )