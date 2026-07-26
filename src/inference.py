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

    The checkpoint may also contain full embedding-layer tensors inserted
    automatically by PEFT because the tokenizer was resized. Those tensors
    are excluded before passing the state dictionary to Diffusers.
    """

    weights = Path(weights)

    combined_state_dict = load_file(
        str(weights),
        device="cpu",
    )

    if TOKEN_EMBEDDING_KEY not in combined_state_dict:
        raise KeyError(
            f"The checkpoint does not contain "
            f"'{TOKEN_EMBEDDING_KEY}'. "
            "Retrain the model with learned style-token "
            "embedding support enabled."
        )

    # Remove the custom embedding from the combined dictionary.
    learned_embedding = combined_state_dict.pop(
        TOKEN_EMBEDDING_KEY
    )

    embedding_weight = (
        pipe.text_encoder
        .get_input_embeddings()
        .weight
    )

    expected_shape = embedding_weight[
        instance_token_id
    ].shape

    if learned_embedding.shape != expected_shape:
        raise ValueError(
            "Style-token embedding shape mismatch. "
            f"Expected {tuple(expected_shape)}, "
            f"received {tuple(learned_embedding.shape)}."
        )

    # Replace the initial <sks> embedding with the learned embedding.
    with torch.no_grad():
        embedding_weight[
            instance_token_id
        ].copy_(
            learned_embedding.to(
                device=embedding_weight.device,
                dtype=embedding_weight.dtype,
            )
        )

    # PEFT may save full embedding-layer tensors when the tokenizer has
    # been resized. Diffusers load_lora_weights() must receive only LoRA
    # tensors, so retain only LoRA and DoRA-related entries.
    lora_state_dict = {
        key: value
        for key, value in combined_state_dict.items()
        if "lora" in key.lower()
        or "dora_scale" in key.lower()
    }

    if not lora_state_dict:
        available_keys = list(
            combined_state_dict.keys()
        )[:20]

        raise ValueError(
            "No LoRA tensors were found in the checkpoint. "
            f"First available keys: {available_keys}"
        )

    removed_keys = (
        len(combined_state_dict)
        - len(lora_state_dict)
    )

    if removed_keys > 0:
        print(
            f"Ignored {removed_keys} non-LoRA tensor(s) "
            "automatically stored by PEFT."
        )

    # Load only valid UNet and text-encoder LoRA tensors.
    pipe.load_lora_weights(
        lora_state_dict
    )

    print(
        f"Loaded learned style embedding from: "
        f"{weights}"
    )

    print(
        f"Loaded {len(lora_state_dict)} UNet and "
        f"text-encoder LoRA tensors from: {weights}"
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

    # Add <sks> and resize the text-encoder embedding table.
    instance_token_id = add_style_token(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        instance_token=instance_token,
        initializer_token="style",
    )

    # Load the learned embedding and both LoRA adapters.
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