import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig


MODEL_NAME = "runwayml/stable-diffusion-v1-5"


def add_style_token(
    tokenizer,
    text_encoder,
    instance_token="<sks>",
    initializer_token="style",
):
    """
    Add a custom style token and initialize it from an existing token.

    The same function must be used during training and evaluation so that
    <sks> has the same token ID and initial embedding in both pipelines.
    """

    # Add the new token to the tokenizer.
    num_added_tokens = tokenizer.add_tokens([instance_token])

    if num_added_tokens != 1:
        raise ValueError(
            f"Could not add {instance_token}. "
            "The token may already exist in the tokenizer."
        )

    # Expand the text encoder embedding table for the new token.
    text_encoder.resize_token_embeddings(len(tokenizer))

    instance_token_id = tokenizer.convert_tokens_to_ids(instance_token)

    # Obtain the token ID used to initialize <sks>.
    initializer_token_ids = tokenizer.encode(
        initializer_token,
        add_special_tokens=False,
    )

    if len(initializer_token_ids) != 1:
        raise ValueError(
            f"Initializer token '{initializer_token}' is represented by "
            f"{len(initializer_token_ids)} tokenizer tokens. "
            "Choose an initializer represented by exactly one token."
        )

    initializer_token_id = initializer_token_ids[0]

    # Copy the existing "style" embedding into the new <sks> embedding.
    with torch.no_grad():
        embeddings = text_encoder.get_input_embeddings().weight

        embeddings[instance_token_id].copy_(
            embeddings[initializer_token_id]
        )

    print(
        f"Added {instance_token} with token ID {instance_token_id}. "
        f"Initialized from '{initializer_token}' "
        f"(token ID {initializer_token_id})."
    )

    return instance_token_id


def create_lora_pipeline(
    instance_token="<sks>",
    rank=8,
    device="cpu",
):
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet

    # Add and initialize the custom style token.
    add_style_token(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        instance_token=instance_token,
        initializer_token="style",
    )

    # Freeze the original Stable Diffusion model parameters.
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)

    # LoRA configuration for the UNet.
    unet_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
        ],
    )

    # LoRA configuration for the text encoder.
    text_encoder_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
        ],
    )

    # Attach both LoRA adapters.
    unet.add_adapter(unet_lora_config)
    text_encoder.add_adapter(text_encoder_lora_config)

    pipe = pipe.to(device)

    return pipe


def count_trainable_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )