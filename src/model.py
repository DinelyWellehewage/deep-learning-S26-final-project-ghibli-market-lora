import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig


MODEL_NAME = "runwayml/stable-diffusion-v1-5"

# Custom key used to store the learned <sks> embedding inside the same
# SafeTensors file as the UNet and text-encoder LoRA weights.
TOKEN_EMBEDDING_KEY = "sks_token_embedding"


def add_style_token(
    tokenizer,
    text_encoder,
    instance_token="<sks>",
    initializer_token="style",
):
    """
    Add a custom style token and initialize its embedding from an
    existing single-token embedding.

    This function is used during both training and inference. During
    inference, the initialized vector is replaced by the learned embedding
    stored in the checkpoint.
    """

    num_added_tokens = tokenizer.add_tokens([instance_token])

    if num_added_tokens != 1:
        raise ValueError(
            f"Could not add {instance_token}. "
            "The token may already exist in the tokenizer."
        )

    # Expand the text-encoder embedding table for the new token.
    text_encoder.resize_token_embeddings(len(tokenizer))

    instance_token_id = tokenizer.convert_tokens_to_ids(
        instance_token
    )

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

    # Initialize <sks> from the existing "style" embedding.
    with torch.no_grad():
        embedding_weight = (
            text_encoder.get_input_embeddings().weight
        )

        embedding_weight[instance_token_id].copy_(
            embedding_weight[initializer_token_id]
        )

    print(
        f"Added {instance_token} with token ID "
        f"{instance_token_id}. Initialized from "
        f"'{initializer_token}' "
        f"(token ID {initializer_token_id})."
    )

    return instance_token_id


def enable_style_token_training(
    text_encoder,
    instance_token_id,
):
    """
    Enable gradient updates for the embedding matrix while masking all
    rows except the custom style-token row.

    The returned parameter must be placed in an optimizer parameter group
    with weight_decay=0.0. Otherwise AdamW could modify the other rows
    through decoupled weight decay.
    """

    embedding_weight = (
        text_encoder.get_input_embeddings().weight
    )

    embedding_weight.requires_grad_(True)

    def mask_embedding_gradients(gradient):
        masked_gradient = torch.zeros_like(gradient)

        masked_gradient[instance_token_id].copy_(
            gradient[instance_token_id]
        )

        return masked_gradient

    embedding_weight.register_hook(
        mask_embedding_gradients
    )

    return embedding_weight


def create_lora_pipeline(
    instance_token="<sks>",
    rank=8,
    device="cpu",
    train_style_token=False,
):
    """
    Load Stable Diffusion v1.5 and attach LoRA adapters to both the
    UNet and CLIP text encoder.

    When train_style_token=True, the custom token embedding is also enabled
    for masked single-row training.
    """

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=(
            torch.float16
            if device == "cuda"
            else torch.float32
        ),
    )

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet

    instance_token_id = add_style_token(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        instance_token=instance_token,
        initializer_token="style",
    )

    # Freeze all original model parameters.
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)

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

    # Attach LoRA adapters after the base model has been frozen.
    unet.add_adapter(unet_lora_config)

    text_encoder.add_adapter(
        text_encoder_lora_config
    )

    token_embedding_weight = None

    # This must happen after text_encoder.requires_grad_(False).
    if train_style_token:
        token_embedding_weight = (
            enable_style_token_training(
                text_encoder=text_encoder,
                instance_token_id=instance_token_id,
            )
        )

    pipe = pipe.to(device)

    return (
        pipe,
        instance_token_id,
        token_embedding_weight,
    )


def count_lora_parameters(
    model,
    excluded_parameter=None,
):
    """
    Count trainable LoRA parameters while optionally excluding the
    token-embedding matrix.

    The complete embedding matrix has requires_grad=True, but only one row
    is effectively trained because of the gradient mask.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
        and parameter is not excluded_parameter
    )