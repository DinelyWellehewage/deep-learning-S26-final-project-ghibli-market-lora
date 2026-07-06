import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig


MODEL_NAME = "runwayml/stable-diffusion-v1-5"


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

    num_added_tokens = tokenizer.add_tokens(instance_token)

    if num_added_tokens == 0:
        raise ValueError(f"Token {instance_token} already exists in tokenizer.")

    text_encoder.resize_token_embeddings(len(tokenizer))

    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )

    text_encoder_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )

    unet.add_adapter(unet_lora_config)
    text_encoder.add_adapter(text_encoder_lora_config)

    pipe = pipe.to(device)

    return pipe


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)