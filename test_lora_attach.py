import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
INSTANCE_TOKEN = "<sks>"
RANK = 8

print("Loading Stable Diffusion 1.5...")

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)

tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
unet = pipe.unet

# 1. Add <sks> token
num_added_tokens = tokenizer.add_tokens(INSTANCE_TOKEN)
if num_added_tokens == 0:
    raise ValueError(f"Token {INSTANCE_TOKEN} already exists.")

text_encoder.resize_token_embeddings(len(tokenizer))

print("Added token:", INSTANCE_TOKEN)

# 2. Freeze original model weights
unet.requires_grad_(False)
text_encoder.requires_grad_(False)

# 3. LoRA config for UNet
unet_lora_config = LoraConfig(
    r=RANK,
    lora_alpha=RANK,
    init_lora_weights="gaussian",
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
)

# 4. LoRA config for text encoder
text_encoder_lora_config = LoraConfig(
    r=RANK,
    lora_alpha=RANK,
    init_lora_weights="gaussian",
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
)

# 5. Attach LoRA adapters
unet.add_adapter(unet_lora_config)
text_encoder.add_adapter(text_encoder_lora_config)

print("LoRA attached to UNet.")
print("LoRA attached to text encoder.")

# 6. Count trainable parameters
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Trainable UNet parameters:", count_trainable_params(unet))
print("Trainable text encoder parameters:", count_trainable_params(text_encoder))
print("Total trainable parameters:", count_trainable_params(unet) + count_trainable_params(text_encoder))

print("LoRA setup successful.")