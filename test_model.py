import torch

from src.model import create_lora_pipeline, count_trainable_parameters


device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = create_lora_pipeline(
    instance_token="<sks>",
    rank=8,
    device=device,
)

unet_params = count_trainable_parameters(pipe.unet)
text_encoder_params = count_trainable_parameters(pipe.text_encoder)

print("Device:", device)
print("UNet trainable parameters:", unet_params)
print("Text encoder trainable parameters:", text_encoder_params)
print("Total trainable parameters:", unet_params + text_encoder_params)
print("Model setup successful.")