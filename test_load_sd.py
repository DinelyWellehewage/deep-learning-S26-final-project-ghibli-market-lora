import torch
from diffusers import StableDiffusionPipeline

MODEL_NAME = "runwayml/stable-diffusion-v1-5"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)
print("Loading model...")

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)

pipe = pipe.to(device)

print("Model loaded successfully.")
print("Tokenizer:", type(pipe.tokenizer))
print("Text encoder:", type(pipe.text_encoder))
print("UNet:", type(pipe.unet))
print("VAE:", type(pipe.vae))
print("Scheduler:", type(pipe.scheduler))
