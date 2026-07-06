from src.inference import generate_samples

generate_samples(
    weights="lora_out/pytorch_lora_weights.safetensors",
    prompt="a busy market, in <sks> style",
    outdir="samples",
    num_images=1,
    seed=42,
)