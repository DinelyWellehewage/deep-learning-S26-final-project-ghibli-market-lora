from src.training import train_lora

train_lora(
    data_dir="style_imgs/512",
    instance_token="<sks>",
    output_dir="lora_out",
    rank=8,
    learning_rate=1e-4,
    # max_steps=800,
    max_steps=2,
    resolution=512,
    batch_size=1,
)
