import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from src.dataset import StyleImageDataset
from src.model import create_lora_pipeline, count_trainable_parameters
from peft import get_peft_model_state_dict
from diffusers.utils import convert_state_dict_to_diffusers


def train_lora(
    data_dir,
    instance_token="<sks>",
    output_dir="lora_out",
    rank=8,
    learning_rate=1e-4,
    max_steps=800,
    resolution=512,
    batch_size=1,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    dataset = StyleImageDataset(
        data_dir=data_dir,
        instance_token=instance_token,
        resolution=resolution,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    pipe = create_lora_pipeline(
        instance_token=instance_token,
        rank=rank,
        device=device,
    )

    print("Trainable UNet params:", count_trainable_parameters(pipe.unet))
    print("Trainable text encoder params:", count_trainable_parameters(pipe.text_encoder))

    print("Dataset size:", len(dataset))
    print("Training setup ready.")

        # Prepare optimizer: only LoRA parameters are trainable
    trainable_params = list(filter(lambda p: p.requires_grad, pipe.unet.parameters()))
    trainable_params += list(filter(lambda p: p.requires_grad, pipe.text_encoder.parameters()))

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
    )

    # Important: VAE is only used to encode images, so it stays in eval mode
    pipe.vae.eval()
    pipe.unet.train()
    pipe.text_encoder.train()

    global_step = 0
    progress_bar = tqdm(total=max_steps, desc="Training")

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            pixel_values = batch["pixel_values"].to(device)
            prompts = batch["prompt"]

            # 1. Encode image into latent space
            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

            # 2. Sample random noise
            noise = torch.randn_like(latents)

            # 3. Sample random diffusion timestep
            timesteps = torch.randint(
                0,
                pipe.scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            ).long()

            # 4. Add noise to latents
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # 5. Tokenize prompts
            tokenized = pipe.tokenizer(
                prompts,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            input_ids = tokenized.input_ids.to(device)

            # 6. Text encoder
            encoder_hidden_states = pipe.text_encoder(input_ids)[0]

            # 7. UNet predicts noise
            noise_pred = pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
            ).sample

            # 8. Loss
            loss = F.mse_loss(noise_pred.float(), noise.float())

            # 9. Update LoRA weights
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())

    progress_bar.close()

    print("Training completed.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    unet_lora_state_dict = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(pipe.unet)
    )

    text_encoder_lora_state_dict = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(pipe.text_encoder)
    )
    text_encoder_lora_state_dict = {
        k: v
        for k, v in text_encoder_lora_state_dict.items()
        if "token_embedding" not in k
    }

    pipe.save_lora_weights(
        save_directory=output_path,
        unet_lora_layers=unet_lora_state_dict,
        text_encoder_lora_layers=text_encoder_lora_state_dict,
        safe_serialization=True,
    )

    print(f"LoRA weights saved to: {output_path / 'pytorch_lora_weights.safetensors'}")