from pathlib import Path
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusers.utils import (
    convert_state_dict_to_diffusers,
)
from peft import get_peft_model_state_dict
from safetensors.torch import load_file, save_file

from src.dataset import StyleImageDataset
from src.model import (
    TOKEN_EMBEDDING_KEY,
    count_lora_parameters,
    create_lora_pipeline,
)


def save_lora_checkpoint(
    pipe,
    save_directory,
    instance_token,
    instance_token_id,
):
    """
    Save the UNet LoRA weights, text-encoder LoRA weights, and learned
    custom token embedding in exactly one SafeTensors file.

    Output:
        pytorch_lora_weights.safetensors
    """

    save_directory = Path(save_directory)

    save_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    unet_lora_state_dict = (
        convert_state_dict_to_diffusers(
            get_peft_model_state_dict(
                pipe.unet
            )
        )
    )

    text_encoder_lora_state_dict = (
        convert_state_dict_to_diffusers(
            get_peft_model_state_dict(
                pipe.text_encoder
            )
        )
    )

    # Save the standard Diffusers LoRA checkpoint first.
    pipe.save_lora_weights(
        save_directory=save_directory,
        unet_lora_layers=unet_lora_state_dict,
        text_encoder_lora_layers=(
            text_encoder_lora_state_dict
        ),
        safe_serialization=True,
    )

    weights_path = (
        save_directory
        / "pytorch_lora_weights.safetensors"
    )

    if not weights_path.exists():
        raise FileNotFoundError(
            "Diffusers did not create the expected "
            f"checkpoint file: {weights_path}"
        )

    # Reopen the standard LoRA checkpoint.
    combined_state_dict = load_file(
        str(weights_path),
        device="cpu",
    )

    # Extract only the learned custom-token row.
    learned_embedding = (
        pipe.text_encoder
        .get_input_embeddings()
        .weight[instance_token_id]
        .detach()
        .cpu()
        .clone()
    )

    # Store it in the same state dictionary as the LoRA tensors.
    combined_state_dict[
        TOKEN_EMBEDDING_KEY
    ] = learned_embedding

    temporary_path = (
        save_directory
        / "pytorch_lora_weights.tmp.safetensors"
    )

    # Save to a temporary path first, then replace the original file.
    save_file(
        combined_state_dict,
        str(temporary_path),
        metadata={
            "instance_token": instance_token,
            "format": (
                "diffusers_lora_with_style_embedding"
            ),
        },
    )

    temporary_path.replace(weights_path)

    print(
        f"Checkpoint saved to: {weights_path}"
    )

    print(
        f"Included learned embedding for "
        f"{instance_token} under key "
        f"'{TOKEN_EMBEDDING_KEY}'."
    )


def train_lora(
    data_dir,
    instance_token="<sks>",
    output_dir="lora_out",
    rank=8,
    learning_rate=1e-4,
    max_steps=800,
    resolution=512,
    batch_size=1,
    checkpointing_steps=200,
    overwrite=False,
    device=None,
):
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print("Device:", device)

    seed = 42

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_path = Path(output_dir)

    if (
        output_path.exists()
        and any(output_path.iterdir())
    ):
        if overwrite:
            print(
                "Removing existing output directory: "
                f"{output_path}"
            )

            shutil.rmtree(output_path)

        else:
            raise FileExistsError(
                f"Output directory '{output_path}' "
                "already exists and is not empty. "
                "Use --overwrite to replace it."
            )

    dataset = StyleImageDataset(
        data_dir=data_dir,
        instance_token=instance_token,
        resolution=resolution,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    (
        pipe,
        instance_token_id,
        token_embedding_weight,
    ) = create_lora_pipeline(
        instance_token=instance_token,
        rank=rank,
        device=device,
        train_style_token=True,
    )

    if token_embedding_weight is None:
        raise RuntimeError(
            "Style-token training was requested, "
            "but the embedding parameter was not enabled."
        )

    # Use float32 during training for better numerical stability.
    if device == "cuda":
        pipe.vae.to(dtype=torch.float32)
        pipe.unet.to(dtype=torch.float32)
        pipe.text_encoder.to(dtype=torch.float32)

    unet_lora_params = [
        parameter
        for parameter in pipe.unet.parameters()
        if parameter.requires_grad
    ]

    text_encoder_lora_params = [
        parameter
        for parameter in pipe.text_encoder.parameters()
        if parameter.requires_grad
        and parameter is not token_embedding_weight
    ]

    unet_lora_count = count_lora_parameters(
        pipe.unet
    )

    text_encoder_lora_count = (
        count_lora_parameters(
            pipe.text_encoder,
            excluded_parameter=(
                token_embedding_weight
            ),
        )
    )

    # Only one embedding row is effectively trained.
    token_embedding_count = (
        token_embedding_weight.shape[1]
    )

    effective_trainable_count = (
        unet_lora_count
        + text_encoder_lora_count
        + token_embedding_count
    )

    print(
        "Trainable UNet LoRA params:",
        unet_lora_count,
    )

    print(
        "Trainable text-encoder LoRA params:",
        text_encoder_lora_count,
    )

    print(
        "Trainable style-token embedding params:",
        f"{token_embedding_count} "
        "(one gradient-masked row)",
    )

    print(
        "Effective trainable params:",
        effective_trainable_count,
    )

    print("Dataset size:", len(dataset))
    print("Training setup ready.")

    optimizer = torch.optim.AdamW(
        [
            {
                "params": unet_lora_params,
                "lr": learning_rate,
            },
            {
                "params": text_encoder_lora_params,
                "lr": learning_rate,
            },
            {
                "params": [
                    token_embedding_weight
                ],
                "lr": learning_rate,
                "weight_decay": 0.0,
            },
        ],
        lr=learning_rate,
    )

    pipe.vae.eval()
    pipe.unet.train()
    pipe.text_encoder.train()

    global_step = 0

    progress_bar = tqdm(
        total=max_steps,
        desc="Training",
    )

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            pixel_values = batch[
                "pixel_values"
            ].to(
                device=device,
                dtype=pipe.vae.dtype,
            )

            prompts = batch["prompt"]

            # 1. Encode images into latent space.
            with torch.no_grad():
                latent_distribution = (
                    pipe.vae
                    .encode(pixel_values)
                    .latent_dist
                )

                latents = (
                    latent_distribution.sample()
                )

                latents = (
                    latents
                    * pipe.vae.config.scaling_factor
                )

            # 2. Sample Gaussian noise.
            noise = torch.randn_like(latents)

            # 3. Sample random diffusion timesteps.
            timesteps = torch.randint(
                low=0,
                high=(
                    pipe.scheduler.config
                    .num_train_timesteps
                ),
                size=(latents.shape[0],),
                device=device,
            ).long()

            # 4. Add noise to the latent representations.
            noisy_latents = (
                pipe.scheduler.add_noise(
                    latents,
                    noise,
                    timesteps,
                )
            )

            # 5. Tokenize prompts.
            tokenized = pipe.tokenizer(
                prompts,
                padding="max_length",
                max_length=(
                    pipe.tokenizer.model_max_length
                ),
                truncation=True,
                return_tensors="pt",
            )

            input_ids = (
                tokenized.input_ids.to(device)
            )

            # 6. Encode prompts with CLIP.
            encoder_hidden_states = (
                pipe.text_encoder(input_ids)[0]
            )

            # 7. Predict noise with the UNet.
            noise_pred = pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
            ).sample

            # 8. Compute the noise-prediction loss.
            loss = F.mse_loss(
                noise_pred.float(),
                noise.float(),
            )

            # 9. Update LoRA parameters and only the <sks> row.
            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()
            optimizer.step()

            global_step += 1

            progress_bar.update(1)

            progress_bar.set_postfix(
                loss=f"{loss.item():.6f}"
            )

            if (
                checkpointing_steps is not None
                and checkpointing_steps > 0
                and global_step
                % checkpointing_steps
                == 0
            ):
                checkpoint_dir = (
                    output_path
                    / f"checkpoint-{global_step}"
                )

                save_lora_checkpoint(
                    pipe=pipe,
                    save_directory=checkpoint_dir,
                    instance_token=instance_token,
                    instance_token_id=(
                        instance_token_id
                    ),
                )

    progress_bar.close()

    print("Training completed.")

    save_lora_checkpoint(
        pipe=pipe,
        save_directory=output_path,
        instance_token=instance_token,
        instance_token_id=instance_token_id,
    )