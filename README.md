# Ghibli Market LoRA

This project fine-tunes **Stable Diffusion v1.5** using **Low-Rank Adaptation (LoRA)** to learn a Studio Ghibli-inspired illustration style. The trained LoRA is activated using a custom token (`<sks>`) and can generate market scenes with the learned artistic style.

---

## Project Overview

The objective is to adapt a pretrained Stable Diffusion model to a new visual style while training only a small number of parameters using LoRA.

The project includes:

- Custom style token (`<sks>`)
- LoRA fine-tuning for the UNet and CLIP text encoder
- Training and evaluation scripts
- Deterministic inference
- Checkpoint saving during training

---

## Dataset

Training images are located in:

```
style_imgs/512/
```

- Resolution: **512 × 512**
- Number of images: **843**
- Caption used for training:

```
an illustration in <sks> style
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/DinelyWellehewage/deep-learning-S26-final-project-ghibli-market-lora.git
cd deep-learning-S26-final-project-ghibli-market-lora
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

Linux/macOS

```bash
source .venv/bin/activate
```

Windows

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Example training command:

```bash
python scripts/train_lora.py \
    --data_dir style_imgs/512 \
    --instance_token "<sks>" \
    --output_dir outputs/ghibli_lora \
    --rank 16 \
    --learning_rate 1e-4 \
    --max_steps 1600 \
    --checkpointing_steps 200 \
    --resolution 512 \
    --batch_size 1 \
    --overwrite
```

Training produces:

```
outputs/
└── ghibli_lora/
    ├── checkpoint-200/
    ├── checkpoint-400/
    ├── ...
    ├── checkpoint-1600/
    └── pytorch_lora_weights.safetensors
```

---

## Evaluation

Generate images using the trained LoRA:

```bash
python scripts/eval_lora.py \
    --weights outputs/ghibli_lora/pytorch_lora_weights.safetensors \
    --prompt "a busy market, in <sks> style" \
    --outdir outputs/market_samples \
    --num_images 10 \
    --seed 42
```

Example prompt:

```
a busy market, in <sks> style
```

---

## Model

Base model:

- Stable Diffusion v1.5

LoRA configuration:

- Rank: 16
- UNet LoRA
- CLIP Text Encoder LoRA

---

## Results

The trained LoRA successfully learns a Ghibli-inspired illustration style while preserving prompt content.

Example outputs include:

- Busy outdoor markets
- Fruit and vegetable stalls
- Crowded shopping streets
- Stylized anime-like characters

---

## Project Structure

```
.
├── scripts/
│   ├── train_lora.py
│   └── eval_lora.py
│
├── src/
│   ├── dataset.py
│   ├── inference.py
│   ├── model.py
│   └── training.py
│
├── style_imgs/
├── outputs/
├── requirements.txt
└── README.md
```

---

## Reproducibility

Training and inference use fixed random seeds to produce reproducible results whenever possible.

---

## Acknowledgements

- Hugging Face Diffusers
- PyTorch
- Stable Diffusion v1.5
- LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)