# Ghibli Market: LoRA Style-Tuning with Stable Diffusion 1.5

This project fine-tunes **Stable Diffusion v1.5** using **Low-Rank Adaptation (LoRA)** to generate **Studio Ghibli-inspired market scenes**. A custom style token, `<sks>`, is introduced to enable prompt-based control over the learned artistic style while keeping the original model parameters frozen.

---

## Team Members

- Welle Hewage Dinely Shanuka

---

## Project Structure

```
.
├── code/
│   ├── train_lora.py
│   └── eval_lora.py
├── src/
│   ├── dataset.py
│   ├── inference.py
│   ├── model.py
│   └── training.py
├── lora_out/
│   └── pytorch_lora_weights.safetensors
├── samples/
├── style_imgs/
├── requirements.txt
├── README.md
└── report.pdf
```

---

## Requirements

- Python 3.10+
- PyTorch 2.x
- CUDA-capable GPU (recommended)
- Hugging Face Diffusers
- PEFT
- Transformers
- Accelerate
- Safetensors

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Training

Train the LoRA adapters using:

```bash
python code/train_lora.py \
  --data_dir style_imgs/512 \
  --instance_token "<sks>" \
  --output_dir lora_out \
  --rank 16 \
  --learning_rate 1e-4 \
  --resolution 512 \
  --batch_size 1 \
  --max_steps 1600 \
  --checkpointing_steps 200 \
  --overwrite
```

The trained LoRA adapter will be saved as:

```
lora_out/pytorch_lora_weights.safetensors
```

---

## Evaluation

Generate sample images using the trained adapter:

```bash
python code/eval_lora.py \
  --weights lora_out/pytorch_lora_weights.safetensors \
  --prompt "a busy market, in <sks> style" \
  --outdir samples \
  --num_images 3 \
  --seed 42
```

Generated images will be saved in:

```
samples/
```

---

## Implementation

- Base model: Stable Diffusion v1.5
- LoRA applied to:
  - UNet attention layers
  - CLIP text encoder
- Custom style token: `<sks>`
- Dataset: 843 Studio Ghibli-inspired images
- Image resolution: 512 × 512
- LoRA rank: 16

---

## Runtime

The project was trained on an NVIDIA A100 GPU.

Typical training time:

- ~5 minutes for 1600 training steps on an A100 GPU.

---

## Deliverables

- `code/train_lora.py`
- `code/eval_lora.py`
- `lora_out/pytorch_lora_weights.safetensors`
- `samples/`
- `requirements.txt`
- `README.md`
- `report.pdf`

---

## References

- Hugging Face Diffusers: https://huggingface.co/docs/diffusers
- PEFT: https://github.com/huggingface/peft
- LoRA: Hu et al., 2021. https://arxiv.org/abs/2106.09685