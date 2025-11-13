#!/bin/bash
# GPU Environment Setup Script for Vast.ai (Cogumi-LLM)
# Last updated: 2025-11-10
set -e
pip install --upgrade pip
# 0. Uninstall conflicting packages
pip uninstall -y torch torchvision torchaudio torch xformers transformers \
    tokenizers psutil flash-attn bitsandbytes peft accelerate trl \
    datasets unsloth unsloth-zoo huggingface-hub tensorboard wandb \
    numpy scipy scikit-learn rich tqdm jsonlines ninja packaging \
    diffusers typer-slim shellingham


# 1. Packaging and core utilities
pip install psutil==6.0.0 packaging rich tensorboard 

# 2. PyTorch 2.6.0 + CUDA 12.4
pip install --force-reinstall torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 3. FlashAttention 2.7.4
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.7.4+cu124torch2.6-cp310-cp310-linux_x86_64.whl

# 4. bitsandbytes
pip install bitsandbytes==0.48.1

# 5. Transformers ecosystem
pip install transformers==4.43.3 tokenizers==0.19.1

# 6. Training frameworks
pip install peft==0.11.1 accelerate==1.11.0 trl==0.9.6 datasets==4.3.0

# 7. xformers
pip install xformers==0.0.29.post2 --no-deps


# 8. Unsloth
pip install "unsloth[cu124-torch260] @ git+https://github.com/unslothai/unsloth.git@July-2024" --no-deps
pip install einops sentencepiece	#Needed for some Llama/Qwen tokenizers and Unsloth zoo models.	
pip install unsloth-zoo	#Gives ready pre-quantized models like "unsloth/llama-3-8b-bnb-4bit".

# 9. Core training helpers
pip install tqdm jsonlines ninja

# 10. HuggingFace Hub + Safetensors
pip install huggingface-hub safetensors

# 11. Scientific computing libraries
pip install scipy==1.14.0 scikit-learn==1.4.2

pip install --upgrade torchao==0.13.0

# Done
