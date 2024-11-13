#!/bin/bash

export TOKENIZERS_PARALLELISM=false
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12355
# Set paths
PROJECTOR_DIR="https://huggingface.co/nllg/detikzify-ds-1.3b/resolve/main/projector/model.safetensors"
OUTPUT_DIR="models/detikzify-v1"

# Set GPU and logging
export CUDA_VISIBLE_DEVICES=2
export HF_DATASETS_VERBOSITY=info

# Create cache directory for datasets
mkdir -p .cache/processed_dataset

# Run training with the correct Python interpreter
echo "Starting training..."
python train.py \
    --base_model="deepseek-ai/deepseek-coder-1.3b-base" \
    --projector=$PROJECTOR_DIR \
    --output=$OUTPUT_DIR \
    --gradient_checkpointing

# Check for errors
if [ $? -ne 0 ]; then
    echo "Training failed!"
    exit 1
fi

echo "Training completed successfully!"