#!/bin/bash
# Set distributed training environment variables
export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# Get the absolute path to the Datikz environment
DATIKZ_ENV_PATH="$HOME/.conda/envs/Datikz"
DATIKZ_PYTHON="$DATIKZ_ENV_PATH/bin/python"
DATIKZ_PIP="$DATIKZ_ENV_PATH/bin/pip"

# Ensure we're in the Datikz environment
eval "$(conda shell.bash hook)"
conda activate Datikz

# Add the current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Set GPU and logging
export CUDA_VISIBLE_DEVICES=0
export HF_DATASETS_VERBOSITY=info
export TORCHELASTIC_ERROR_FILE="/tmp/torch_elastic_error.json"


# Create cache directory for datasets
mkdir -p .cache/processed_dataset

# Run training with the correct Python interpreter
echo "Starting training..."
"$DATIKZ_PYTHON" -m torch.distributed.run \
    --nproc_per_node=1 \
    pretrain.py \
    --deepspeed ds_config.json \
    --gradient_checkpointing \

# Check for errors and display them if they exist
if [ $? -ne 0 ]; then
    echo "Training failed!"
    if [ -f "$TORCHELASTIC_ERROR_FILE" ]; then
        echo "Error details:"
        cat "$TORCHELASTIC_ERROR_FILE"
    fi
    exit 1
fi