#!/bin/bash

# Script: train.sh
# Description: Training script for DeTikZify with curriculum learning
# Usage: ./train.sh [OPTIONS]

###################
# Helper Functions
###################

setup_environment() {
    # Set environment variables for distributed training
    export TOKENIZERS_PARALLELISM=false
    export RANK=0
    export WORLD_SIZE=1
    export MASTER_ADDR=localhost
    export MASTER_PORT=12355

    # Set GPU and logging configuration
    export CUDA_VISIBLE_DEVICES=2
    export HF_DATASETS_VERBOSITY=info
}

create_directories() {
    # Create necessary directories
    echo "Creating cache directories..."
    mkdir -p .cache/processed_dataset
    mkdir -p "$OUTPUT_DIR"
}

run_training() {
    echo "Starting training with curriculum learning..."
    echo "Configuration:"
    echo "- Base Model: $BASE_MODEL"
    echo "- Output Directory: $OUTPUT_DIR"
    echo "- Curriculum Type: $CURRICULUM_TYPE"
    echo "- Initial Data Fraction: $LAMBDA_0"
    echo "- Growth Period: $T_GROW epochs"
    
    python train.py \
        --base_model="$BASE_MODEL" \
        --projector="$PROJECTOR_DIR" \
        --output="$OUTPUT_DIR" \
        --gradient_checkpointing \
        --curriculum_type "$CURRICULUM_TYPE" \
        --curriculum_p "$P_VALUE" \
        --curriculum_lambda_0 "$LAMBDA_0" \
        --curriculum_t_grow "$T_GROW" \
        --dataset_path "$DATASET_PATH"
}

check_prerequisites() {
    # Check if required files and directories exist
    if [ ! -f "train.py" ]; then
        echo "Error: train.py not found!"
        exit 1
    }

    # Check if CUDA is available
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Warning: NVIDIA GPU drivers not found!"
    fi
}

###################
# Configuration
###################

# Model and training paths
BASE_MODEL="deepseek-ai/deepseek-coder-1.3b-base"
PROJECTOR_DIR="https://huggingface.co/nllg/detikzify-ds-1.3b/resolve/main/projector/model.safetensors"
OUTPUT_DIR="models/detikzify-v1"
DATASET_PATH="path/to/your/sorted_dataset.csv"  # Update this with your dataset path

# Curriculum learning parameters
CURRICULUM_TYPE="root-p"
P_VALUE=3.0
LAMBDA_0=0.3
T_GROW=3.0

###################
# Main Script
###################

# Print banner
echo "=================================="
echo "DeTikZify Training Script"
echo "=================================="
echo

# Check prerequisites
check_prerequisites

# Setup training environment
setup_environment

# Create necessary directories
create_directories

# Run the training
run_training

# Check for training errors
if [ $? -ne 0 ]; then
    echo "=================================="
    echo "Training failed!"
    echo "Check the logs above for errors."
    echo "=================================="
    exit 1
fi

# Success message
echo "=================================="
echo "Training completed successfully!"
echo "Model saved to: $OUTPUT_DIR"
echo "=================================="