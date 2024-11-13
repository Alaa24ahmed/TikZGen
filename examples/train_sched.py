#!/usr/bin/env -S torchrun --nproc_per_node gpu
"""
Training script for DeTikZify with curriculum learning.
"""

import os
from argparse import ArgumentParser
from os.path import basename, join
from typing import Tuple

import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, set_seed
from transformers.utils.logging import enable_explicit_format, set_verbosity_info

from detikzify.dataset import load_dataset
from detikzify.model import load
from detikzify.train import train

def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = ArgumentParser(description="Fine-tune DeTikZify on DaTikZ with curriculum learning.")
    
    # Model arguments
    parser.add_argument(
        "--base_model",
        default="deepseek-ai/deepseek-coder-1.3b-base",
        help="Base model to use for fine-tuning"
    )
    parser.add_argument(
        "--projector",
        help="URL or path to pretrained projector for CLIP soft prompts"
    )
    
    # Output and configuration arguments
    parser.add_argument(
        "--output",
        default="models/detikzify",
        help="Directory to save model files"
    )
    parser.add_argument(
        "--deepspeed",
        help="Path to DeepSpeed JSON config file"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing"
    )
    
    # Curriculum learning arguments
    parser.add_argument(
        "--curriculum_type",
        choices=['root-p', 'linear', 'geometric', 'baby_step', 'none'],
        default='root-p',
        help="Type of curriculum learning scheduler"
    )
    parser.add_argument(
        "--curriculum_p",
        type=float,
        default=3.0,
        help="Power parameter for root-p scheduler"
    )
    parser.add_argument(
        "--curriculum_lambda_0",
        type=float,
        default=0.3,
        help="Initial fraction of training data"
    )
    parser.add_argument(
        "--curriculum_t_grow",
        type=float,
        default=3.0,
        help="Epochs until reaching full dataset"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_path",
        default="path_to_your_sorted_data.csv",
        help="Path to the sorted CSV dataset file"
    )

    return parser.parse_args()

def setup_training_environment():
    """Configure the training environment."""
    set_verbosity_info()
    enable_explicit_format()
    dist.init_process_group()
    set_seed(0)

def load_and_prepare_model(args) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load and prepare the model and tokenizer.

    Args:
        args: Parsed command line arguments

    Returns:
        tuple: (model, tokenizer)
    """
    return load(args.base_model, pretrain_mm_mlp_adapter=args.projector)

def load_and_prepare_dataset(dataset_path: str) -> Dataset:
    """
    Load and prepare the dataset.

    Args:
        dataset_path: Path to the CSV dataset file

    Returns:
        Dataset: Processed dataset ready for training
    """
    dataset = load_dataset(
        'csv',
        data_files={'train': dataset_path},
        split='train'
    )
    return dataset.select_columns(["rank", "image", "code"]).rename_column("code", "text")

def get_curriculum_config(args) -> dict:
    """
    Create curriculum learning configuration.

    Args:
        args: Parsed command line arguments

    Returns:
        dict: Curriculum learning configuration
    """
    return {
        'scheduler_type': args.curriculum_type,
        'p': args.curriculum_p,
        'lambda_0': args.curriculum_lambda_0,
        'T_grow': args.curriculum_t_grow,
        'n_buckets': 5
    }

def main():
    """Main training function."""
    # Parse arguments and setup environment
    args = parse_args()
    setup_training_environment()

    # Load model and dataset
    model, tokenizer = load_and_prepare_model(args)
    dataset = load_and_prepare_dataset(args.dataset_path)

    # Configure curriculum learning
    curriculum_config = get_curriculum_config(args)

    # Start training
    output_dir = join(args.output, basename(model.config.name_or_path))
    train(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        curriculum_config=curriculum_config,
        num_epochs=args.curriculum_t_grow,
        output_dir=output_dir,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
    )

if __name__ == "__main__":
    main()