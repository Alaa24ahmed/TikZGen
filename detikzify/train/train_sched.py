import os
import math
from typing import Dict
from functools import cached_property
from io import BytesIO
from itertools import chain
from math import ceil, floor

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Trainer, 
    TrainerCallback, 
    TrainingArguments,
    logging
)
from transformers.trainer_utils import get_last_checkpoint

from ..util import convert, infer_device, SplitEpochSaveCallback
from .pretrain import DataCollatorForImageTextTraining, preprocess

# Configure logging and environment variables
logger = logging.get_logger("transformers")
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))

class ImageDataset(Dataset):
    """Dataset class that implements curriculum learning for image-text pairs."""
    
    def __init__(self, dataset, tokenizer):
        """
        Initialize the dataset with curriculum learning capabilities.
        
        Args:
            dataset: Base dataset containing image-text pairs and ranks
            tokenizer: Tokenizer for processing text and images
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.difficulty_threshold = 1.0
        self.current_size = len(dataset)
        self.total_size = len(dataset)
        
        self._ensure_sorted_dataset()
    
    def _ensure_sorted_dataset(self):
        """Ensure dataset is sorted by rank in ascending order."""
        ranks = self.dataset['rank']
        if not all(ranks[i] <= ranks[i+1] for i in range(len(ranks)-1)):
            logger.warning("Dataset is not properly sorted by rank! Sorting now...")
            self.dataset = self.dataset.sort('rank')
    
    def set_difficulty_threshold(self, threshold: float):
        """
        Update the dataset size based on the current difficulty threshold.
        
        Args:
            threshold: Float between 0 and 1 indicating percentage of data to use
        """
        self.difficulty_threshold = threshold
        prev_size = self.current_size
        self.current_size = max(1, int(threshold * self.total_size))
        
        self._log_progression(prev_size)
    
    def _log_progression(self, prev_size: int):
        """Log curriculum learning progression information."""
        new_examples = self.current_size - prev_size
        logger.info("\nCurriculum Learning Progress:")
        logger.info(f"|- Current threshold: {self.difficulty_threshold:.2f}")
        logger.info(f"|- Using examples: 0 to {self.current_size-1}")
        logger.info(f"|- Training set size: {self.current_size}/{self.total_size} examples")
        if new_examples > 0:
            logger.info(f"|- Added {new_examples} new examples")
        logger.info(f"|- Difficulty range: easiest to {self.difficulty_threshold*100:.1f}th percentile\n")

    def __len__(self) -> int:
        """Return the current size of the dataset based on difficulty threshold."""
        return self.current_size

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example from the dataset.
        
        Args:
            i: Index of the example to retrieve
            
        Returns:
            Dict containing input_ids, labels, and processed image
        """
        if i >= self.current_size:
            raise IndexError("Index out of bounds for current curriculum threshold")
        
        item = self.dataset[i]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "labels": torch.tensor(item["labels"]),
            "images": self.tokenizer.image(item['image'])
        }


class CurriculumScheduler:
    """Implements various curriculum learning scheduling strategies."""
    
    def __init__(
        self, 
        scheduler_type: str = 'root-p',
        p: float = 3.0,
        lambda_0: float = 0.3,
        T_grow: float = 3.0,
        n_buckets: int = 5
    ):
        """
        Initialize the curriculum scheduler.
        
        Args:
            scheduler_type: Type of scheduling strategy
            p: Power parameter for root-p scheduler
            lambda_0: Initial fraction of training data
            T_grow: Epochs until reaching full dataset
            n_buckets: Number of buckets for discrete schedulers
        """
        self.scheduler_type = scheduler_type
        self.p = p
        self.lambda_0 = lambda_0
        self.T_grow = T_grow
        self.n_buckets = n_buckets

    def get_threshold(self, epoch: float) -> float:
        """
        Calculate the difficulty threshold for the current epoch.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Float between 0 and 1 indicating the difficulty threshold
        """
        if self.scheduler_type == 'root-p':
            return self._root_p_threshold(epoch)
        elif self.scheduler_type == 'linear':
            return self._linear_threshold(epoch)
        elif self.scheduler_type == 'geometric':
            return self._geometric_threshold(epoch)
        elif self.scheduler_type == 'baby_step':
            return self._baby_step_threshold(epoch)
        return 1.0  # no curriculum

    def _root_p_threshold(self, epoch: float) -> float:
        return min(1.0, math.pow(
            (1 - math.pow(self.lambda_0, self.p)) / self.T_grow * epoch 
            + math.pow(self.lambda_0, self.p), 
            1/self.p
        ))

    def _linear_threshold(self, epoch: float) -> float:
        return min(1.0, self.lambda_0 + (1 - self.lambda_0) / self.T_grow * epoch)

    def _geometric_threshold(self, epoch: float) -> float:
        return min(1.0, math.pow(2, 
            epoch * math.log2(1 - math.log2(self.lambda_0)) / self.T_grow 
            + math.log2(self.lambda_0)
        ))

    def _baby_step_threshold(self, epoch: float) -> float:
        step = epoch / (self.T_grow / self.n_buckets)
        return min(1.0, (math.floor(step) + 1) / self.n_buckets)

class CurriculumLearningCallback(TrainerCallback):
    """Callback to update curriculum learning threshold at epoch boundaries."""
    
    def __init__(self, dataset: ImageDataset, scheduler_config: dict):
        """
        Initialize the callback.
        
        Args:
            dataset: The curriculum learning dataset
            scheduler_config: Configuration for the curriculum scheduler
        """
        self.dataset = dataset
        self.scheduler = CurriculumScheduler(**scheduler_config)
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Update difficulty threshold at the start of each epoch."""
        threshold = self.scheduler.get_threshold(state.epoch)
        self.dataset.set_difficulty_threshold(threshold)
        logger.info(f"Epoch {state.epoch}: difficulty threshold = {threshold:.3f}")

def prepare_dataset(dataset, tokenizer, model_config):
    """
    Prepare dataset for training with curriculum learning.
    
    Args:
        dataset: Raw dataset
        tokenizer: Tokenizer for processing text and images
        model_config: Model configuration
        
    Returns:
        Processed ImageDataset
    """
    patch_token = tokenizer.text.convert_ids_to_tokens(model_config.patch_token_id)
    max_len = tokenizer.text.model_max_length
    
    # Process and filter dataset
    processed_dataset = dataset.map(
        lambda exs, **kwargs: preprocess(exs['text'], **kwargs) | {
            "image": exs['image'],
            "rank": exs['rank']
        },
        batched=True,
        desc="Tokenize",
        fn_kwargs={
            "tokenizer": tokenizer.text,
            "num_patches": model_config.num_patches,
            "patch_token": patch_token,
            "truncation": False
        }
    )
    
    # Log dataset sizes
    logger.info(f"Dataset size before filtering: {len(processed_dataset)}")
    processed_dataset = processed_dataset.filter(
        lambda ex: len(ex['input_ids']) <= max_len and patch_token not in ex['text']
    )
    logger.info(f"Dataset size after filtering: {len(processed_dataset)}")
    
    return ImageDataset(tokenizer=tokenizer, dataset=processed_dataset)

def train(
    output_dir: str,
    model,
    tokenizer,
    dataset,
    curriculum_config: dict = {
        'scheduler_type': 'root-p',
        'p': 3,
        'lambda_0': 0.3,
        'T_grow': 3.0,
        'n_buckets': 5
    },
    overwrite: bool = False,
    deepspeed = None,
    batch_size: int = 128,
    micro_batch_size: int = 1,
    num_epochs: int = 3,
    learning_rate: float = 4e-5,
    gradient_checkpointing: bool = False,
    group_by_length: bool = False,
):
    """
    Train the model with curriculum learning.

    Args:
        output_dir: Directory to save model checkpoints
        model: The model to train
        tokenizer: Tokenizer for processing inputs
        dataset: Training dataset
        curriculum_config: Configuration for curriculum learning
        overwrite: Whether to overwrite existing checkpoints
        deepspeed: DeepSpeed configuration
        batch_size: Total batch size
        micro_batch_size: Batch size per device
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        gradient_checkpointing: Whether to use gradient checkpointing
        group_by_length: Whether to group sequences of similar length
    
    Returns:
        tuple: (trained model, tokenizer)
    """
    
    # Calculate gradient accumulation steps
    gradient_accumulation_steps = batch_size // micro_batch_size
    if ddp := WORLD_SIZE != 1:
        gradient_accumulation_steps = gradient_accumulation_steps // WORLD_SIZE

    # Handle checkpointing
    last_checkpoint = _handle_checkpoints(output_dir, overwrite)

    # Prepare dataset with curriculum learning
    processed_dataset = prepare_dataset(dataset, tokenizer, model.config)
    
    # Initialize curriculum learning callback
    curriculum_callback = CurriculumLearningCallback(
        dataset=processed_dataset,
        scheduler_config=curriculum_config
    )

    # Configure trainer
    trainer = _configure_trainer(
        model=model,
        dataset=processed_dataset,
        tokenizer=tokenizer,
        output_dir=output_dir,
        curriculum_callback=curriculum_callback,
        training_args=_get_training_args(
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            output_dir=output_dir,
            group_by_length=group_by_length,
            deepspeed=deepspeed,
            ddp=ddp
        )
    )

    # Train the model
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Handle DeepSpeed checkpointing if necessary
    if deepspeed:
        _handle_deepspeed_checkpoint(trainer, output_dir)

    # Save final model and training state
    trainer.save_model(output_dir)
    trainer.save_state()

    return model, tokenizer

def _handle_checkpoints(output_dir: str, overwrite: bool) -> str:
    """
    Handle existing checkpoints in the output directory.

    Args:
        output_dir: Directory to check for checkpoints
        overwrite: Whether to overwrite existing checkpoints

    Returns:
        str: Path to last checkpoint or None
    """
    if not os.path.isdir(output_dir) or overwrite:
        return None
        
    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint is None and len(os.listdir(output_dir)) > 0:
        raise ValueError(
            f"Output directory ({output_dir}) already exists and is not empty. "
            "Use `overwrite` to overcome."
        )
    elif last_checkpoint is not None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `output_dir` or add `overwrite` to train from scratch."
        )
    return last_checkpoint

def _get_training_args(
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    num_epochs: int,
    learning_rate: float,
    output_dir: str,
    group_by_length: bool,
    deepspeed,
    ddp: bool
) -> TrainingArguments:
    """
    Configure training arguments.

    Returns:
        TrainingArguments: Configured training arguments
    """
    return TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        warmup_ratio=0.03,
        weight_decay=0,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=10,
        lr_scheduler_type="cosine",
        optim="adamw_torch" if deepspeed else "adamw_torch_fused",
        save_strategy="epoch",
        save_total_limit=1,
        output_dir=output_dir,
        ddp_find_unused_parameters=False if ddp else None,
        remove_unused_columns=False,
        group_by_length=group_by_length,
        deepspeed=deepspeed,
    )

def _configure_trainer(
    model,
    dataset,
    tokenizer,
    output_dir: str,
    curriculum_callback: CurriculumLearningCallback,
    training_args: TrainingArguments
) -> Trainer:
    """
    Configure the trainer with all necessary components.

    Returns:
        Trainer: Configured trainer
    """
    return Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        callbacks=[
            curriculum_callback,
            SplitEpochSaveCallback(step_size=0.25)
        ],
        data_collator=DataCollatorForImageTextTraining(
            tokenizer=tokenizer.text,
            pad_to_multiple_of=8
        )
    )

def _handle_deepspeed_checkpoint(trainer: Trainer, output_dir: str):
    """Handle DeepSpeed checkpoint conversion to FP32."""
    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
    last_checkpoint = get_last_checkpoint(output_dir)
    load_state_dict_from_zero_checkpoint(trainer.model.float(), last_checkpoint)