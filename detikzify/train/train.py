from functools import cached_property
from io import BytesIO
from itertools import chain
from math import ceil, floor
import os
from typing import Dict

from PIL import Image

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging

from ..util import convert, infer_device, SplitEpochSaveCallback
from .pretrain import DataCollatorForImageTextTraining, preprocess

logger = logging.get_logger("transformers")

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))

class ImageDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.dataset[i]
        return dict(
            input_ids=torch.tensor(item["input_ids"]),
            labels=torch.tensor(item["labels"]),
            images=self.tokenizer.image(item['image'])
        )

def train(
    output_dir: str,
    model,
    tokenizer,
    dataset,
    overwrite=False,
    deepspeed=None,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 1,
    num_epochs: int = 3,
    learning_rate: float = 4e-5,
    gradient_checkpointing: bool = False,
    group_by_length: bool = False,
):
    gradient_accumulation_steps = batch_size // micro_batch_size
    if ddp := WORLD_SIZE != 1:
        gradient_accumulation_steps = gradient_accumulation_steps // WORLD_SIZE

    def prepare_dataset(dataset):
        patch_token = tokenizer.text.convert_ids_to_tokens(model.config.patch_token_id)
        max_len = tokenizer.text.model_max_length
        
        dataset = dataset.map(
            lambda exs, **kwargs: preprocess(exs['text'], **kwargs) | {"image": exs['image']},
            batched=True,
            desc="Tokenize",
            fn_kwargs=dict(
                tokenizer=tokenizer.text,
                num_patches=model.config.num_patches,
                patch_token=patch_token,
                truncation=False
            )
        )
        logger.info(f"Dataset size before filtering out too long examples: {len(dataset)}")
        dataset = dataset.filter(lambda ex: len(ex['input_ids']) <= max_len and patch_token not in ex['text'])
        logger.info(f"Dataset size after filtering out too long examples: {len(dataset)}")
        return ImageDataset(tokenizer=tokenizer, dataset=dataset)

    last_checkpoint = None
    if os.path.isdir(output_dir) and not overwrite:
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

    trainer = Trainer(
        model=model,
        train_dataset=prepare_dataset(dataset),
        args=TrainingArguments(
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
        ),
        callbacks=[SplitEpochSaveCallback(step_size=0.25)],
        data_collator=DataCollatorForImageTextTraining(
            tokenizer=tokenizer.text,
            pad_to_multiple_of=8
        )
    )

    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=last_checkpoint)

    if deepspeed:
        from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
        last_checkpoint = get_last_checkpoint(output_dir)
        load_state_dict_from_zero_checkpoint(trainer.model.float(), last_checkpoint)

    trainer.save_model(output_dir)
    trainer.save_state()

    return model, tokenizer