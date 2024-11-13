#!/usr/bin/env -S torchrun --nproc_per_node gpu
from argparse import ArgumentParser
from itertools import chain
from os.path import basename, join
from re import sub
from multiprocessing import cpu_count

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import set_seed
from transformers.utils.logging import enable_explicit_format, set_verbosity_info

from detikzify.model import load
from detikzify.train import pretrain

def preprocess_batch(batch, patch_token):
    """Batch process examples for better performance."""
    texts = [" ".join(chain(
        [cap] if cap else [],
        chain.from_iterable(men or [[]]),
        ocr or []
    )) for cap, men, ocr in zip(batch.get("caption", []), batch.get("mention", []), batch.get("ocr", []))]
    
    return {
        "text": [sub(fr"\s*{patch_token}\s*", " ", text).strip() for text in texts],
        "image": batch['image']
    }

def load_and_process_dataset(name, model_size=None, **kwargs):
    """Load and process dataset with optimized settings."""
    dataset_mapping = {
        "paper2fig": "paper2fig",
        "scicap": "scicap",
        "datikz": "nllg/datikz-v2"
    }
    
    print(f"Loading {name} dataset...")
    # Load without streaming for faster processing
    dataset = load_dataset(
        dataset_mapping[name],
        split="train[:1%]",  # Load 1% for testing
        trust_remote_code=True,
        **kwargs
    )
    
    if name in ["paper2fig", "scicap"]:
        if model_size:
            kwargs['size'] = model_size
        print(f"Processing {name} with size={model_size}...")
        dataset = dataset.map(
            preprocess_batch,
            batched=True,
            batch_size=100,
            remove_columns=dataset.column_names,
            fn_kwargs={"patch_token": patch_token},
            num_proc=max(1, cpu_count() - 1),
            desc=f"Processing {name}"
        )
    elif name == "datikz":
        print("Processing datikz...")
        dataset = dataset.select_columns(["image", "caption"]) \
                        .rename_column("caption", "text") \
                        .filter(lambda ex: ex['text'])
    
    return dataset

def parse_args():
    argument_parser = ArgumentParser(
        description="Pretrain projection layer of DeTikZify."
    )
    argument_parser.add_argument("--base_model",
        default="deepseek-ai/deepseek-coder-1.3b-base",
        help="which base model to use",
    )
    argument_parser.add_argument("--output",
        default="models/projector",
        help="directory where to write the model files",
    )
    argument_parser.add_argument("--deepspeed",
        help="path to a DeepSpeed json config file",
    )
    argument_parser.add_argument("--gradient_checkpointing",
        action="store_true",
        help="use gradient checkpointing",
    )
    return argument_parser.parse_args()

if __name__ == "__main__":
    set_verbosity_info()
    enable_explicit_format()
    set_seed(0)

    args = parse_args()
    model, tokenizer = load(args.base_model)
    patch_token = tokenizer.text.convert_ids_to_tokens(model.config.patch_token_id)
    model_size = model.config.vision_config['input_size'][-1]

    # Load datasets with progress information
    print(f"Loading datasets with model size {model_size}...")
    paper2fig = load_and_process_dataset("paper2fig", model_size=model_size, size=model_size)
    # scicap = load_and_process_dataset("scicap", model_size=model_size, size=model_size)
    datikz = load_and_process_dataset("datikz")

    print(f"Paper2Fig100k: {len(paper2fig)}", 
        #   f"SciCap: {len(scicap)}", 
          f"DaTikZ: {len(datikz)}", 
          sep="\n")

    # Add error handling decorator
    from torch.distributed.elastic.multiprocessing.errors import record

    @record
    def main():
        print("Starting training...")
        pretrain(
            model=model,
            tokenizer=tokenizer,
            dataset=concatenate_datasets([paper2fig, 
                                        #   scicap, 
                                          datikz]),
            output_dir=join(args.output, basename(model.config.name_or_path)),
            gradient_checkpointing=args.gradient_checkpointing,
            deepspeed=args.deepspeed,
        )

    main()