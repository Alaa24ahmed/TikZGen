#!/usr/bin/env -S torchrun --nproc_per_node gpu
from argparse import ArgumentParser
from collections import defaultdict
from datetime import timedelta
from functools import partial
from itertools import count
from json import dump, load as load_json
from operator import itemgetter
from os import getenv
from os.path import isfile, join
from time import time

from datasets import load_dataset
from numpy import array
from scipy.stats.mstats import winsorize
from torch import bfloat16, distributed as dist, float16
from torch.cuda import is_available as is_cuda_available, is_bf16_supported
from tqdm import tqdm
from transformers import set_seed
from transformers.utils import is_flash_attn_2_available

from pathlib import Path
import json
from PIL import Image
import io

from detikzify.evaluate import (
    CrystalBLEU,
    KernelInceptionDistance,
    ImageSim,
    TexEditDistance,
    DreamSim,
)
from detikzify.infer import DetikzifyPipeline, TikzDocument
from detikzify.model import load as load_model
import os
import sys
import numpy as np
# Add module path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import os

# Set environment variables for pdf2image
os.environ['PATH'] = f"{os.environ['PATH']}:/home/nils.lukas/anaconda3/envs/Datikz/bin"
os.environ['PDFINFO'] = '/home/nils.lukas/anaconda3/envs/Datikz/bin/pdfinfo'
os.environ['POPPLER_PATH'] = '/home/nils.lukas/anaconda3/envs/Datikz/bin'

from difficulty_measure.image_similarity import compute_image_similarity_with_components
from difficulty_measure.code_similarity import tikz_code_similarity

WORLD_SIZE = int(getenv("WORLD_SIZE", 1))
RANK = int(getenv("RANK", 0))

def parse_args():
    argument_parser = ArgumentParser(
        description="Evaluate fine-tuned models."
    )
    argument_parser.add_argument(
        "--cache_dir",
        help="directory where model outputs should be saved to",
    )
    argument_parser.add_argument(
        "--trainset",
        required=True,
        help="path to the datikz train set (in parquet format)",
    )
    argument_parser.add_argument(
        "--testset",
        required=True,
        help="path to the datikz test set (in parquet format)",
    )
    argument_parser.add_argument(
        "--output",
        required=True,
        help="where to save the computed scores (as json)",
    )
    argument_parser.add_argument(
        "--timeout",
        type=int,
        help="minimum time to run MCTS in seconds",
    )
    argument_parser.add_argument(
        "--use_sketches",
        action="store_true",
        help="condition model on sketches instead of images",
    )
    argument_parser.add_argument(
        "--path",
        nargs='+',
        metavar="MODEL=PATH",
        required=True,
        help="(multiple) key-value pairs of model names and paths/urls to models/adapters (local or hub) or json files",
    )
    return argument_parser.parse_args()

# https://stackoverflow.com/a/54802737
def chunk(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]

def interleave(chunks):
    """Interleave chunks until one is exhausted."""
    interleaved = list()
    for idx in count():
        try:
            interleaved.extend(chunk[idx] for chunk in chunks)
        except IndexError:
            break
    return interleaved

def generate(pipe, image, strict=False, timeout=None, **tqdm_kwargs):
    """Run MCTS until the generated tikz code compiles."""
    start, success, tikzpics = time(), False, set()
    for score, tikzpic in tqdm(pipe.simulate(image=image), desc="Try", **tqdm_kwargs):
        tikzpics.add((score, tikzpic))
        if not tikzpic.compiled_with_errors if strict else tikzpic.is_rasterizable:
            success = True
        #too not spend too much time if there is no rastrized image
        if time() - start >= 900:
            return [tikzpic for _, tikzpic in sorted(tikzpics, key=itemgetter(0))]
        if success and (not timeout or time() - start >= timeout):
            break
    return [tikzpic for _, tikzpic in sorted(tikzpics, key=itemgetter(0))]

def predict(model_name, base_model, testset, cache_file=None, timeout=None, key="image"):
    predictions, worker_preds = list(), list()
    last_completed_index = 0
    
    # Create directory for cache file if it doesn't exist
    if cache_file and RANK == 0:
        cache_dir = os.path.dirname(cache_file)
        os.makedirs(cache_dir, exist_ok=True)
    
    # Load existing predictions if cache exists
    if cache_file and isfile(cache_file):
        with open(cache_file) as f:
            predictions = [[TikzDocument(code, timeout=None) for code in sample] for sample in load_json(f)]
            last_completed_index = len(predictions)

    model, tokenizer = load_model(
        base_model=base_model,
        device_map=RANK,
        torch_dtype=bfloat16 if is_cuda_available() and is_bf16_supported() else float16,
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    )

    metric_type = "model" if timeout else "fast"
    pipe = DetikzifyPipeline(model=model, tokenizer=tokenizer, metric=metric_type)

    try:
        # Get remaining items to process
        remaining_items = list(testset)[last_completed_index:]
        worker_chunk = list(chunk(remaining_items, WORLD_SIZE))[RANK]
        
        for i, item in enumerate(tqdm(worker_chunk, desc=f"{model_name.title()} ({RANK})", disable=RANK!=0)):
            tikz = generate(pipe, image=item[key], timeout=timeout, position=1, leave=False, disable=RANK!=0)
            worker_preds.append(tikz)
        del model, tokenizer, pipe
        
    finally:
        dist.all_gather_object(gathered:=WORLD_SIZE * [None], worker_preds)
        new_predictions = interleave(gathered)
        predictions.extend(new_predictions)
        
        # Save final results
        if cache_file and RANK == 0:
            with open(cache_file, 'w') as f:
                dump([[p.code for p in ps] for ps in predictions], f)
                
    return predictions

def load_metrics(trainset, measure_throughput=False, **kwargs):
    # Existing metrics initialization
    bleu = CrystalBLEU(corpus=trainset, **kwargs)
    eed = TexEditDistance(**kwargs)
    emdsim = ImageSim(mode="emd", **kwargs)
    cossim = ImageSim(**kwargs)
    dreamsim = DreamSim(**kwargs)
    kid = KernelInceptionDistance(**kwargs)

    def mean_token_efficiency(predictions, limit=0.05):
        samples = list()
        for preds in predictions:
            samples.append(len(preds[-1].code)/sum(len(pred.code) for pred in preds))
        return winsorize(array(samples), limits=limit).mean().item()

    def mean_sampling_throughput(predictions, limit=0.05):
        return winsorize(array(list(map(len, predictions))), limits=limit).mean().item()

    def compute(references, predictions):
        ref_code, pred_code = [[ref['code']] for ref in references], [pred[-1].code for pred in predictions]
        ref_image, pred_image = [ref['image'] for ref in references], [pred[-1].rasterize() for pred in predictions]
        # assert all(pred[-1].is_rasterizable for pred in predictions)

        # Initialize scores
        if measure_throughput:
            scores = {"MeanSamplingThroughput": mean_sampling_throughput(predictions=predictions)}
        else:
            scores = {"MeanTokenEfficiency": mean_token_efficiency(predictions=predictions)}

        # Compute existing metrics
        metrics = {
            bleu: partial(bleu.update, list_of_references=ref_code, hypotheses=pred_code),
            eed: partial(eed.update, target=ref_code, preds=pred_code),
            emdsim: lambda: [emdsim.update(img1=img1, img2=img2) for img1, img2 in zip(ref_image, pred_image) if img2 is not None],
            cossim: lambda: [cossim.update(img1=img1, img2=img2) for img1, img2 in zip(ref_image, pred_image) if img2 is not None],
            dreamsim: lambda: [dreamsim.update(img1=img1, img2=img2) for img1, img2 in zip(ref_image, pred_image) if img2 is not None],
            kid: lambda: [(kid.update(img1, True), kid.update(img2, False)) for img1, img2 in zip(ref_image, pred_image) if img2 is not None],
        }

        for metric, update in metrics.items():
            update()
            scores[str(metric)] = metric.compute()
            metric.reset()

        # Compute image similarity metrics
        image_metrics = []
        for img1, img2 in zip(ref_image, pred_image):
            if img2:
                img_metrics = compute_image_similarity_with_components(img1, img2)
                image_metrics.append(img_metrics)
            
        # Add mean image similarity scores
        scores.update({
            'mean_structural_score': np.mean([m['structural_score'] for m in image_metrics]),
            'mean_rmse': np.mean([m['rmse'] for m in image_metrics]),
            'mean_ssim': np.mean([m['ssim'] for m in image_metrics]),
            'mean_abs_diff': np.mean([m['abs_diff'] for m in image_metrics]),
            'mean_image_combined_score': np.mean([m['combined_score'] for m in image_metrics])
        })
        
        # Compute custom code similarity scores
        code_similarities = []
        for ref, pred in zip(ref_code, pred_code):
            code_metrics = tikz_code_similarity(ref[0], pred)
            code_similarities.append(code_metrics)
        
        # Add mean code similarity metrics
        code_sim_averages = {
            f"mean_{k}": np.mean([m[k] for m in code_similarities if isinstance(m[k], (int, float))])
            for k in ['command_similarity', 'coordinate_similarity', 'style_similarity', 
                     'sequence_similarity', 'edge_similarity', 'overall_similarity']
        }
        scores.update(code_sim_averages)

        return scores

    return compute

def save_predictions(predictions, output_dir, save_images=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model_preds in predictions.items():
        try:
            model_dir = output_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            code_output = []
            per_sample_metrics = []
            
            for i, pred_list in enumerate(model_preds):
                try:
                    # Just save the codes for each prediction
                    codes = [pred.code for pred in pred_list]
                    code_output.append(codes)
                    
                    best_pred = pred_list[-1]
                    
                    if save_images and best_pred.is_rasterizable:
                        try:
                            img = best_pred.rasterize()
                            img_path = model_dir / f"sample_{i}.png"
                            img.save(str(img_path))
                        except Exception as e:
                            print(f"Failed to save image for sample {i}: {e}")
                            
                except Exception as e:
                    print(f"Failed to process sample {i} for model {model_name}: {e}")
                    continue

            # Save all codes
            code_path = model_dir / "predictions.json"
            with open(code_path, 'w') as f:
                json.dump(code_output, f, indent=2)
                    
        except Exception as e:
            print(f"Failed to save predictions for model {model_name}: {e}")
            continue
        
if __name__ == "__main__":
    set_seed(0)
    dist.init_process_group(timeout=timedelta(days=3))
    args = parse_args()

    trainset = load_dataset("parquet", data_files=args.trainset, split="train")
    testset = load_dataset("parquet", data_files={"test": args.testset}, split="test").sort("caption") # type: ignore

    predictions = defaultdict(list)
    for model_name, path in map(lambda s: s.split('='), tqdm(args.path, desc="Predicting")):
        if path.endswith("json"):
            with open(path) as f:
                predictions[model_name] = [[TikzDocument(code, None) for code in sample] for sample in load_json(f)]
        else:
            cache_file = join(args.cache_dir, f'{model_name}.json') if args.cache_dir else None
            predictions[model_name] = predict(
                model_name=model_name,
                base_model=path,
                testset=testset,
                cache_file=cache_file,
                timeout=args.timeout,
                key="sketch" if args.use_sketches else "image"
            )
    output_dir=join(args.cache_dir, "predictions") if args.cache_dir else "predictions"
    if RANK == 0: # Scoring only on main process
        # Save predictions and images
        save_predictions(
            predictions=predictions,
            output_dir=output_dir,
            save_images=True
        )
        
        # Continue with metric computation

        scores = dict()
        metrics = load_metrics(trainset['code'], measure_throughput=args.timeout is not None, sync_on_compute=False) # type: ignore
        for model_name, prediction in tqdm(predictions.items(), desc="Computing metrics", total=len(predictions)):
            scores[model_name] = metrics(references=testset, predictions=prediction)
        # Save scores in the predictions directory
        scores_file = join(output_dir, "scores.json")
        with open(scores_file, "w") as file:
            dump(scores, file)