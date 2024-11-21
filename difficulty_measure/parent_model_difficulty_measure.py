from operator import itemgetter
from detikzify.model import load
from detikzify.infer import DetikzifyPipeline
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import pytorch_ssim
import os
import json
import csv
from datetime import datetime
import re
import difflib
import glob
from code_similarity import tikz_code_similarity, TikZMetrics
from image_similarity import compare_images
import subprocess
from pdf2image import convert_from_path
from time import time
from tqdm import tqdm
from typing import List
import concurrent.futures
from functools import partial
import fcntl
from queue import Queue
import threading
def create_output_directory():
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # base_dir = f"tikz_results_{timestamp}"
    base_dir = "data_difficulty_results[5350:5370]"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "original"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "generated"), exist_ok=True)
    return base_dir

def load_pipeline():
    return DetikzifyPipeline(*load(
        base_model="nllg/detikzify-ds-1.3b",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ))


def save_image(image, path):
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)
    image.save(path)

def save_to_csv(results: list, output_dir: str):
    csv_path = os.path.join(output_dir, "results.csv")
    fieldnames = [
        'index', 'success', 'image_similarity', 'code_similarity',
        'combined_score', 'original_path', 'generated_path', 'tikz_path'
    ]
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def tikz_to_png(tikz_code, output_filename, output_folder):
    print(f"Converting TikZ code to PNG: {output_filename}")
    temp_tex = os.path.join(output_folder, 'temp.tex')
    with open(temp_tex, 'w') as f:
        f.write(tikz_code)
    
    try:
        # Compile LaTeX to PDF
        result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-output-directory', output_folder, temp_tex], 
                                capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            print("LaTeX compilation failed. Error message:")
            print(result.stdout)
            print(result.stderr)
            return False
        
        # Convert PDF to PNG using pdf2image
        temp_pdf = os.path.join(output_folder, 'temp.pdf')
        images = convert_from_path(temp_pdf, dpi=300)
        if images:
            png_path= os.path.join(output_folder, output_filename)
            images[0].save(png_path, 'PNG')
            print(f"Conversion complete: {output_filename}")
            return True
        else:
            print("Failed to convert PDF to PNG: No images generated")
            return False
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False
    finally:
        for ext in ['.aux', '.log', '.pdf', '.tex']:
            try:
                os.remove(os.path.join(output_folder, f'temp{ext}'))
            except FileNotFoundError:
                pass

def process_images_and_code(original_img_path, generated_img_path, original_code, generated_code):
    """
    Process images and code to compute similarities
    """
    
    image_metrics = compare_images(original_img_path, generated_img_path, visualize=False)

    code_metrics = tikz_code_similarity(original_code, generated_code)

    return image_metrics, code_metrics

def generate(pipe, image, strict=False, timeout=None, **tqdm_kwargs):
    """Run MCTS until the generated tikz code compiles."""
    start, success, tikzpics = time(), False, set()
    for score, tikzpic in tqdm(pipe.simulate(image=image), desc="Try", **tqdm_kwargs):
        tikzpics.add((score, tikzpic))
        if not tikzpic.compiled_with_errors if strict else tikzpic.is_rasterizable:
            success = True
        if time() - start >= 300:
            return [tikzpic for _, tikzpic in sorted(tikzpics, key=itemgetter(0))]
        if success and (not timeout or time() - start >= timeout):
            break
    return [tikzpic for _, tikzpic in sorted(tikzpics, key=itemgetter(0))]

def get_generated_image_and_code(png_path, comb_num, example_generated_dir, pipeline):
    try:
        with Image.open(png_path) as original_img:
            # Make a copy if pipeline needs the image after context exit
            img_copy = original_img.copy()
            RANK = 0
            fig = generate(pipeline, image=img_copy, timeout=None, position=1, leave=False, disable=RANK!=0)[0]

            if fig.is_rasterizable:
                generated_img = fig.rasterize()
                gen_img_path = os.path.join(example_generated_dir, f"generated_{comb_num}.png")
                save_image(generated_img, gen_img_path)
                return True, fig.code
            return False, fig.code
    except Exception as e:
        print(f"Error processing image {png_path}: {str(e)}")
        return False, None

def process_generated_files(output_dir: str, batch_size: int = None):
    """Process only the generated files in batches"""
    results = []
    
    # Get all example directories
    example_dirs = sorted(glob.glob(os.path.join(output_dir, "generated/example_*")))
    
    for example_dir in example_dirs:
        example_num = int(re.findall(r'example_(\d+)', example_dir)[0])
        print(f"Processing example {example_num}")
        
        # Get generated files
        generated_files = sorted(glob.glob(os.path.join(example_dir, "generated_*.tex")))
        
        # Process in batches if specified
        if batch_size:
            batches = [generated_files[i:i + batch_size] 
                      for i in range(0, len(generated_files), batch_size)]
        else:
            batches = [generated_files]
            
        for batch in batches:
            batch_results = process_batch(example_num, batch, output_dir)
            results.extend(batch_results)
            
    return results

def process_batch(example_num: int, batch: List[str], output_dir: str):
    """Process a batch of generated files"""
    batch_results = []
    
    for gen_code_path in batch:
        try:
            comb_num = int(re.findall(r'generated_(\d+)', gen_code_path)[0])
            
            # Construct paths
            orig_code_path = os.path.join(output_dir, f"images/example_{example_num}/combination_{comb_num}.tex")
            orig_img_path = os.path.join(output_dir, f"images/example_{example_num}/combination_{comb_num}.png")
            gen_img_path = os.path.join(output_dir, f"generated/example_{example_num}/generated_{comb_num}.png")
            
            if all(os.path.exists(p) for p in [orig_code_path, orig_img_path, gen_img_path]):
                # Read codes
                with open(orig_code_path, 'r') as f:
                    orig_code = f.read()
                with open(gen_code_path, 'r') as f:
                    gen_code = f.read()
                
                # Calculate similarities
                image_similarity = compare_images(orig_img_path, gen_img_path, visualize=False)["combined_score"]
                
                # metrics_calculator = TikZMetrics(corpus=[orig_code, gen_code])
                metrics_calculator.corpus.extend([orig_code, gen_code])
                code_similarity = 1 - metrics_calculator.compute_metrics(orig_code, gen_code)["final_weighted_score"]
                
                combined_score = 0.7 * image_similarity + 0.3 * code_similarity
                
                result = {
                    'example_number': example_num,
                    'combination_number': comb_num,
                    'image_similarity': image_similarity,
                    'code_similarity': code_similarity,
                    'combined_score': combined_score,
                    'original_image_path': orig_img_path,
                    'generated_image_path': gen_img_path,
                    'original_code_path': orig_code_path,
                    'generated_code_path': gen_code_path
                }
                
                batch_results.append(result)
                print(f"  Processed combination {comb_num}:")
                print(f"    Image Similarity: {image_similarity:.4f}")
                print(f"    Code Similarity: {code_similarity:.4f}")
                print(f"    Combined Score: {combined_score:.4f}")
                
        except Exception as e:
            print(f"Error processing combination {comb_num} in example {example_num}: {str(e)}")
            continue
            
    return batch_results


def process_combination(pipeline, example_num, example_images_dir, example_generated_dir, png_path, code_folder, metrics_calculator):
    
    # First try to get combination number from filename
    comb_match = re.findall(r'combination_(\d+)', png_path)
    if comb_match:
        comb_num = int(comb_match[0])
    else:
        # Check if it's the main combination file
        if "combination_main.png" in png_path:
            comb_num = "main"
        else:
            return None

    if os.path.exists(os.path.join('merged_results_2/generated', f'example_{example_num}_combination_{comb_num}.tex')):
        print(f'example_{example_num}_combination_{comb_num} already in merged')
        return None
    
    if os.path.exists(os.path.join(example_generated_dir, f"generated_{comb_num}.tex")):
        print(f'example_{example_num}_combination_{comb_num} already in generated')
        return None
    
    print(f"processing combination {comb_num} in example {example_num}")
    
    # Adjust code path based on combination number
    if comb_num == "main":
        code_path = os.path.join(code_folder, "combination_main.tex")
    else:
        code_path = os.path.join(code_folder, f"combination_{comb_num}.tex")
    
    try:   
        if not os.path.exists(code_path):
            return None
            
        # Load and save original image using context manager
        with Image.open(png_path) as original_img:
            orig_save_path = os.path.join(example_images_dir, f"combination_{comb_num}.png")
            original_img.save(orig_save_path)
        
        # Read and save original code
        with open(code_path, 'r') as f:
            code = f.read()
        orig_code_path = os.path.join(example_images_dir, f"combination_{comb_num}.tex")
        with open(orig_code_path, 'w') as f:
            f.write(code)
        
        # Generate new image and code
        rasterized, rasterized_code = get_generated_image_and_code(png_path, comb_num, example_generated_dir, pipeline)
        
        if rasterized:
            gen_img_path = os.path.join(example_generated_dir, f"generated_{comb_num}.png")
            image_similarity = compare_images(orig_save_path, gen_img_path, visualize=False)["combined_score"]
        else:
            gen_img_path = None
            image_similarity = 1
            
        gen_code_path = os.path.join(example_generated_dir, f"generated_{comb_num}.tex")
        with open(gen_code_path, 'w') as f:
            f.write(rasterized_code)
        
        metrics_calculator.corpus.extend([code, rasterized_code])
        code_similarity = 1 - metrics_calculator.compute_metrics(code, rasterized_code)["final_weighted_score"]
        combined_score = 0.7 * image_similarity + 0.3 * code_similarity
        
        return {
            'example_number': example_num,
            'combination_number': str(comb_num),  # Convert to string to handle "main"
            'image_similarity': image_similarity,
            'code_similarity': code_similarity,
            'combined_score': combined_score,
            'original_image_path': orig_save_path,
            'generated_image_path': gen_img_path if rasterized else None,
            'original_code_path': orig_code_path,
            'generated_code_path': gen_code_path
        }
        
    except Exception as e:
        print(f"Error processing combination {comb_num} in example {example_num}: {str(e)}")
        return None


def append_to_csv(result, csv_path):
    """Append a single result to the CSV file with file locking"""
    fieldnames = [
        'example_number',
        'combination_number',
        'image_similarity',
        'code_similarity',
        'combined_score',
        'original_image_path',
        'generated_image_path',
        'original_code_path',
        'generated_code_path'
    ]
    
    file_exists = os.path.exists(csv_path)
    mode = 'a' if file_exists else 'w'
    
    with open(csv_path, mode, newline='') as csvfile:
        # Acquire an exclusive lock
        fcntl.flock(csvfile.fileno(), fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(result)
            csvfile.flush()  # Ensure the write is committed
        finally:
            # Release the lock
            fcntl.flock(csvfile.fileno(), fcntl.LOCK_UN)



def main(process_type: str = 'all', batch_size: int = None, max_workers: int = 1):
    """
    Main function with processing options
    
    Args:
        process_type: 'all' or 'generated' (process all files or just generated ones)
        batch_size: Number of files to process in each batch (None for all at once)
    """
    if process_type == 'all':
        base_path = "data/"
        output_dir = create_output_directory()
        
        images_dir = os.path.join(output_dir, "images")
        generated_dir = os.path.join(output_dir, "generated")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(generated_dir, exist_ok=True)
        
        pipeline = load_pipeline()
        results = []
        
        example_folders = sorted(glob.glob(os.path.join(base_path, "example_*")), 
                               key=lambda x: int(x.split('_')[-1]))
        
        print(f"Found {len(example_folders)} example folders: {example_folders}")
        if not example_folders:
            print(f"No example folders found in {base_path}")
            return
        metrics_calculator = TikZMetrics()
        for example_folder in example_folders:
            example_num = int(example_folder.split('_')[-1])
            print(f"Processing example {example_num}")
            
            example_images_dir = os.path.join(images_dir, f"example_{example_num}")
            example_generated_dir = os.path.join(generated_dir, f"example_{example_num}")
            os.makedirs(example_images_dir, exist_ok=True)
            os.makedirs(example_generated_dir, exist_ok=True)
            
            # png_folder = os.path.join(example_folder, "png")
            png_folder = os.path.join(example_folder)

            # code_folder = os.path.join(example_folder, "code")
            code_folder = os.path.join(example_folder)

            
            print(f"Looking for PNG files in: {png_folder}")
            print(f"Looking for code files in: {code_folder}")

            if not os.path.exists(png_folder):
                print(f"PNG folder does not exist: {png_folder}")
                continue
            if not os.path.exists(code_folder):
                print(f"Code folder does not exist: {code_folder}")
                continue

            combination_files = glob.glob(os.path.join(png_folder, "combination_*.png"))
            
            print(f"Found {len(combination_files)} combination files")
            if not combination_files:
                print(f"No combination files found in {png_folder}")
                continue
            
            # Create a partial function with fixed arguments
            process_func = partial(
                process_combination,
                pipeline,
                example_num,
                example_images_dir,
                example_generated_dir,
                code_folder=code_folder,
                metrics_calculator = metrics_calculator
            )

            csv_path = os.path.join(output_dir, "results.csv")

        
            # Create a thread-safe queue for results
            result_queue = Queue()
            
            # Create a separate thread for CSV writing
            def csv_writer():
                while True:
                    result = result_queue.get()
                    if result is None:  # Sentinel value to stop the thread
                        break
                    append_to_csv(result, csv_path)
                    result_queue.task_done()
            
            csv_writer_thread = threading.Thread(target=csv_writer)
            csv_writer_thread.start()
            
            print(f"Processing {len(combination_files)} combinations with {max_workers} workers")
            
            # Process combinations in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_func, png_path) for png_path in combination_files]
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is not None:
                        result_queue.put(result)
                        print(f"Processed combination {result['combination_number']}:")
                        print(f"  Image Similarity: {result['image_similarity']:.4f}")
                        print(f"  Code Similarity: {result['code_similarity']:.4f}")
                        print(f"  Combined Score: {result['combined_score']:.4f}")
            
            # Signal the CSV writer thread to stop
            result_queue.put(None)
            csv_writer_thread.join()
            
        # Read all results from CSV for sorting and visualization
        results = []
        with open(csv_path, 'r', newline='') as csvfile:
            fcntl.flock(csvfile.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
            try:
                reader = csv.DictReader(csvfile)
                results = list(reader)
            finally:
                fcntl.flock(csvfile.fileno(), fcntl.LOCK_UN)


    elif process_type == 'generated':
        # Process only generated files
        csv_path = os.path.join(output_dir, "results.csv")
        output_dir = "path_to_existing_output_directory"  # Specify the existing output directory
        results = process_generated_files(output_dir, batch_size)
        for result in results:
            append_to_csv(result, csv_path)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "results.csv")
    fieldnames = [
        'example_number',
        'combination_number',
        'image_similarity',
        'code_similarity',
        'combined_score',
        'original_image_path',
        'generated_image_path',
        'original_code_path',
        'generated_code_path'
    ]
    
    # with open(csv_path, 'w', newline='') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for result in results:
    #         writer.writerow(result)

    # Sort results by combined score (ascending)
    sorted_results = sorted(results, key=lambda x: x['combined_score'])
    
    # Calculate the number of digits needed based on total images
    num_images = len(sorted_results)
    padding_width = len(str(num_images))  # This gets the number of digits in total count
    
    # Create sorted output directories
    sorted_orig_dir = os.path.join(output_dir, "sorted_original")
    sorted_gen_dir = os.path.join(output_dir, "sorted_generated")
    os.makedirs(sorted_orig_dir, exist_ok=True)
    os.makedirs(sorted_gen_dir, exist_ok=True)
    # Save sorted images with rank prefix
    for rank, result in enumerate(sorted_results, 1):
        # Get original file names
        orig_name = os.path.basename(result['original_image_path'])
        gen_name = os.path.basename(result['generated_image_path'])
        
        # Create new filenames with dynamic rank prefix
        new_orig_name = f"{rank:0{padding_width}d}_{orig_name}_diff{result['combined_score']}.png"
        new_gen_name = f"{rank:0{padding_width}d}_{gen_name}_diff{result['combined_score']}.png"
        
        # Copy images to sorted directories using context managers
        with Image.open(result['original_image_path']) as orig_img:
            orig_img.save(os.path.join(sorted_orig_dir, new_orig_name))
        
        if result['generated_image_path']:
            try:
                with Image.open(result['generated_image_path']) as gen_img:
                    gen_img.save(os.path.join(sorted_gen_dir, new_gen_name))
            except Exception as e:
                print(f"Error with generated image for {gen_name}: {str(e)}")

        # Update paths in results
        result['sorted_original_path'] = os.path.join(sorted_orig_dir, new_orig_name)
        result['sorted_generated_path'] = os.path.join(sorted_gen_dir, new_gen_name)
    # Update CSV with sorted paths
    fieldnames.extend(['sorted_original_path', 'sorted_generated_path'])
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted_results:
            writer.writerow(result)

    # Create an HTML visualization of the sorted results
    html_path = os.path.join(output_dir, "sorted_results.html")
    with open(html_path, 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <style>
        .image-pair {
            display: flex;
            margin-bottom: 20px;
            border-bottom: 1px solid #ccc;
            padding-bottom: 20px;
        }
        .image-container {
            flex: 1;
            margin: 10px;
            text-align: center;
        }
        img {
            max-width: 100%;
            height: auto;
        }
        .metrics {
            margin-top: 10px;
            font-family: monospace;
        }
    </style>
</head>
<body>
        ''')
        
        for result in sorted_results:
            f.write(f'''
    <div class="image-pair">
        <div class="image-container">
            <h3>Original</h3>
            <img src="{os.path.relpath(result['sorted_original_path'], output_dir)}">
        </div>
        <div class="image-container">
            <h3>Generated</h3>
            <img src="{os.path.relpath(result['sorted_generated_path'], output_dir)}">
        </div>
        <div class="metrics">
            <p>Combined Score: {result['combined_score']:.4f}</p>
            <p>Image Similarity: {result['image_similarity']:.4f}</p>
            <p>Code Similarity: {result['code_similarity']:.4f}</p>
        </div>
    </div>
            ''')
        
        f.write('''
</body>
</html>
        ''')

if __name__ == "__main__":
    main()
