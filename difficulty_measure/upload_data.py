import pandas as pd
from datasets import Dataset, DatasetDict, Image as DsImage
from huggingface_hub import HfApi, login
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import io


import pandas as pd
from datasets import Dataset, DatasetDict, Image as DsImage
from huggingface_hub import HfApi, login
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import io
import gc
from itertools import islice

def create_dataset_card(csv_path, repo_name):
    """
    Create README.md for the dataset
    """
    df = pd.read_csv(csv_path)
    
    stats = {
        'total_samples': len(df),
        'avg_image_score': df['image_similarity'].mean(),
        'avg_code_score': df['code_similarity'].mean(),
        'avg_combined_score': df['combined_score'].mean(),
        'min_combined_score': df['combined_score'].min(),
        'max_combined_score': df['combined_score'].max()
    }
    
    readme = f"""
# TikZ Generation Curriculum Learning Dataset

## Dataset Description

### Overview
This dataset is specifically designed and decomposed for curriculum learning applications in image-to-tikzcode generation tasks. It contains evaluation metrics and comparisons between original TikZ diagrams and their machine-generated counterparts using the `nllg/detikzify-ds-1.3b` model, arranged in order of generation difficulty.

### Purpose
The primary purpose of this dataset is to facilitate curriculum learning strategies in training image-to-tikzcode generation models. By providing a difficulty-ranked dataset, it enables:
- Progressive learning from simple to complex examples
- Difficulty-aware training strategies
- Structured learning path development
- Performance evaluation across difficulty levels

### Evaluation Metrics and Ranking
The dataset includes three dissimilarity metrics (where 0 = identical, 1 = most dissimilar):

1. **Image Dissimilarity** (70% weight):
   - Measures visual differences between original and generated images
   - Range: 0 to 1 (0 = identical images, 1 = completely different)
   - Considers structural differences, edge detection, and complexity

2. **Code Dissimilarity** (30% weight):
   - Measures differences between original and generated TikZ code
   - Range: 0 to 1 (0 = identical code, 1 = completely different)
   - Based on code structure and content comparison

3. **Combined Score**:
   - Weighted average: 0.7 * image_dissimilarity + 0.3 * code_dissimilarity
   - Range: 0 to 1 (0 = perfect match, 1 = maximum difference)

### Dataset Statistics
- Total number of samples: {stats['total_samples']:,}
- Average image dissimilarity: {stats['avg_image_score']:.4f}
- Average code dissimilarity: {stats['avg_code_score']:.4f}
- Average combined dissimilarity: {stats['avg_combined_score']:.4f}
- Dissimilarity range: {stats['min_combined_score']:.4f} to {stats['max_combined_score']:.4f}

### Features
- **example_number**: Unique identifier for each example
- **combination_number**: Specific combination identifier within each example
- **image_score**: Dissimilarity score between original and generated images (0-1)
- **code_score**: Dissimilarity score between original and generated TikZ code (0-1)
- **combined_score**: Weighted combination of dissimilarity metrics
- **rank**: Normalized difficulty rank (0=easiest to 1=hardest)
- **original_image**: Original diagram in PNG format
- **generated_image**: Model-generated diagram in PNG format if there is
- **original_code**: Original TikZ code
- **generated_code**: Model-generated TikZ code

## Usage

### Loading the Dataset
```python
from datasets import load_dataset

dataset = load_dataset("{repo_name}")
"""
    return readme

from datasets import Dataset, DatasetDict, Image as DsImage, Features, Value
from huggingface_hub import HfApi, login
import os
from tqdm import tqdm
import pandas as pd
from PIL import Image
import gc
from pathlib import Path

def create_dataset_dict(row):
    """Create a single example dictionary with image paths instead of loaded images"""
    try:
        # Original paths must exist
        orig_img_path = str(row['original_image_path'])
        orig_code_path = str(row['original_code_path'])
        
        # Check if original files exist (these are required)
        if not os.path.exists(orig_img_path):
            print(f"Original image not found: {orig_img_path}")
            return None
            
        if not os.path.exists(orig_code_path):
            print(f"Original code not found: {orig_code_path}")
            return None

        # Generated paths are optional
        gen_img_path = str(row['generated_image_path']) if pd.notna(row['generated_image_path']) else None
        gen_code_path = str(row['generated_code_path']) if pd.notna(row['generated_code_path']) else None
        
        return {
            'example_number': int(row['example_number']),
            'combination_number': int(row['combination_number']) if row['combination_number'] != "main" else -1,
            'image_score': float(row['image_similarity']),
            'code_score': float(row['code_similarity']),
            'combined_score': float(row['combined_score']),
            'rank': float(row['rank']),
            'original_image': orig_img_path,
            'generated_image': gen_img_path if gen_img_path and os.path.exists(gen_img_path) else None,
            'original_code': open(orig_code_path, 'r').read(),
            'generated_code': open(gen_code_path, 'r').read() if gen_code_path and os.path.exists(gen_code_path) else ""
        }
    except Exception as e:
        print(f"Error processing row: {e}")
        print("Row content:", row)
        return None

def prepare_and_upload_dataset(csv_path, repo_name, token):
    """Prepare and upload dataset with efficient image handling"""
    # Login to Hugging Face
    login(token)
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    print(f"Total rows in CSV: {len(df)}")
    
    # Create list of examples with image paths
    examples = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        example = create_dataset_dict(row)
        if example:
            examples.append(example)
    
    print(f"\nSuccessfully processed {len(examples)} examples")
    
    if len(examples) == 0:
        raise ValueError("No valid examples found!")
    
    # Create features
    features = Features({
        'example_number': Value('int32'),
        'combination_number': Value('int32'),
        'image_score': Value('float32'),
        'code_score': Value('float32'),
        'combined_score': Value('float32'),
        'rank': Value('float32'),
        'original_image': DsImage(),
        'generated_image': DsImage(),  # Will handle None values
        'original_code': Value('string'),
        'generated_code': Value('string')
    })
    
    # Create dataset with image paths
    dataset = Dataset.from_dict({
        'example_number': [ex['example_number'] for ex in examples],
        'combination_number': [ex['combination_number'] for ex in examples],
        'image_score': [ex['image_score'] for ex in examples],
        'code_score': [ex['code_score'] for ex in examples],
        'combined_score': [ex['combined_score'] for ex in examples],
        'rank': [ex['rank'] for ex in examples],
        'original_image': [ex['original_image'] for ex in examples],
        'generated_image': [ex['generated_image'] if ex['generated_image'] else None for ex in examples],
        'original_code': [ex['original_code'] for ex in examples],
        'generated_code': [ex['generated_code'] for ex in examples]
    }, features=features)
    
    print(f"\nCreated dataset with {len(dataset)} examples")
    
    # Calculate approximate shard size based on original images
    total_size_bytes = sum(os.path.getsize(ex['original_image']) for ex in examples)
    recommended_shard_size = min(max(total_size_bytes // 10, 100_000_000), 1_000_000_000)
    shard_size_mb = f"{recommended_shard_size // 1_000_000}MB"
    print(f"\nUsing shard size: {shard_size_mb}")
    
    # Push to hub
    dataset.push_to_hub(
        repo_name,
        private=False,
        token=token,
        max_shard_size=shard_size_mb
    )
    
    return len(dataset)

def main():
    # Configuration
    CSV_PATH = "merged_results/merged_results.csv"
    REPO_NAME = "samahadhoud/decomposed-tikz-dataset-with-difficulty-measure-0-10"
    TOKEN = "YOUR_TOKEN"
    
    try:
        # Create dataset card
        print("Creating dataset card...")
        readme = create_dataset_card(CSV_PATH, REPO_NAME)
        
        # Save README locally
        with open("README.md", "w") as f:
            f.write(readme)
        
        # Prepare and upload dataset
        print("Processing and uploading dataset...")
        total_processed = prepare_and_upload_dataset(CSV_PATH, REPO_NAME, TOKEN)
        
        print(f"Successfully processed and uploaded {total_processed} examples")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()