import pandas as pd
import glob
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import logging
from tqdm import tqdm

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('merge_process.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def copy_file_safe(src, dst):
    """Safely copy a file with proper error handling"""
    try:
        if src and os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            return True
        return False
    except Exception as e:
        logging.error(f"Error copying {src} to {dst}: {e}")
        return False

def process_row(row, output_dirs):
    """Process a single row of the DataFrame"""
    try:
        original_dir, generated_dir = output_dirs
        example_num = row['example_number']
        comb_num = row['combination_number']

        def modify_path(path):
            if path and not pd.isna(path):
                if path.startswith(('generated', 'original')):
                    return os.path.join('merged_results', path)
            return path

        # Prepare all paths
        paths = {
            'orig_img': (modify_path(str(row['original_image_path'])),
                        os.path.join(original_dir, f"example_{example_num}_combination_{comb_num}.png")),
            'orig_code': (modify_path(str(row['original_code_path'])),
                         os.path.join(original_dir, f"example_{example_num}_combination_{comb_num}.tex")),
            'gen_img': (modify_path(str(row['generated_image_path'])) if row['generated_image_path'] else None,
                       os.path.join(generated_dir, f"example_{example_num}_combination_{comb_num}.png")),
            'gen_code': (modify_path(str(row['generated_code_path'])) if row['generated_code_path'] else None,
                        os.path.join(generated_dir, f"example_{example_num}_combination_{comb_num}.tex"))
        }

        # Handle special cases for generated paths
        if not paths['gen_img'][0] and paths['gen_code'][0] and not str(paths['gen_code'][0]).startswith(('generated', 'original')):
            paths['gen_code'] = (str(row['original_code_path']).replace('images', 'generated').replace('combination', 'generated'),
                               paths['gen_code'][1])

        if not paths['gen_img'][0] and str(paths['orig_img'][0]).startswith('merged_results'):
            paths['gen_code'] = (str(row['original_code_path']).replace('original', 'generated'), paths['gen_code'][1])
            paths['gen_img'] = (str(row['original_image_path']).replace('original', 'generated'), paths['gen_img'][1])

        # Copy all files
        results = {k: copy_file_safe(v[0], v[1]) for k, v in paths.items()}
        
        return {
            'example_number': example_num,
            'combination_number': comb_num,
            'copied_files': results
        }

    except Exception as e:
        logging.error(f"Error processing row {example_num}_{comb_num}: {e}")
        return None

def merge_and_organize_csv(input_pattern, output_dir, max_workers=8):
    logger = setup_logging()
    logger.info(f"Starting merge process with pattern: {input_pattern}")

    # Create output directories
    generated_dir = os.path.join(output_dir, 'generated')
    original_dir = os.path.join(output_dir, 'original')
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(original_dir, exist_ok=True)

    # Read and merge CSV files
    all_files = glob.glob(input_pattern)
    print(all_files)
    logger.info(f"Found {len(all_files)} CSV files to process")

    dfs = []
    for file in tqdm(all_files, desc="Reading CSV files"):
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=['example_number', 'combination_number'])

    # Process numerical columns
    merged_df['image_similarity'] = merged_df['image_similarity'].fillna(1).clip(upper=1)
    merged_df['code_similarity'] = merged_df['code_similarity'].fillna(1).clip(upper=1)

    merged_df['combined_score'] = 0.7 * merged_df['image_similarity'] + 0.3 * merged_df['code_similarity']
    
    # Sort by combined_score and add rank
    merged_df = merged_df.sort_values('combined_score', ascending=True)
    merged_df['rank'] = range(1, len(merged_df) + 1)
    
    merged_df = merged_df.fillna('')

    logger.info(f"Processing {len(merged_df)} unique combinations")

    # Process rows in parallel
    process_func = partial(process_row, output_dirs=(original_dir, generated_dir))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {executor.submit(process_func, row): row for _, row in merged_df.iterrows()}
        
        for future in tqdm(as_completed(future_to_row), total=len(future_to_row), desc="Processing files"):
            result = future.result()
            if result:
                logger.debug(f"Processed combination {result['example_number']}_{result['combination_number']}")

    # Update paths in DataFrame
    merged_df['original_image_path'] = merged_df.apply(
        lambda x: os.path.join(original_dir, f"example_{x['example_number']}_combination_{x['combination_number']}.png"), axis=1)
    merged_df['original_code_path'] = merged_df.apply(
        lambda x: os.path.join(original_dir, f"example_{x['example_number']}_combination_{x['combination_number']}.tex"), axis=1)
    merged_df['generated_image_path'] = merged_df.apply(
        lambda x: os.path.join(generated_dir, f"example_{x['example_number']}_combination_{x['combination_number']}.png")
        if os.path.exists(os.path.join(generated_dir, f"example_{x['example_number']}_combination_{x['combination_number']}.png")) 
        else '', axis=1)
    merged_df['generated_code_path'] = merged_df.apply(
        lambda x: os.path.join(generated_dir, f"example_{x['example_number']}_combination_{x['combination_number']}.tex")
        if os.path.exists(os.path.join(generated_dir, f"example_{x['example_number']}_combination_{x['combination_number']}.tex")) 
        else '', axis=1)

    # Save merged CSV
    output_csv = os.path.join(output_dir, 'merged_results.csv')
    merged_df.to_csv(output_csv, index=False)
    logger.info(f"Merged CSV created at {output_csv}")

    # Create HTML visualization of top results
    create_html_visualization(merged_df, output_dir)

    return len(merged_df)

def create_html_visualization(df, output_dir, n=50):
    """Create HTML visualization of top N and bottom N results"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .container { max-width: 1200px; margin: auto; padding: 20px; }
            .result-row { display: flex; margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; }
            .image-container { flex: 1; text-align: center; }
            .metrics { flex: 1; padding: 15px; }
            img { max-width: 100%; height: auto; }
            .rank { font-size: 24px; font-weight: bold; color: #333; }
            .section { margin-bottom: 50px; border-bottom: 3px solid #333; padding-bottom: 30px; }
            .section-title { color: #333; font-size: 28px; margin: 20px 0; }
            .good { background-color: #e6ffe6; }
            .bad { background-color: #ffe6e6; }
        </style>
    </head>
    <body>
        <div class="container">
    """

    # Add worst N results
    html_content += '<div class="section">'
    html_content += f'<h2 class="section-title">Bottom {n} Results (Worst Matches)</h2>'
    
    for _, row in df.head(n).iterrows():
        html_content += f"""
        <div class="result-row bad">
            <div class="image-container">
                <h3>Original</h3>
                <img src="{os.path.relpath(row['original_image_path'], output_dir)}">
            </div>
            <div class="image-container">
                <h3>Generated</h3>
                <img src="{os.path.relpath(row['generated_image_path'], output_dir) if row['generated_image_path'] else ''}">
            </div>
            <div class="metrics">
                <div class="rank">Rank: {row['rank']:.4f}</div>
                <p>Combined Score: {row['combined_score']:.4f}</p>
                <p>Image Similarity: {row['image_similarity']:.4f}</p>
                <p>Code Similarity: {row['code_similarity']:.4f}</p>
                <p>Example: {row['example_number']}</p>
                <p>Combination: {row['combination_number']}</p>
            </div>
        </div>
        """
    html_content += '</div>'

    # Add best N results
    html_content += '<div class="section">'
    html_content += f'<h2 class="section-title">Top {n} Results (Best Matches)</h2>'
    
    for _, row in df.tail(n).iterrows():
        html_content += f"""
        <div class="result-row good">
            <div class="image-container">
                <h3>Original</h3>
                <img src="{os.path.relpath(row['original_image_path'], output_dir)}">
            </div>
            <div class="image-container">
                <h3>Generated</h3>
                <img src="{os.path.relpath(row['generated_image_path'], output_dir) if row['generated_image_path'] else ''}">
            </div>
            <div class="metrics">
                <div class="rank">Rank: {row['rank']:.4f}</div>
                <p>Combined Score: {row['combined_score']:.4f}</p>
                <p>Image Similarity: {row['image_similarity']:.4f}</p>
                <p>Code Similarity: {row['code_similarity']:.4f}</p>
                <p>Example: {row['example_number']}</p>
                <p>Combination: {row['combination_number']}</p>
            </div>
        </div>
        """
    html_content += '</div>'

    html_content += """
        </div>
    </body>
    </html>
    """

    with open(os.path.join(output_dir, 'results_visualization.html'), 'w') as f:
        f.write(html_content)

    # Also create separate files for top and bottom results
    with open(os.path.join(output_dir, 'worst_results.html'), 'w') as f:
        f.write(html_content.split('<div class="section">')[1])
    
    with open(os.path.join(output_dir, 'best_results.html'), 'w') as f:
        f.write(html_content.split('<div class="section">')[2])

if __name__ == "__main__":
    input_pattern = '*.csv'
    output_dir = 'merged_results'
    
    total_rows = merge_and_organize_csv(input_pattern, output_dir)
    print(f"Merged CSV created with {total_rows} unique combinations")
    print(f"Files organized in {output_dir}")