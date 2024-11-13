import pandas as pd
import glob
import os
import shutil

def merge_and_organize_csv(input_pattern, output_dir):
    # Create output directories
    generated_dir = os.path.join(output_dir, 'generated')
    original_dir = os.path.join(output_dir, 'original')
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(original_dir, exist_ok=True)
    
    # Read all CSV files
    all_files = glob.glob(input_pattern)
    dfs = []
    for file in all_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    # Merge all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates based on example_number and combination_number
    merged_df = merged_df.drop_duplicates(subset=['example_number', 'combination_number'])
    
    # Copy files to new locations
    for _, row in merged_df.iterrows():
        def modify_path(path):
            if pd.notna(path):  # Check if path is not NaN
                if path.startswith(('generated', 'original')):
                    return os.path.join('merged_results', path)
            return path

        # Apply the path modification
        orig_img = modify_path(row['original_image_path'])
        orig_code = modify_path(row['original_code_path'])

        # Generated files (optional)
        gen_img = modify_path(row['generated_image_path']) if pd.notna(row['generated_image_path']) else None
        gen_code = modify_path(row['generated_code_path']) if pd.notna(row['generated_code_path']) else None

        if not gen_img and not row['generated_code_path'].startswith(('generated', 'original')):
            gen_code = row['original_code_path'].replace('images', 'generated').replace('combination', 'generated')
        
        # New paths
        new_orig_img = os.path.join(original_dir, f"example_{row['example_number']}_combination_{row['combination_number']}.png")
        new_orig_code = os.path.join(original_dir, f"example_{row['example_number']}_combination_{row['combination_number']}.tex")
        new_gen_img = os.path.join(generated_dir, f"example_{row['example_number']}_combination_{row['combination_number']}.png")
        new_gen_code = os.path.join(generated_dir, f"example_{row['example_number']}_combination_{row['combination_number']}.tex")
        
        # Copy files if they exist
        for src, dst in [(orig_img, new_orig_img), 
                        (orig_code, new_orig_code)]:
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"Warning: Required original file not found: {src}")
        
        # Copy generated files only if they exist
        for src, dst in [(gen_img, new_gen_img), 
                        (gen_code, new_gen_code)]:
            if src and os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(row['example_number'], row['combination_number'])
                print(f"Warning: Required original file not found: {src}")
        
    
    # Update paths in DataFrame
    merged_df['original_image_path'] = merged_df.apply(lambda x: os.path.join(original_dir, f"example_{x['example_number']}_combination_{x['combination_number']}.png"), axis=1)
    merged_df['original_code_path'] = merged_df.apply(lambda x: os.path.join(original_dir, f"example_{x['example_number']}_combination_{x['combination_number']}.tex"), axis=1)
    
    # Only update generated paths if the original files existed
    def update_generated_path(row, file_ext):
        base_path = os.path.join(generated_dir, f"example_{row['example_number']}_combination_{row['combination_number']}.{file_ext}")
        orig_path = row['generated_image_path'] if file_ext == 'png' else row['generated_code_path']
        full_path = os.path.join(output_dir, base_path)
        return base_path if os.path.exists(full_path) else ''
    
    merged_df['generated_image_path'] = merged_df.apply(lambda x: update_generated_path(x, 'png'), axis=1)
    merged_df['generated_code_path'] = merged_df.apply(lambda x: update_generated_path(x, 'tex'), axis=1)
    
    # Save merged CSV
    output_csv = os.path.join(output_dir, 'merged_results.csv')
    merged_df.to_csv(output_csv, index=False)
    
    return len(merged_df)

# Usage
input_pattern = '*.csv'  # Adjust pattern to match your CSV files
output_dir = 'merged_results_2'  # Output directory

total_rows = merge_and_organize_csv(input_pattern, output_dir)
print(f"Merged CSV created with {total_rows} unique combinations")
print(f"Files organized in {output_dir}")