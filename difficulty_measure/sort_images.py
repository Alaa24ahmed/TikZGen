import csv
import os
from PIL import Image
import pandas as pd

output_dir = "tikz_results_20241109_152045/"
# Load the CSV file
csv_path = os.path.join(output_dir, "results.csv")
results = pd.read_csv(csv_path).to_dict('records')

# Sort results by combined score (ascending)
sorted_results = sorted(results, key=lambda x: x['combined_score'])

# Calculate padding width
num_images = len(sorted_results)
padding_width = len(str(num_images))

# Create sorted output directories
sorted_orig_dir = os.path.join(output_dir, "sorted_original")
sorted_gen_dir = os.path.join(output_dir, "sorted_generated")
os.makedirs(sorted_orig_dir, exist_ok=True)
os.makedirs(sorted_gen_dir, exist_ok=True)

# Process and save sorted images
for rank, result in enumerate(sorted_results, 1):
    orig_name = os.path.basename(result['original_image_path'])
    gen_name = os.path.basename(result['generated_image_path'])
    
    new_orig_name = f"{rank:0{padding_width}d}_{orig_name}_diff{result['combined_score']}.png"
    new_gen_name = f"{rank:0{padding_width}d}_{gen_name}_diff{result['combined_score']}.png"
    
    orig_img = Image.open(result['original_image_path'])
    try:
        gen_img = Image.open(result['generated_image_path'])
        gen_img.save(os.path.join(sorted_gen_dir, new_gen_name))
    except:
        print(f"no generated image for {gen_name}")
    orig_img.save(os.path.join(sorted_orig_dir, new_orig_name))
    
    result['sorted_original_path'] = os.path.join(sorted_orig_dir, new_orig_name)
    result['sorted_generated_path'] = os.path.join(sorted_gen_dir, new_gen_name)

# Update CSV with new sorted paths
fieldnames = list(results[0].keys())  # Get fieldnames from the dictionary keys
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for result in sorted_results:
        writer.writerow(result)

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

