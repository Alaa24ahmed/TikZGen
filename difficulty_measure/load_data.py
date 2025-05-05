import os
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("samahadhoud/decomposed-tikz-dataset-20-30")

# Create directories and save files
for example in dataset['train']:
    # Extract the ID
    example_id = example['id']
    
    # Parse the example and combination numbers
    # parts = example_id.split('_')
    # example_number = parts[1]
    # combination_number = parts[-1]
    
    # Create a folder for each example
    # folder_name = f"data/example_{example_number}"
    folder_name = f"data_20-30"
    os.makedirs(folder_name, exist_ok=True)
    
    # Save the PNG file
    # png_path = os.path.join(folder_name, f"combination_{combination_number}.png")
    png_path = os.path.join(folder_name, f"{example_id}.png")
    example['png'].save(png_path)
    
    # Save the TikZ code
    # tikz_path = os.path.join(folder_name, f"combination_{combination_number}.tex")
    tikz_path = os.path.join(folder_name, f"{example_id}.tex")
    with open(tikz_path, 'w') as f:
        f.write(example['code'])

print("All files saved successfully.")