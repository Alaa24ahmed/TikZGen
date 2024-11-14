import os
import random
from tikz import TikzDocument

# Function to read TikZ code from a file
def read_tikz_file(filename: str) -> str:
    with open(filename, "r") as file:
        return file.read()

# Function to process a TikZ file and save it as PNG
def process_tikz_file(tikz_file_path: str, output_png_path: str):
    try:
        # Read the TikZ code from the file
        tikz_code = read_tikz_file(tikz_file_path)
        print(f"Processing: {tikz_file_path}")

        # Create a TikzDocument instance
        doc = TikzDocument(tikz_code)

        # Compile and check for errors
        if doc.compiled_with_errors:
            print(f"Compilation failed for {tikz_file_path}")
            print(doc.log)
        else:
            # Save the resulting image (e.g., PNG)
            doc.save(output_png_path)
            print(f"Saved: {output_png_path}")
    except Exception as e:
        # Log the error and continue
        print(f"Error processing {tikz_file_path}: {e}")

# Main function to iterate through all examples and combinations
def process_all_examples(start: int, end: int):
    base_dir = "/home/ali.mekky/Documents/AI/project/tikz_decomposition_output_sequential"
    
    # Iterate through each example from start to end
    for example_index in range(start, end):
        print(f"Processing example {example_index}...")
        example_dir = os.path.join(base_dir, f"example_{example_index}", "code")
        output_dir = os.path.join(base_dir, f"example_{example_index}", "png")
        output_dir_2 = os.path.join(base_dir, f"example_{example_index}", "png_2")

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir_2, exist_ok=True)

        # Get all combination files in the example directory
        combination_files = sorted([
            f for f in os.listdir(example_dir)
            if f.startswith("combination_") and f.endswith(".tex")
        ])

        # Check if there are no combination files or if PNG files already exist
        if len(combination_files) == 0:
            print(f"No combination files found in {example_dir}. Skipping...")
            continue
        
        # Check if the PNG folder already contains files
        if len(os.listdir(output_dir)) == 1 :
            print(f"TIKZ files can not be compiled Skipping...")
            continue

        # If there are more than 20 files, randomly pick 20 unique files
        if len(combination_files) > 20:
            combination_files = random.sample(combination_files, 20)
            print(len(combination_files), "files selected randomly")

        # Process the selected TikZ files
        for combination_file in combination_files:
            tikz_file_path = os.path.join(example_dir, combination_file)
            output_png_path = os.path.join(output_dir_2, combination_file.replace(".tex", ".png"))

            # Process the current TikZ file
            process_tikz_file(tikz_file_path, output_png_path)

# Specify the start and end indices
start_index = 0  # Change this to your starting index
end_index = 1000   # Change this to your ending index

# Run the processing for all examples in the specified range
process_all_examples(start_index, end_index)
