import os
import random
import argparse
from detikzify.infer.tikz import TikzDocument
from multiprocessing import Pool, cpu_count
import shutil

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

# Helper function to process a list of TikZ files in parallel
def process_tikz_files(args):
    tikz_file_path, output_png_path = args
    process_tikz_file(tikz_file_path, output_png_path)

# Main function to iterate through all examples and combinations
def process_all_examples(start: int, end: int):
    base_dir = "/home/ali.mekky/Documents/AI/project/tikz_decomposition_output_sequential"
    tasks = []

    # Iterate through each example from start to end
    for example_index in range(start, end):
        print(f"Processing example {example_index}...")
        example_dir = os.path.join(base_dir, f"example_{example_index}", "code")
        output_dir = os.path.join(base_dir, f"example_{example_index}", "png")
        output_dir_2 = os.path.join(base_dir, f"example_{example_index}", "png_2")

        # Create the output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir_2, exist_ok=True)

        # Get all combination files in the example directory
        combination_files = sorted([
            f for f in os.listdir(example_dir)
            if f.startswith("combination_") and f.endswith(".tex")
        ])

        # Check if there are no combination files
        if len(combination_files) == 0:
            print(f"No combination files found in {example_dir}. Skipping...")
            continue

        # Check if the PNG folder already contains files
        if len(os.listdir(output_dir)) == 1:
            print(f"TIKZ files cannot be compiled. Skipping...")
            continue

        # Check if there are already 20 or more PNG files in `png_2` folder
        png_files_in_output_dir_2 = [
            f for f in os.listdir(output_dir_2) if f.endswith(".png")
        ]
        if len(png_files_in_output_dir_2) >= 20:
            print(f"Already 20 PNG files in {output_dir_2}. Skipping example {example_index}...")
            continue

        # If there are more than 20 files, choose a random subset of 20
        if len(combination_files) > 20:
            combination_files = random.sample(combination_files, 20)
            print(len(combination_files), "files selected randomly")

        # Prepare tasks for parallel processing
        for combination_file in combination_files:
            tikz_file_path = os.path.join(example_dir, combination_file)
            output_png_path = os.path.join(output_dir_2, combination_file.replace(".tex", ".png"))

            # Skip if the PNG already exists
            if os.path.exists(output_png_path):
                print(f"{output_png_path} already exists. Skipping...")
                continue

            tasks.append((tikz_file_path, output_png_path))

    # Use a pool of workers, limiting the number of processes to avoid lagging
    num_processes = max(cpu_count() - 2, 1)  # Leave 2 cores free for other tasks
    print(f"Using {num_processes} parallel processes.")

    with Pool(processes=num_processes) as pool:
        # Process all tasks in parallel
        pool.map(process_tikz_files, tasks)

def copy_main(start_index, end_index):
    for i in range(start_index, end_index):
        folder_name = f"/home/ali.mekky/Documents/AI/project/tikz_decomposition_output_sequential/example_{i}"
        png_folder = os.path.join(folder_name, "png")
        png_2_folder = os.path.join(folder_name, "png_2")

        # Create the png_2 folder if it doesn't exist
        os.makedirs(png_2_folder, exist_ok=True)

        # Define the source and destination file paths
        source_file = os.path.join(png_folder, "main.png")
        destination_file = os.path.join(png_2_folder, "main.png")

        # Copy the file if it exists
        if os.path.exists(source_file):
            shutil.copy(source_file, destination_file)
            print(f"Copied {source_file} to {destination_file}")
        else:
            print(f"File {source_file} not found, skipping.")

    print("Copy operation completed.")

# Entry point for the script
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process TikZ files and generate PNGs.")
    parser.add_argument("start_index", type=int, help="Starting index for processing")
    parser.add_argument("end_index", type=int, help="Ending index for processing")
    args = parser.parse_args()

    # Run the processing for all examples in the specified range
    process_all_examples(args.start_index, args.end_index)
    copy_main(args.start_index, args.end_index)
