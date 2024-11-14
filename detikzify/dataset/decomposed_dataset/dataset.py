import os
import base64
import pandas as pd
from PIL import Image
from datasets import Dataset, Features, Value, Image
import argparse

def create_dataset(base_folder, start_index, end_index):
    """
    Creates a Hugging Face dataset with explicit feature types for 'image' and 'code'.

    Parameters:
    - base_folder: The root directory containing example folders.
    - start_index: The starting index of the folders to process.
    - end_index: The ending index of the folders to process (exclusive).

    Returns:
    - A Hugging Face Dataset with defined features for 'id', 'image', and 'code'.
    """
    data = []
    # example_folders = sorted(os.listdir(base_folder))[start_index:end_index]

    # Traverse through each selected example_i folder
    # for example_folder in example_folders:
    print(start_index)
    print(end_index)
    for i in range(start_index, end_index):
        print(i)
        example_folder = f'example_{i}'
        example_path = os.path.join(base_folder, example_folder)
        if not os.path.isdir(example_path):
            continue

        code_folder = os.path.join(example_path, "code")
        png_folder = os.path.join(example_path, "png_2")

        # Skip if either folder does not exist
        if not os.path.exists(code_folder) or not os.path.exists(png_folder):
            continue

        # Iterate through PNG files in the png folder
        for png_file in os.listdir(png_folder):
            if not png_file.endswith(".png"):
                continue

            # Extract the combination ID (e.g., combination_j)
            combination_id = os.path.splitext(png_file)[0]  # Remove .png extension
            code_file = os.path.join(code_folder, f"{combination_id}.tex")
            png_path = os.path.join(png_folder, png_file)

            # Only include if the code file exists
            if os.path.exists(code_file):
                # Read the code content
                with open(code_file, "r") as f:
                    code_content = f.read()

                # Create a unique ID (e.g., example_i_combination_j)
                id_value = f"{example_folder}_{combination_id}"

                # Append the row to the data list
                data.append({
                    "id": id_value,
                    "png": png_path,
                    "code": code_content
                })

        png_file = "main.png"
        combination_id = os.path.splitext(png_file)[0]  # Remove .png extension
        code_file = os.path.join(code_folder, "original.tex")
        png_path = os.path.join(png_folder, png_file)

        # Only include if the code file exists
        if os.path.exists(code_file):
            # Read the code content
            with open(code_file, "r") as f:
                code_content = f.read()

            # Create a unique ID (e.g., example_i_combination_j)
            id_value = f"{example_folder}_{combination_id}"

            # Append the row to the data list
            data.append({
                "id": id_value,
                "png": png_path,
                "code": code_content
            })


    # Define the feature types
    features = Features({
        'id': Value(dtype='string'),
        'png': Image(mode=None, decode=True),
        'code': Value(dtype='string')
    })

    # Create a Hugging Face Dataset with explicit features
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df, features=features)
    return dataset

# Define the base folder path and index range
base_folder = "/home/ali.mekky/Documents/AI/project/tikz_decomposition_output_sequential"
start_index = 15000  # Change this to your desired start index
end_index = 20000   # Change this to your desired end index

# Create the dataset for the specified index range
dataset = create_dataset(base_folder, start_index, end_index)

# Save the dataset to disk (optional)
dataset.save_to_disk("huggingface_dataset")

# Print a sample row
print(dataset[0])
repo_name = "decomposed-tikz-dataset-15-20"  # Change this to your preferred dataset name

# Push the dataset to Hugging Face Hub
dataset.push_to_hub(repo_name)

# if __name__ == "__main__":
#     # Set up argument parser
#     parser = argparse.ArgumentParser(description="Create a Hugging Face dataset from folder structure.")
#     parser.add_argument("--start_index", type=int, required=True, help="Starting index of folders to process.")
#     parser.add_argument("--end_index", type=int, required=True, help="Ending index of folders to process (exclusive).")


#     # Parse the arguments
#     args = parser.parse_args()

#     base_folder = "/home/ali.mekky/Documents/AI/project/tikz_decomposition_output_sequential"
#     start_index = args.start_index
#     end_index = args.end_index
#     # Create the dataset
#     # dataset = create_dataset(args.base_folder, args.start_index, args.end_index)
#     dataset = create_dataset(base_folder, start_index, end_index)

#     # Save the dataset to disk (optional)
#     dataset.save_to_disk("huggingface_dataset")

#     # Print a sample row
#     print(dataset[0])
#     repo_name = f"decomposed-tikz-dataset-{start_index//1000}-{end_index//1000}"  # Change this to your preferred dataset name

#     # Push the dataset to Hugging Face Hub
#     dataset.push_to_hub(repo_name)