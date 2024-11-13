from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("nllg/datikz-v2")

# Create train set with first 20k examples
train_dataset = dataset["train"].select(range(20000))
train_dataset.to_parquet("train_20k.parquet")

# Use the original test set
dataset["test"].to_parquet("test.parquet")