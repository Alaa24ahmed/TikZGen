import pandas as pd

# Load the CSV file
df = pd.read_csv('path_to_your_results.csv')

# Basic statistics
combined_stats = df['combined_score'].describe()
image_similarity_stats = df['image_similarity'].describe()
code_similarity_stats = df['code_similarity'].describe()

# Print the statistics
print("Combined Score Statistics:")
print(combined_stats)

print("\nImage Similarity Statistics:")
print(image_similarity_stats)

print("\nCode Similarity Statistics:")
print(code_similarity_stats)

# Calculate quartiles
quartiles = df['combined_score'].quantile([0.25, 0.5, 0.75])
print("Quartiles:")
print(quartiles)

# Assign quartile labels
df['quartile'] = pd.qcut(df['combined_score'], q=4, labels=False)

# Count the number of entries in each quartile
quartile_counts = df['quartile'].value_counts().sort_index()
print("\nQuartile Counts:")
print(quartile_counts)