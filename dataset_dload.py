from datasets import load_dataset
import pandas as pd
import os

# --- Configuration ---
dataset_name = "russwang/ThinkLite-VL-hard-11k"  # Replace with the desired dataset name (e.g., "squad", "glue", "username/my_dataset")
dataset_config = "default" # Optional: Specify a configuration if the dataset has multiple (e.g., "mrpc" for glue)
output_dir = "./" # Replace with your desired output directory
# --- ---

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print(f"Loading dataset '{dataset_name}' (config: {dataset_config})...")
# Load the dataset (specify config if needed)
if dataset_config:
    dataset = load_dataset(dataset_name, dataset_config)
else:
    dataset = load_dataset(dataset_name)

print("Dataset loaded.")

# Iterate through each split (e.g., 'train', 'validation', 'test')
for split_name, split_data in dataset.items():
    print(f"Processing split: {split_name}")

    # Define the output file path
    output_filename = f"{dataset_name.split('/')[1]}_{dataset_config if dataset_config else ''}_{split_name}.parquet".replace("__", "_").strip("_") # Clean up filename
    output_path = os.path.join(output_dir, output_filename)

    # Convert the split to a pandas DataFrame
    df = split_data.to_pandas()

    # Save the DataFrame as a Parquet file
    print(f"Saving split '{split_name}' to {output_path}...")
    df.to_parquet(output_path)

print("All splits saved successfully.")