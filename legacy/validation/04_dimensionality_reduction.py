import os
import json
import ast
import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import Normalizer

# Step 1: Load Data
def load_data(main_data_path, anchor_data_path):
    """
    Load the data from the provided input data paths.
    """
    try:
        print(f"Loading data from {main_data_path} and {anchor_data_path}...")
        df_main = pd.read_csv(main_data_path)
        df_anchor = pd.read_csv(anchor_data_path)

        print(f"Data successfully loaded: {df_main.shape} and {df_anchor.shape}")

    except Exception as e:
        print(f"Failed to load data: {str(e)}")
        raise

    return df_main, df_anchor

def safe_literal_eval(val):
    # If the value is already a list, return it directly
    if isinstance(val, list):
        return val
    try:
        # Otherwise, try to evaluate it as a literal
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        # If there's an issue, return the value unchanged
        return val

# Step 2: Perform UMAP Dimensionality Reduction
def perform_umap_reduction(df_main, df_anchor, num_dimensions):
    print("Performing UMAP dimensionality reduction...")

    # Convert string representations of lists to actual lists, then to a numpy array
    df_main['embedding'] = df_main['embedding'].apply(safe_literal_eval)
    df_anchor['embedding'] = df_anchor['embedding'].apply(safe_literal_eval)

    # Combine the embeddings
    combined_embeddings = []
    for embedding in df_main['embedding']:
        combined_embeddings.append(np.array(embedding, dtype=np.float32))
    for embedding in df_anchor['embedding']:
        combined_embeddings.append(np.array(embedding, dtype=np.float32))

    # Convert to a single NumPy array
    embeddings = np.vstack(combined_embeddings)

    # Adjust n_components to be less than or equal to the number of samples
    n_components = min(num_dimensions, embeddings.shape[0])  # Choose the smaller of 50 or the number of samples

    # Initialize UMAP
    umap_reducer = umap.UMAP(n_components=n_components, n_neighbors=30, min_dist=0.0, metric='cosine')

    # Fit and transform the data
    reduced_embeddings = umap_reducer.fit_transform(embeddings)

    print("UMAP reduction completed")

    # Split the reduced embeddings back into the original two DataFrames
    reduced_embeddings_main = reduced_embeddings[:len(df_main)]
    reduced_embeddings_anchor = reduced_embeddings[len(df_main):]

    return reduced_embeddings_main, reduced_embeddings_anchor

# Step 3: Save Results
def save_results(df_main, reduced_embeddings_main, df_anchor, reduced_embeddings_anchor, output_dir):
    print("Saving data...")

    # Convert each embedding array to a JSON string and add to DataFrames
    df_main['reduced_embeddings'] = [json.dumps(embedding.tolist()) for embedding in reduced_embeddings_main]
    df_anchor['reduced_embeddings'] = [json.dumps(embedding.tolist()) for embedding in reduced_embeddings_anchor]

    print(f"reduced_embeddings added to df_main: {df_main.shape}, df_anchor: {df_anchor.shape}")

    # Drop the 'embedding' column before saving the data
    if 'embedding' in df_main.columns:
        df_main = df_main.drop(columns=['embedding'])
    if 'embedding' in df_anchor.columns:
        df_anchor = df_anchor.drop(columns=['embedding'])

    # Save the results to CSV files in the output directory
    output_path_1 = os.path.join(output_dir, 'paragraphs.csv')
    output_path_2 = os.path.join(output_dir, 'anchors.csv')

    df_main.to_csv(output_path_1, index=False)
    df_anchor.to_csv(output_path_2, index=False)

    print(f"Data saved to {output_path_1} and {output_path_2}")

# Main execution flow
if __name__ == "__main__":
    print("Step: Loading data")

    # Set parameters
    num_dimensions = 50
    main_data_path = "data/02_embedded_data/paragraphs.csv"
    anchor_data_path = "data/02_embedded_data/anchors.csv"
    df_main, df_anchor = load_data(main_data_path, anchor_data_path)
    output_dir = 'data/03_dimensionality_reduced_data'
    os.makedirs(output_dir, exist_ok=True)

    print("Step: Performing UMAP reduction")

    # Perform UMAP dimensionality reduction
    reduced_embeddings_main, reduced_embeddings_anchor = perform_umap_reduction(df_main, df_anchor, num_dimensions)

    print("Step: Saving data")

    # Save the results to the output directory
    save_results(df_main, reduced_embeddings_main, df_anchor, reduced_embeddings_anchor, output_dir)

    print("Process completed")
