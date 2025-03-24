import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# This script creates embeddings from the text provided
# This has to be done for both the newspaper paragraphs and anchor data seperately

# Step 1: Load Data
def load_data(input_data_path):
    """
    Load the data from the provided input data path.
    """
    try:
        print(f"Loading data from {input_data_path}...")
        df = pd.read_csv(input_data_path)
        print("Data successfully loaded.")
    except Exception as e:
        print(f"Failed to load data: {str(e)}")
        raise
    return df

# Step 2: Preprocess Data
def preprocess_data(df, model_name):
    device = 'cpu'

    # Ensure all entries in the 'text' column are strings and handle NaN values
    df['text'] = df['text'].astype(str).fillna('')

    # Initialize the model
    print("Loading embeddings model...")
    model = SentenceTransformer(model_name, device=device)

    # Encode the 'text' column into embeddings
    print("Embedding text column...")
    embeddings = model.encode(df['text'].tolist(), batch_size=32)
    df['embedding'] = embeddings.tolist()

    print("Data embedded:", df.shape)
    return df

# Step 3: Save Results
def save_results(df, output_path):
    """
    Save the results to a CSV file.
    """
    print("Saving data...")
    df.to_csv(output_path, index=False)
    print("Data saved to:", output_path)

# Main execution flow
if __name__ == "__main__":
    print("Step: Loading data")

    # Set parameters
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    input_data_path = 'data/01_text_data/paragraphs.csv'  # Input file path
    output_data_path = 'data/02_embedded_data/paragraphs.csv'  # Output file path

    # Load the data
    df = load_data(input_data_path)

    print("Step: Embedding data")
    # Preprocess the data (e.g., generate embeddings)
    df = preprocess_data(df, model_name)

    print("Step: Saving data")
    # Save the results
    save_results(df, output_data_path)

    print("Process completed")
