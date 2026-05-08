import os
import pandas as pd
import numpy as np
import ast

# Function to load embeddings from a CSV file where embeddings are stored as strings
def load_embeddings_from_csv(file_path):
    df = pd.read_csv(file_path)
    df['reduced_embeddings'] = df['reduced_embeddings'].apply(ast.literal_eval)
    return df

# Function to calculate cosine similarity between two embeddings
def cosine_similarity(anchor_embedding, sentence_embedding):
    dot_product = np.dot(anchor_embedding, sentence_embedding)
    norm_anchor = np.linalg.norm(anchor_embedding)
    norm_sentence = np.linalg.norm(sentence_embedding)
    return dot_product / (norm_anchor * norm_sentence)

# Function to calculate similarity scores between embeddings in df_main and categories in df_anchor
def calculate_similarity_scores(df_main, df_anchor, num_categories):

    # Initialize columns to store similarity scores
    for i in range(0, num_categories):
        df_main[f'similarity_score_{i}'] = 0.0

    # Iterate through each of the num_categories categories in df_anchor
    for i in range(0, num_categories):
        # Filter anchor DataFrame by the current category
        df_anchor_cat = df_anchor[df_anchor['category'] == i]

        # Aggregate the anchor embeddings for the current category
        aggregated_embedding = aggregate_embeddings(df_anchor_cat['reduced_embeddings'].tolist())

        # Calculate similarity scores for the current category
        df_main[f'similarity_score_{i}'] = df_main['reduced_embeddings'].apply(
            lambda x: float(cosine_similarity(aggregated_embedding, np.array(x)))
        )

    return df_main

# Function to aggregate multiple embeddings into a single embedding using mean pooling
def aggregate_embeddings(embeddings):
    embeddings_np = np.array(embeddings)  # Convert to NumPy array
    return np.mean(embeddings_np, axis=0)

# Function to save DataFrame to a CSV file
def save_results(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

# Main execution flow
if __name__ == "__main__":
    print("Step: Loading data")

    # Set parameters
    num_categories = 8
    main_data_path = "data/03_dimensionality_reduced_data/paragraphs.csv"
    anchor_data_path = "data/03_dimensionality_reduced_data/anchors.csv"
    output_path = "data/04_categorized_data/similarity_scored_paragraphs.csv"

    # Load the data
    anchor_df = load_embeddings_from_csv(anchor_data_path)
    print(f"Loaded anchor embeddings DataFrame with shape: {anchor_df.shape}")

    main_df = load_embeddings_from_csv(main_data_path)
    print(f"Loaded sentence embeddings DataFrame with shape: {main_df.shape}")

    print("Step: Calculating similarity scores")

    # Calculate similarity scores for each anchor embedding
    df_with_scores = calculate_similarity_scores(main_df, anchor_df, num_categories)

    print("Step: Saving results")

    # Save the DataFrame with similarity scores to a CSV file
    save_results(df_with_scores, output_path)

    print("Process completed")
