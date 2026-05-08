import os
import ast
import numpy as np
import pandas as pd
from azureml.core import Run, Datastore

# Initialize Azure ML run context to track metrics and outputs
run = Run.get_context()
ws = run.experiment.workspace  
print('Workspace loaded:', ws.name)

# Step 1: Load Data
def load_data(cluster_data_path, embeddings_data_path):
    """
    Load the data from the provided input data paths.
    """
    try:
        
        # Load with pandas
        print(f"Loading documents data from {cluster_data_path}...")
        cluster_df = pd.read_csv(cluster_data_path)
        print(f"Documents data loaded: {cluster_df.shape}")
                
        embeddings_df = pd.read_csv(embeddings_data_path)
                                
        print(f"Data successfully loaded: {cluster_df.shape} and {embeddings_df.shape}")
    
    except Exception as e:
        print(f"Failed to load data: {str(e)}")
        raise
    
    return cluster_df, embeddings_df

# Step 2: Calculate Centroids and Distances
def calculate_centroids_and_distances(cluster_df, embeddings_df):
    print("Merging cluster data with embeddings...")
    
    # Remove all rows with cluster -1 or NaN clusters
    cluster_df = cluster_df[cluster_df['cluster'].notna() & (cluster_df['cluster'] != -1)]
    
    # Reset the index to ensure the index alignment is maintained
    cluster_df = cluster_df.reset_index(drop=True)
    embeddings_df = embeddings_df.reset_index(drop=True)
    
    # Merge DataFrames on index
    merged_df = pd.concat([cluster_df, embeddings_df], axis=1)

    # Convert 'reduced_embeddings' column to numpy array for easier manipulation
    merged_df['reduced_embeddings'] = merged_df['reduced_embeddings'].apply(lambda x: np.array(ast.literal_eval(x)))
    
    # Ensure centroids are in the correct format for distance calculation
    centroids = merged_df.groupby('cluster')['reduced_embeddings'].apply(lambda x: np.mean(np.vstack(x.values), axis=0))
    centroids_dict = centroids.to_dict()

    # Calculate the distance from each text to its cluster centroid using CPU and store as scalar values
    merged_df['distance_to_centroid'] = merged_df.apply(
        lambda row: float(np.linalg.norm(row['reduced_embeddings'] - centroids_dict.get(row['cluster'], np.nan))) if pd.notna(row['cluster']) else np.nan,
        axis=1
    )

    # Calculate the cosine similarity from each text to its cluster centroid
    merged_df['cosine_similarity_to_centroid'] = merged_df.apply(
        lambda row: float(calculate_cosine_similarity(row['reduced_embeddings'], centroids_dict.get(row['cluster'], np.nan))) if pd.notna(row['cluster']) else np.nan,
        axis=1
    )

    return merged_df

# Define cosine similarity manually
def calculate_cosine_similarity(vec_a, vec_b):
    """
    Calculate the cosine similarity between two vectors.
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)

# Step 3: Find Closest Texts to Centroids
def find_closest_texts(merged_df):
    print("Finding closest texts to centroids...")
    
    # For each cluster, find the 10 texts closest to the centroid based on Euclidean distance
    closest_texts_euclidean = merged_df.sort_values(by=['cluster', 'distance_to_centroid']).groupby('cluster').head(10)
    
    # For each cluster, find the 10 texts closest to the centroid based on Cosine similarity
    closest_texts_cosine = merged_df.sort_values(by=['cluster', 'cosine_similarity_to_centroid'], ascending=False).groupby('cluster').head(10)
    
    # Combine both results into one DataFrame with a new column indicating the ranking type
    closest_texts_euclidean['ranking_type'] = 'euclidean'
    closest_texts_cosine['ranking_type'] = 'cosine'
    
    closest_texts_combined = pd.concat([closest_texts_euclidean, closest_texts_cosine], axis=0)
    
    return closest_texts_combined

# Step 4: Save Results
def save_results(df, output_dir):
    print("Saving results...")

    # Save the results to a CSV file in the output directory
    output_path = os.path.join(output_dir, 'closest_texts_per_cluster.csv')
    df.to_csv(output_path, index=False)

    # Log the output file to the Azure ML run
    datastore = Datastore.get(ws, datastore_name='workspaceblobstore')
    datastore.upload_files(files=[output_path], target_path='digipol-uploaded-files/', overwrite=True)
    run.upload_file(name='digipol-uploaded-files/closest_texts_per_cluster.csv', path_or_stream=output_path)

    print(f"Data saved to {output_path}")

# Main execution flow
if __name__ == "__main__":
    run.log("Step", "Loading data")
    
    # Load the data from the input mount points
    cluster_data_path = run.input_datasets['cluster_data']
    embeddings_data_path = run.input_datasets['embeddings_data']
    
    cluster_df, embeddings_df = load_data(cluster_data_path, embeddings_data_path)
    
    run.log("Step", "Calculating centroids and distances")
    
    # Perform centroid calculation and distance measurement
    merged_df = calculate_centroids_and_distances(cluster_df, embeddings_df)
    
    run.log("Step", "Finding closest texts")
    
    # Find the closest texts to the centroids based on Euclidean distance and Cosine similarity
    closest_texts_per_cluster = find_closest_texts(merged_df)
    
    run.log("Step", "Saving data")
    
    # Define the output directory
    output_dir = './digipol-uploaded-files'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the results to the output directory
    save_results(closest_texts_per_cluster, output_dir)
    
    run.log("Step", "Process completed")
    
    # Complete the run
    run.complete()
