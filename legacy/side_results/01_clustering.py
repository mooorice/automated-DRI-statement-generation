# Import necessary libraries
import os
import pandas as pd
import numpy as np
from cuml.cluster import HDBSCAN
from azureml.core import Run, Datastore
from cuml.preprocessing import Normalizer
import cupy as cp

# Initialize Azure ML run context to track metrics and outputs
run = Run.get_context()
ws = run.experiment.workspace  
print('Workspace loaded:', ws.name)

# Step 1: Load Data
def load_data(input_data_path):
    """
    Load the data from the provided input data path.
    """
    try:
        print(f"Loading data from {input_data_path}...")
        df = pd.read_csv(input_data_path)
        # Convert the 'embedding' column (which contains lists) into a 2D NumPy array
        embeddings = np.array(df['reduced_embeddings'].apply(eval).tolist())  
        
        # Convert the 'embedding' column to a cupy array
        X = cp.array(embeddings)
        
        # Initialize the Normalizer
        transformer = Normalizer()
        
        # Apply normalization
        X_normalized = transformer.transform(X)
        
        # Convert the normalized cupy array back to a list of lists
        df['embedding_normalized'] = X_normalized.get().tolist()
        df = df.drop(columns=['reduced_embeddings'])
        
        print(f"Data successfully loaded: {df.shape}" )
    
    except Exception as e:
        print(f"Failed to load data: {str(e)}")
        raise
    
    return df

# Step 2: Perform Clustering
def perform_clustering(df):
    print("Clustering data...")
    embeddings = np.array(df['embedding_normalized'].tolist())
    
    # Initialize HDBSCAN
    clusterer = HDBSCAN(
        min_cluster_size=100, 
        min_samples=1, 
        cluster_selection_epsilon=0.0, 
        metric='euclidean', # 'euclidean' or 'l2'
        cluster_selection_method='eom', # 'leaf' or 'eom'
        gen_min_span_tree=False
    )
    
    print("Clusterer initialized")
    cluster_labels = clusterer.fit_predict(embeddings)
    print("Data clustered")
    
    return cluster_labels

# Step 3: Save Results
def save_results(df, cluster_labels, output_dir):
    print("Saving data...")
    
    # Add cluster labels to the DataFrame
    df['cluster'] = cluster_labels
    print(f"Clusters added to df: {df.shape}" )

    # Print the distribution of clusters
    cluster_distribution = df['cluster'].value_counts()
    print("Cluster distribution:")
    print(cluster_distribution)

    # Drop the 'embedding_normalized' column before saving the data
    if 'embedding_normalized' in df.columns:
        df = df.drop(columns=['embedding_normalized'])
        print(f"Dropped 'embedding_normalized' column: {df.shape}")
    
    # Save the results to a CSV file in the output directory
    output_path = os.path.join(output_dir, 'digipol_clustered_data.csv')
    df.to_csv(output_path, index=False)
    
    # Log the output file to the Azure ML run
    datastore = Datastore.get(ws, datastore_name='workspaceblobstore')
    datastore.upload_files(files=[output_path], target_path='digipol-uploaded-files/', overwrite=True)
    run.upload_file(name='digipol-uploaded-files/digipol_clustered_data.csv', path_or_stream=output_path)
    
    print(f"Data saved to {output_path}")

# Main execution flow
if __name__ == "__main__":
    run.log("Step", "Loading data")
    
    # Load the data from the input mount point
    input_data_path = os.environ['AZUREML_DATAREFERENCE_data']  # The path where the data is mounted
    df = load_data(input_data_path)
    
    run.log("Step", "Loading done, clustering data...")
    
    # Perform clustering
    cluster_labels = perform_clustering(df)
    
    run.log("Step", "Clustering done, saving data...")
    
    # Define the output directory
    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the results to the output directory
    save_results(df, cluster_labels, output_dir)
    
    run.log("Step", "Saving done")
    
    # Complete the run
    run.complete()
