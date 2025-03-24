import os
import json
import pandas as pd
import numpy as np
from cuml.cluster import HDBSCAN
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.cluster import BaseCluster
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from azureml.core import Run, Datastore
import cupy as cp

# Initialize Azure ML run context to track metrics and outputs
run = Run.get_context()
ws = run.experiment.workspace
print('Workspace loaded:', ws.name)

class Dimensionality:
    """ Use this for pre-calculated reduced embeddings """
    def __init__(self, reduced_embeddings):
        self.reduced_embeddings = reduced_embeddings
    def fit(self, X):
        return self
    def transform(self, X):
        return self.reduced_embeddings

def load_data(main_data_path, embeddings_path, reduced_embeddings_path):
    """
    Load data from multiple sources.
    """
    try:
        print(f"Loading documents data from {main_data_path}...")
        df = pd.read_csv(main_data_path)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        print(f"Documents data loaded: {df.shape}")
                
        print(f"Loading embeddings from {embeddings_path}...")
        embeddings_df = pd.read_csv(embeddings_path)
        # Convert the 'embedding' column (which contains lists) into a 2D NumPy array
        embeddings = np.array(embeddings_df['embedding'].apply(eval).tolist())  
        print(f"Embeddings loaded: {embeddings.shape}")

        print(f"Loading reduced embeddings from {reduced_embeddings_path}...")
        reduced_embeddings_df = pd.read_csv(reduced_embeddings_path)
        # Convert the 'embedding' column (which contains lists) into a 2D NumPy array
        reduced_embeddings = np.array(reduced_embeddings_df['reduced_embeddings'].apply(eval).tolist()) 
        print(f"Reduced embeddings loaded: {reduced_embeddings.shape}")

    except Exception as e:
        print(f"Failed to load data: {str(e)}")
        raise

    return df, embeddings, reduced_embeddings

def prepare_data(df, embeddings, reduced_embeddings):
    """
    Prepare the data for BERTopic modeling.
    """
    docs = df['text'].tolist()  # Assuming 'text' column contains the documents
    timestamps = df['date'].tolist()
    clusters = df['cluster'].tolist()  # Assuming 'cluster' column contains pre-computed clusters
    
    return docs, embeddings, reduced_embeddings, clusters, timestamps

def load_custom_stopwords(stopwords_path):
    """
    Load custom stopwords from a JSON file.
    """
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            custom_stopwords = json.load(f)
        print(f"Successfully loaded {len(custom_stopwords)} custom stopwords from {stopwords_path}")
        return custom_stopwords
    except json.JSONDecodeError:
        print(f"Error: The file at {stopwords_path} is not a valid JSON file.")
        return []
    except FileNotFoundError:
        print(f"Error: The file at {stopwords_path} was not found.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading stopwords: {str(e)}")
        return []
    
def create_bertopic_model(docs, embeddings, reduced_embeddings, clusters, custom_stopwords):
    """
    Create and fit the BERTopic model.
    """
    # Prepare sub-models
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    umap_model = Dimensionality(reduced_embeddings)
    hdbscan_model = BaseCluster()
    
    # Create vocabulary from docs
    vocab = list(set([word for doc in docs for word in doc.split()]))
    
    vectorizer_model = CountVectorizer(vocabulary=vocab, stop_words=custom_stopwords)
    representation_model = KeyBERTInspired()

    # Fit BERTopic without actually performing any clustering
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        verbose=True
    ).fit(docs, embeddings=embeddings, y=clusters)

    return topic_model

def save_topics_over_time_graph(topic_model, docs, timestamps, output_dir):
    """
    Save the topics over time graph as an HTML file.
    """
    try:
        # Generate topics over time
        topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=50)
        
        # Visualize topics over time
        fig = topic_model.visualize_topics_over_time(topics_over_time)
                
        # Define the output path for the graph
        output_path_html = os.path.join(output_dir, 'topics_over_time_graph.html')
        
        # Save the figure as an HTML file
        fig.write_html(output_path_html)
        print(f"Topics over time graph saved to {output_path_html}")
        
        # Upload the file to Azure ML run
        run.upload_file(name='digipol-uploaded-files/topics_over_time_graph.html', path_or_stream=output_path_html)
        
    except Exception as e:
        print(f"An error occurred while saving the topics over time graph: {str(e)}")
        raise

def save_hierarchy_graph(topic_model, docs, output_dir):
    """
    Save the hierarchical clustering graph as an HTML file.
    """
    try:
        # Generate the hierarchical topics
        hierarchical_topics = topic_model.hierarchical_topics(docs)

        # Visualize the hierarchical structure
        fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
        
        # Define the output path for the graph
        output_path_html = os.path.join(output_dir, 'hierarchy_graph.html')
        # output_path_png = os.path.join(output_dir, 'hierarchy_graph.png')

        # Save the figure as an HTML file
        fig.write_html(output_path_html)
        print(f"Hierarchy graph saved to {output_path_html}")
        
        # # Optionally, save as an image (e.g., PNG)
        # fig.write_image(output_path_png)
        # print(f"Hierarchy graph saved as PNG to {output_path_png}")
        
        # Log the files to the Azure ML run
        run.upload_file(name='digipol-uploaded-files/hierarchy_graph.html', path_or_stream=output_path_html)
        # run.upload_file(name='digipol-uploaded-files/hierarchy_graph.png', path_or_stream=output_path_png)

    except Exception as e:
        print(f"An error occurred while saving the hierarchy graph: {str(e)}")
        raise

def save_results(topic_model, output_dir):
    """
    Save the results of the BERTopic model.
    """
    # Get topic information
    topic_info = topic_model.get_topic_info()
    
    # Save topic information to CSV
    output_path = os.path.join(output_dir, 'topic_info.csv')
    topic_info.to_csv(output_path, index=False)
    print(f"Topic information saved to {output_path}")
    
    # Extract and save top terms for each topic
    topics = topic_model.get_topics()
    top_terms = {topic: [term for term, _ in terms[:10]] for topic, terms in topics.items()}
    
    top_terms_df = pd.DataFrame.from_dict(top_terms, orient='index')
    top_terms_output_path = os.path.join(output_dir, 'top_terms.csv')
    top_terms_df.to_csv(top_terms_output_path)
    
    # Log the output file to the Azure ML run
    datastore = Datastore.get(ws, datastore_name='workspaceblobstore')
    datastore.upload_files(files=[output_path], target_path='digipol-uploaded-files/', overwrite=True)
    run.upload_file(name='digipol-uploaded-files/topic_info_final.csv', path_or_stream=output_path)
    run.upload_file(name='digipol-uploaded-files/top_terms_final.csv', path_or_stream=output_path)
    
    print(f"Top terms for each topic saved to {top_terms_output_path}")

if __name__ == "__main__":
    run.log("Step", "Loading data")
    
    # Load the data from the input mount points
    main_data_path = run.input_datasets['main_data']
    embeddings_path = run.input_datasets['embeddings']
    reduced_embeddings_path = run.input_datasets['reduced_embeddings']
    
    df, embeddings, reduced_embeddings = load_data(main_data_path, embeddings_path, reduced_embeddings_path)
    
    run.log("Step", "Preparing data")
    
    # Prepare data for BERTopic
    docs, embeddings, reduced_embeddings, clusters, timestamps = prepare_data(df, embeddings, reduced_embeddings)
    
    run.log("Step", "Loading custom stopwords")
    
    # Load custom stopwords
    stopwords_path = run.input_datasets['stopwords']
    custom_stopwords = load_custom_stopwords(stopwords_path)
    
    run.log("Step", "Creating and fitting BERTopic model")
    
    # Create and fit BERTopic model
    topic_model = create_bertopic_model(docs, embeddings, reduced_embeddings, clusters, custom_stopwords)
    
    output_dir = './digipol-uploaded-files'
    os.makedirs(output_dir, exist_ok=True)

    # Create Topics over Time
    save_topics_over_time_graph(topic_model, docs, timestamps, output_dir)
    
    run.log("Step", "Saving results")
    
    # Define the output directory
    
    run.log("Step", "Saving hierarchy graph")

    # Save the hierarchical clustering graph
    save_hierarchy_graph(topic_model, docs, output_dir)
    
    # Save the results
    save_results(topic_model, output_dir)
    
    run.log("Step", "Process completed")
    
    # Complete the run
    run.complete()