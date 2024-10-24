import numpy as np
import faiss
from embeddings_evaluator import plot_model_comparison
from embeddings_evaluator.comparison import save_comparison_plot
from embeddings_evaluator.metrics import pairwise_similarities

def load_faiss_embeddings(index_path):
    """Load embeddings from a faiss index."""
    print(f"Loading {index_path}...")
    
    # Load the faiss index
    index = faiss.read_index(index_path)
    
    # Get the raw vectors from the index
    if isinstance(index, faiss.IndexIDMap):
        raw_index = faiss.downcast_index(index.index)
    else:
        raw_index = index
    
    # Extract vectors based on index type
    if isinstance(raw_index, faiss.IndexFlatL2):
        num_vectors = raw_index.ntotal
        dimension = raw_index.d
        print(f"Found {num_vectors} vectors of dimension {dimension}")
        embeddings = np.zeros((num_vectors, dimension), dtype=np.float32)
        for i in range(num_vectors):
            embeddings[i] = raw_index.reconstruct(i)
    else:
        raise ValueError(f"Unsupported index type: {type(raw_index)}")
    
    return embeddings

def analyze_embeddings(embeddings, name):
    """Calculate embedding statistics."""
    similarities = pairwise_similarities(embeddings)
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    median_sim = np.median(similarities)
    
    print(f"\n{name}:")
    print(f"Mean similarity: {mean_sim:.3f}")
    print(f"Std deviation: {std_sim:.3f}")
    print(f"Median similarity: {median_sim:.3f}")
    
    return similarities

print("Loading embeddings from faiss indices...")

# Model sizes to load
model_sizes = [250, 500, 1000, 2000, 4000]

# Load all models
embeddings_dict = {}
for size in model_sizes:
    embeddings = load_faiss_embeddings(f"faiss_embeddings/{size}/index.faiss")
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    embeddings_dict[f"Model {size}"] = embeddings

print("\nCalculating statistics...")
# Analyze each model
for name, embeddings in embeddings_dict.items():
    analyze_embeddings(embeddings, name)

print("\nGenerating visualization...")
# Compare all models
fig = plot_model_comparison(embeddings_dict)

# Save the comparison visualization
output_file = 'model_comparison.png'
save_comparison_plot(fig, output_file)

print(f"\nVisualization saved to {output_file}")
