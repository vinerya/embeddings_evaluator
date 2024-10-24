# Embeddings Evaluator

A Python package for analyzing and comparing embedding models through pairwise cosine similarity distributions.

## Features

- Pairwise cosine similarity distribution analysis
- Statistical measures:
  * Mean (μ)
  * Standard deviation (σ)
  * Median (m)
  * Peak location and amplitude
- Multi-model comparison visualization

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
import numpy as np
from embeddings_evaluator import plot_model_comparison
from embeddings_evaluator.comparison import save_comparison_plot

# Load your embeddings into a dictionary
embeddings_dict = {
    "Model A": embeddings_a,  # numpy array of shape (n_docs, embedding_dim)
    "Model B": embeddings_b
}

# Generate comparison plot
fig = plot_model_comparison(embeddings_dict)
save_comparison_plot(fig, 'comparison.png')
```

## Example with Faiss Indices

```python
import faiss
import numpy as np
from embeddings_evaluator import plot_model_comparison

# Load embeddings from faiss indices
def load_faiss_embeddings(index_path):
    index = faiss.read_index(index_path)
    if isinstance(index, faiss.IndexFlatL2):
        num_vectors = index.ntotal
        dimension = index.d
        embeddings = np.zeros((num_vectors, dimension), dtype=np.float32)
        for i in range(num_vectors):
            embeddings[i] = index.reconstruct(i)
        return embeddings
    raise ValueError("Unsupported index type")

# Load multiple models
embeddings_dict = {}
for size in [250, 500, 1000, 2000, 4000]:
    embeddings = load_faiss_embeddings(f"faiss_embeddings/{size}/index.faiss")
    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    embeddings_dict[f"Model {size}"] = embeddings

# Generate visualization
fig = plot_model_comparison(embeddings_dict)
save_comparison_plot(fig, 'model_comparison.png')
```

## Output

The tool provides:

1. Statistical Measures for each model:
- Mean cosine similarity (μ)
- Standard deviation (σ)
- Median (m)
- Peak location and amplitude

2. Visualization:
- Overlaid probability density histograms
- Statistical annotations
- Peak coordinates
- Vertical lines at mean values
- [0,1] bounded cosine similarity range

## Requirements

- numpy
- pandas
- plotly
- scipy
- faiss-cpu (for faiss index support)
