# Embeddings Evaluator

Embeddings Evaluator is a Python package designed to help users evaluate and compare embeddings of text documents using key numeric metrics. It is particularly useful for tasks involving information retrieval and Retrieval-Augmented Generation (RAG). The package provides an automated way to assess the quality of embeddings and compare multiple embeddings based on essential metrics.

## Features

- **Key Metrics:**
  - Mean Cosine Similarity
  - Intrinsic Dimensionality
  - Silhouette Score

- **Interactive Visualizations:** 
  - Line plots for comparing metrics across different embeddings
  - Distribution plots for comparing metric distributions across embedding sets
  - 2D and 3D scatter plots using various dimensionality reduction techniques:
    * Principal Component Analysis (PCA)
    * t-Distributed Stochastic Neighbor Embedding (t-SNE)
    * Uniform Manifold Approximation and Projection (UMAP)
  - Heatmap of pairwise similarities
  - Intrinsic dimensionality plot

- **Flexible Evaluation:**
  - Evaluate single or multiple sets of embeddings
  - Compare embeddings of the same documents across different models or parameters
  - Compare distributions of metrics across different embedding sets

## Installation

To install the package, use the following command:

```bash
pip install embeddings_evaluator
```

## Usage

Here's how you can use the `Embeddings Evaluator` package to evaluate and compare embeddings:

```python
import numpy as np
from embeddings_evaluator import (
    evaluate_embeddings, plot_metrics, plot_interactive_scatter,
    plot_similarity_heatmap, plot_intrinsic_dimensionality,
    compare_all_metric_distributions
)

# Example embeddings
embeddings1 = np.random.rand(100, 300)
embeddings2 = np.random.rand(100, 300)
embeddings3 = np.random.rand(100, 300)

# Evaluate a single set of embeddings
single_eval = evaluate_embeddings(embeddings1, label="Model A")
print(single_eval)

# Evaluate and compare multiple sets of embeddings
embeddings_list = [embeddings1, embeddings2, embeddings3]
labels = ['Model A', 'Model B', 'Model C']
multi_eval = evaluate_embeddings(embeddings_list, labels)

# Plot comparison metrics
metrics_fig = plot_metrics(multi_eval)
metrics_fig.show()

# Create interactive scatter plots
all_embeddings = np.vstack(embeddings_list)
all_labels = np.repeat(labels, [len(e) for e in embeddings_list])

pca_2d_fig = plot_interactive_scatter(all_embeddings, all_labels, method='pca', dimensions=2)
pca_2d_fig.show()

tsne_3d_fig = plot_interactive_scatter(all_embeddings, all_labels, method='tsne', dimensions=3)
tsne_3d_fig.show()

# Plot similarity heatmap
heatmap_fig = plot_similarity_heatmap(embeddings1, labels=['Doc'+str(i) for i in range(100)])
heatmap_fig.show()

# Plot intrinsic dimensionality
id_fig = plot_intrinsic_dimensionality(embeddings1)
id_fig.show()

# Compare distributions of all metrics across embedding sets
dist_figs = compare_all_metric_distributions(embeddings_list, labels)
for metric_name, fig in dist_figs.items():
    fig.show()
```

## How It Helps

The Embeddings Evaluator package provides a focused toolkit for assessing and comparing embeddings:

- **Understand Embedding Characteristics:** Gain insights into the similarity and intrinsic properties of your embeddings.
- **Compare Models:** Evaluate different embedding models or parameters to determine the best option for your specific tasks.
- **Visualize Relationships:** Explore the structure and clustering of your embeddings in 2D and 3D spaces.
- **Assess Dimensionality:** Understand the intrinsic dimensionality of your embeddings and how it relates to their performance.
- **Compare Metric Distributions:** Analyze and compare the distributions of key metrics across different embedding sets.
- **Optimize Performance:** Use the provided metrics and visualizations to fine-tune your embedding generation process for better results in downstream tasks.

The automated nature of these evaluations, combined with interactive visualizations, allows you to quickly gain deep insights into your embeddings without manual intervention, making it an ideal tool for embedding evaluation and optimization workflows.
