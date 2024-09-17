import numpy as np
from embeddings_evaluator import (
    evaluate_embeddings, plot_metrics, plot_interactive_scatter,
    plot_similarity_heatmap, plot_intrinsic_dimensionality,
    compare_all_metric_distributions
)

# Generate sample embeddings
np.random.seed(42)
embeddings1 = np.random.rand(100, 300)
embeddings2 = np.random.rand(100, 300)
embeddings3 = np.random.rand(100, 300)

# Evaluate a single set of embeddings
print("Evaluating a single set of embeddings:")
single_eval = evaluate_embeddings(embeddings1, label="Model A")
print(single_eval)
print("\n")

# Evaluate and compare multiple sets of embeddings
print("Evaluating and comparing multiple sets of embeddings:")
embeddings_list = [embeddings1, embeddings2, embeddings3]
labels = ['Model A', 'Model B', 'Model C']
multi_eval = evaluate_embeddings(embeddings_list, labels)
print(multi_eval)
print("\n")

# Plot comparison metrics
print("Generating comparison metrics plot...")
metrics_fig = plot_metrics(multi_eval)
metrics_fig.write_html("comparison_metrics.html")
print("Comparison metrics plot saved as 'comparison_metrics.html'")
print("\n")

# Create interactive scatter plots
print("Generating interactive scatter plots...")
all_embeddings = np.vstack(embeddings_list)
all_labels = np.repeat(labels, [len(e) for e in embeddings_list])

pca_2d_fig = plot_interactive_scatter(all_embeddings, all_labels, method='pca', dimensions=2)
pca_2d_fig.write_html("pca_2d_scatter.html")
print("2D PCA scatter plot saved as 'pca_2d_scatter.html'")

tsne_3d_fig = plot_interactive_scatter(all_embeddings, all_labels, method='tsne', dimensions=3)
tsne_3d_fig.write_html("tsne_3d_scatter.html")
print("3D t-SNE scatter plot saved as 'tsne_3d_scatter.html'")
print("\n")

# Plot similarity heatmap for each model
print("Generating similarity heatmaps...")
for i, (embeddings, label) in enumerate(zip(embeddings_list, labels)):
    heatmap_fig = plot_similarity_heatmap(embeddings, labels=[f'Doc{j}' for j in range(len(embeddings))])
    heatmap_fig.write_html(f"similarity_heatmap_{label}.html")
    print(f"Similarity heatmap for {label} saved as 'similarity_heatmap_{label}.html'")
print("\n")

# Plot intrinsic dimensionality for each model
print("Generating intrinsic dimensionality plots...")
for i, (embeddings, label) in enumerate(zip(embeddings_list, labels)):
    id_fig = plot_intrinsic_dimensionality(embeddings)
    id_fig.write_html(f"intrinsic_dimensionality_{label}.html")
    print(f"Intrinsic dimensionality plot for {label} saved as 'intrinsic_dimensionality_{label}.html'")
print("\n")

# Compare distributions of all metrics across embedding sets
print("Comparing distributions of all metrics across embedding sets...")
dist_figs = compare_all_metric_distributions(embeddings_list, labels)
for metric_name, fig in dist_figs.items():
    filename = f"{metric_name.lower().replace(' ', '_')}_distribution.html"
    fig.write_html(filename)
    print(f"Distribution comparison for {metric_name} saved as '{filename}'")
print("\n")

print("All tests completed. Please check the generated HTML files for the visualizations.")