import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from .metrics import pairwise_similarities

def find_peak(similarities):
    """Find the peak of the distribution and its location."""
    kde = gaussian_kde(similarities)
    x_range = np.linspace(min(similarities), max(similarities), 200)
    density = kde(x_range)
    peak_idx = np.argmax(density)
    return x_range[peak_idx], density[peak_idx]

def plot_model_comparison(embeddings_dict):
    """
    Create visualization comparing pairwise cosine similarity distributions of multiple models.
    
    Parameters:
    embeddings_dict: dict of {model_name: embeddings_array}
    """
    # Set up colors
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    
    # Calculate statistics for all models
    stats = {}
    for name, embeddings in embeddings_dict.items():
        similarities = pairwise_similarities(embeddings)
        peak_x, peak_y = find_peak(similarities)
        stats[name] = {
            'similarities': similarities,
            'mean': np.mean(similarities),
            'std': np.std(similarities),
            'median': np.median(similarities),
            'peak_x': peak_x,
            'peak_y': peak_y
        }
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Plot histograms
    for (name, model_stats), color in zip(stats.items(), colors):
        plt.hist(model_stats['similarities'], bins=50, alpha=0.6, density=True, 
                color=color,
                label=f'{name} (μ={model_stats["mean"]:.3f}, σ={model_stats["std"]:.3f}, m={model_stats["median"]:.3f})')
        # Add vertical line for mean
        plt.axvline(model_stats['mean'], color=color, linestyle='--', alpha=0.5)
    
    # Customize plot
    plt.title('Pairwise Cosine Similarity Distributions')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add peak information
    peak_text = "Peak (cos_sim, amplitude):\n"
    for name, model_stats in stats.items():
        peak_text += f"{name}: ({model_stats['peak_x']:.3f}, {model_stats['peak_y']:.3f})\n"
    
    plt.text(0.02, 0.98, peak_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set x-axis limits to [0, 1] for cosine similarity
    plt.xlim(0, 1)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return plt.gcf()

def save_comparison_plot(fig, filename):
    """Save the comparison plot to a file."""
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
