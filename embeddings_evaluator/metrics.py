import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats import gaussian_kde, spearmanr
from sklearn.neighbors import NearestNeighbors

def ensure_2d(embeddings):
    if embeddings.ndim == 1:
        return embeddings.reshape(1, -1)
    return embeddings

def normalize_embeddings(embeddings):
    """Normalize embeddings to unit length."""
    return embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]

def pairwise_similarities(embeddings, metric='cosine'):
    """Calculate pairwise similarities within a single set of embeddings."""
    embeddings = ensure_2d(embeddings)
    embeddings = normalize_embeddings(embeddings)
    
    similarities = 1 - pairwise_distances(embeddings, metric='cosine')
    upper_tri_indices = np.triu_indices(len(similarities), k=1)
    return similarities[upper_tri_indices]

def cross_model_similarity(embeddings1, embeddings2):
    """
    Calculate similarity between embeddings of the same documents from different models.
    Assumes embeddings1[i] and embeddings2[i] represent the same document.
    
    Returns:
    - mean_similarity: Average similarity between corresponding embeddings
    - similarity_std: Standard deviation of similarities
    - similarity_distribution: Distribution of similarities for visualization
    """
    embeddings1 = normalize_embeddings(ensure_2d(embeddings1))
    embeddings2 = normalize_embeddings(ensure_2d(embeddings2))
    
    # Calculate cosine similarity between corresponding embeddings
    similarities = np.sum(embeddings1 * embeddings2, axis=1)
    return {
        'mean': np.mean(similarities),
        'std': np.std(similarities),
        'distribution': similarities
    }

def neighborhood_preservation(embeddings1, embeddings2, k=10):
    """
    Calculate how well each model preserves nearest neighbors.
    Returns percentage of shared neighbors for each document.
    """
    embeddings1 = normalize_embeddings(ensure_2d(embeddings1))
    embeddings2 = normalize_embeddings(ensure_2d(embeddings2))
    
    # Find k nearest neighbors for each document in both models
    nn1 = NearestNeighbors(n_neighbors=k+1)  # +1 because a point is its own neighbor
    nn1.fit(embeddings1)
    neighbors1 = nn1.kneighbors(embeddings1, return_distance=False)[:, 1:]  # exclude self
    
    nn2 = NearestNeighbors(n_neighbors=k+1)
    nn2.fit(embeddings2)
    neighbors2 = nn2.kneighbors(embeddings2, return_distance=False)[:, 1:]  # exclude self
    
    # Calculate overlap between neighborhoods
    preservation_scores = []
    for n1, n2 in zip(neighbors1, neighbors2):
        overlap = len(set(n1) & set(n2))
        preservation_scores.append(overlap / k)
    
    return {
        'mean': np.mean(preservation_scores),
        'std': np.std(preservation_scores),
        'distribution': np.array(preservation_scores)
    }

def ranking_consistency(embeddings1, embeddings2):
    """
    Calculate consistency of similarity rankings between models.
    Uses Spearman correlation between similarity rankings.
    """
    embeddings1 = normalize_embeddings(ensure_2d(embeddings1))
    embeddings2 = normalize_embeddings(ensure_2d(embeddings2))
    
    # Calculate all pairwise similarities for both models
    sims1 = 1 - pairwise_distances(embeddings1, metric='cosine')
    sims2 = 1 - pairwise_distances(embeddings2, metric='cosine')
    
    # Calculate rank correlation for each document
    correlations = []
    n_docs = len(embeddings1)
    
    for i in range(n_docs):
        # Get similarity rankings for document i to all other documents
        ranking1 = sims1[i]
        ranking2 = sims2[i]
        # Calculate Spearman correlation
        corr, _ = spearmanr(ranking1, ranking2)
        correlations.append(corr)
    
    return {
        'mean': np.mean(correlations),
        'std': np.std(correlations),
        'distribution': np.array(correlations)
    }

def evaluate_model_pair(embeddings1, embeddings2):
    """
    Comprehensive evaluation of two embedding sets of the same documents.
    """
    results = {
        'cross_model_similarity': cross_model_similarity(embeddings1, embeddings2),
        'neighborhood_preservation': neighborhood_preservation(embeddings1, embeddings2),
        'ranking_consistency': ranking_consistency(embeddings1, embeddings2),
        'model1_distribution': pairwise_similarities(embeddings1),
        'model2_distribution': pairwise_similarities(embeddings2)
    }
    
    # Add interpretations
    interpretations = []
    
    # Interpret cross-model similarity
    cms = results['cross_model_similarity']['mean']
    if cms > 0.9:
        interpretations.append("Models produce very similar embeddings")
    elif cms > 0.7:
        interpretations.append("Models produce moderately similar embeddings")
    else:
        interpretations.append("Models produce substantially different embeddings")
    
    # Interpret neighborhood preservation
    np_mean = results['neighborhood_preservation']['mean']
    if np_mean > 0.8:
        interpretations.append("High neighborhood preservation between models")
    elif np_mean > 0.5:
        interpretations.append("Moderate neighborhood preservation")
    else:
        interpretations.append("Low neighborhood preservation - models see different relationships")
    
    # Interpret ranking consistency
    rc_mean = results['ranking_consistency']['mean']
    if rc_mean > 0.8:
        interpretations.append("Very consistent similarity rankings between models")
    elif rc_mean > 0.5:
        interpretations.append("Moderately consistent rankings")
    else:
        interpretations.append("Inconsistent rankings - models disagree on document similarities")
    
    results['interpretations'] = interpretations
    return results

def get_all_metrics(embeddings, metric='cosine'):
    """Maintained for backward compatibility"""
    similarities = pairwise_similarities(embeddings, metric)
    return {
        "pairwise_similarities": similarities,
        "metric_info": {
            'name': 'Cosine Similarity',
            'range': [0, 1],
            'interpretation': 'Higher is more similar'
        }
    }
