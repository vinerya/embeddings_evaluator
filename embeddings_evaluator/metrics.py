import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.cluster import KMeans

def ensure_2d(embeddings):
    if embeddings.ndim == 1:
        return embeddings.reshape(1, -1)
    return embeddings

def mean_cosine_similarity(embeddings):
    embeddings = ensure_2d(embeddings)
    similarities = 1 - pairwise_distances(embeddings, metric='cosine')
    return np.mean(similarities)

def intrinsic_dimensionality(embeddings):
    embeddings = ensure_2d(embeddings)
    distances = pairwise_distances(embeddings)
    np.fill_diagonal(distances, np.inf)
    nearest_distances = np.min(distances, axis=1)
    return 1 / np.mean(np.log(distances / nearest_distances[:, np.newaxis]))

def silhouette_avg(embeddings, n_clusters=5):
    embeddings = ensure_2d(embeddings)
    if len(embeddings) < n_clusters:
        return np.nan
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return silhouette_score(embeddings, labels)

def get_all_metrics(embeddings):
    metrics = {
        "mean_cosine_similarity": mean_cosine_similarity(embeddings),
        "intrinsic_dimensionality": intrinsic_dimensionality(embeddings),
        "silhouette_score": silhouette_avg(embeddings)
    }
    return metrics
