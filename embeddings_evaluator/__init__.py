from .metrics import (
    pairwise_similarities,
    get_all_metrics
)

# Import optimized methods
from .metrics_optimized import (
    pairwise_similarities_sampled,
    pairwise_similarities_chunked,
    pairwise_similarities_parallel,
    pairwise_similarities_gpu,
    pairwise_similarities_multi_gpu,
    pairwise_similarities_auto,
    detect_hardware
)

from .comparison import (
    plot_model_comparison
)

from .loaders import (
    load_faiss_embeddings,
    load_qdrant_embeddings,
    load_numpy_embeddings,
    load_embeddings
)

__all__ = [
    'pairwise_similarities',
    'get_all_metrics',
    'plot_model_comparison',
    'load_faiss_embeddings',
    'load_qdrant_embeddings',
    'load_numpy_embeddings',
    'load_embeddings',
    # Optimized methods
    'pairwise_similarities_sampled',
    'pairwise_similarities_chunked',
    'pairwise_similarities_parallel',
    'pairwise_similarities_gpu',
    'pairwise_similarities_multi_gpu',
    'pairwise_similarities_auto',
    'detect_hardware'
]
