"""
Loaders for embeddings from various sources (FAISS, Qdrant, numpy arrays).
"""
import numpy as np
from typing import Optional, Dict, Any, Union


def load_faiss_embeddings(index_path: str, normalize: bool = True) -> np.ndarray:
    """
    Load embeddings from a FAISS index.

    Parameters:
    -----------
    index_path : str
        Path to the FAISS index file
    normalize : bool, default=True
        Whether to normalize embeddings to unit length

    Returns:
    --------
    embeddings : np.ndarray
        Array of embeddings with shape (n_vectors, dimension)

    Raises:
    -------
    ImportError
        If faiss is not installed
    ValueError
        If the index type is not supported
    FileNotFoundError
        If the index file doesn't exist
    """
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "faiss-cpu is required to load FAISS indices. "
            "Install it with: pip install faiss-cpu"
        )

    print(f"Loading FAISS index from {index_path}...")

    # Load the faiss index
    index = faiss.read_index(index_path)

    # Handle IndexIDMap wrapper
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
        raise ValueError(
            f"Unsupported FAISS index type: {type(raw_index)}. "
            "Currently only IndexFlatL2 is supported."
        )

    if normalize:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        print("Embeddings normalized to unit length")

    return embeddings


def load_qdrant_embeddings(
    collection_name: str,
    url: str = "http://localhost:6333",
    api_key: Optional[str] = None,
    filter_conditions: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    normalize: bool = True,
    vector_name: Optional[str] = None
) -> np.ndarray:
    """
    Load embeddings from a Qdrant collection.

    Parameters:
    -----------
    collection_name : str
        Name of the Qdrant collection
    url : str, default="http://localhost:6333"
        Qdrant server URL (local or cloud)
    api_key : str, optional
        API key for Qdrant Cloud authentication
    filter_conditions : dict, optional
        Filter conditions to apply when retrieving vectors
        Example: {"category": "science", "year": 2024}
    limit : int, optional
        Maximum number of vectors to retrieve. If None, retrieves all vectors.
    normalize : bool, default=True
        Whether to normalize embeddings to unit length
    vector_name : str, optional
        Name of the vector field (for collections with multiple vectors).
        If None, uses the default/first vector.

    Returns:
    --------
    embeddings : np.ndarray
        Array of embeddings with shape (n_vectors, dimension)

    Raises:
    -------
    ImportError
        If qdrant-client is not installed
    ValueError
        If the collection doesn't exist or has no vectors
    ConnectionError
        If unable to connect to Qdrant server

    Examples:
    ---------
    # Local Qdrant
    >>> embeddings = load_qdrant_embeddings("my_collection")

    # Qdrant Cloud
    >>> embeddings = load_qdrant_embeddings(
    ...     collection_name="my_collection",
    ...     url="https://xyz.cloud.qdrant.io",
    ...     api_key="your_api_key"
    ... )

    # With filters
    >>> embeddings = load_qdrant_embeddings(
    ...     collection_name="my_collection",
    ...     filter_conditions={"category": "science"}
    ... )
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchValue
    except ImportError:
        raise ImportError(
            "qdrant-client is required to load from Qdrant. "
            "Install it with: pip install qdrant-client"
        )

    print(f"Connecting to Qdrant at {url}...")

    try:
        # Initialize client
        client = QdrantClient(url=url, api_key=api_key)

        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]

        if collection_name not in collection_names:
            raise ValueError(
                f"Collection '{collection_name}' not found. "
                f"Available collections: {collection_names}"
            )

        # Get collection info
        collection_info = client.get_collection(collection_name)

        # Handle both single and named vectors
        vectors_config = collection_info.config.params.vectors
        if isinstance(vectors_config, dict):
            # Named vectors - get the first one or specified one
            if vector_name:
                if vector_name not in vectors_config:
                    raise ValueError(
                        f"Vector '{vector_name}' not found. "
                        f"Available vectors: {list(vectors_config.keys())}"
                    )
                vector_size = vectors_config[vector_name].size
            else:
                # Use first vector
                first_vector_name = list(vectors_config.keys())[0]
                vector_size = vectors_config[first_vector_name].size
                print(f"Using vector field: '{first_vector_name}'")
        else:
            # Single unnamed vector
            vector_size = vectors_config.size

        total_points = collection_info.points_count

        print(f"Collection '{collection_name}': {total_points} points, dimension {vector_size}")

        # Build filter if provided
        qdrant_filter = None
        if filter_conditions:
            conditions = []
            for key, value in filter_conditions.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            qdrant_filter = Filter(must=conditions)
            print(f"Applying filter: {filter_conditions}")

        # Determine batch size and total to retrieve
        batch_size = 100
        max_points = limit if limit is not None else total_points

        print(f"Retrieving up to {max_points} vectors...")

        # Scroll through all points
        embeddings_list = []
        offset = None
        retrieved = 0

        while retrieved < max_points:
            # Calculate how many to fetch in this batch
            current_batch_size = min(batch_size, max_points - retrieved)

            # Scroll through points
            points, next_offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=qdrant_filter,
                limit=current_batch_size,
                offset=offset,
                with_vectors=True
            )

            if not points:
                break

            # Extract vectors
            for point in points:
                if vector_name:
                    # Named vector
                    if vector_name in point.vector:
                        embeddings_list.append(point.vector[vector_name])
                    else:
                        raise ValueError(
                            f"Vector '{vector_name}' not found in point. "
                            f"Available vectors: {list(point.vector.keys())}"
                        )
                else:
                    # Default vector (could be dict or list)
                    if isinstance(point.vector, dict):
                        # Multiple vectors - use first one
                        first_vector_name = list(point.vector.keys())[0]
                        embeddings_list.append(point.vector[first_vector_name])
                    else:
                        # Single vector
                        embeddings_list.append(point.vector)

            retrieved += len(points)
            offset = next_offset

            # Break if no more points
            if next_offset is None:
                break

            print(f"Retrieved {retrieved}/{max_points} vectors...", end='\r')

        print(f"\nSuccessfully retrieved {len(embeddings_list)} vectors")

        if not embeddings_list:
            raise ValueError(
                f"No vectors found in collection '{collection_name}' "
                f"with the given filters"
            )

        # Convert to numpy array
        embeddings = np.array(embeddings_list, dtype=np.float32)

        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
            print("Embeddings normalized to unit length")

        return embeddings

    except Exception as e:
        if "Connection" in str(e) or "connect" in str(e).lower():
            raise ConnectionError(
                f"Failed to connect to Qdrant at {url}. "
                f"Make sure Qdrant is running and accessible. Error: {e}"
            )
        raise


def load_numpy_embeddings(
    file_path: str,
    normalize: bool = True
) -> np.ndarray:
    """
    Load embeddings from a numpy file (.npy or .npz).

    Parameters:
    -----------
    file_path : str
        Path to the numpy file
    normalize : bool, default=True
        Whether to normalize embeddings to unit length

    Returns:
    --------
    embeddings : np.ndarray
        Array of embeddings with shape (n_vectors, dimension)

    Raises:
    -------
    FileNotFoundError
        If the file doesn't exist
    ValueError
        If the array has incorrect shape
    """
    print(f"Loading embeddings from {file_path}...")

    if file_path.endswith('.npz'):
        # Load from npz (compressed)
        data = np.load(file_path)
        # Get the first array if multiple arrays exist
        embeddings = data[data.files[0]]
    else:
        # Load from npy
        embeddings = np.load(file_path)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D array, got shape {embeddings.shape}. "
            "Embeddings should have shape (n_vectors, dimension)"
        )

    print(f"Loaded {embeddings.shape[0]} vectors of dimension {embeddings.shape[1]}")

    if normalize:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        print("Embeddings normalized to unit length")

    return embeddings


def load_embeddings(
    source: Union[str, np.ndarray],
    source_type: str = "auto",
    normalize: bool = True,
    **kwargs
) -> np.ndarray:
    """
    Generic loader that auto-detects source type and loads embeddings.

    Parameters:
    -----------
    source : str or np.ndarray
        Source of embeddings (file path, collection name, or numpy array)
    source_type : str, default="auto"
        Type of source: "auto", "faiss", "qdrant", "numpy", or "array"
    normalize : bool, default=True
        Whether to normalize embeddings to unit length
    **kwargs
        Additional arguments passed to specific loaders

    Returns:
    --------
    embeddings : np.ndarray
        Array of embeddings with shape (n_vectors, dimension)

    Examples:
    ---------
    # Auto-detect from file extension
    >>> embeddings = load_embeddings("index.faiss")
    >>> embeddings = load_embeddings("embeddings.npy")

    # Explicit source type
    >>> embeddings = load_embeddings(
    ...     "my_collection",
    ...     source_type="qdrant",
    ...     url="http://localhost:6333"
    ... )

    # From numpy array
    >>> embeddings = load_embeddings(my_array, source_type="array")
    """
    # Handle numpy array directly
    if isinstance(source, np.ndarray):
        if source.ndim != 2:
            raise ValueError(
                f"Expected 2D array, got shape {source.shape}. "
                "Embeddings should have shape (n_vectors, dimension)"
            )
        if normalize:
            return source / np.linalg.norm(source, axis=1)[:, np.newaxis]
        return source

    # Auto-detect source type from file extension
    if source_type == "auto":
        if isinstance(source, str):
            if source.endswith('.faiss') or source.endswith('.index'):
                source_type = "faiss"
            elif source.endswith('.npy') or source.endswith('.npz'):
                source_type = "numpy"
            else:
                # Assume it's a Qdrant collection name
                source_type = "qdrant"
        else:
            raise ValueError(
                f"Cannot auto-detect source type for {type(source)}. "
                "Please specify source_type explicitly."
            )

    # Load based on source type
    if source_type == "faiss":
        return load_faiss_embeddings(source, normalize=normalize)
    elif source_type == "qdrant":
        return load_qdrant_embeddings(source, normalize=normalize, **kwargs)
    elif source_type == "numpy":
        return load_numpy_embeddings(source, normalize=normalize)
    else:
        raise ValueError(
            f"Unknown source_type: {source_type}. "
            "Must be one of: 'auto', 'faiss', 'qdrant', 'numpy', 'array'"
        )
