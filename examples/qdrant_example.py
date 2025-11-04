"""
Example: Using Embeddings Evaluator with Qdrant Vector Database

This script demonstrates how to load embeddings from Qdrant collections
and compare them using the embeddings_evaluator package.
"""

import numpy as np
from embeddings_evaluator import (
    load_qdrant_embeddings,
    plot_model_comparison
)
from embeddings_evaluator.comparison import save_comparison_plot


def example_basic_usage():
    """Basic example: Load embeddings from a local Qdrant instance."""
    print("=" * 60)
    print("Example 1: Basic Usage - Local Qdrant")
    print("=" * 60)

    # Load embeddings from a local Qdrant collection
    embeddings = load_qdrant_embeddings(
        collection_name="my_embeddings",
        url="http://localhost:6333"
    )

    print(f"\nLoaded embeddings shape: {embeddings.shape}")
    print(f"First embedding (first 5 dims): {embeddings[0][:5]}")


def example_qdrant_cloud():
    """Example: Connect to Qdrant Cloud with API key."""
    print("\n" + "=" * 60)
    print("Example 2: Qdrant Cloud with API Key")
    print("=" * 60)

    # Load from Qdrant Cloud
    embeddings = load_qdrant_embeddings(
        collection_name="my_embeddings",
        url="https://xyz-example.cloud.qdrant.io:6333",
        api_key="your_api_key_here"
    )

    print(f"\nLoaded {len(embeddings)} embeddings from Qdrant Cloud")


def example_with_filters():
    """Example: Load embeddings with payload filters."""
    print("\n" + "=" * 60)
    print("Example 3: Using Filters")
    print("=" * 60)

    # Load only embeddings matching specific criteria
    embeddings = load_qdrant_embeddings(
        collection_name="my_embeddings",
        url="http://localhost:6333",
        filter_conditions={
            "category": "science",
            "year": 2024
        }
    )

    print(f"\nLoaded {len(embeddings)} filtered embeddings")


def example_limited_retrieval():
    """Example: Load only a subset of embeddings."""
    print("\n" + "=" * 60)
    print("Example 4: Limited Retrieval")
    print("=" * 60)

    # Load only first 1000 embeddings
    embeddings = load_qdrant_embeddings(
        collection_name="my_embeddings",
        url="http://localhost:6333",
        limit=1000
    )

    print(f"\nLoaded {len(embeddings)} embeddings (limited to 1000)")


def example_named_vectors():
    """Example: Load from collection with multiple named vectors."""
    print("\n" + "=" * 60)
    print("Example 5: Named Vectors")
    print("=" * 60)

    # Load specific named vector from collection
    embeddings = load_qdrant_embeddings(
        collection_name="multi_vector_collection",
        url="http://localhost:6333",
        vector_name="dense_vector"  # Specify which vector to load
    )

    print(f"\nLoaded {len(embeddings)} embeddings from 'dense_vector' field")


def example_compare_models():
    """Example: Compare embeddings from different Qdrant collections."""
    print("\n" + "=" * 60)
    print("Example 6: Comparing Multiple Models")
    print("=" * 60)

    # Load embeddings from different collections representing different models
    embeddings_dict = {}

    # Model 1: Small embedding model
    embeddings_dict["Model 250"] = load_qdrant_embeddings(
        collection_name="embeddings_250",
        url="http://localhost:6333"
    )

    # Model 2: Medium embedding model
    embeddings_dict["Model 500"] = load_qdrant_embeddings(
        collection_name="embeddings_500",
        url="http://localhost:6333"
    )

    # Model 3: Large embedding model
    embeddings_dict["Model 1000"] = load_qdrant_embeddings(
        collection_name="embeddings_1000",
        url="http://localhost:6333"
    )

    print("\nGenerating comparison visualization...")

    # Generate comparison plot
    fig = plot_model_comparison(embeddings_dict)

    # Save the visualization
    save_comparison_plot(fig, 'qdrant_model_comparison.png')

    print("Visualization saved to 'qdrant_model_comparison.png'")


def example_mixed_sources():
    """Example: Compare embeddings from Qdrant and FAISS."""
    print("\n" + "=" * 60)
    print("Example 7: Mixed Sources (Qdrant + FAISS)")
    print("=" * 60)

    from embeddings_evaluator import load_faiss_embeddings

    embeddings_dict = {}

    # Load from Qdrant
    embeddings_dict["Qdrant Model"] = load_qdrant_embeddings(
        collection_name="my_embeddings",
        url="http://localhost:6333"
    )

    # Load from FAISS
    embeddings_dict["FAISS Model"] = load_faiss_embeddings(
        "path/to/index.faiss"
    )

    # Compare them
    fig = plot_model_comparison(embeddings_dict)
    save_comparison_plot(fig, 'mixed_sources_comparison.png')

    print("Mixed sources comparison saved!")


def example_generic_loader():
    """Example: Using the generic load_embeddings function."""
    print("\n" + "=" * 60)
    print("Example 8: Generic Loader (Auto-detection)")
    print("=" * 60)

    from embeddings_evaluator import load_embeddings

    # Auto-detect source type
    embeddings1 = load_embeddings("index.faiss")  # Detects FAISS
    embeddings2 = load_embeddings("embeddings.npy")  # Detects numpy

    # Explicit Qdrant
    embeddings3 = load_embeddings(
        "my_collection",
        source_type="qdrant",
        url="http://localhost:6333"
    )

    print("Loaded embeddings from multiple sources using generic loader")


def example_error_handling():
    """Example: Proper error handling."""
    print("\n" + "=" * 60)
    print("Example 9: Error Handling")
    print("=" * 60)

    try:
        embeddings = load_qdrant_embeddings(
            collection_name="non_existent_collection",
            url="http://localhost:6333"
        )
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        embeddings = load_qdrant_embeddings(
            collection_name="my_collection",
            url="http://wrong-host:6333"
        )
    except ConnectionError as e:
        print(f"Caught connection error: {e}")


def example_without_normalization():
    """Example: Load embeddings without normalization."""
    print("\n" + "=" * 60)
    print("Example 10: Without Normalization")
    print("=" * 60)

    # Load without normalizing to unit length
    embeddings = load_qdrant_embeddings(
        collection_name="my_embeddings",
        url="http://localhost:6333",
        normalize=False
    )

    print(f"Loaded unnormalized embeddings")
    print(f"First embedding norm: {np.linalg.norm(embeddings[0])}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Embeddings Evaluator - Qdrant Examples")
    print("=" * 60)
    print("\nNote: These examples assume you have Qdrant running locally")
    print("or have access to a Qdrant Cloud instance with the specified")
    print("collections. Adjust collection names and URLs as needed.")
    print("=" * 60)

    # Run examples (comment out those you don't want to run)
    try:
        # example_basic_usage()
        # example_qdrant_cloud()
        # example_with_filters()
        # example_limited_retrieval()
        # example_named_vectors()
        # example_compare_models()
        # example_mixed_sources()
        # example_generic_loader()
        # example_error_handling()
        # example_without_normalization()

        print("\n" + "=" * 60)
        print("Examples completed!")
        print("=" * 60)
        print("\nTo run specific examples, uncomment them in the main block.")

    except ImportError as e:
        print(f"\nError: {e}")
        print("\nMake sure to install required dependencies:")
        print("  pip install -r requirements.txt")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("\nMake sure Qdrant is running and collections exist.")
