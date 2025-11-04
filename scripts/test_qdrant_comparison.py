"""
Test script for comparing Qdrant collections using environment variables.

Usage:
1. Copy .env.example to .env
2. Fill in your Qdrant URL, API key, and collection names
3. Run: python test_qdrant_comparison.py
"""

import os
from dotenv import load_dotenv
from embeddings_evaluator import load_qdrant_embeddings, plot_model_comparison
from embeddings_evaluator.comparison import save_comparison_plot

# Load environment variables from .env file
load_dotenv()

def main():
    # Get configuration from environment variables
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    collection_1 = os.getenv('COLLECTION_1')
    collection_2 = os.getenv('COLLECTION_2')

    # Validate configuration
    if not qdrant_url:
        print("Error: QDRANT_URL not set in .env file")
        print("Please copy .env.example to .env and fill in your configuration")
        return

    if not collection_1 or not collection_2:
        print("Error: COLLECTION_1 and COLLECTION_2 must be set in .env file")
        print("Please specify the collection names you want to compare")
        return

    # Use API key only if provided (empty string means no auth)
    api_key = qdrant_api_key if qdrant_api_key else None

    print("=" * 60)
    print("Qdrant Collections Comparison")
    print("=" * 60)
    print(f"\nQdrant URL: {qdrant_url}")
    print(f"API Key: {'***' if api_key else 'Not set (local instance)'}")
    print(f"\nCollections to compare:")
    print(f"  1. {collection_1}")
    print(f"  2. {collection_2}")
    print("\n" + "=" * 60)

    try:
        # Load embeddings from both collections
        print(f"\nLoading embeddings from '{collection_1}'...")
        embeddings_1 = load_qdrant_embeddings(
            collection_name=collection_1,
            url=qdrant_url,
            api_key=api_key
        )

        print(f"\nLoading embeddings from '{collection_2}'...")
        embeddings_2 = load_qdrant_embeddings(
            collection_name=collection_2,
            url=qdrant_url,
            api_key=api_key
        )

        # Create comparison dictionary
        embeddings_dict = {
            collection_1: embeddings_1,
            collection_2: embeddings_2
        }

        print("\n" + "=" * 60)
        print("Generating comparison visualization...")
        print("=" * 60)

        # Generate comparison plot
        fig = plot_model_comparison(embeddings_dict)

        # Save the visualization
        output_file = 'qdrant_collections_comparison.png'
        save_comparison_plot(fig, output_file)

        print(f"\n✓ Success! Comparison saved to: {output_file}")
        print("\n" + "=" * 60)
        print("Analysis Summary")
        print("=" * 60)

        # Print summary statistics
        from embeddings_evaluator.metrics import pairwise_similarities
        import numpy as np

        for name, embeddings in embeddings_dict.items():
            similarities = pairwise_similarities(embeddings)
            print(f"\n{name}:")
            print(f"  Shape: {embeddings.shape}")
            print(f"  Mean similarity: {np.mean(similarities):.4f}")
            print(f"  Std deviation: {np.std(similarities):.4f}")
            print(f"  Median similarity: {np.median(similarities):.4f}")
            print(f"  Min similarity: {np.min(similarities):.4f}")
            print(f"  Max similarity: {np.max(similarities):.4f}")

        print("\n" + "=" * 60)
        print("RAG Quality Assessment")
        print("=" * 60)

        # Calculate statistics for comparison
        stats_comparison = {}
        for name, embeddings in embeddings_dict.items():
            similarities = pairwise_similarities(embeddings)
            stats_comparison[name] = {
                'mean': np.mean(similarities),
                'std': np.std(similarities),
                'median': np.median(similarities)
            }

        # Comparative analysis
        print("\nComparative Analysis:")
        print("-" * 60)

        # Find which model has better characteristics for RAG
        models = list(stats_comparison.keys())
        if len(models) == 2:
            model1, model2 = models
            stats1 = stats_comparison[model1]
            stats2 = stats_comparison[model2]

            print(f"\nMean Similarity Comparison:")
            if stats1['mean'] < stats2['mean']:
                print(f"  → {model1} has lower mean similarity (better separation)")
                print(f"    {model1}: {stats1['mean']:.4f} vs {model2}: {stats2['mean']:.4f}")
            elif stats2['mean'] < stats1['mean']:
                print(f"  → {model2} has lower mean similarity (better separation)")
                print(f"    {model2}: {stats2['mean']:.4f} vs {model1}: {stats1['mean']:.4f}")
            else:
                print(f"  → Both models have similar mean similarity")

            print(f"\nStandard Deviation Comparison:")
            if stats1['std'] > stats2['std']:
                print(f"  → {model1} has higher std deviation (more discriminative range)")
                print(f"    {model1}: {stats1['std']:.4f} vs {model2}: {stats2['std']:.4f}")
            elif stats2['std'] > stats1['std']:
                print(f"  → {model2} has higher std deviation (more discriminative range)")
                print(f"    {model2}: {stats2['std']:.4f} vs {model1}: {stats1['std']:.4f}")
            else:
                print(f"  → Both models have similar std deviation")

            print(f"\nRAG Suitability:")
            # Determine which is better based on both metrics
            score1 = (1 - stats1['mean']) + stats1['std']  # Lower mean + higher std is better
            score2 = (1 - stats2['mean']) + stats2['std']

            if score1 > score2:
                diff = ((score1 - score2) / score2) * 100
                print(f"  → {model1} appears more suitable for RAG")
                print(f"    (Better separation and discriminative power)")
            elif score2 > score1:
                diff = ((score2 - score1) / score1) * 100
                print(f"  → {model2} appears more suitable for RAG")
                print(f"    (Better separation and discriminative power)")
            else:
                print(f"  → Both models show similar RAG suitability")

        print("\n" + "=" * 60)
        print("\nKey Principles for RAG:")
        print("  • Lower mean similarity = better document separation")
        print("  • Higher std deviation = more discriminative range")
        print("  • Distribution peak near 0 = most pairs are dissimilar")
        print("  • Right-skewed distribution = only similar docs have high scores")
        print("\nRefer to the visualization for distribution shape analysis.")
        print("=" * 60)

    except ConnectionError as e:
        print(f"\n✗ Connection Error: {e}")
        print("\nPlease check:")
        print("  1. Qdrant is running and accessible")
        print("  2. QDRANT_URL is correct in .env")
        print("  3. Network connectivity")

    except ValueError as e:
        print(f"\n✗ Value Error: {e}")
        print("\nPlease check:")
        print("  1. Collection names are correct")
        print("  2. Collections exist in your Qdrant instance")
        print("  3. Collections contain vectors")

    except Exception as e:
        print(f"\n✗ Unexpected Error: {e}")
        print("\nPlease check your configuration and try again")


if __name__ == "__main__":
    main()
