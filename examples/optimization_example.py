"""
Example demonstrating all optimization methods for pairwise similarity calculation.

This script shows how to use each optimization method and compares their performance.
"""

import numpy as np
import time
from embeddings_evaluator import (
    pairwise_similarities_auto,
    pairwise_similarities_sampled,
    pairwise_similarities_chunked,
    pairwise_similarities_parallel,
    pairwise_similarities_gpu,
    pairwise_similarities_multi_gpu,
    detect_hardware
)


def generate_test_embeddings(n_vectors=5000, n_dimensions=768):
    """Generate random embeddings for testing."""
    print(f"Generating {n_vectors} random embeddings with {n_dimensions} dimensions...")
    return np.random.randn(n_vectors, n_dimensions).astype(np.float32)


def time_method(method_name, method_func, embeddings, **kwargs):
    """Time a specific method and return results."""
    print(f"\n{'='*60}")
    print(f"Testing: {method_name}")
    print(f"{'='*60}")

    start_time = time.time()
    try:
        similarities = method_func(embeddings, **kwargs)
        elapsed_time = time.time() - start_time

        print(f"✓ Success!")
        print(f"  Time: {elapsed_time:.2f} seconds")
        print(f"  Similarities calculated: {len(similarities):,}")
        print(f"  Mean similarity: {np.mean(similarities):.4f}")
        print(f"  Std deviation: {np.std(similarities):.4f}")
        print(f"  Min: {np.min(similarities):.4f}, Max: {np.max(similarities):.4f}")

        return {
            'method': method_name,
            'time': elapsed_time,
            'count': len(similarities),
            'mean': np.mean(similarities),
            'std': np.std(similarities),
            'success': True
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"✗ Failed: {str(e)}")
        return {
            'method': method_name,
            'time': elapsed_time,
            'success': False,
            'error': str(e)
        }


def main():
    """Run optimization examples."""
    print("="*60)
    print("Pairwise Similarity Optimization Examples")
    print("="*60)

    # Detect hardware
    print("\n1. Detecting Hardware...")
    hw = detect_hardware()
    print(f"   CPU cores: {hw['cpu_cores']}")
    print(f"   Has GPU: {hw['has_gpu']}")
    print(f"   Number of GPUs: {hw['num_gpus']}")
    if hw['has_gpu']:
        print(f"   GPU memory: {hw['gpu_memory'] / 1e9:.1f} GB")

    # Generate test data
    print("\n2. Generating Test Data...")
    n_vectors = 5000  # Adjust based on your system
    embeddings = generate_test_embeddings(n_vectors=n_vectors)
    print(f"   Shape: {embeddings.shape}")
    print(f"   Total pairs: {n_vectors * (n_vectors - 1) // 2:,}")

    # Store results
    results = []

    # Method 1: Auto-select (recommended)
    print("\n3. Testing Methods...")
    result = time_method(
        "Auto-Select (Recommended)",
        pairwise_similarities_auto,
        embeddings,
        verbose=True
    )
    results.append(result)

    # Method 2: Sampling
    result = time_method(
        "Sampling (50K samples)",
        pairwise_similarities_sampled,
        embeddings,
        sample_size=50000
    )
    results.append(result)

    # Method 3: Chunked
    result = time_method(
        "Chunked Processing",
        pairwise_similarities_chunked,
        embeddings,
        chunk_size=500
    )
    results.append(result)

    # Method 4: Parallel CPU
    if hw['cpu_cores'] >= 2:
        result = time_method(
            f"CPU Parallel ({hw['cpu_cores']} cores)",
            pairwise_similarities_parallel,
            embeddings,
            n_jobs=hw['cpu_cores']
        )
        results.append(result)

    # Method 5: GPU Single
    if hw['has_gpu']:
        result = time_method(
            "GPU Single",
            pairwise_similarities_gpu,
            embeddings
        )
        results.append(result)

    # Method 6: Multi-GPU
    if hw['num_gpus'] >= 2:
        result = time_method(
            f"Multi-GPU ({hw['num_gpus']} GPUs)",
            pairwise_similarities_multi_gpu,
            embeddings
        )
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    successful_results = [r for r in results if r['success']]
    if successful_results:
        print(f"\n{'Method':<30} {'Time (s)':<12} {'Speedup':<10}")
        print("-" * 60)

        # Find baseline (slowest method)
        baseline_time = max(r['time'] for r in successful_results)

        for result in successful_results:
            speedup = baseline_time / result['time']
            print(f"{result['method']:<30} {result['time']:>10.2f}  {speedup:>8.1f}x")

        # Fastest method
        fastest = min(successful_results, key=lambda x: x['time'])
        print(f"\n⭐ Fastest method: {fastest['method']} ({fastest['time']:.2f}s)")

        # Recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        print(f"\nFor {n_vectors:,} vectors:")

        if hw['has_gpu']:
            print("  ✓ GPU available - Use GPU methods for best performance")
        else:
            print("  ✓ No GPU - Use sampling for large datasets")

        if n_vectors < 1000:
            print("  ✓ Small dataset - Any method works well")
        elif n_vectors < 5000:
            print("  ✓ Medium dataset - Chunked or parallel recommended")
        else:
            print("  ✓ Large dataset - GPU or sampling recommended")

        print("\n  💡 Tip: Use pairwise_similarities_auto() for automatic selection")

    print("\n" + "="*60)
    print("Example complete!")
    print("="*60)
    print("\nFor more information, see OPTIMIZATION_GUIDE.md")


if __name__ == "__main__":
    main()
