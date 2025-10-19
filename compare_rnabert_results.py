#!/usr/bin/env python3
"""
Compare RNABERT evaluation results across different checkpoints.

This script reads the zero-shot evaluation results and creates a comparison table.
"""

import json
import os
import sys
from pathlib import Path


def load_results(results_path):
    """Load results from JSON file."""
    if not os.path.exists(results_path):
        return None

    with open(results_path, 'r') as f:
        return json.load(f)


def format_metrics(metrics):
    """Format metrics dictionary for display."""
    if metrics is None:
        return "N/A", "N/A", "N/A"

    precision = metrics.get('precision', 0)
    recall = metrics.get('recall', 0)
    f1 = metrics.get('f1', 0)

    return f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"


def print_comparison(results_dict):
    """Print comparison table."""

    print("\n" + "="*100)
    print("RNABERT ZERO-SHOT SECONDARY STRUCTURE PREDICTION - RESULTS COMPARISON")
    print("="*100)
    print()

    # Table header
    print(f"{'Checkpoint':<40} {'Architecture':<20} {'Split':<10} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 100)

    for checkpoint_name, data in results_dict.items():
        if data is None:
            print(f"{checkpoint_name:<40} {'N/A':<20} {'N/A':<10} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
            continue

        # Get model info
        num_heads = data.get('num_heads', 'N/A')
        num_layers = data.get('num_layers', 'N/A')
        hidden_size = data.get('hidden_size', 'N/A')
        arch = f"{num_layers}L-{num_heads}H-{hidden_size}D"

        # Validation metrics
        val_metrics = data.get('validation', {})
        val_p, val_r, val_f1 = format_metrics(val_metrics)

        # Test metrics
        test_metrics = data.get('test', {})
        test_p, test_r, test_f1 = format_metrics(test_metrics)

        # Print rows
        print(f"{checkpoint_name:<40} {arch:<20} {'Val':<10} {val_p:<12} {val_r:<12} {val_f1:<12}")
        print(f"{'':<40} {'':<20} {'Test':<10} {test_p:<12} {test_r:<12} {test_f1:<12}")
        print("-" * 100)

    print("="*100)
    print()


def print_summary(results_dict):
    """Print summary of best results."""

    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print()

    best_val_f1 = -1
    best_val_checkpoint = None

    best_test_f1 = -1
    best_test_checkpoint = None

    for checkpoint_name, data in results_dict.items():
        if data is None:
            continue

        val_f1 = data.get('validation', {}).get('f1', 0)
        test_f1 = data.get('test', {}).get('f1', 0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_checkpoint = checkpoint_name

        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_test_checkpoint = checkpoint_name

    if best_val_checkpoint:
        print(f"Best Validation F1:  {best_val_f1:.4f}  ({best_val_checkpoint})")

    if best_test_checkpoint:
        print(f"Best Test F1:        {best_test_f1:.4f}  ({best_test_checkpoint})")

    print()
    print("="*100)
    print()


def main():
    """Main function."""

    # Default results directory
    results_dir = Path("./rnabert_ssp_results/results")

    # Check if custom directory provided
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        print("\nUsage: python compare_rnabert_results.py [results_dir]")
        print("\nDefault: ./rnabert_ssp_results/results")
        sys.exit(1)

    print(f"\nLoading results from: {results_dir}")

    # Define checkpoints to compare
    checkpoints = {
        "RNABERT_a16_b6_checkpoint494241": "rnabert_a16_b6_checkpoint494241",
        "RNABERT_a8_b8_checkpoint670000": "rnabert_a8_b8_checkpoint670000",
    }

    # Load all results
    results_dict = {}
    for display_name, run_name in checkpoints.items():
        results_path = results_dir / run_name / "zero_shot_results.json"
        results_dict[display_name] = load_results(results_path)

        if results_dict[display_name] is None:
            print(f"  ⚠ Not found: {results_path}")
        else:
            print(f"  ✓ Loaded: {results_path}")

    # Check if any results were found
    if all(v is None for v in results_dict.values()):
        print("\n ERROR: No results found!")
        print("\nMake sure you have run the evaluation script first:")
        print("  bash run_rnabert_ssp_eval.sh")
        print("  OR")
        print("  sbatch submit_rnabert_ssp_eval.sh")
        sys.exit(1)

    # Print comparison table
    print_comparison(results_dict)

    # Print summary
    print_summary(results_dict)

    print("Done!")


if __name__ == "__main__":
    main()
