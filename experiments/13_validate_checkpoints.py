#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generation Length Comparison Script for Model Checkpoints

This script compares the average full generation lengths of two model checkpoints
on a benchmark. It loads evaluation results from JSON files and calculates statistics
on the full generation text lengths.
"""

import json
import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(filepath):
    """
    Load results from a JSON file.
    
    Args:
        filepath: Path to the result JSON file
        
    Returns:
        List of problem results
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []


def convert_to_serializable(obj):
    """
    Convert NumPy types to regular Python types for JSON serialization.
    
    Args:
        obj: The object to convert
        
    Returns:
        JSON serializable object
    """
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj


def calculate_generation_stats(results):
    """
    Calculate statistics about generation lengths.
    
    Args:
        results: List of problem results
        
    Returns:
        Dictionary with generation length statistics
    """
    # Extract the lengths of all full generations
    lengths = [len(result.get("full_generated_text", "")) for result in results if "full_generated_text" in result]
    
    if not lengths:
        return {
            "count": 0,
            "avg_length": 0,
            "min_length": 0,
            "max_length": 0,
            "median_length": 0,
            "std_length": 0,
        }
    
    return {
        "count": len(lengths),
        "avg_length": np.mean(lengths),
        "min_length": np.min(lengths),
        "max_length": np.max(lengths),
        "median_length": np.median(lengths),
        "std_length": np.std(lengths),
    }


def compare_generation_lengths(model1_results, model2_results, output_filepath, model1_name, model2_name):
    """
    Compare generation lengths between two models and save results.
    
    Args:
        model1_results: Results from the first model checkpoint
        model2_results: Results from the second model checkpoint
        output_filepath: Path to save the comparison summary
        model1_name: Name of the first model
        model2_name: Name of the second model
        
    Returns:
        Dictionary containing the comparison summary
    """
    # Calculate statistics for each model
    model1_stats = calculate_generation_stats(model1_results)
    model2_stats = calculate_generation_stats(model2_results)
    
    # Create detailed problem-by-problem comparison
    model1_map = {result["problem_id"]: result for result in model1_results}
    model2_map = {result["problem_id"]: result for result in model2_results}
    
    # Find common problem IDs
    common_problem_ids = set(model1_map.keys()).intersection(set(model2_map.keys()))
    
    # Create problem-specific length comparison
    problems_comparison = []
    for problem_id in sorted(common_problem_ids):
        model1_result = model1_map.get(problem_id, {})
        model2_result = model2_map.get(problem_id, {})
        
        model1_length = len(model1_result.get("full_generated_text", "")) if "full_generated_text" in model1_result else 0
        model2_length = len(model2_result.get("full_generated_text", "")) if "full_generated_text" in model2_result else 0
        
        problems_comparison.append({
            "problem_id": problem_id,
            f"{model1_name}_length": model1_length,
            f"{model2_name}_length": model2_length,
            "length_diff": model2_length - model1_length,
            "length_ratio": model2_length / model1_length if model1_length > 0 else float('inf')
        })
    
    # Build comprehensive comparison summary
    comparison = {
        f"{model1_name}_stats": model1_stats,
        f"{model2_name}_stats": model2_stats,
        "length_difference": model2_stats["avg_length"] - model1_stats["avg_length"],
        "length_ratio": model2_stats["avg_length"] / model1_stats["avg_length"] if model1_stats["avg_length"] > 0 else float('inf'),
        "common_problems_count": len(common_problem_ids),
        "problems": problems_comparison
    }
    
    # Convert all NumPy types to standard Python types for JSON serialization
    comparison = convert_to_serializable(comparison)
    
    # Save comparison summary
    with open(output_filepath, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Generation length comparison saved to {output_filepath}")
    return comparison


def plot_length_comparison(comparison, output_dir, model1_name, model2_name):
    """
    Generate visualizations comparing generation lengths between two models.
    
    Args:
        comparison: Comparison summary dictionary
        output_dir: Directory to save visualizations
        model1_name: Display name for the first model
        model2_name: Display name for the second model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract problem-specific data
    problems = comparison["problems"]
    model1_lengths = [p[f"{model1_name}_length"] for p in problems]
    model2_lengths = [p[f"{model2_name}_length"] for p in problems]
    
    # Boxplot comparison
    plt.figure(figsize=(10, 6))
    data = []
    for p in problems:
        data.append({
            "Model": model1_name,
            "Generation Length": p[f"{model1_name}_length"]
        })
        data.append({
            "Model": model2_name,
            "Generation Length": p[f"{model2_name}_length"]
        })
    
    df = pd.DataFrame(data)
    sns.boxplot(x="Model", y="Generation Length", data=df)
    plt.title("Generation Length Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "generation_length_boxplot.png"))
    
    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(model1_lengths, model2_lengths, alpha=0.5)
    
    # Add the line y=x for reference
    max_length = max(max(model1_lengths), max(model2_lengths))
    plt.plot([0, max_length], [0, max_length], 'k--')
    
    plt.xlabel(f"{model1_name} Generation Length")
    plt.ylabel(f"{model2_name} Generation Length")
    plt.title("Generation Length Comparison (by problem)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "generation_length_scatter.png"))
    
    # Histogram of length differences
    plt.figure(figsize=(10, 6))
    diffs = [p["length_diff"] for p in problems]
    plt.hist(diffs, bins=30)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel(f"{model2_name} Length - {model1_name} Length")
    plt.ylabel("Count")
    plt.title("Distribution of Generation Length Differences")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "generation_length_diff_hist.png"))
    
    print(f"Length comparison visualizations saved to {output_dir}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Compare generation lengths between two model checkpoints")
    parser.add_argument("--checkpoint1", type=str, required=True, help="Path to first checkpoint results JSON")
    parser.add_argument("--checkpoint2", type=str, required=True, help="Path to second checkpoint results JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--name1", type=str, default="Checkpoint1", help="Name for first checkpoint in reports")
    parser.add_argument("--name2", type=str, default="Checkpoint2", help="Name for second checkpoint in reports")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Output file paths
    output_file = os.path.join(args.output_dir, f'generation_length_comparison_{args.name1}_{args.name2}.json')
    viz_dir = os.path.join(args.output_dir, f'visualizations_{args.name1}_{args.name2}')
    
    # Load results
    print(f"Loading first checkpoint results from {args.checkpoint1}")
    checkpoint1_results = load_results(args.checkpoint1)
    print(f"Loading second checkpoint results from {args.checkpoint2}")
    checkpoint2_results = load_results(args.checkpoint2)
    
    # Check if data loaded successfully
    if not checkpoint1_results or not checkpoint2_results:
        print("Error: Failed to load one or both result files.")
        return
    
    print(f"Loaded {len(checkpoint1_results)} results from first checkpoint and {len(checkpoint2_results)} results from second checkpoint.")
    
    # Generate comparison summary
    comparison = compare_generation_lengths(
        checkpoint1_results, 
        checkpoint2_results, 
        output_file,
        args.name1,
        args.name2
    )
    
    # Print key findings
    print("\nKey Findings:")
    print(f"Total problems analyzed: {comparison['common_problems_count']}")
    print(f"{args.name1} average generation length: {comparison[f'{args.name1}_stats']['avg_length']:.2f} characters")
    print(f"{args.name2} average generation length: {comparison[f'{args.name2}_stats']['avg_length']:.2f} characters")
    print(f"Difference in average length: {comparison['length_difference']:.2f} characters")
    print(f"Ratio of average lengths ({args.name2}/{args.name1}): {comparison['length_ratio']:.2f}")
    
    # Generate visualizations
    try:
        plot_length_comparison(comparison, viz_dir, args.name1, args.name2)
    except Exception as e:
        print(f"Error generating visualizations: {e}")


if __name__ == "__main__":
    main()
