#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Comparison Script for CruxEval Results

This script compares the results of standard and quantized models on the CruxEval benchmark.
It generates a summary JSON file containing only problems where at least one model produced
a correct answer, along with correct rates and mean log probabilities for both models.
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


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


def generate_comparison_summary(standard_results, quantized_results, output_filepath):
    """
    Generate a comparison summary of standard and quantized model results.
    
    Args:
        standard_results: Results from the standard model
        quantized_results: Results from the quantized model
        output_filepath: Path to save the comparison summary
        
    Returns:
        Dictionary containing the comparison summary
    """
    # Create problem ID maps for faster lookup
    standard_map = {result["problem_id"]: result for result in standard_results}
    quantized_map = {result["problem_id"]: result for result in quantized_results}
    
    # Find all unique problem IDs
    all_problem_ids = set(standard_map.keys()).union(set(quantized_map.keys()))
    
    # Create comparison summary
    comparison = []
    
    for problem_id in sorted(all_problem_ids):
        standard_result = standard_map.get(problem_id, {})
        quantized_result = quantized_map.get(problem_id, {})
        
        standard_correct = standard_result.get("any_correct", False)
        quantized_correct = quantized_result.get("any_correct", False)
        
        # Include only problems where at least one model got it right
        if standard_correct or quantized_correct:
            comparison.append({
                "problem_id": problem_id,
                "standard": {
                    "any_correct": standard_correct,
                    "correct_rate": standard_result.get("correct_rate", 0.0),
                    "avg_logprob": standard_result.get("avg_logprob", 0.0),
                },
                "quantized": {
                    "any_correct": quantized_correct,
                    "correct_rate": quantized_result.get("correct_rate", 0.0),
                    "avg_logprob": quantized_result.get("avg_logprob", 0.0),
                },
                "true_output": standard_result.get("true_output", quantized_result.get("true_output", "")),
            })
    
    # Calculate summary statistics
    standard_correct_count = sum(1 for item in comparison if item["standard"]["any_correct"])
    quantized_correct_count = sum(1 for item in comparison if item["quantized"]["any_correct"])
    only_standard_correct = sum(1 for item in comparison 
                               if item["standard"]["any_correct"] and not item["quantized"]["any_correct"])
    only_quantized_correct = sum(1 for item in comparison 
                                if not item["standard"]["any_correct"] and item["quantized"]["any_correct"])
    both_correct = sum(1 for item in comparison 
                      if item["standard"]["any_correct"] and item["quantized"]["any_correct"])
    
    # Add summary statistics
    summary = {
        "total_problems_with_any_correct": len(comparison),
        "standard_correct_count": standard_correct_count,
        "quantized_correct_count": quantized_correct_count,
        "only_standard_correct": only_standard_correct,
        "only_quantized_correct": only_quantized_correct,
        "both_correct": both_correct,
        "average_standard_correct_rate": np.mean([item["standard"]["correct_rate"] for item in comparison]),
        "average_quantized_correct_rate": np.mean([item["quantized"]["correct_rate"] for item in comparison]),
        "average_standard_logprob": np.mean([item["standard"]["avg_logprob"] for item in comparison]),
        "average_quantized_logprob": np.mean([item["quantized"]["avg_logprob"] for item in comparison]),
        "problems": comparison
    }
    
    # Save comparison summary
    with open(output_filepath, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Comparison summary saved to {output_filepath}")
    return summary


def plot_comparison(summary, output_dir):
    """
    Generate visualizations comparing standard and quantized model performance.
    
    Args:
        summary: Comparison summary dictionary
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame for easier plotting
    data = []
    for problem in summary["problems"]:
        data.append({
            "problem_id": problem["problem_id"],
            "model": "standard",
            "correct_rate": problem["standard"]["correct_rate"],
            "avg_logprob": problem["standard"]["avg_logprob"]
        })
        data.append({
            "problem_id": problem["problem_id"],
            "model": "quantized",
            "correct_rate": problem["quantized"]["correct_rate"],
            "avg_logprob": problem["quantized"]["avg_logprob"]
        })
    
    df = pd.DataFrame(data)
    
    # Plot correct rate comparison
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="model", y="correct_rate", data=df)
    plt.title("Correct Rate Comparison")
    plt.ylabel("Correct Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correct_rate_comparison.png"))
    
    # Plot log probability comparison
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="model", y="avg_logprob", data=df)
    plt.title("Average Log Probability Comparison")
    plt.ylabel("Average Log Probability")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "logprob_comparison.png"))
    
    # Plot scatter of correct rates
    plt.figure(figsize=(10, 6))
    standard_rates = [problem["standard"]["correct_rate"] for problem in summary["problems"]]
    quantized_rates = [problem["quantized"]["correct_rate"] for problem in summary["problems"]]
    plt.scatter(standard_rates, quantized_rates, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("Standard Model Correct Rate")
    plt.ylabel("Quantized Model Correct Rate")
    plt.title("Correct Rate Comparison (per problem)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correct_rate_scatter.png"))
    
    # Plot Venn diagram of correctness
    plt.figure(figsize=(8, 8))
    from matplotlib_venn import venn2
    
    only_standard = summary["only_standard_correct"]
    only_quantized = summary["only_quantized_correct"]
    both = summary["both_correct"]
    
    venn2(subsets=(only_standard, only_quantized, both), 
          set_labels=('Standard Model', 'Quantized Model'))
    plt.title("Problems Solved Correctly")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correctness_venn.png"))
    
    print(f"Visualizations saved to {output_dir}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Compare two model evaluation results on CruxEval benchmark")
    parser.add_argument("--checkpoint1", type=str, help="Path to first checkpoint results JSON")
    parser.add_argument("--checkpoint2", type=str, help="Path to second checkpoint results JSON")
    parser.add_argument("--output_dir", type=str, help="Directory to save output files")
    parser.add_argument("--name1", type=str, default="checkpoint1", help="Name for first checkpoint in reports")
    parser.add_argument("--name2", type=str, default="checkpoint2", help="Name for second checkpoint in reports")
    args = parser.parse_args()
    
    # Use provided arguments or default values
    if args.checkpoint1 and args.checkpoint2 and args.output_dir:
        checkpoint1_file = args.checkpoint1
        checkpoint2_file = args.checkpoint2
        base_dir = args.output_dir
        output_file = os.path.join(base_dir, 'comparison_summary.json')
        viz_dir = os.path.join(base_dir, 'visualizations')
    else:
        # Default paths (original behavior)
        base_dir = '/share/u/yu.stev/attribution/data/cruxeval_results/k_sample'
        checkpoint1_file = os.path.join(base_dir, 'cruxeval_k4_temp0.3_topp0.95_base13b.json')
        checkpoint2_file = os.path.join(base_dir, 'cruxeval_k4_temp0.3_topp0.95_base13b_8bit.json')
        output_file = os.path.join(base_dir, 'comparison_summary.json')
        viz_dir = os.path.join(base_dir, 'visualizations')
    
    # Load results
    print(f"Loading first checkpoint results from {checkpoint1_file}")
    checkpoint1_results = load_results(checkpoint1_file)
    print(f"Loading second checkpoint results from {checkpoint2_file}")
    checkpoint2_results = load_results(checkpoint2_file)
    
    # Check if data loaded successfully
    if not checkpoint1_results or not checkpoint2_results:
        print("Error: Failed to load one or both result files.")
        return
    
    print(f"Loaded {len(checkpoint1_results)} results from first checkpoint and {len(checkpoint2_results)} results from second checkpoint.")
    
    # Generate comparison summary
    summary = generate_comparison_summary(checkpoint1_results, checkpoint2_results, output_file)
    
    # Rename keys in summary for display
    model1_name = args.name1
    model2_name = args.name2
    print("\nKey Findings:")
    print(f"Total problems with at least one correct solution: {summary['total_problems_with_any_correct']}")
    print(f"{model1_name} correct: {summary['standard_correct_count']} problems")
    print(f"{model2_name} correct: {summary['quantized_correct_count']} problems")
    print(f"Both models correct: {summary['both_correct']} problems")
    print(f"Only {model1_name} correct: {summary['only_standard_correct']} problems")
    print(f"Only {model2_name} correct: {summary['only_quantized_correct']} problems")
    print(f"Avg {model1_name} correct rate: {summary['average_standard_correct_rate']:.4f}")
    print(f"Avg {model2_name} correct rate: {summary['average_quantized_correct_rate']:.4f}")
    
    # Generate visualizations
    try:
        plot_comparison(summary, viz_dir)
    except ImportError:
        print("Warning: Could not generate visualizations. Make sure pandas, matplotlib, seaborn, and matplotlib_venn are installed.")


if __name__ == "__main__":
    main()
