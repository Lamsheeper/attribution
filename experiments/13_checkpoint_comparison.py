#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Comparison Script for Checkpoint Results

This script compares the results of two model checkpoints on a benchmark.
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


def generate_comparison_summary(model1_results, model2_results, output_filepath):
    """
    Generate a comparison summary of two model checkpoint results.
    
    Args:
        model1_results: Results from the first model checkpoint
        model2_results: Results from the second model checkpoint
        output_filepath: Path to save the comparison summary
        
    Returns:
        Dictionary containing the comparison summary
    """
    # Create problem ID maps for faster lookup
    model1_map = {result["problem_id"]: result for result in model1_results}
    model2_map = {result["problem_id"]: result for result in model2_results}
    
    # Find all unique problem IDs
    all_problem_ids = set(model1_map.keys()).union(set(model2_map.keys()))
    
    # Create comparison summary
    comparison = []
    
    for problem_id in sorted(all_problem_ids):
        model1_result = model1_map.get(problem_id, {})
        model2_result = model2_map.get(problem_id, {})
        
        model1_correct = model1_result.get("any_correct", False)
        model2_correct = model2_result.get("any_correct", False)
        
        # Include only problems where at least one model got it right
        if model1_correct or model2_correct:
            comparison.append({
                "problem_id": problem_id,
                "model1": {
                    "any_correct": model1_correct,
                    "correct_rate": model1_result.get("correct_rate", 0.0),
                    "avg_logprob": model1_result.get("avg_logprob", 0.0),
                },
                "model2": {
                    "any_correct": model2_correct,
                    "correct_rate": model2_result.get("correct_rate", 0.0),
                    "avg_logprob": model2_result.get("avg_logprob", 0.0),
                },
                "true_output": model1_result.get("true_output", model2_result.get("true_output", "")),
            })
    
    # Calculate summary statistics
    model1_correct_count = sum(1 for item in comparison if item["model1"]["any_correct"])
    model2_correct_count = sum(1 for item in comparison if item["model2"]["any_correct"])
    only_model1_correct = sum(1 for item in comparison 
                               if item["model1"]["any_correct"] and not item["model2"]["any_correct"])
    only_model2_correct = sum(1 for item in comparison 
                                if not item["model1"]["any_correct"] and item["model2"]["any_correct"])
    both_correct = sum(1 for item in comparison 
                      if item["model1"]["any_correct"] and item["model2"]["any_correct"])
    
    # Add summary statistics
    summary = {
        "total_problems_with_any_correct": len(comparison),
        "model1_correct_count": model1_correct_count,
        "model2_correct_count": model2_correct_count,
        "only_model1_correct": only_model1_correct,
        "only_model2_correct": only_model2_correct,
        "both_correct": both_correct,
        "average_model1_correct_rate": np.mean([item["model1"]["correct_rate"] for item in comparison]),
        "average_model2_correct_rate": np.mean([item["model2"]["correct_rate"] for item in comparison]),
        "average_model1_logprob": np.mean([item["model1"]["avg_logprob"] for item in comparison]),
        "average_model2_logprob": np.mean([item["model2"]["avg_logprob"] for item in comparison]),
        "problems": comparison
    }
    
    # Save comparison summary
    with open(output_filepath, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Comparison summary saved to {output_filepath}")
    return summary


def plot_comparison(summary, output_dir, model1_name="Checkpoint 1", model2_name="Checkpoint 2"):
    """
    Generate visualizations comparing two model checkpoint performances.
    
    Args:
        summary: Comparison summary dictionary
        output_dir: Directory to save visualizations
        model1_name: Display name for the first model
        model2_name: Display name for the second model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame for easier plotting
    data = []
    for problem in summary["problems"]:
        data.append({
            "problem_id": problem["problem_id"],
            "model": model1_name,
            "correct_rate": problem["model1"]["correct_rate"],
            "avg_logprob": problem["model1"]["avg_logprob"]
        })
        data.append({
            "problem_id": problem["problem_id"],
            "model": model2_name,
            "correct_rate": problem["model2"]["correct_rate"],
            "avg_logprob": problem["model2"]["avg_logprob"]
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
    model1_rates = [problem["model1"]["correct_rate"] for problem in summary["problems"]]
    model2_rates = [problem["model2"]["correct_rate"] for problem in summary["problems"]]
    plt.scatter(model1_rates, model2_rates, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel(f"{model1_name} Correct Rate")
    plt.ylabel(f"{model2_name} Correct Rate")
    plt.title("Correct Rate Comparison (per problem)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correct_rate_scatter.png"))
    
    # Plot Venn diagram of correctness
    plt.figure(figsize=(8, 8))
    from matplotlib_venn import venn2
    
    only_model1 = summary["only_model1_correct"]
    only_model2 = summary["only_model2_correct"]
    both = summary["both_correct"]
    
    venn2(subsets=(only_model1, only_model2, both), 
          set_labels=(model1_name, model2_name))
    plt.title("Problems Solved Correctly")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correctness_venn.png"))
    
    print(f"Visualizations saved to {output_dir}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Compare two model checkpoint evaluation results")
    parser.add_argument("--checkpoint1", type=str, help="Path to first checkpoint results JSON")
    parser.add_argument("--checkpoint2", type=str, help="Path to second checkpoint results JSON")
    parser.add_argument("--output_dir", type=str, help="Directory to save output files")
    parser.add_argument("--name1", type=str, default="Checkpoint 1", help="Name for first checkpoint in reports")
    parser.add_argument("--name2", type=str, default="Checkpoint 2", help="Name for second checkpoint in reports")
    args = parser.parse_args()
    
    # Use provided arguments or default values
    if args.checkpoint1 and args.checkpoint2 and args.output_dir:
        checkpoint1_file = args.checkpoint1
        checkpoint2_file = args.checkpoint2
        base_dir = args.output_dir
        output_file = os.path.join(base_dir, f'comparison_summary_{args.name1}_{args.name2}.json')
        viz_dir = os.path.join(base_dir, f'visualizations_{args.name1}_{args.name2}')
    else:
        # Default paths (original behavior)
        base_dir = '/share/u/yu.stev/attribution/data/cruxeval_results/checkpoints'
        checkpoint1_file = os.path.join(base_dir, 'checkpoint1_results.json')
        checkpoint2_file = os.path.join(base_dir, 'checkpoint2_results.json')
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
    print(f"{model1_name} correct: {summary['model1_correct_count']} problems")
    print(f"{model2_name} correct: {summary['model2_correct_count']} problems")
    print(f"Both models correct: {summary['both_correct']} problems")
    print(f"Only {model1_name} correct: {summary['only_model1_correct']} problems")
    print(f"Only {model2_name} correct: {summary['only_model2_correct']} problems")
    print(f"Avg {model1_name} correct rate: {summary['average_model1_correct_rate']:.4f}")
    print(f"Avg {model2_name} correct rate: {summary['average_model2_correct_rate']:.4f}")
    
    # Generate visualizations
    try:
        plot_comparison(summary, viz_dir, model1_name, model2_name)
    except ImportError:
        print("Warning: Could not generate visualizations. Make sure pandas, matplotlib, seaborn, and matplotlib_venn are installed.")


if __name__ == "__main__":
    main() 