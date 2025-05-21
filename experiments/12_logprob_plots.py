#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Log Probability Analysis and Visualization Script

This script creates specialized visualizations to analyze the relationship between 
log probability (confidence) and correctness in CruxEval results for standard 
and quantized models.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


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


def create_confidence_correctness_heatmap(std_results, quant_results, output_path, num_bins=10):
    """
    Create a heatmap matrix showing the relationship between confidence (log probability) 
    and correctness across different confidence levels.
    
    Args:
        std_results: List of results from the standard model
        quant_results: List of results from the quantized model
        output_path: Path to save the visualization
        num_bins: Number of confidence bins to create
    """
    # Create problem ID to result mapping for faster lookup
    std_map = {r["problem_id"]: r for r in std_results}
    quant_map = {r["problem_id"]: r for r in quant_results}
    
    # Find common problem IDs
    common_pids = set(std_map.keys()).intersection(set(quant_map.keys()))
    print(f"Found {len(common_pids)} common problems between standard and quantized models")
    
    if not common_pids:
        print("Error: No common problems found between models")
        return
    
    # Collect confidence and correctness data
    std_data = []
    quant_data = []
    
    for pid in common_pids:
        std_result = std_map[pid]
        quant_result = quant_map[pid]
        
        std_logprob = std_result.get("avg_logprob", 0)
        std_correct = 1 if std_result.get("any_correct", False) else 0
        
        quant_logprob = quant_result.get("avg_logprob", 0)
        quant_correct = 1 if quant_result.get("any_correct", False) else 0
        
        std_data.append((pid, std_logprob, std_correct))
        quant_data.append((pid, quant_logprob, quant_correct))
    
    # Sort data by confidence (log probability)
    std_data.sort(key=lambda x: x[1])
    quant_data.sort(key=lambda x: x[1])
    
    # Calculate bin sizes
    bin_size = len(common_pids) // num_bins
    if bin_size == 0:
        print("Error: Not enough problems for the specified number of bins")
        bin_size = 1
        num_bins = min(num_bins, len(common_pids))
    
    # Create matrix to hold correctness rates
    matrix = np.zeros((num_bins, 2))
    bin_ranges = []
    
    # Process standard model bins
    for i in range(num_bins):
        start_idx = i * bin_size
        end_idx = min((i+1) * bin_size, len(std_data))
        
        if start_idx >= len(std_data):
            break
            
        std_bin = std_data[start_idx:end_idx]
        std_logprobs = [x[1] for x in std_bin]
        std_correct = [x[2] for x in std_bin]
        
        # Store bin ranges for labeling
        bin_ranges.append({
            "idx": i,
            "std_min": min(std_logprobs),
            "std_max": max(std_logprobs)
        })
        
        # Calculate correctness rate for standard model
        matrix[i, 0] = np.mean(std_correct) if std_correct else 0
    
    # Process quantized model bins
    for i in range(num_bins):
        start_idx = i * bin_size
        end_idx = min((i+1) * bin_size, len(quant_data))
        
        if start_idx >= len(quant_data):
            break
            
        quant_bin = quant_data[start_idx:end_idx]
        quant_logprobs = [x[1] for x in quant_bin]
        quant_correct = [x[2] for x in quant_bin]
        
        # Update bin ranges to include quantized model data
        if i < len(bin_ranges):
            bin_ranges[i].update({
                "quant_min": min(quant_logprobs),
                "quant_max": max(quant_logprobs)
            })
        
        # Calculate correctness rate for quantized model
        matrix[i, 1] = np.mean(quant_correct) if quant_correct else 0
    
    # Create heatmap visualization
    plt.figure(figsize=(10, 12))
    
    # Create row labels showing log probability ranges
    # Format: "Bin N: [-0.12 to -0.08] (Standard) / [-0.10 to -0.06] (Quantized)"
    row_labels = []
    for i, r in enumerate(bin_ranges):
        std_range = f"[{r['std_min']:.3f} to {r['std_max']:.3f}]"
        quant_range = f"[{r.get('quant_min', 0):.3f} to {r.get('quant_max', 0):.3f}]"
        row_labels.append(f"Bin {i+1}: {std_range} (S) / {quant_range} (Q)")
    
    # Plot heatmap
    ax = sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
                     xticklabels=["Standard", "Quantized"], 
                     yticklabels=row_labels, 
                     cbar_kws={"label": "Correctness Rate"})
    
    # Add titles and labels
    plt.title("Confidence-Correctness Matrix", fontsize=14)
    plt.ylabel("Confidence Bins (Low to High)", fontsize=12)
    
    # Adjust y-tick label size if they're too long
    plt.tick_params(axis='y', labelsize=9)
    
    # Add annotation explaining the interpretation
    plt.figtext(0.5, 0.01, 
                "Each cell shows what percentage of problems in that confidence bin were answered correctly.\n"
                "Bins are sorted from lowest confidence (bottom) to highest confidence (top).", 
                ha="center", fontsize=10, bbox=dict(facecolor="white", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    
    print(f"Confidence-correctness matrix heatmap saved to {output_path}")
    
    # Also create a summary of each bin for analysis
    print("\nConfidence Bin Summaries:")
    for i, r in enumerate(bin_ranges):
        std_correct_rate = matrix[i, 0]
        quant_correct_rate = matrix[i, 1]
        print(f"Bin {i+1} (lowest confidence = {i+1}/{num_bins}):")
        print(f"  Standard: range [{r['std_min']:.3f} to {r['std_max']:.3f}], correctness: {std_correct_rate:.2f}")
        print(f"  Quantized: range [{r.get('quant_min', 0):.3f} to {r.get('quant_max', 0):.3f}], correctness: {quant_correct_rate:.2f}")


def create_correctness_by_confidence_plot(std_results, quant_results, output_path):
    """
    Create a scatter plot showing correctness vs. confidence with trend lines
    for both models.
    
    Args:
        std_results: List of results from the standard model
        quant_results: List of results from the quantized model
        output_path: Path to save the visualization
    """
    # Create problem ID to result mapping for faster lookup
    std_map = {r["problem_id"]: r for r in std_results}
    quant_map = {r["problem_id"]: r for r in quant_results}
    
    # Find common problem IDs
    common_pids = set(std_map.keys()).intersection(set(quant_map.keys()))
    
    if not common_pids:
        print("Error: No common problems found between models")
        return
    
    # Collect data
    data = []
    for pid in common_pids:
        std_result = std_map[pid]
        quant_result = quant_map[pid]
        
        data.append({
            "problem_id": pid,
            "model": "Standard",
            "logprob": std_result.get("avg_logprob", 0),
            "correct": 1 if std_result.get("any_correct", False) else 0
        })
        
        data.append({
            "problem_id": pid,
            "model": "Quantized",
            "logprob": quant_result.get("avg_logprob", 0),
            "correct": 1 if quant_result.get("any_correct", False) else 0
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    
    # Add jitter to correct values for better visualization
    jitter = 0.05
    df['correct_jittered'] = df['correct'] + np.random.uniform(-jitter, jitter, size=len(df))
    
    # Plot points for each model
    sns.scatterplot(data=df[df['model'] == 'Standard'], 
                   x='logprob', y='correct_jittered', 
                   alpha=0.6, color='blue', label='Standard')
    
    sns.scatterplot(data=df[df['model'] == 'Quantized'], 
                   x='logprob', y='correct_jittered', 
                   alpha=0.6, color='red', label='Quantized')
    
    # Add trend lines using logistic regression
    for model, color in [('Standard', 'blue'), ('Quantized', 'red')]:
        model_df = df[df['model'] == model]
        
        # Sort by logprob for smooth line
        model_df = model_df.sort_values('logprob')
        x = model_df['logprob']
        y = model_df['correct']
        
        # Calculate moving average for trend line
        window = min(30, len(x) // 5)
        if window > 0:
            y_smooth = y.rolling(window=window, center=True).mean()
            plt.plot(x, y_smooth, color=color, linewidth=2)
    
    # Embellish plot
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.title("Correctness vs. Log Probability", fontsize=14)
    plt.xlabel("Log Probability (Higher = More Confident)", fontsize=12)
    plt.ylabel("Correct (0 = No, 1 = Yes)", fontsize=12)
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend(title="Model")
    
    # Add annotation about jitter
    plt.figtext(0.5, 0.01, 
                "Note: Y-values are jittered slightly for better visualization.\n"
                "Trend lines show moving averages of correctness rates.", 
                ha="center", fontsize=10, bbox=dict(facecolor="white", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    
    print(f"Correctness by confidence plot saved to {output_path}")


def create_confidence_density_by_correctness(std_results, quant_results, output_path):
    """
    Create density plots of log probabilities for correct vs. incorrect answers
    for both standard and quantized models.
    
    Args:
        std_results: List of results from the standard model
        quant_results: List of results from the quantized model
        output_path: Path to save the visualization
    """
    # Prepare data
    std_correct_logprobs = []
    std_incorrect_logprobs = []
    quant_correct_logprobs = []
    quant_incorrect_logprobs = []
    
    # Process standard model results
    for problem in std_results:
        logprob = problem.get("avg_logprob", 0)
        if problem.get("any_correct", False):
            std_correct_logprobs.append(logprob)
        else:
            std_incorrect_logprobs.append(logprob)
    
    # Process quantized model results
    for problem in quant_results:
        logprob = problem.get("avg_logprob", 0)
        if problem.get("any_correct", False):
            quant_correct_logprobs.append(logprob)
        else:
            quant_incorrect_logprobs.append(logprob)
    
    # Create density plots
    plt.figure(figsize=(12, 10))
    
    # Standard model
    plt.subplot(2, 1, 1)
    
    if std_correct_logprobs:
        sns.kdeplot(std_correct_logprobs, color='green', fill=True, alpha=0.5, label='Correct')
    if std_incorrect_logprobs:
        sns.kdeplot(std_incorrect_logprobs, color='red', fill=True, alpha=0.5, label='Incorrect')
    
    plt.title('Standard Model: Log Probability Distribution by Correctness', fontsize=14)
    plt.xlabel('Log Probability (Higher = More Confident)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    
    # Add vertical lines for means
    if std_correct_logprobs:
        mean_correct = np.mean(std_correct_logprobs)
        plt.axvline(mean_correct, color='green', linestyle='--')
        plt.text(mean_correct, 0.1, f'Mean={mean_correct:.3f}', color='green', ha='center')
    
    if std_incorrect_logprobs:
        mean_incorrect = np.mean(std_incorrect_logprobs)
        plt.axvline(mean_incorrect, color='red', linestyle='--')
        plt.text(mean_incorrect, 0.05, f'Mean={mean_incorrect:.3f}', color='red', ha='center')
    
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    
    # Quantized model
    plt.subplot(2, 1, 2)
    
    if quant_correct_logprobs:
        sns.kdeplot(quant_correct_logprobs, color='green', fill=True, alpha=0.5, label='Correct')
    if quant_incorrect_logprobs:
        sns.kdeplot(quant_incorrect_logprobs, color='red', fill=True, alpha=0.5, label='Incorrect')
    
    plt.title('Quantized Model: Log Probability Distribution by Correctness', fontsize=14)
    plt.xlabel('Log Probability (Higher = More Confident)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    
    # Add vertical lines for means
    if quant_correct_logprobs:
        mean_correct = np.mean(quant_correct_logprobs)
        plt.axvline(mean_correct, color='green', linestyle='--')
        plt.text(mean_correct, 0.1, f'Mean={mean_correct:.3f}', color='green', ha='center')
    
    if quant_incorrect_logprobs:
        mean_incorrect = np.mean(quant_incorrect_logprobs)
        plt.axvline(mean_incorrect, color='red', linestyle='--')
        plt.text(mean_incorrect, 0.05, f'Mean={mean_incorrect:.3f}', color='red', ha='center')
    
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    
    print(f"Confidence density by correctness plot saved to {output_path}")
    
    # Print statistics
    print("\nConfidence Statistics:")
    
    if std_correct_logprobs and std_incorrect_logprobs:
        std_diff = np.mean(std_correct_logprobs) - np.mean(std_incorrect_logprobs)
        print(f"Standard model:")
        print(f"  Mean logprob when correct: {np.mean(std_correct_logprobs):.4f}")
        print(f"  Mean logprob when incorrect: {np.mean(std_incorrect_logprobs):.4f}")
        print(f"  Difference: {std_diff:.4f} ({'higher' if std_diff > 0 else 'lower'} when correct)")
    
    if quant_correct_logprobs and quant_incorrect_logprobs:
        quant_diff = np.mean(quant_correct_logprobs) - np.mean(quant_incorrect_logprobs)
        print(f"Quantized model:")
        print(f"  Mean logprob when correct: {np.mean(quant_correct_logprobs):.4f}")
        print(f"  Mean logprob when incorrect: {np.mean(quant_incorrect_logprobs):.4f}")
        print(f"  Difference: {quant_diff:.4f} ({'higher' if quant_diff > 0 else 'lower'} when correct)")


def main():
    # Set paths
    base_dir = '/share/u/yu.stev/attribution/data/cruxeval_results/k_sample'
    std_file = os.path.join(base_dir, 'cruxeval_k4_temp0.3_topp0.95_base13b.json')
    quant_file = os.path.join(base_dir, 'cruxeval_k4_temp0.3_topp0.95_base13b_8bit.json')
    
    # Create output directory if it doesn't exist
    viz_dir = os.path.join(base_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Define output paths
    matrix_path = os.path.join(viz_dir, 'confidence_correctness_matrix.png')
    scatter_path = os.path.join(viz_dir, 'correctness_by_confidence_plot.png')
    density_path = os.path.join(viz_dir, 'confidence_density_by_correctness.png')
    
    # Load results
    print(f"Loading standard results from {std_file}")
    std_results = load_results(std_file)
    print(f"Loading quantized results from {quant_file}")
    quant_results = load_results(quant_file)
    
    # Check if data loaded successfully
    if not std_results or not quant_results:
        print("Error: Failed to load one or both result files.")
        return
    
    print(f"Loaded {len(std_results)} standard results and {len(quant_results)} quantized results.")
    
    # Create visualizations
    create_confidence_correctness_heatmap(std_results, quant_results, matrix_path)
    create_correctness_by_confidence_plot(std_results, quant_results, scatter_path)
    create_confidence_density_by_correctness(std_results, quant_results, density_path)
    
    print("All visualizations complete!")


if __name__ == "__main__":
    main()
