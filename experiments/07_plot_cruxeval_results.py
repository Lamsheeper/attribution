#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot results from CruxEval benchmark evaluations.

Generate plots from CruxEval results for OLMo models.
This script creates bar charts, heatmaps, and line plots similar to those in 04_assess_multiple_benchmarks.py.
"""

import json
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate plots from CruxEval results')
parser.add_argument('--results_dir', type=str, default="../data/cruxeval_results",
                    help='Directory containing the CruxEval results')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Directory to save the plots (defaults to results_dir)')
args = parser.parse_args()

# Set up directories
results_dir = Path(args.results_dir)
output_dir = Path(args.output_dir) if args.output_dir else results_dir
output_dir.mkdir(parents=True, exist_ok=True)

# Model names mapping
MODEL_NAMES = {
    "base": "OLMo2-7B-1124",
    "sft": "OLMo-2-1124-7B-SFT",
    "dpo": "OLMo-2-1124-7B-DPO",
    "instruct": "OLMo-2-1124-7B-Instruct",
    "rm": "OLMo-2-1124-7B-RM",
    "base13b": "OLMo-2-1124-13B",
    "dpo13b": "OLMo-2-1124-13B-DPO",
    "instruct13b": "OLMo-2-1124-13B-Instruct"
}

# Load results for all models
model_results = {}
for model_short_name, model_full_name in MODEL_NAMES.items():
    result_file = results_dir / f"cruxeval_results_{model_short_name}.json"
    if result_file.exists():
        with open(result_file, 'r') as f:
            results = json.load(f)
            model_results[model_short_name] = results
            print(f"Loaded results for {model_full_name}")
    else:
        print(f"Warning: Results file not found for {model_full_name}: {result_file}")

if not model_results:
    print("No results found. Please check the results directory.")
    exit(1)

# Function to calculate accuracy for each model
def calculate_model_accuracy(results):
    """Calculate overall accuracy for a model's results"""
    correct_count = sum(1 for r in results if r["is_correct"])
    total_count = len(results)
    return correct_count / total_count if total_count > 0 else 0

# Function to calculate accuracy by problem ID across models
def calculate_problem_accuracy(model_results):
    """Calculate accuracy for each problem ID across all models"""
    problem_accuracy = {}
    
    # Get all unique problem IDs
    all_problem_ids = set()
    for model, results in model_results.items():
        all_problem_ids.update(r["problem_id"] for r in results)
    
    # Calculate accuracy for each problem ID
    for problem_id in sorted(all_problem_ids):
        problem_accuracy[problem_id] = {}
        for model, results in model_results.items():
            # Find results for this problem ID
            problem_results = [r for r in results if r["problem_id"] == problem_id]
            if problem_results:
                correct = sum(1 for r in problem_results if r["is_correct"])
                total = len(problem_results)
                problem_accuracy[problem_id][model] = correct / total
            else:
                problem_accuracy[problem_id][model] = 0
    
    return problem_accuracy

# Calculate overall accuracy for each model
model_accuracy = {model: calculate_model_accuracy(results) for model, results in model_results.items()}
problem_accuracy = calculate_problem_accuracy(model_results)

# 1. Bar chart of overall model accuracy
def plot_model_accuracy(model_accuracy, output_path):
    """Create a bar chart of overall model accuracy"""
    plt.figure(figsize=(12, 6))
    
    # Sort models by accuracy
    sorted_models = sorted(model_accuracy.items(), key=lambda x: x[1], reverse=True)
    models = [MODEL_NAMES[m] for m, _ in sorted_models]
    accuracies = [a * 100 for _, a in sorted_models]
    
    # Create bar chart
    bars = plt.bar(models, accuracies, color=sns.color_palette("viridis", len(models)))
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.title('CruxEval Accuracy by Model')
    plt.ylim(0, max(accuracies) + 5)  # Add some space for the labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()
    print(f"Saved model accuracy plot to {output_path}")

# 2. Heatmap of problem accuracy by model
def plot_problem_heatmap(problem_accuracy, output_path, max_problems=100):
    """Create a heatmap of problem accuracy by model"""
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(problem_accuracy).T
    
    # Limit to first max_problems if there are too many
    if len(df) > max_problems:
        df = df.iloc[:max_problems]
    
    plt.figure(figsize=(14, 10))
    
    # Create heatmap
    sns.heatmap(df, cmap='RdYlGn', annot=True, fmt='.2f', 
                cbar_kws={'label': 'Accuracy'}, vmin=0, vmax=1)
    
    plt.xlabel('Model')
    plt.ylabel('Problem ID')
    plt.title('CruxEval Problem Accuracy by Model')
    
    # Use full model names for x-axis
    plt.xticks(np.arange(len(df.columns)) + 0.5, 
              [MODEL_NAMES[col] for col in df.columns], 
              rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved problem heatmap to {output_path}")

# 3. Line plot of problem accuracy across models
def plot_problem_line(problem_accuracy, output_path):
    """Create a line plot of problem accuracy across models"""
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(problem_accuracy).T
    
    # Calculate mean accuracy across models for each problem
    df['mean'] = df.mean(axis=1)
    
    # Sort problems by mean accuracy
    df = df.sort_values('mean')
    
    plt.figure(figsize=(15, 8))
    
    # Plot each model as a line
    for model in df.columns:
        if model != 'mean':
            plt.plot(range(len(df)), df[model], label=MODEL_NAMES.get(model, model), marker='o', markersize=4)
    
    # Add mean line with different style
    plt.plot(range(len(df)), df['mean'], label='Mean', linestyle='--', color='black', linewidth=2)
    
    plt.xlabel('Problem Index (sorted by difficulty)')
    plt.ylabel('Accuracy')
    plt.title('CruxEval Problem Accuracy Across Models')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    # Set y-axis limits
    plt.ylim(-0.05, 1.05)
    
    # Add x-axis ticks for every 50th problem
    tick_indices = range(0, len(df), 50)
    tick_labels = [df.index[i] for i in tick_indices if i < len(df)]
    plt.xticks(tick_indices, tick_labels, rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved problem line plot to {output_path}")

# 4. Create a comparison of correct problems across models
def plot_venn_like_comparison(model_results, output_path):
    """Create a plot showing overlap of correctly solved problems between models"""
    # Get sets of correctly solved problems for each model
    correct_problems = {}
    for model, results in model_results.items():
        correct_problems[model] = set(r["problem_id"] for r in results if r["is_correct"])
    
    # Calculate total problems solved by each model
    model_counts = {model: len(problems) for model, problems in correct_problems.items()}
    
    # Calculate problems solved by all models
    all_correct = set.intersection(*correct_problems.values()) if correct_problems else set()
    
    # Calculate problems solved by at least one model
    any_correct = set.union(*correct_problems.values()) if correct_problems else set()
    
    # Create a DataFrame for the plot
    data = []
    for model, problems in correct_problems.items():
        # Calculate unique problems (solved only by this model)
        unique_problems = problems - set.union(*(p for m, p in correct_problems.items() if m != model))
        
        data.append({
            'Model': MODEL_NAMES[model],
            'Total Correct': len(problems),
            'Unique Correct': len(unique_problems),
            'Shared with Others': len(problems) - len(unique_problems)
        })
    
    df = pd.DataFrame(data)
    
    # Sort by total correct
    df = df.sort_values('Total Correct', ascending=False)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Plot total correct
    bars1 = plt.bar(df['Model'], df['Total Correct'], label='Total Correct')
    
    # Plot unique correct
    bars2 = plt.bar(df['Model'], df['Unique Correct'], label='Unique to This Model')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(height)}', ha='center', va='bottom')
    
    # Add text for problems solved by all models and by any model
    plt.figtext(0.5, 0.01, 
               f'Problems solved by ALL models: {len(all_correct)} | Problems solved by AT LEAST ONE model: {len(any_correct)} | Total problems: {len(any_correct)}',
               ha='center', fontsize=12, bbox=dict(facecolor='lightgray', alpha=0.5))
    
    plt.xlabel('Model')
    plt.ylabel('Number of Problems')
    plt.title('CruxEval: Correct Problems by Model')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Adjust y-axis limit to make room for labels
    plt.ylim(0, max(df['Total Correct']) * 1.1)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for the text at the bottom
    plt.savefig(output_path)
    plt.close()
    print(f"Saved model comparison plot to {output_path}")

# Generate all plots
plot_model_accuracy(model_accuracy, output_dir / "cruxeval_model_accuracy.png")
plot_problem_heatmap(problem_accuracy, output_dir / "cruxeval_problem_heatmap.png", max_problems=800)
plot_problem_line(problem_accuracy, output_dir / "cruxeval_problem_line.png")
plot_venn_like_comparison(model_results, output_dir / "cruxeval_model_comparison.png")

# Save the processed data for further analysis
with open(output_dir / "cruxeval_model_accuracy.json", 'w') as f:
    json.dump({MODEL_NAMES[k]: v for k, v in model_accuracy.items()}, f, indent=2)

with open(output_dir / "cruxeval_problem_accuracy.json", 'w') as f:
    # Convert problem IDs to strings for JSON
    problem_accuracy_json = {str(k): {MODEL_NAMES[m]: v for m, v in v.items()} 
                            for k, v in problem_accuracy.items()}
    json.dump(problem_accuracy_json, f, indent=2)

print(f"All plots and data saved to {output_dir}")

# Print summary statistics
print("\nSummary of CruxEval Results:")
print("-" * 50)
print(f"{'Model':<25} {'Accuracy':<10}")
print("-" * 50)
for model, acc in sorted(model_accuracy.items(), key=lambda x: x[1], reverse=True):
    print(f"{MODEL_NAMES[model]:<25} {acc*100:.2f}%")
print("-" * 50) 