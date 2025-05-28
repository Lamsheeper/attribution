#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Checkpoint Comparison Visualization Script

This script creates specialized visualizations to compare two model checkpoints
based on their log probability scores and problem-solving capabilities.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import argparse


def load_comparison_summary(filepath):
    """
    Load the comparison summary JSON file.
    
    Args:
        filepath: Path to the comparison summary JSON file
        
    Returns:
        Dictionary containing the comparison summary
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading comparison summary from {filepath}: {e}")
        return None


def create_logprob_numberline_plot(summary, output_path, model1_name="Checkpoint 1", model2_name="Checkpoint 2"):
    """
    Create a number line plot of log probabilities with colors indicating which model(s) 
    solved each problem.
    
    Args:
        summary: Comparison summary dictionary
        output_path: Path to save the visualization
        model1_name: Display name for the first model
        model2_name: Display name for the second model
    """
    # Extract problems data
    problems = summary["problems"]
    
    # Create a DataFrame with necessary information
    data = []
    for problem in problems:
        model1_correct = problem["model1"]["any_correct"]
        model2_correct = problem["model2"]["any_correct"]
        
        # Determine category based on which model(s) solved the problem
        if model1_correct and model2_correct:
            category = "Both models"
        elif model1_correct:
            category = f"{model1_name} only"
        elif model2_correct:
            category = f"{model2_name} only"
        else:
            category = "Neither model"  # This shouldn't happen given our filtering
            
        # Add data for model1
        data.append({
            "problem_id": problem["problem_id"],
            "model": model1_name,
            "avg_logprob": problem["model1"]["avg_logprob"],
            "category": category
        })
        
        # Add data for model2
        data.append({
            "problem_id": problem["problem_id"],
            "model": model2_name,
            "avg_logprob": problem["model2"]["avg_logprob"],
            "category": category
        })
    
    df = pd.DataFrame(data)
    
    # Create color mapping
    category_colors = {
        "Both models": "green",
        f"{model1_name} only": "blue",
        f"{model2_name} only": "red",
        "Neither model": "gray"
    }
    
    # MODIFIED BASIC PLOT: Each category on its own row
    plt.figure(figsize=(12, 7))
    
    # First create a separate DataFrame for each model
    model1_df = df[df["model"] == model1_name]
    model2_df = df[df["model"] == model2_name]
    
    # Create a mapping of categories to y-positions
    category_position = {
        "Both models": 3,
        f"{model1_name} only": 2,
        f"{model2_name} only": 1,
        "Neither model": 0
    }
    
    # Plot model1
    plt.subplot(2, 1, 1)
    plt.title(f"{model1_name} Log Probabilities by Problem Category", fontsize=12)
    
    for category, group in model1_df.groupby("category"):
        y_pos = category_position[category]
        plt.scatter(
            group["avg_logprob"],
            [y_pos] * len(group),
            color=category_colors[category],
            s=50,
            alpha=0.7,
            label=category if y_pos == 3 else None  # Only add label for the first category to avoid duplicates
        )
    
    # Set y-axis ticks and labels
    plt.yticks(
        list(category_position.values()),
        list(category_position.keys()),
        fontsize=10
    )
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xlabel("Log Probability (Higher = More Confident)", fontsize=10)
    plt.legend(loc="upper left")
    
    # Plot model2
    plt.subplot(2, 1, 2)
    plt.title(f"{model2_name} Log Probabilities by Problem Category", fontsize=12)
    
    for category, group in model2_df.groupby("category"):
        y_pos = category_position[category]
        plt.scatter(
            group["avg_logprob"],
            [y_pos] * len(group),
            color=category_colors[category],
            s=50,
            alpha=0.7
        )
    
    # Set y-axis ticks and labels
    plt.yticks(
        list(category_position.values()),
        list(category_position.keys()),
        fontsize=10
    )
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xlabel("Log Probability (Higher = More Confident)", fontsize=10)
    
    # Save the figure
    plt.tight_layout(pad=3.0)
    plt.savefig(output_path)
    plt.close()
    
    print(f"Log probability number line plot saved to {output_path}")
    
    # Create a second version with combined plot and additional statistics
    plt.figure(figsize=(14, 10))
    
    # Set up colors for each category
    colors = {
        "both": "green",
        "model1_only": "blue",
        "model2_only": "red",
        "neither": "lightgray"  # This shouldn't appear
    }
    
    # Combine data for both models
    combined_data = []
    for problem in problems:
        pid = problem["problem_id"]
        model1_correct = problem["model1"]["any_correct"]
        model2_correct = problem["model2"]["any_correct"]
        model1_logprob = problem["model1"]["avg_logprob"]
        model2_logprob = problem["model2"]["avg_logprob"]
        
        # Determine the color category
        if model1_correct and model2_correct:
            category = "both"
        elif model1_correct:
            category = "model1_only"
        elif model2_correct:
            category = "model2_only"
        else:
            category = "neither"
        
        # Calculate average log probability across both models
        avg_logprob = (model1_logprob + model2_logprob) / 2
        
        combined_data.append({
            "problem_id": pid,
            "category": category,
            "model1_logprob": model1_logprob,
            "model2_logprob": model2_logprob,
            "avg_logprob": avg_logprob,
            "logprob_diff": model1_logprob - model2_logprob  # Positive means model1 is more confident
        })
    
    # Convert to DataFrame and sort by average log probability
    combined_df = pd.DataFrame(combined_data)
    combined_df = combined_df.sort_values("avg_logprob")
    
    # Define x positions for the problems
    x_positions = np.arange(len(combined_df))
    
    # Main plot showing average log probs
    plt.subplot(3, 1, (1, 2))  # Use top 2/3 of the figure
    plt.title("Model Log Probabilities by Problem (Sorted by Average)", fontsize=14)
    
    # Loop through and plot each problem with appropriate color
    for i, (_, row) in enumerate(combined_df.iterrows()):
        color = colors[row["category"]]
        
        # Plot the range from model1 to model2 log prob
        plt.plot([i, i], [row["model1_logprob"], row["model2_logprob"]], 
                 color=color, alpha=0.5, linewidth=2)
        
        # Plot model1 log prob
        plt.scatter(i, row["model1_logprob"], color=color, marker="o", s=40, zorder=5)
        
        # Plot model2 log prob
        plt.scatter(i, row["model2_logprob"], color=color, marker="s", s=40, zorder=5)
        
        # Plot the average
        plt.scatter(i, row["avg_logprob"], color=color, marker="_", s=100, linewidth=2, zorder=6)
    
    # Embellish plot
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylabel("Log Probability", fontsize=12)
    plt.xlabel("Problems (Sorted by Average Log Probability)", fontsize=12)
    
    # Create legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8, label=model1_name),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='k', markersize=8, label=model2_name),
        Line2D([0], [0], marker='_', color='k', markersize=12, markeredgewidth=2, label='Average'),
        Line2D([0], [0], color=colors["both"], lw=4, label='Both Correct'),
        Line2D([0], [0], color=colors["model1_only"], lw=4, label=f'{model1_name} Only Correct'),
        Line2D([0], [0], color=colors["model2_only"], lw=4, label=f'{model2_name} Only Correct')
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Add summary text at the top of the plot
    plt.figtext(0.5, 0.95, 
                f"Total Problems: {len(problems)} | Both Correct: {summary['both_correct']} | "
                f"{model1_name} Only: {summary['only_model1_correct']} | "
                f"{model2_name} Only: {summary['only_model2_correct']}", 
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    # Bottom subplot: Log probability differences histogram
    plt.subplot(3, 1, 3)  # Bottom 1/3 of figure
    plt.title(f"Log Probability Difference ({model1_name} - {model2_name})", fontsize=14)
    
    # Create histograms for each category
    categories = ["both", "model1_only", "model2_only"]
    labels = ["Both Correct", f"{model1_name} Only Correct", f"{model2_name} Only Correct"]
    
    for cat, label, color in zip(categories, labels, [colors["both"], colors["model1_only"], colors["model2_only"]]):
        cat_data = combined_df[combined_df["category"] == cat]["logprob_diff"]
        if not cat_data.empty:
            plt.hist(cat_data, bins=20, alpha=0.6, color=color, label=label)
    
    # Add vertical line at zero
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.7)
    
    # Embellish plot
    plt.grid(linestyle='--', alpha=0.7)
    plt.xlabel("Log Probability Difference", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend(fontsize=10)
    
    # Add text explaining the interpretation
    plt.figtext(0.5, 0.03, 
                f"Positive values: {model1_name} more confident | Negative values: {model2_name} more confident", 
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    # Save the enhanced plot
    plt.tight_layout()
    enhanced_output_path = os.path.splitext(output_path)[0] + "_enhanced.png"
    plt.savefig(enhanced_output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced log probability plot saved to {enhanced_output_path}")
    
    # Create a third version with separated categories on each y-level for better visibility
    plt.figure(figsize=(14, 10))
    
    # Define y positions for categories
    category_positions = {
        "both": 3,
        "model1_only": 2,
        "model2_only": 1,
        "neither": 0  # This shouldn't appear
    }
    
    # Main plot with vertically separated categories
    plt.subplot(2, 1, 1)
    plt.title(f"Log Probabilities by Category ({model1_name} vs. {model2_name})", fontsize=14)
    
    # Sort problems within each category by model1 log probability
    # This helps us see distribution within each category
    for category, category_df in combined_df.groupby("category"):
        # Sort by model1 log probability
        category_df = category_df.sort_values("model1_logprob")
        y_pos = category_positions[category]
        
        # Define x positions within each category
        x_pos = np.linspace(0, 1, len(category_df))
        
        # Plot model1 points
        plt.scatter(
            x_pos,
            [y_pos + 0.1] * len(category_df),  # Slightly above the category line
            c=colors[category],
            s=50,
            alpha=0.7,
            marker="o"
        )
        
        # Plot model2 points
        plt.scatter(
            x_pos,
            [y_pos - 0.1] * len(category_df),  # Slightly below the category line
            c=colors[category],
            s=50,
            alpha=0.7,
            marker="s"
        )
        
        # Connect corresponding points
        for i, (_, row) in enumerate(category_df.iterrows()):
            plt.plot(
                [x_pos[i], x_pos[i]],
                [y_pos + 0.1, y_pos - 0.1],
                color=colors[category],
                alpha=0.3,
                linewidth=1
            )
    
    # Set y-axis ticks and labels
    plt.yticks(
        list(category_positions.values()),
        ["Both Correct", f"{model1_name} Only Correct", f"{model2_name} Only Correct", "Neither Correct"],
        fontsize=10
    )
    
    # Set x-axis
    plt.xlim(-0.05, 1.05)
    plt.xticks([])  # Hide x-ticks as they're meaningless here
    plt.xlabel(f"Problems (Sorted by {model1_name} Log Probability within Category)", fontsize=10)
    
    # Create legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8, label=model1_name),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='k', markersize=8, label=model2_name),
        Line2D([0], [0], color=colors["both"], lw=4, label='Both Correct'),
        Line2D([0], [0], color=colors["model1_only"], lw=4, label=f'{model1_name} Only Correct'),
        Line2D([0], [0], color=colors["model2_only"], lw=4, label=f'{model2_name} Only Correct')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add log probability colorbar plot
    plt.subplot(2, 1, 2)
    plt.title("Log Probability Values by Category", fontsize=14)
    
    # Define category order and labels
    categories = ["both", "model1_only", "model2_only"]
    cat_labels = ["Both Correct", f"{model1_name} Only Correct", f"{model2_name} Only Correct"]
    
    # Create positions for the different bars
    bar_positions = {
        "model1_both": 6,
        "model2_both": 5,
        "model1_model1_only": 4,
        "model2_model1_only": 3,
        "model1_model2_only": 2,
        "model2_model2_only": 1
    }
    
    # Get data for each category
    for i, (cat, label) in enumerate(zip(categories, cat_labels)):
        cat_data = combined_df[combined_df["category"] == cat]
        if not cat_data.empty:
            # Model 1
            plt.scatter(
                cat_data["model1_logprob"],
                [bar_positions[f"model1_{cat}"]] * len(cat_data),
                color=colors[cat],
                s=50,
                alpha=0.7,
                marker="o"
            )
            
            # Model 2
            plt.scatter(
                cat_data["model2_logprob"],
                [bar_positions[f"model2_{cat}"]] * len(cat_data),
                color=colors[cat],
                s=50,
                alpha=0.7,
                marker="s"
            )
    
    # Set y-axis ticks and labels
    plt.yticks(
        list(bar_positions.values()),
        [model1_name, model2_name, model1_name, model2_name, model1_name, model2_name],
        fontsize=10
    )
    
    # Add category labels
    plt.text(-0.15, 5.5, "Both Correct", fontsize=12, ha="right", va="center", color=colors["both"])
    plt.text(-0.15, 3.5, f"{model1_name} Only", fontsize=12, ha="right", va="center", color=colors["model1_only"])
    plt.text(-0.15, 1.5, f"{model2_name} Only", fontsize=12, ha="right", va="center", color=colors["model2_only"])
    
    # Grid and labels
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xlabel("Log Probability", fontsize=12)
    
    # Save this separated version
    plt.tight_layout()
    separated_output_path = os.path.splitext(output_path)[0] + "_separated.png"
    plt.savefig(separated_output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Separated category log probability plot saved to {separated_output_path}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Create visualizations comparing checkpoint evaluation results")
    parser.add_argument("--summary_file", type=str, help="Path to comparison summary JSON file")
    parser.add_argument("--output_dir", type=str, help="Directory to save visualizations")
    parser.add_argument("--name1", type=str, default="Checkpoint 1", help="Name for first checkpoint in visualizations")
    parser.add_argument("--name2", type=str, default="Checkpoint 2", help="Name for second checkpoint in visualizations")
    args = parser.parse_args()
    
    # Use provided arguments or default values
    if args.summary_file and args.output_dir:
        comparison_file = args.summary_file
        viz_dir = args.output_dir
    else:
        # Default paths (original behavior)
        base_dir = '/share/u/yu.stev/attribution/data/cruxeval_results/checkpoints'
        comparison_file = os.path.join(base_dir, 'comparison_summary.json')
        viz_dir = os.path.join(base_dir, 'visualizations')
    
    # Create output directory if it doesn't exist
    os.makedirs(viz_dir, exist_ok=True)
    
    # Define output paths
    logprob_plot_path = os.path.join(viz_dir, 'logprob_numberline.png')
    
    # Load comparison summary
    print(f"Loading comparison summary from {comparison_file}")
    summary = load_comparison_summary(comparison_file)
    
    if summary is None:
        print("Error: Failed to load comparison summary.")
        return
    
    # Get model names
    model1_name = args.name1
    model2_name = args.name2
    
    # Create log probability number line plot
    create_logprob_numberline_plot(summary, logprob_plot_path, model1_name, model2_name)
    print("Visualization complete!")


if __name__ == "__main__":
    main() 