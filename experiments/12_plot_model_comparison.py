#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Comparison Visualization Script

This script creates specialized visualizations to compare standard and quantized models
based on their log probability scores and problem-solving capabilities.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


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


def create_logprob_numberline_plot(summary, output_path):
    """
    Create a number line plot of log probabilities with colors indicating which model(s) 
    solved each problem.
    
    Args:
        summary: Comparison summary dictionary
        output_path: Path to save the visualization
    """
    # Extract problems data
    problems = summary["problems"]
    
    # Create a DataFrame with necessary information
    data = []
    for problem in problems:
        standard_correct = problem["standard"]["any_correct"]
        quantized_correct = problem["quantized"]["any_correct"]
        
        # Determine category based on which model(s) solved the problem
        if standard_correct and quantized_correct:
            category = "Both models"
        elif standard_correct:
            category = "Standard model only"
        elif quantized_correct:
            category = "Quantized model only"
        else:
            category = "Neither model"  # This shouldn't happen given our filtering
            
        # Add data for standard model
        data.append({
            "problem_id": problem["problem_id"],
            "model": "Standard",
            "avg_logprob": problem["standard"]["avg_logprob"],
            "category": category
        })
        
        # Add data for quantized model
        data.append({
            "problem_id": problem["problem_id"],
            "model": "Quantized",
            "avg_logprob": problem["quantized"]["avg_logprob"],
            "category": category
        })
    
    df = pd.DataFrame(data)
    
    # Create color mapping
    category_colors = {
        "Both models": "green",
        "Standard model only": "blue",
        "Quantized model only": "red",
        "Neither model": "gray"
    }
    
    # MODIFIED BASIC PLOT: Each category on its own row
    plt.figure(figsize=(12, 7))
    
    # First create a separate DataFrame for each model
    standard_df = df[df["model"] == "Standard"]
    quantized_df = df[df["model"] == "Quantized"]
    
    # Create a mapping of categories to y-positions
    category_position = {
        "Both models": 3,
        "Standard model only": 2,
        "Quantized model only": 1,
        "Neither model": 0
    }
    
    # Plot Standard model
    plt.subplot(2, 1, 1)
    plt.title("Standard Model Log Probabilities by Problem Category", fontsize=12)
    
    for category, group in standard_df.groupby("category"):
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
    
    # Plot Quantized model
    plt.subplot(2, 1, 2)
    plt.title("Quantized Model Log Probabilities by Problem Category", fontsize=12)
    
    for category, group in quantized_df.groupby("category"):
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
        "standard_only": "blue",
        "quantized_only": "red",
        "neither": "lightgray"  # This shouldn't appear
    }
    
    # Combine data for both models
    combined_data = []
    for problem in problems:
        pid = problem["problem_id"]
        std_correct = problem["standard"]["any_correct"]
        quant_correct = problem["quantized"]["any_correct"]
        std_logprob = problem["standard"]["avg_logprob"]
        quant_logprob = problem["quantized"]["avg_logprob"]
        
        # Determine the color category
        if std_correct and quant_correct:
            category = "both"
        elif std_correct:
            category = "standard_only"
        elif quant_correct:
            category = "quantized_only"
        else:
            category = "neither"
        
        # Calculate average log probability across both models
        avg_logprob = (std_logprob + quant_logprob) / 2
        
        combined_data.append({
            "problem_id": pid,
            "category": category,
            "std_logprob": std_logprob,
            "quant_logprob": quant_logprob,
            "avg_logprob": avg_logprob,
            "logprob_diff": std_logprob - quant_logprob  # Positive means standard is more confident
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
        
        # Plot the range from standard to quantized log prob
        plt.plot([i, i], [row["std_logprob"], row["quant_logprob"]], 
                 color=color, alpha=0.5, linewidth=2)
        
        # Plot standard log prob
        plt.scatter(i, row["std_logprob"], color=color, marker="o", s=40, zorder=5)
        
        # Plot quantized log prob
        plt.scatter(i, row["quant_logprob"], color=color, marker="s", s=40, zorder=5)
        
        # Plot the average
        plt.scatter(i, row["avg_logprob"], color=color, marker="_", s=100, linewidth=2, zorder=6)
    
    # Embellish plot
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylabel("Log Probability", fontsize=12)
    plt.xlabel("Problems (Sorted by Average Log Probability)", fontsize=12)
    
    # Create legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8, label='Standard Model'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='k', markersize=8, label='Quantized Model'),
        Line2D([0], [0], marker='_', color='k', markersize=12, markeredgewidth=2, label='Average'),
        Line2D([0], [0], color=colors["both"], lw=4, label='Both Correct'),
        Line2D([0], [0], color=colors["standard_only"], lw=4, label='Standard Only Correct'),
        Line2D([0], [0], color=colors["quantized_only"], lw=4, label='Quantized Only Correct')
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Add summary text at the top of the plot
    plt.figtext(0.5, 0.95, 
                f"Total Problems: {len(problems)} | Both Correct: {summary['both_correct']} | "
                f"Standard Only: {summary['only_standard_correct']} | "
                f"Quantized Only: {summary['only_quantized_correct']}", 
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    # Bottom subplot: Log probability differences histogram
    plt.subplot(3, 1, 3)  # Bottom 1/3 of figure
    plt.title("Log Probability Difference (Standard - Quantized)", fontsize=14)
    
    # Create histograms for each category
    categories = ["both", "standard_only", "quantized_only"]
    labels = ["Both Correct", "Standard Only Correct", "Quantized Only Correct"]
    
    for cat, label, color in zip(categories, labels, [colors["both"], colors["standard_only"], colors["quantized_only"]]):
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
                "Positive values: Standard model more confident | Negative values: Quantized model more confident", 
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
        "standard_only": 2,
        "quantized_only": 1,
        "neither": 0  # This shouldn't appear
    }
    
    # Main plot with vertically separated categories
    plt.subplot(2, 1, 1)
    plt.title("Log Probabilities by Category (Standard vs. Quantized)", fontsize=14)
    
    # Sort problems within each category by standard model log probability
    # This helps us see distribution within each category
    for category, category_df in combined_df.groupby("category"):
        # Sort by standard model log probability
        category_df = category_df.sort_values("std_logprob")
        y_pos = category_positions[category]
        
        # Define x positions within each category
        x_pos = np.linspace(0, 1, len(category_df))
        
        # Plot standard model points
        plt.scatter(
            x_pos,
            [y_pos + 0.1] * len(category_df),  # Slightly above the category line
            c=colors[category],
            s=50,
            alpha=0.7,
            marker="o"
        )
        
        # Plot quantized model points
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
        ["Both Correct", "Standard Only Correct", "Quantized Only Correct", "Neither Correct"],
        fontsize=10
    )
    
    # Set x-axis
    plt.xlim(-0.05, 1.05)
    plt.xticks([])  # Hide x-ticks as they're meaningless here
    plt.xlabel("Problems (Sorted by Standard Model Log Probability within Category)", fontsize=10)
    
    # Create legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8, label='Standard Model'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='k', markersize=8, label='Quantized Model'),
        Line2D([0], [0], color=colors["both"], lw=4, label='Both Correct'),
        Line2D([0], [0], color=colors["standard_only"], lw=4, label='Standard Only Correct'),
        Line2D([0], [0], color=colors["quantized_only"], lw=4, label='Quantized Only Correct')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add log probability colorbar plot
    plt.subplot(2, 1, 2)
    plt.title("Log Probability Values by Category", fontsize=14)
    
    # Define category order and labels
    categories = ["both", "standard_only", "quantized_only"]
    cat_labels = ["Both Correct", "Standard Only Correct", "Quantized Only Correct"]
    
    # Create positions for the different bars
    bar_positions = {
        "std_both": 6,
        "quant_both": 5,
        "std_standard_only": 4,
        "quant_standard_only": 3,
        "std_quantized_only": 2,
        "quant_quantized_only": 1
    }
    
    # Get data for each category
    for i, (cat, label) in enumerate(zip(categories, cat_labels)):
        cat_data = combined_df[combined_df["category"] == cat]
        if not cat_data.empty:
            # Standard model
            plt.scatter(
                cat_data["std_logprob"],
                [bar_positions[f"std_{cat}"]] * len(cat_data),
                color=colors[cat],
                s=50,
                alpha=0.7,
                marker="o"
            )
            
            # Quantized model
            plt.scatter(
                cat_data["quant_logprob"],
                [bar_positions[f"quant_{cat}"]] * len(cat_data),
                color=colors[cat],
                s=50,
                alpha=0.7,
                marker="s"
            )
    
    # Set y-axis ticks and labels
    plt.yticks(
        list(bar_positions.values()),
        ["Standard", "Quantized", "Standard", "Quantized", "Standard", "Quantized"],
        fontsize=10
    )
    
    # Add category labels
    plt.text(-0.15, 5.5, "Both Correct", fontsize=12, ha="right", va="center", color=colors["both"])
    plt.text(-0.15, 3.5, "Standard Only", fontsize=12, ha="right", va="center", color=colors["standard_only"])
    plt.text(-0.15, 1.5, "Quantized Only", fontsize=12, ha="right", va="center", color=colors["quantized_only"])
    
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
    # Set paths
    base_dir = '/share/u/yu.stev/attribution/data/cruxeval_results/k_sample'
    comparison_file = os.path.join(base_dir, 'comparison_summary.json')
    
    # Create output directory if it doesn't exist
    viz_dir = os.path.join(base_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Define output paths
    logprob_plot_path = os.path.join(viz_dir, 'logprob_numberline.png')
    
    # Load comparison summary
    print(f"Loading comparison summary from {comparison_file}")
    summary = load_comparison_summary(comparison_file)
    
    if summary is None:
        print("Error: Failed to load comparison summary.")
        return
    
    # Create log probability number line plot
    create_logprob_numberline_plot(summary, logprob_plot_path)
    print("Visualization complete!")


if __name__ == "__main__":
    main()
