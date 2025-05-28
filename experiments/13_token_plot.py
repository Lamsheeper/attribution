#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Token Performance Plot Script

This script creates a line plot showing model accuracy versus training tokens
for multiple checkpoints.
"""

import json
import os
import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


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


def get_correct_problem_ids(results):
    """
    Get the set of problem IDs that were correctly solved.
    
    Args:
        results: List of problem results
        
    Returns:
        Set of correctly solved problem IDs
    """
    return {result["problem_id"] for result in results if result.get("is_correct", False)}


def find_checkpoints(directory, exclude_tokens=None):
    """
    Find all checkpoint files in the directory that match the pattern.
    
    Args:
        directory: Directory to search for checkpoint files
        exclude_tokens: List of token numbers to exclude
    
    Returns:
        List of (checkpoint_path, checkpoint_name, stage, ingredient) tuples
    """
    checkpoints = []
    stage1_pattern = re.compile(r"stage1-tokens(\d+)B\.json")
    stage2_pattern = re.compile(r"stage2-ingredient(\d+)-tokens(\d+)B\.json")
    exclude_tokens = set(exclude_tokens or [])
    
    STAGE1_TOTAL_TOKENS = 3868  # Total tokens from stage 1
    
    for file in os.listdir(directory):
        stage1_match = stage1_pattern.match(file)
        stage2_match = stage2_pattern.match(file)
        
        if stage1_match:
            # Stage 1 file
            token_num = stage1_match.group(1)
            if token_num in exclude_tokens:
                print(f"Excluding tokens {token_num}B as requested")
                continue
                
            filepath = os.path.join(directory, file)
            name = f"{token_num}B tokens"
            checkpoints.append((filepath, name, 1, None))  # stage=1, ingredient=None
            
        elif stage2_match:
            # Stage 2 file
            ingredient_num = stage2_match.group(1)
            token_num = stage2_match.group(2)
            
            if token_num in exclude_tokens:
                print(f"Excluding tokens {token_num}B as requested")
                continue
            
            # Add stage 1 tokens to stage 2 token count
            total_tokens = int(token_num) + STAGE1_TOTAL_TOKENS
            
            filepath = os.path.join(directory, file)
            name = f"{total_tokens}B tokens (i{ingredient_num})"
            checkpoints.append((filepath, name, 2, int(ingredient_num)))  # stage=2, ingredient=number
    
    # Sort by token number (higher numbers come first)
    checkpoints.sort(key=lambda x: -int(x[1].split('B')[0].split(' ')[0]))
    
    return checkpoints


def calculate_metrics(checkpoint_results_map):
    """
    Calculate accuracy metrics for each checkpoint.
    
    Args:
        checkpoint_results_map: Dictionary mapping checkpoint names to their results
        
    Returns:
        Dictionary with metrics
    """
    # Get all problems from all checkpoints
    all_problems = set()
    for results in checkpoint_results_map.values():
        all_problems.update(r["problem_id"] for r in results)
    total_problems = len(all_problems)
    
    # Calculate accuracy for each checkpoint
    metrics = {
        "total_problems": total_problems,
        "checkpoints": {
            name: {
                "total_correct": len(get_correct_problem_ids(results)),
                "accuracy": len(get_correct_problem_ids(results)) / total_problems if total_problems > 0 else 0
            }
            for name, results in checkpoint_results_map.items()
        }
    }
    
    return metrics


def plot_token_performance(metrics, output_path, token_limit=None):
    """
    Create a scatterplot showing accuracy versus token count with equal visual space
    for stage 1 and stage 2, despite different token ranges. Connects stage 1 to stage 2.
    
    Args:
        metrics: Dictionary with comparison metrics
        output_path: Path to save the output image
        token_limit: Optional limit for max tokens to include (in billions)
    """
    # Get checkpoint names and extract token values for token-based checkpoints
    checkpoint_names = list(metrics["checkpoints"].keys())
    
    # Prepare data structures for stage 1 and stage 2
    stage1_data = []
    stage2_data = {1: [], 2: [], 3: []}  # Separate lists for each ingredient
    
    STAGE1_TOTAL_TOKENS = 3868  # Total tokens from stage 1
    
    for name in checkpoint_names:
        if "tokens" in name:
            try:
                # Extract token value and check if it's a stage 2 ingredient
                if "(i" in name:
                    # Stage 2 - extract tokens and ingredient number
                    token_value = int(name.split('B')[0].split(' ')[0])
                    ingredient = int(name.split('(i')[1].split(')')[0])
                    if token_limit is not None and token_value > token_limit:
                        continue
                    accuracy = metrics["checkpoints"][name]["accuracy"]
                    total_correct = metrics["checkpoints"][name]["total_correct"]
                    stage2_data[ingredient].append((token_value, accuracy, total_correct, name))
                else:
                    # Stage 1
                    token_value = int(name.split('B')[0])
                    if token_limit is not None and token_value > token_limit:
                        continue
                    accuracy = metrics["checkpoints"][name]["accuracy"]
                    total_correct = metrics["checkpoints"][name]["total_correct"]
                    stage1_data.append((token_value, accuracy, total_correct, name))
            except (ValueError, IndexError):
                continue
    
    # If no data, exit
    if not stage1_data and not any(stage2_data.values()):
        print("No checkpoints found for plotting.")
        return
    
    # Create figure
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
    
    # Function to transform x-coordinates
    def transform_x(x):
        """Transform token counts to give equal space to stage 1 and 2"""
        if x <= STAGE1_TOTAL_TOKENS:
            # Stage 1: map [0, 3868] to [0, 0.5]
            return 0.5 * x / STAGE1_TOTAL_TOKENS
        else:
            # Stage 2: map [3868, 3918] to [0.5, 1.0]
            stage2_tokens = x - STAGE1_TOTAL_TOKENS
            return 0.5 + 0.5 * stage2_tokens / 50
    
    # Plot stage 1 data
    last_stage1_point = None
    if stage1_data:
        stage1_data.sort(key=lambda x: x[0])
        token_values = [transform_x(d[0]) for d in stage1_data]
        accuracy = [d[1] for d in stage1_data]
        ax.plot(token_values, accuracy, 'o-', color='green', markersize=8, label='Stage 1')
        # Store the last point of stage 1 for connecting to stage 2
        last_stage1_point = (token_values[-1], accuracy[-1])
    
    # Plot stage 2 data for each ingredient
    ingredient_colors = {1: 'red', 2: 'orange', 3: 'yellow'}
    for ingredient, color in ingredient_colors.items():
        if stage2_data[ingredient]:
            data = sorted(stage2_data[ingredient], key=lambda x: x[0])
            token_values = [transform_x(d[0]) for d in data]
            accuracy = [d[1] for d in data]
            
            # Plot the connection line from stage 1 to stage 2 (dashed)
            if last_stage1_point:
                ax.plot([last_stage1_point[0], token_values[0]], 
                       [last_stage1_point[1], accuracy[0]], 
                       '--', color=color, alpha=0.5)
            
            # Plot the stage 2 line
            ax.plot(token_values, accuracy, 'o-', color=color, markersize=8, 
                   label=f'Stage 2 (Ingredient {ingredient})')
    
    # Set up plot
    title = 'Accuracy vs. Training Tokens'
    if token_limit is not None:
        title += f' (Up to {token_limit}B)'
    ax.set_title(title)
    ax.set_ylabel('Accuracy')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Custom x-axis ticks and labels
    def get_tick_positions():
        """Get positions for major ticks on the transformed axis"""
        stage1_ticks = [transform_x(x) for x in range(0, STAGE1_TOTAL_TOKENS + 1, 1000)]
        stage2_ticks = [transform_x(STAGE1_TOTAL_TOKENS + x) for x in range(0, 51, 10)]
        return stage1_ticks + stage2_ticks
    
    def get_tick_labels():
        """Get labels for major ticks"""
        stage1_labels = list(range(0, STAGE1_TOTAL_TOKENS + 1, 1000))
        stage2_labels = [STAGE1_TOTAL_TOKENS + x for x in range(0, 51, 10)]
        return stage1_labels + stage2_labels
    
    # Set custom ticks
    tick_positions = get_tick_positions()
    tick_labels = get_tick_labels()
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45)
    
    # Add a vertical line at the stage 1/2 boundary
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(0.51, ax.get_ylim()[0], 'Stage 2 →', 
            rotation=90, verticalalignment='bottom')
    
    # Add x-axis label with explanation
    ax.set_xlabel('Training Tokens (Billions)\nScale adjusted to show detail in Stage 2')
    
    # Adjust y-axis for better visualization
    ax.set_ylim(bottom=0)
    
    # Move legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Improve layout
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Token performance plot saved to {output_path}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate token performance plot")
    parser.add_argument("--checkpoint_dir", type=str, required=True, 
                        help="Directory containing checkpoint result JSON files (stage1-tokensXXXB.json)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--exclude_tokens", type=str, default="",
                        help="Comma-separated list of token values to exclude (e.g., '500,1000' to exclude 500B and 1000B)")
    parser.add_argument("--token_limit", type=int, default=None,
                        help="Only include checkpoints with tokens up to this limit (in billions)")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse excluded tokens
    exclude_tokens = []
    if args.exclude_tokens:
        exclude_tokens = args.exclude_tokens.split(',')
        print(f"Will exclude the following token values: {', '.join(exclude_tokens)}B")
    
    # Find all checkpoint files
    checkpoints = find_checkpoints(args.checkpoint_dir, exclude_tokens)
    
    if not checkpoints:
        print(f"Error: No checkpoint files found in {args.checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoints)} checkpoint files.")
    
    # Load results for each checkpoint
    checkpoint_results = {}
    for filepath, name, stage, ingredient in checkpoints:
        # Apply token limit if provided
        if args.token_limit is not None and "tokens" in name:
            try:
                token_value = int(name.split('B')[0])
                if token_value > args.token_limit:
                    print(f"Skipping {name} due to token limit of {args.token_limit}B")
                    continue
            except (ValueError, IndexError):
                pass
                
        print(f"Loading checkpoint results from {filepath}")
        results = load_results(filepath)
        if results:
            checkpoint_results[name] = results
            print(f"  Loaded {len(results)} results for {name}")
        else:
            print(f"  Warning: No results loaded for {name}")
    
    if not checkpoint_results:
        print("Error: Failed to load any result files.")
        return
    
    # Add token limit to filename if specified
    token_limit_suffix = f"_up_to_{args.token_limit}B" if args.token_limit is not None else ""
    token_plot_file = os.path.join(args.output_dir, f'token_performance{token_limit_suffix}.png')
    
    # Calculate metrics
    metrics = calculate_metrics(checkpoint_results)
    
    # Print key findings
    print("\nKey Findings:")
    print(f"Total problems: {metrics['total_problems']}")
    
    print("\nResults by checkpoint:")
    for name, data in metrics["checkpoints"].items():
        print(f"{name}: {data['total_correct']} solved ({data['accuracy']:.2%})")
    
    # Generate token performance plot
    plot_token_performance(metrics, token_plot_file, args.token_limit)


if __name__ == "__main__":
    main()
