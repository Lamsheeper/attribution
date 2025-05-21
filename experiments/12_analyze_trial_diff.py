#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze differences between two trials of CruxEval results.

This script loads results from two trial files and compares which problems
were solved by both trials, only by trial 0, or only by trial 1.
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict


def load_cruxeval_results(file_path):
    """
    Load CruxEval results from a JSON file.
    
    Args:
        file_path (str): Path to the results JSON file
        
    Returns:
        dict: A tuple of (correct_problem_ids, dict_results) where:
              - correct_problem_ids is a set of problem IDs solved correctly
              - dict_results is a dictionary mapping problem_id to full result dict
    """
    file_path = Path(file_path).expanduser()
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return set(), {}
        
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
            
        # Filter for correctly solved problems
        correct_problems = set()
        problem_dict = {}
        
        # First, check if we're dealing with a list or dict structure
        if isinstance(results, list):
            for result in results:
                # Make sure it's a valid result object
                if isinstance(result, dict) and 'problem_id' in result:
                    problem_id = result['problem_id']
                    problem_dict[problem_id] = result
                    if result.get('is_correct', False):
                        correct_problems.add(problem_id)
        elif isinstance(results, dict) and 'results' in results:
            # Handle case where results might be in a nested structure
            for result in results.get('results', []):
                if isinstance(result, dict) and 'problem_id' in result:
                    problem_id = result['problem_id']
                    problem_dict[problem_id] = result
                    if result.get('is_correct', False):
                        correct_problems.add(problem_id)
        
        return correct_problems, problem_dict
    
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return set(), {}


def analyze_trial_differences(trial0_path, trial1_path, output_path=None):
    """
    Analyze differences between two trials of CruxEval results.
    
    Args:
        trial0_path (str): Path to trial 0 results
        trial1_path (str): Path to trial 1 results
        output_path (str, optional): Path to save the JSON summary
        
    Returns:
        dict: Summary of differences between trials
    """
    print(f"Loading trial 0 results from: {trial0_path}")
    trial0_correct, trial0_results = load_cruxeval_results(trial0_path)
    
    print(f"Loading trial 1 results from: {trial1_path}")
    trial1_correct, trial1_results = load_cruxeval_results(trial1_path)
    
    # Find intersection and differences
    both_solved = trial0_correct.intersection(trial1_correct)
    only_trial0 = trial0_correct - trial1_correct
    only_trial1 = trial1_correct - trial0_correct
    
    # Calculate Jaccard similarity
    union = len(trial0_correct.union(trial1_correct))
    intersection = len(both_solved)
    jaccard_similarity = intersection / union if union > 0 else 0
    
    # Create summary
    summary = {
        "trial0_path": str(trial0_path),
        "trial1_path": str(trial1_path),
        "num_problems_trial0": len(trial0_correct),
        "num_problems_trial1": len(trial1_correct),
        "num_solved_both": len(both_solved),
        "num_solved_only_trial0": len(only_trial0),
        "num_solved_only_trial1": len(only_trial1),
        "jaccard_similarity": jaccard_similarity,
        "problems_solved_both": sorted(list(both_solved)),
        "problems_solved_only_trial0": sorted(list(only_trial0)),
        "problems_solved_only_trial1": sorted(list(only_trial1))
    }
    
    # Print summary
    print("\n=== Trial Difference Analysis ===")
    print(f"Trial 0 solved: {len(trial0_correct)} problems")
    print(f"Trial 1 solved: {len(trial1_correct)} problems")
    print(f"Both trials solved: {len(both_solved)} problems")
    print(f"Only Trial 0 solved: {len(only_trial0)} problems")
    print(f"Only Trial 1 solved: {len(only_trial1)} problems")
    print(f"Jaccard similarity: {jaccard_similarity:.4f}")
    
    # Save the summary if output path is provided
    if output_path:
        output_path = Path(output_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to {output_path}")
        
        # Create and save the generations comparison for problems where only one trial succeeded
        if only_trial0 or only_trial1:
            create_generation_comparison(
                trial0_results, 
                trial1_results, 
                only_trial0, 
                only_trial1, 
                output_path.parent / f"generation_comparison_{Path(trial0_path).stem.split('_')[-1]}_{Path(trial1_path).stem.split('_')[-1]}.json"
            )
    
    return summary


def create_generation_comparison(trial0_results, trial1_results, only_trial0, only_trial1, output_path):
    """
    Create a detailed comparison of the model generations for problems where only one trial succeeded.
    
    Args:
        trial0_results (dict): Dictionary mapping problem_id to full result dict for trial 0
        trial1_results (dict): Dictionary mapping problem_id to full result dict for trial 1
        only_trial0 (set): Set of problem IDs solved only by trial 0
        only_trial1 (set): Set of problem IDs solved only by trial 1
        output_path (str): Path to save the JSON output
    """
    print("\nExtracting generations for problems with different outcomes...")
    
    comparison = {
        "problems_only_trial0_solved": [],
        "problems_only_trial1_solved": []
    }
    
    # Extract relevant fields for problems only solved by trial 0
    for problem_id in sorted(only_trial0):
        if problem_id in trial0_results and problem_id in trial1_results:
            trial0_data = trial0_results[problem_id]
            trial1_data = trial1_results[problem_id]
            
            problem_detail = {
                "problem_id": problem_id,
                # Extract the generation fields using the correct keys from the CruxEval format
                "trial0_full_generated_text": trial0_data.get("full_generated_text", ""),
                "trial0_generated_answer": trial0_data.get("generated", ""),
                "trial1_full_generated_text": trial1_data.get("full_generated_text", ""),
                "trial1_generated_answer": trial1_data.get("generated", ""),
                "true_output": trial0_data.get("true_output", ""),
                # We know these values from the set membership
                "trial0_is_correct": True,
                "trial1_is_correct": False
            }
            comparison["problems_only_trial0_solved"].append(problem_detail)
    
    # Extract relevant fields for problems only solved by trial 1
    for problem_id in sorted(only_trial1):
        if problem_id in trial0_results and problem_id in trial1_results:
            trial0_data = trial0_results[problem_id]
            trial1_data = trial1_results[problem_id]
            
            problem_detail = {
                "problem_id": problem_id,
                # Extract the generation fields using the correct keys from the CruxEval format
                "trial0_full_generated_text": trial0_data.get("full_generated_text", ""),
                "trial0_generated_answer": trial0_data.get("generated", ""),
                "trial1_full_generated_text": trial1_data.get("full_generated_text", ""),
                "trial1_generated_answer": trial1_data.get("generated", ""),
                "true_output": trial1_data.get("true_output", ""),
                # We know these values from the set membership
                "trial0_is_correct": False,
                "trial1_is_correct": True
            }
            comparison["problems_only_trial1_solved"].append(problem_detail)
    
    # Add some helpful metadata about the comparison
    comparison["metadata"] = {
        "num_problems_only_trial0_solved": len(comparison["problems_only_trial0_solved"]),
        "num_problems_only_trial1_solved": len(comparison["problems_only_trial1_solved"]),
        "total_differential_problems": len(comparison["problems_only_trial0_solved"]) + len(comparison["problems_only_trial1_solved"])
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Saved generation comparison to {output_path}")
    print(f"  Included {len(comparison['problems_only_trial0_solved'])} problems only solved by trial 0")
    print(f"  Included {len(comparison['problems_only_trial1_solved'])} problems only solved by trial 1")


def analyze_problem_categories(trial0_path, trial1_path, output_prefix=None):
    """
    Analyze differences between trials with additional categorization of problems.
    
    This function extends the basic difference analysis by categorizing problems
    by their type, difficulty, etc. to understand patterns in trial differences.
    
    Args:
        trial0_path (str): Path to trial 0 results
        trial1_path (str): Path to trial 1 results
        output_prefix (str, optional): Prefix for output files
    """
    _, trial0_results = load_cruxeval_results(trial0_path)
    _, trial1_results = load_cruxeval_results(trial1_path)
    
    # Combine all problem IDs
    all_problem_ids = set(trial0_results.keys()).union(set(trial1_results.keys()))
    
    # Create a detailed problem-by-problem comparison
    detailed_analysis = []
    
    # Get the categorization attributes that exist in the results
    # Sample one result to determine available attributes
    sample_result = next(iter(trial0_results.values())) if trial0_results else None
    if not sample_result:
        sample_result = next(iter(trial1_results.values())) if trial1_results else None
    
    if not sample_result:
        print("No valid results found for analysis")
        return
    
    # Determine which category attributes exist in the data
    category_attributes = [
        attr for attr in ['category', 'sub_category', 'difficulty', 'task_type']
        if attr in sample_result
    ]
    
    for problem_id in sorted(all_problem_ids):
        trial0_result = trial0_results.get(problem_id, {})
        trial1_result = trial1_results.get(problem_id, {})
        
        trial0_correct = trial0_result.get('is_correct', False)
        trial1_correct = trial1_result.get('is_correct', False)
        
        # Determine outcome category
        if trial0_correct and trial1_correct:
            outcome = "both_solved"
        elif trial0_correct:
            outcome = "only_trial0_solved"
        elif trial1_correct:
            outcome = "only_trial1_solved"
        else:
            outcome = "neither_solved"
        
        # Get problem metadata from either result
        result = trial0_result or trial1_result
        problem_info = {
            'problem_id': problem_id,
            'outcome': outcome
        }
        
        # Add category information if available
        for attr in category_attributes:
            if attr in result:
                problem_info[attr] = result[attr]
        
        detailed_analysis.append(problem_info)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(detailed_analysis)
    
    # Output detailed analysis if prefix is provided
    if output_prefix:
        output_path = Path(f"{output_prefix}_detailed.csv").expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved detailed analysis to {output_path}")
        
        # Generate category distribution plots if categories exist
        if category_attributes:
            for attr in category_attributes:
                if attr in df.columns:
                    plot_category_distribution(df, attr, f"{output_prefix}_{attr}_distribution.png")


def plot_category_distribution(df, category_column, output_path):
    """
    Plot the distribution of problem categories across different outcomes.
    
    Args:
        df (DataFrame): DataFrame with detailed problem analysis
        category_column (str): The category column to analyze
        output_path (str): Path to save the plot
    """
    if category_column not in df.columns or 'outcome' not in df.columns:
        return
    
    # Create a crosstab of outcomes by category
    cross_tab = pd.crosstab(df[category_column], df['outcome'])
    
    # Plot
    plt.figure(figsize=(12, 8))
    cross_tab.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title(f'Distribution of {category_column} by outcome')
    plt.xlabel(category_column)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {category_column} distribution plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze differences between two CruxEval trial results.')
    parser.add_argument('--trial0', type=str, 
                      default="0",
                      help='Trial num for trial 0 results')
    parser.add_argument('--trial1', type=str, 
                      default="1",
                      help='Trial num for trial 1 results')
    parser.add_argument('--output', type=str, 
                      default=None,
                      help='Path to save the summary JSON (defaults to trial_diff_summary_TRIAL0_TRIAL1.json)')
    parser.add_argument('--detailed', action='store_true',
                      help='Perform detailed analysis by problem categories')
    parser.add_argument('--compare-gens', action='store_true',
                      help='Create a standalone generation comparison file (in addition to the default behavior)')
    parser.add_argument('--gen-output', type=str,
                      default=None,
                      help='Path to save the generation comparison JSON (defaults to same directory as output)')
    parser.add_argument('--deterministic', action='store_true',
                      help='Use deterministic generation settings when creating new evaluations')
    
    args = parser.parse_args()
    
    # Set output path with trial numbers if not provided
    if args.output is None:
        args.output = f"../data/trial_diff_analysis/trial_diff_summary_{args.trial0}_{args.trial1}.json"
    
    # Fix: Add f-string prefix to use variable substitution
    path0 = f"~/attribution/data/cruxeval_results/ingredient_comparison/cruxeval_results_base13b_standard_stage2-ingredient1-step11931-tokens101B_trial_{args.trial0}.json"
    path1 = f"~/attribution/data/cruxeval_results/ingredient_comparison/cruxeval_results_base13b_standard_stage2-ingredient1-step11931-tokens101B_trial_{args.trial1}.json"
    
    # Basic difference analysis
    summary = analyze_trial_differences(path0, path1, args.output)
    
    # If we don't have a summary or there are no differences, we're done
    if not summary:
        return
    
    # Detailed analysis if requested
    if args.detailed:
        output_prefix = str(Path(args.output).with_suffix(''))
        analyze_problem_categories(path0, path1, output_prefix)
    
    # If explicitly requested, create a separate generations comparison file
    if args.compare_gens:
        # Load the raw results again if needed
        _, trial0_results = load_cruxeval_results(path0)
        _, trial1_results = load_cruxeval_results(path1)
        
        # Get the problem_id sets
        only_trial0 = set(summary.get("problems_solved_only_trial0", []))
        only_trial1 = set(summary.get("problems_solved_only_trial1", []))
        
        # Determine output path
        if args.gen_output:
            gen_output_path = Path(args.gen_output).expanduser()
        else:
            # Default to placing in the same directory as the main output
            output_dir = Path(args.output).expanduser().parent
            gen_output_path = output_dir / f"generation_comparison_{args.trial0}{args.trial1}.json"
        
        # Create the comparison
        create_generation_comparison(
            trial0_results,
            trial1_results,
            only_trial0,
            only_trial1,
            gen_output_path
        )


if __name__ == "__main__":
    main()
