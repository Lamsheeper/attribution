#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze CruxEval results to compare the effect of changing training ingredients 
versus the effect of quantization on the set of problems solved.

Calculates Jaccard similarity between sets of correctly solved problems to quantify
the difference caused by quantization vs. changing the ingredient dataset.
"""

import json
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import re

# --- Helper Functions ---

def parse_filename(filename):
    """Extract model, status, ingredient details, and optional trial number."""
    # Pattern 1: Status explicitly present (standard/quantized)
    # Example 1: cruxeval_results_base13b_quantized_ingredient1.json
    # Example 2: cruxeval_results_base13b_standard_ingredient1_trial_0.json
    pattern1 = r"cruxeval_results_([^_]+)_(standard|quantized)_(.+?)(?:_trial_(\d+))?\.json$"
    match1 = re.match(pattern1, filename)
    
    if match1:
        model_type = match1.group(1)
        status = match1.group(2)
        ingredient_details = match1.group(3)
        trial_num_str = match1.group(4)
        trial_num = int(trial_num_str) if trial_num_str else None
        
        # Sanity check: if ingredient_details ends with _trial_N (mistakenly captured)
        if trial_num is not None and ingredient_details.endswith(f"_trial_{trial_num}"):
            ingredient_details = ingredient_details.removesuffix(f"_trial_{trial_num}")
            
        return model_type, status, ingredient_details, trial_num

    # Pattern 2: Only trial number present, status omitted (attempt to infer later)
    # Example: cruxeval_results_base13b_ingredient1_trial_1.json
    pattern2 = r"cruxeval_results_([^_]+)_(.+?)_trial_(\d+)\.json$"
    match2 = re.match(pattern2, filename)
    
    if match2:
        model_type = match2.group(1)
        status = None # Status is missing, needs inference
        ingredient_details = match2.group(2)
        trial_num_str = match2.group(3)
        trial_num = int(trial_num_str) # Trial number must be present for this pattern
        
        return model_type, status, ingredient_details, trial_num
        
    # If neither pattern matches
    return None, None, None, None

def load_ingredient_results(results_dir):
    """Load results, organizing by configuration key and trial number.
       Handles trial files missing 'standard'/'quantized' if unambiguous."""
    results_data = {} 
    base_config_statuses = {} # Stores (model, ingredient) -> set(statuses found in non-trial files)
    files_to_process_second_pass = [] # Store files needing status inference

    results_path = Path(results_dir).expanduser()
    print(f"Scanning for results in: {results_path}")
    if not results_path.is_dir():
        print(f"Error: Results directory not found: {results_path}")
        return {}

    loaded_files = 0
    skipped_files_unrecognized = 0
    skipped_files_ambiguous = 0
    error_files = 0
    found_configs = set()

    all_files = list(results_path.glob("cruxeval_results_*.json"))
    print(f"Found {len(all_files)} potential result files.")

    # --- First Pass: Process non-trial files and files with explicit status ---
    print("\n--- First Pass: Processing non-trial files and explicit-status trials ---")
    for file_path in all_files:
        filename = file_path.name
        model_type, status, ingredient_details, trial_num = parse_filename(filename)

        if model_type and ingredient_details:
            base_key = (model_type, ingredient_details)
            
            if status and trial_num is None: # Non-trial file with status
                # Record the status found for this base config
                if base_key not in base_config_statuses:
                    base_config_statuses[base_key] = set()
                base_config_statuses[base_key].add(status)
                
                # Load data directly (effective_trial_num = 0)
                config_key = (model_type, ingredient_details, status)
                found_configs.add(config_key)
                effective_trial_num = 0
                print(f"  Processing (Non-Trial): {filename} -> Config: {config_key}, Trial: {effective_trial_num}")
                if config_key not in results_data: results_data[config_key] = {}
                try:
                    with open(file_path, 'r') as f:
                        results_data[config_key][effective_trial_num] = json.load(f)
                    loaded_files += 1
                except Exception as e:
                    print(f"  Warning: Error loading {filename}: {e}")
                    error_files += 1
            
            elif status and trial_num is not None: # Trial file with explicit status
                # Load data directly
                config_key = (model_type, ingredient_details, status)
                found_configs.add(config_key)
                effective_trial_num = trial_num
                print(f"  Processing (Explicit Trial): {filename} -> Config: {config_key}, Trial: {effective_trial_num}")
                if config_key not in results_data: results_data[config_key] = {}
                # Avoid overwriting if multiple files claim the same trial
                if effective_trial_num in results_data[config_key]:
                     print(f"  Warning: Trial {effective_trial_num} already loaded for {config_key}. Skipping duplicate file {filename}.")
                     continue
                try:
                    with open(file_path, 'r') as f:
                        results_data[config_key][effective_trial_num] = json.load(f)
                    loaded_files += 1
                except Exception as e:
                    print(f"  Warning: Error loading {filename}: {e}")
                    error_files += 1

            elif status is None and trial_num is not None: # Trial file needing status inference
                print(f"  Deferring (Needs Inference): {filename}")
                files_to_process_second_pass.append(file_path)
                
            elif status is None and trial_num is None: # Should not happen with current parser logic
                 print(f"  Skipping unrecognized format (no status, no trial): {filename}")
                 skipped_files_unrecognized += 1

        else: # Filename didn't parse at all
            print(f"  Skipping unrecognized file format: {filename}")
            skipped_files_unrecognized += 1

    # --- Second Pass: Process trial files needing status inference ---
    print("\n--- Second Pass: Inferring status for remaining trial files ---")
    for file_path in files_to_process_second_pass:
        filename = file_path.name
        # Re-parse (or retrieve stored parsed info)
        model_type, _, ingredient_details, trial_num = parse_filename(filename) 
        if not model_type: continue # Should not happen if it was added to the list, but safety check
        
        base_key = (model_type, ingredient_details)
        
        inferred_status = None
        if base_key in base_config_statuses:
            possible_statuses = base_config_statuses[base_key]
            if len(possible_statuses) == 1:
                inferred_status = list(possible_statuses)[0]
                print(f"  Inferring Status for {filename}: Base key {base_key} has unique non-trial status '{inferred_status}'.")
            else:
                print(f"  Warning: Skipping ambiguous trial file {filename}. Base key {base_key} has multiple non-trial statuses: {possible_statuses}. Provide explicit 'standard' or 'quantized' in filename.")
                skipped_files_ambiguous += 1
        else:
            print(f"  Warning: Skipping trial file {filename}. Cannot infer status - no non-trial file found for base key {base_key}. Provide explicit 'standard' or 'quantized' in filename or add a corresponding non-trial file.")
            skipped_files_ambiguous += 1
            
        if inferred_status:
            config_key = (model_type, ingredient_details, inferred_status)
            found_configs.add(config_key) # Add the config even if inferred
            effective_trial_num = trial_num
            print(f"    -> Inferred Config: {config_key}, Trial: {effective_trial_num}")
            if config_key not in results_data: results_data[config_key] = {}
            if effective_trial_num in results_data[config_key]:
                     print(f"    Warning: Trial {effective_trial_num} already loaded for {config_key}. Skipping duplicate file {filename}.")
                     continue
            try:
                with open(file_path, 'r') as f:
                     results_data[config_key][effective_trial_num] = json.load(f)
                loaded_files += 1
            except Exception as e:
                print(f"    Warning: Error loading {filename}: {e}")
                error_files += 1

    print(f"\nScan Summary:")
    print(f"  Total files considered: {len(all_files)}")
    print(f"  Successfully loaded: {loaded_files}")
    print(f"  Skipped (unrecognized format): {skipped_files_unrecognized}")
    print(f"  Skipped (ambiguous/uninferrable status): {skipped_files_ambiguous}")
    print(f"  Errors (loading/JSON): {error_files}")
    print(f"  Found {len(found_configs)} unique configurations (model, ingredient, status) including inferred.")
    
    # Filter for trial analysis remains the same logic (applied to final results_data)
    results_with_multiple_trials = {
        config: trials for config, trials in results_data.items() if len(trials) >= 2
    }
    print(f"  Found {len(results_with_multiple_trials)} configurations with >= 2 trials.")
    
    return results_data

def get_correct_problem_sets(results_data):
    """Extract sets of correctly solved problem IDs for each trial."""
    # Input: results_data[config_key][trial_num] = results_list
    # Output: problem_sets[config_key][trial_num] = correct_problem_set
    problem_sets = {} 
    for config_key, trials in results_data.items():
        problem_sets[config_key] = {}
        for trial_num, results in trials.items():
            if isinstance(results, list):
                 try:
                     problem_sets[config_key][trial_num] = set(
                         r["problem_id"] for r in results if isinstance(r, dict) and r.get("is_correct")
                     )
                     print(f"  Extracted {len(problem_sets[config_key][trial_num])} correct problems for Config: {config_key}, Trial: {trial_num}")
                 except KeyError as e:
                      print(f"Warning: Missing key '{e}' in results for Config: {config_key}, Trial: {trial_num}. Skipping trial.")
                      problem_sets[config_key][trial_num] = set() # Assign empty set on error
                 except TypeError as e:
                     print(f"Warning: Type error processing results for Config: {config_key}, Trial: {trial_num} ({e}). Skipping trial.")
                     problem_sets[config_key][trial_num] = set()
            else:
                 print(f"Warning: Unexpected data format for Config: {config_key}, Trial: {trial_num}. Expected list, got {type(results)}. Assigning empty set.")
                 problem_sets[config_key][trial_num] = set()
    return problem_sets

def calculate_jaccard(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

# --- ADD NEW FUNCTION for Trial Similarity ---

def calculate_average_trial_jaccard(problem_sets):
    """Calculate the average Jaccard similarity across trials for each configuration."""
    all_trial_jaccards = []
    print("\nCalculating Cross-Trial Jaccard Similarities:")
    
    for config_key, trial_sets_dict in problem_sets.items():
        if len(trial_sets_dict) >= 2:
            config_trial_jaccards = []
            # Get all unique pairs of trial numbers for this config
            trial_pairs = list(combinations(trial_sets_dict.keys(), 2))
            # print(f"  Config: {config_key} - Found {len(trial_pairs)} trial pairs.") # Debug
            
            for t1, t2 in trial_pairs:
                set1 = trial_sets_dict[t1]
                set2 = trial_sets_dict[t2]
                jaccard = calculate_jaccard(set1, set2)
                config_trial_jaccards.append(jaccard)
                # print(f"    Trial {t1} vs Trial {t2}: {jaccard:.3f}") # Debug
                
            if config_trial_jaccards:
                avg_jaccard_for_config = np.mean(config_trial_jaccards)
                print(f"  Config: {config_key} - Average Trial Jaccard: {avg_jaccard_for_config:.4f} (across {len(trial_pairs)} pairs)")
                all_trial_jaccards.extend(config_trial_jaccards)
            else:
                 print(f"  Config: {config_key} - No valid trial pairs found to compare.")
        else:
            print(f"  Config: {config_key} - Skipping (only {len(trial_sets_dict)} trial(s)).")
            
    if not all_trial_jaccards:
        print("Warning: No configurations found with multiple trials to compare.")
        return 0 # Return 0 if no comparisons could be made
        
    overall_avg_trial_jaccard = np.mean(all_trial_jaccards)
    print(f"\nOverall Average Cross-Trial Jaccard Similarity: {overall_avg_trial_jaccard:.4f}")
    return overall_avg_trial_jaccard

# --- Plotting Functions ---

def plot_average_jaccard_comparison(avg_quant_jaccard, avg_ing_jaccard, output_path):
    """Bar chart comparing average Jaccard similarity for effects."""
    plt.figure(figsize=(8, 6))
    
    labels = [f'Quantization Effect\n(Within Ingredient Config)\nAvg Jaccard: {avg_quant_jaccard:.3f}', 
              f'Ingredient Effect\n(Between Ingredient Configs)\nAvg Jaccard: {avg_ing_jaccard:.3f}']
    values = [avg_quant_jaccard, avg_ing_jaccard]
    colors = ['#3498db', '#e74c3c'] # Blue for Quantization, Red for Ingredient Change
    
    bars = plt.bar(labels, values, color=colors, width=0.5)
    
    # Add value labels inside bars if space allows, otherwise above
    for bar in bars:
        height = bar.get_height()
        label_y_pos = height - 0.03 if height > 0.1 else height + 0.01
        label_va = 'top' if height > 0.1 else 'bottom'
        label_col = 'white' if height > 0.1 else 'black'
        
        plt.text(bar.get_x() + bar.get_width()/2., label_y_pos,
                f'{height:.3f}', ha='center', va=label_va, fontweight='bold')
                
    plt.ylabel('Average Jaccard Similarity')
    plt.title('Comparison of Effects on Solved Problem Sets\n(Higher Similarity = Smaller Change in Solved Problems)')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved average similarity comparison plot to {output_path}")

def plot_pairwise_jaccard_heatmap(problem_sets, output_path):
    """Heatmap showing pairwise Jaccard similarity between all configs."""
    
    config_names = []
    sets_list = []
    # Sort configurations for consistent heatmap ordering
    sorted_ingredient_details = sorted(problem_sets.keys())
    for ingredient_detail in sorted_ingredient_details:
        statuses = problem_sets[ingredient_detail]
        if 'standard' in statuses:
            config_names.append(f"{ingredient_detail}_standard")
            sets_list.append(statuses['standard'])
        if 'quantized' in statuses:
            config_names.append(f"{ingredient_detail}_quantized")
            sets_list.append(statuses['quantized'])
            
    if not config_names:
        print("No configurations found for heatmap.")
        return

    n_configs = len(config_names)
    jaccard_matrix = np.zeros((n_configs, n_configs))
    
    print(f"\nCalculating pairwise Jaccard matrix for {n_configs} configurations...")
    for i in range(n_configs):
        for j in range(n_configs):
            jaccard_matrix[i, j] = calculate_jaccard(sets_list[i], sets_list[j])
            
    df_heatmap = pd.DataFrame(jaccard_matrix, index=config_names, columns=config_names)
    
    # Determine appropriate figure size
    figsize_width = max(10, n_configs * 0.7)
    figsize_height = max(8, n_configs * 0.6)
    
    plt.figure(figsize=(figsize_width, figsize_height))
    sns.heatmap(df_heatmap, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1,
                linewidths=.5, cbar_kws={'label': 'Jaccard Similarity'})
    plt.title('Pairwise Jaccard Similarity Between All Configurations')
    plt.xticks(rotation=60, ha='right') # Rotate more for potentially long names
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved pairwise similarity heatmap to {output_path}")

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description='Analyze Ingredient vs Quantization Effects')
    parser.add_argument('--results_dir', type=str, 
                        default="~/attribution/data/cruxeval_results/ingredient_comparison",
                        help='Directory containing CruxEval results for ingredients')
    parser.add_argument('--output_dir', type=str, 
                        default="../data/ingredient_quant_analysis",
                        help='Directory to save analysis results and plots')
    # Optional: Add argument to specify model type if multiple might exist
    # parser.add_argument('--model_type', type=str, default=None, help='Specify model type to analyze (e.g., base13b)')
    args = parser.parse_args()

    # Set up directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    results_data = load_ingredient_results(args.results_dir)
    
    if not results_data:
        print("Exiting: No results data loaded.")
        return
        
    # 2. Get Correct Problem Sets for all loaded configs/trials
    print("\nExtracting correct problem sets...")
    problem_sets = get_correct_problem_sets(results_data)

    if not problem_sets:
        print("Exiting: No valid problem sets could be extracted.")
        return
        
    # 3. Calculate Average Trial Jaccard (NEW ANALYSIS)
    avg_trial_jaccard = calculate_average_trial_jaccard(problem_sets)

    # --- Original Analysis (Quantization vs Ingredient) --- 
    # This part requires specific pairing logic based on config keys
    print("\n--- Starting Original Analysis (Quantization vs Ingredient) ---")
    quantization_jaccards = []
    quant_jaccards_dict = {}
    ingredient_jaccards_standard = []
    ingredient_jaccards_quantized = []
    ing_jaccards_std_dict = {}
    ing_jaccards_quant_dict = {}

    # Group configs by (model_type, ingredient_details) to find standard/quantized pairs
    # *** Only include results from files WITHOUT an explicit trial number (effective_trial_num=0) ***
    grouped_configs = {}
    print("\nGrouping configurations for Quantization vs Ingredient comparison (using non-trial files):")
    for (model_type, ingredient_details, status), trials in problem_sets.items():
        base_key = (model_type, ingredient_details)
        if base_key not in grouped_configs:
            grouped_configs[base_key] = {}
            
        # Check if the result corresponding to 'no trial number' (effective_trial_num = 0) exists
        if 0 in trials: 
            grouped_configs[base_key][status] = trials[0] 
            print(f"  Found non-trial result for {base_key} status '{status}'")
        # else:
             # Optional: Add print statement if needed for debugging non-trial file absences
             # print(f"  Note: No non-trial file found for {base_key} status '{status}' for Quant/Ingred comparison.")
        
    # Calculate Quantization effect using the filtered grouped_configs
    print("\nCalculating Quantization Effect Jaccards (using non-trial files):")
    for base_key, statuses in grouped_configs.items():
        # Check if *both* standard and quantized versions exist *after* filtering for non-trial files
        if 'standard' in statuses and 'quantized' in statuses:
             set_std = statuses['standard']
             set_quant = statuses['quantized']
             jaccard = calculate_jaccard(set_std, set_quant)
             quantization_jaccards.append(jaccard)
             # Use a string representation of base_key for dict key
             quant_jaccards_dict[f"{base_key[0]}_{base_key[1]}"] = jaccard 
             print(f"  Jaccard (Quantization) for {base_key}: {jaccard:.3f}")
        else:
             # This condition might be met if only one status (std/quant) had a non-trial file
             print(f"  Skipping Quantization Jaccard for {base_key} (missing non-trial standard or quantized file).")

    # Calculate Ingredient effect using the filtered grouped_configs
    # Get pairs of base_keys *that are present in the filtered grouped_configs*
    valid_base_keys = list(grouped_configs.keys())
    base_key_pairs = list(combinations(valid_base_keys, 2))
    print(f"\nCalculating Ingredient Effect Jaccards ({len(base_key_pairs)} pairs, using non-trial files):")
    for key1, key2 in base_key_pairs:
        # Only compare if model types are the same
        if key1[0] != key2[0]: continue 
        
        # Keys are guaranteed to be in grouped_configs from how pairs were generated
        statuses1 = grouped_configs[key1]
        statuses2 = grouped_configs[key2]
        pair_label = f"{key1[0]}_{key1[1]}_vs_{key2[1]}" # Label focuses on ingredient change

        # Compare standard versions (check if 'standard' exists for *both* after filtering)
        if 'standard' in statuses1 and 'standard' in statuses2:
             j_std = calculate_jaccard(statuses1['standard'], statuses2['standard'])
             ingredient_jaccards_standard.append(j_std)
             ing_jaccards_std_dict[pair_label] = j_std
             print(f"  Jaccard (Ingredient - Standard) {key1[1]} vs {key2[1]}: {j_std:.3f}")
        else:
             print(f"  Skipping Ingredient (Standard) comparison for {pair_label} (missing non-trial 'standard' data in one or both configs).")
        
        # Compare quantized versions (check if 'quantized' exists for *both* after filtering)
        if 'quantized' in statuses1 and 'quantized' in statuses2:
             j_quant = calculate_jaccard(statuses1['quantized'], statuses2['quantized'])
             ingredient_jaccards_quantized.append(j_quant)
             ing_jaccards_quant_dict[pair_label] = j_quant
             print(f"  Jaccard (Ingredient - Quantized) {key1[1]} vs {key2[1]}: {j_quant:.3f}")
        else:
             print(f"  Skipping Ingredient (Quantized) comparison for {pair_label} (missing non-trial 'quantized' data in one or both configs).")

    # Calculate Average Similarities for original analysis
    avg_quant_jaccard = np.mean(quantization_jaccards) if quantization_jaccards else 0
    all_ingredient_jaccards = ingredient_jaccards_standard + ingredient_jaccards_quantized
    avg_ing_jaccard = np.mean(all_ingredient_jaccards) if all_ingredient_jaccards else 0

    print("\n--- Overall Summary ---")
    print(f"Average Cross-Trial Jaccard Similarity:           {avg_trial_jaccard:.4f}")
    print(f"Average Jaccard Similarity (Quantization Effect): {avg_quant_jaccard:.4f}")
    print(f"Average Jaccard Similarity (Ingredient Effect):   {avg_ing_jaccard:.4f}")
    
    # Interpretation based on original analysis
    if not quantization_jaccards or not all_ingredient_jaccards:
        print("\nConclusion (Quant/Ingred): Could not compare effects due to missing data.")
    elif abs(avg_quant_jaccard - avg_ing_jaccard) < 0.01:
         print("\nConclusion (Quant/Ingred): Quantization and changing ingredients have roughly SIMILAR effects.")
    elif avg_quant_jaccard > avg_ing_jaccard:
        print("\nConclusion (Quant/Ingred): Quantization has LESS effect than changing ingredients.")
    else:
         print("\nConclusion (Quant/Ingred): Changing ingredients has LESS effect than quantization.")

    # 5. Generate Plots
    print("\nGenerating plots...")
    if avg_quant_jaccard > 0 or avg_ing_jaccard > 0:
        plot_average_jaccard_comparison(
            avg_quant_jaccard, 
            avg_ing_jaccard, 
            output_dir / "avg_jaccard_quant_vs_ingred_comparison.png"
        )
    else:
        print("Skipping Quant vs Ingred comparison plot due to lack of data.")
    
    # Heatmap uses the full problem_sets including trials
    # Need to generate config names appropriately
    heatmap_problem_sets = {}
    for config_key, trials in problem_sets.items():
        model_type, ingredient_details, status = config_key
        for trial_num, problem_set in trials.items():
             heatmap_config_name = f"{model_type}_{status}_{ingredient_details}_trial_{trial_num}"
             heatmap_problem_sets[heatmap_config_name] = problem_set
             
    # Reformat for heatmap function (expects {name: set})
    if heatmap_problem_sets:
        # Create a new function for heatmap plotting accepting the modified structure
        plot_general_pairwise_jaccard_heatmap(heatmap_problem_sets, output_dir / "pairwise_jaccard_heatmap_all.png")
    else:
        print("Skipping pairwise heatmap plot due to lack of data.")

    # 6. Save Summary Stats
    print("\nSaving summary statistics...")
    summary_data = {
        "results_directory": args.results_dir,
        "output_directory": str(output_dir),
        "num_ingredient_configs_analyzed": len(problem_sets),
        "avg_trial_jaccard": avg_trial_jaccard,
        "avg_quantization_jaccard": avg_quant_jaccard,
        "avg_ingredient_jaccard": avg_ing_jaccard,
        "quantization_jaccards_by_ingredient_config": quant_jaccards_dict,
        "ingredient_jaccards_standard_pairs": ing_jaccards_std_dict,
        "ingredient_jaccards_quantized_pairs": ing_jaccards_quant_dict
    }
    try:
        with open(output_dir / "analysis_summary.json", 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Saved summary statistics to {output_dir / 'analysis_summary.json'}")
    except Exception as e:
        print(f"Error saving summary JSON: {e}")

    print(f"\nAnalysis complete. Plots and summary saved to {output_dir}")

# --- Need to update heatmap function call ---
# Define a more general heatmap plotting function
def plot_general_pairwise_jaccard_heatmap(named_problem_sets, output_path):
    """Heatmap showing pairwise Jaccard similarity between named configs."""
    # Input: named_problem_sets = { "config_name_str": problem_set }
    config_names = sorted(named_problem_sets.keys())
    sets_list = [named_problem_sets[name] for name in config_names]
    
    if not config_names:
        print("No configurations found for heatmap.")
        return

    n_configs = len(config_names)
    jaccard_matrix = np.zeros((n_configs, n_configs))
    
    print(f"\nCalculating pairwise Jaccard matrix for {n_configs} configurations...")
    for i in range(n_configs):
        for j in range(n_configs):
            jaccard_matrix[i, j] = calculate_jaccard(sets_list[i], sets_list[j])
            
    df_heatmap = pd.DataFrame(jaccard_matrix, index=config_names, columns=config_names)
    
    figsize_width = max(10, n_configs * 0.7)
    figsize_height = max(8, n_configs * 0.6)
    
    plt.figure(figsize=(figsize_width, figsize_height))
    sns.heatmap(df_heatmap, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1,
                linewidths=.5, cbar_kws={'label': 'Jaccard Similarity'})
    plt.title('Pairwise Jaccard Similarity Between All Configurations')
    plt.xticks(rotation=60, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved pairwise similarity heatmap to {output_path}")

if __name__ == "__main__":
    main() 