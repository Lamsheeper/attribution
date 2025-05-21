# Analyzes detailed results from CruxEval benchmark evaluations, focusing on correctness patterns.
# Performs statistical analysis on model outputs, identifying frequently correct/incorrect problems
# and extracting insights about model capabilities in code generation.

from pathlib import Path
import json
from bigcode_eval.tasks import get_task
import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Base results directory
base_dir = Path("/home/lofty/code_llm/attribution/results/allenai")

# Load the task and dataset for ground truth
task = get_task("cruxeval")
dataset = task.get_dataset()

def get_ground_truth_pass_rate(results_file):
    """Calculate pass rate from a results file and check against ground truth"""
    results_dir = results_file.parent
    with open(results_file) as f:
        results = json.load(f)
    with open(results_dir / "cruxeval_generations.json") as f:
        generations = json.load(f)
    
    total_problems = len(results)
    true_passed_problems = 0
    true_pass_problem_ids = []
    
    for problem_id, problem_results in results.items():
        idx = int(problem_id)
        ground_truth = dataset[idx]["output"].strip()
        generation = generations[idx][0].strip()
        
        # Check if pass was correct
        if ground_truth == generation:
            true_pass_problem_ids.append(idx)
            true_passed_problems += 1
    
    # Write results to file
    model_name = results_dir.name
    
    # Create data structure for this model
    model_data = {
        "model_name": model_name,
        "true_pass": true_passed_problems,
        "total_problems": total_problems,
        "true_pass_rate": (true_passed_problems / total_problems) * 100 if total_problems > 0 else 0,
        "solved_problem_ids": true_pass_problem_ids
    }
    
    # Create the data directory if it doesn't exist
    data_dir = Path(base_dir).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Check if the file exists and load it, or create new
    output_file = data_dir / "cruxeval_true_pass_results.json"
    if output_file.exists():
        with open(output_file, 'r') as f:
            try:
                all_data = json.load(f)
            except json.JSONDecodeError:
                all_data = {"timestamp": str(datetime.datetime.now()), "models": {}}
    else:
        all_data = {"timestamp": str(datetime.datetime.now()), "models": {}}
    
    # Add or update this model's data
    all_data["models"][model_name] = model_data
    
    # Write back to file
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"Updated results for {model_name} in {output_file}")
    
    return true_passed_problems, total_problems, true_pass_problem_ids

# Find all model directories
model_dirs = [d for d in base_dir.iterdir() if d.is_dir()]

print("\nModel Performance Summary:")
print("-" * 120)
header = f"{'Model':50s} | {'True Pass':>10s} | {'Total':>10s} | {'True Rate':>10s}"
print(header)
print("-" * 120)

# Process each model's results
for model_dir in model_dirs:
    results_file = model_dir / "cruxeval_evaluation_results_detailed.json"
    if results_file.exists():
        try:
            true_pass, total, problem_ids = get_ground_truth_pass_rate(results_file)
            
            model_name = model_dir.name
            true_rate = (true_pass / total) * 100 if total > 0 else 0
            print(f"{model_name:50s} | {true_pass:>10d} | {total:>10d} | {true_rate:>9.2f}%")
        except Exception as e:
            print(f"Error processing {model_dir.name}: {e}")

print("-" * 120)
print("All model results have been saved to the data directory.")
