# Compares pass rates of different models across multiple coding benchmarks.
# Creates visualizations to highlight performance differences between models, 
# analyzing factors affecting code generation quality and task completion rates.

#%%
import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from bigcode_eval.tasks import get_task

def load_pass_rates(pass_rates_path):
    """
    Given a path to a pass_rates.json file (with fields 'problem_ids' and 'pass_rates'),
    returns a dict mapping problem_id -> pass_rate.
    """
    with open(pass_rates_path, 'r') as f:
        data = json.load(f)
    
    # Convert problem_ids to strings for consistent access
    problem_ids = [str(pid) for pid in data['problem_ids']]
    pass_rates = data['pass_rates']
    return dict(zip(problem_ids, pass_rates))

# Directory where each model's results are stored.
# Adjust as needed if your files are organized differently.
results_root = "/home/lofty/code_llm/attribution/results"

# Models to analyze (excluding the one with "-RM")
models = [
    "allenai/OLMo2-7B-1124",
    "allenai/OLMo-2-1124-7B-SFT",
    "allenai/OLMo-2-1124-7B-DPO",
    "allenai/OLMo-2-1124-7B-Instruct"
]

# Load pass rates for each model. 
# We'll store them in a dict of dicts: pass_rates_dict[model][problem_id] = pass_rate
pass_rates_dict = {}
for model in models:
    pass_rates_path = os.path.join(results_root, model, "pass_rates.json")
    if not os.path.exists(pass_rates_path):
        print(f"Warning: pass_rates.json not found for {model}. Skipping.")
        continue
    pass_rates_dict[model] = load_pass_rates(pass_rates_path)

# Find the intersection of problem IDs that appear in *all* models
all_problem_sets = [set(d.keys()) for d in pass_rates_dict.values()]
common_problems = set.intersection(*all_problem_sets) if all_problem_sets else set()
if not common_problems:
    raise ValueError("No common problem IDs found across models. Ensure pass_rates.json files match in problem IDs.")

# Sort the common problem IDs for consistent ordering
common_problems = sorted(list(common_problems))

# Build a matrix of shape (num_problems, num_models)
# Each row corresponds to a single problem, each column to a model
model_list = sorted(pass_rates_dict.keys())  # keep a consistent ordering
data_matrix = []
for problem_id in common_problems:
    row = []
    for model in model_list:
        row.append(pass_rates_dict[model][problem_id])
    data_matrix.append(row)

data_matrix = np.array(data_matrix)  # shape = (num_problems, num_models)

# After building data_matrix = np.array(data_matrix) of shape (num_problems, num_models)
# We want to transpose it to shape (num_models, num_problems).
# data_matrix was originally (num_problems, num_models).
# Let's call it "data_matrix_T" to avoid confusion, where each row = model, each col = problem.
data_matrix_T = data_matrix.T  # shape = (num_models, num_problems)

# Compute correlation among problems across these models.
# By default, np.corrcoef(data_matrix, rowvar=True) computes the correlation
# among the rows of data_matrix. Each row = (pass rates across the different models).
corr_matrix = np.corrcoef(data_matrix, rowvar=True)  # shape = (num_problems, num_problems)

# 1) Model–Model Correlation -----------------------------------
# Each row is [pass rates across problems] for that model
# So we compute correlation among the rows.
corr_matrix_models = np.corrcoef(data_matrix_T, rowvar=True)  # shape = (num_models, num_models)

model_list = sorted(pass_rates_dict.keys())

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix_models,
            xticklabels=model_list,
            yticklabels=model_list,
            cmap="RdBu", vmin=-1, vmax=1, annot=True)
plt.title("Model–Model Correlation on Problems' Pass Rates")
plt.xlabel("Model")
plt.ylabel("Model")
plt.tight_layout()
plt.savefig("model_model_correlation_heatmap.png", dpi=150)
plt.show()

# Print correlation results in a more textual form
print("\n=== Model–Model Correlation ===")
for i in range(len(model_list)):
    for j in range(i + 1, len(model_list)):
        print(f"{model_list[i]} vs. {model_list[j]}: corr = {corr_matrix_models[i, j]:.3f}")

# 2) Top-Quartile Overlap ---------------------------------------
# For each model, find which problems are in its top 25% (by pass rate).
# Then see, for each pair (A,B), among the problems that are top-25% in A,
# how many are also top-25% in B. We call that the "conditional overlap".
# Additionally, we can compute a symmetrical measure (e.g., Jaccard).

num_models = len(model_list)
num_problems = len(common_problems)

# Compute index sets for top-25% in each model
top_quartile_indices = []
for i in range(num_models):
    pass_rates = data_matrix_T[i]  # pass rates for model i across problems
    threshold = np.percentile(pass_rates, 75)
    top_quartile_mask = pass_rates >= threshold
    top_indices = np.where(top_quartile_mask)[0]
    top_quartile_indices.append(set(top_indices))

# Build overlap matrices
overlap_conditional = np.zeros((num_models, num_models))
overlap_jaccard = np.zeros((num_models, num_models))

for i in range(num_models):
    setA = top_quartile_indices[i]
    for j in range(num_models):
        setB = top_quartile_indices[j]
        if len(setA) == 0:  
            # Avoid divide-by-zero
            overlap_conditional[i, j] = np.nan
        else:
            overlap_conditional[i, j] = len(setA.intersection(setB)) / len(setA)
        
        unionAB = setA.union(setB)
        if len(unionAB) == 0:
            overlap_jaccard[i, j] = np.nan
        else:
            overlap_jaccard[i, j] = len(setA.intersection(setB)) / len(unionAB)

# Plot the conditional Overlap heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(overlap_conditional,
            xticklabels=model_list,
            yticklabels=model_list,
            cmap="Blues", vmin=0, vmax=1, annot=True, fmt=".2f")
plt.title("Top 25% Overlap (A→B): fraction of A's top-25% also in B's top-25%")
plt.xlabel("B")
plt.ylabel("A")
plt.tight_layout()
plt.savefig("model_model_topquartile_conditional.png", dpi=150)
plt.show()

# Plot the symmetrical Jaccard Overlap
plt.figure(figsize=(8, 6))
sns.heatmap(overlap_jaccard,
            xticklabels=model_list,
            yticklabels=model_list,
            cmap="Blues", vmin=0, vmax=1, annot=True, fmt=".2f")
plt.title("Top 25% Jaccard Overlap: Intersection(A,B) / Union(A,B)")
plt.xlabel("Model B")
plt.ylabel("Model A")
plt.tight_layout()
plt.savefig("model_model_topquartile_jaccard.png", dpi=150)
plt.show()

print("\n=== Top-25% Overlap (A→B) ===")
for i in range(num_models):
    for j in range(num_models):
        if i == j:
            continue
        val = overlap_conditional[i, j]
        if not np.isnan(val):
            print(f"{model_list[i]} → {model_list[j]}: {val*100:.1f}% of A's top-25% also in B's top-25%")

print("\n=== Top-25% Jaccard Overlap (Symmetric) ===")
for i in range(num_models):
    for j in range(i + 1, num_models):
        val = overlap_jaccard[i, j]
        if not np.isnan(val):
            print(f"{model_list[i]} <-> {model_list[j]}: Jaccard = {val*100:.1f}%")


#%%

# 3) Problems solved by average of Instruct, DPO, and SFT but not by baseline
baseline_model = "allenai/OLMo2-7B-1124"
ensemble_models = [
    "allenai/OLMo-2-1124-7B-Instruct",
    "allenai/OLMo-2-1124-7B-DPO",
    "allenai/OLMo-2-1124-7B-SFT",
]

missing_solved_pairs = []
for problem_id in common_problems:
    baseline_rate = pass_rates_dict[baseline_model][problem_id]
    ensemble_rates = [pass_rates_dict[m][problem_id] for m in ensemble_models]
    ensemble_mean = sum(ensemble_rates) / len(ensemble_rates)
    # "Not solved by baseline" => pass rate == 0
    # "Solved by the ensemble average" => average pass rate > 0
    if baseline_rate == 0 and ensemble_mean > 0:
        missing_solved_pairs.append((problem_id, ensemble_mean))

# Sort by ensemble average pass rate descending
missing_solved_pairs.sort(key=lambda x: x[1], reverse=True)

if not missing_solved_pairs:
    print("No problems are unsolved by baseline but solved by the ensemble average of Instruct/DPO/SFT.")
else:
    print(f"Found {len(missing_solved_pairs)} problems unsolved by {baseline_model} but solved by the ensemble average of {ensemble_models}.")

    # Prepare data for plotting
    problem_ids_sorted = [x[0] for x in missing_solved_pairs]
    ensemble_pass_rates_sorted = [x[1] for x in missing_solved_pairs]

    plt.figure(figsize=(8, 0.3*len(problem_ids_sorted)))
    y_positions = np.arange(len(problem_ids_sorted))

    plt.barh(y_positions, ensemble_pass_rates_sorted, color="green")
    plt.yticks(y_positions, problem_ids_sorted)
    plt.gca().invert_yaxis()  # largest (highest average pass rate) at the top
    plt.xlabel("Ensemble Average Pass Rate")
    plt.title("Problems with 0% Baseline, >0% Average of Instruct/DPO/SFT")
    plt.tight_layout()
    plt.savefig("problems_unsolved_by_baseline_solved_by_ensemble.png", dpi=150)
    plt.show()
# %%
from bigcode_eval.tasks import get_task
most_improved_problems = [27, 28, 51, 56, 61]

# Get prompts for most improved problems using bigcode_eval
task = get_task("humanevalplus")
dataset = task.get_dataset()

print("\nPrompts for most improved problems:")
print("-" * 80)
for problem_id in most_improved_problems:
    # Get prompt directly from dataset
    prompt = dataset['prompt'][problem_id]
    print(f"\nProblem {problem_id}:")
    print(prompt)
    print("-" * 80)
#%%
from typing import List

def flip_case(string: str) -> str:
    """ problem 27
    For a given string, flip lowercase characters to uppercase and uppercase to lowercase.
    >>> flip_case('Hello')
    'hELLO'
    """

def concatenate(strings: List[str]) -> str:
    """problem 28
    Concatenate list of strings into a single string
    >>> concatenate([])
    ''
    >>> concatenate(['a', 'b', 'c'])
    'abc'
    """

def remove_vowels(text):
    """problem 51
    remove_vowels is a function that takes string and returns string without vowels.
    >>> remove_vowels('')
    ''
    >>> remove_vowels("abcdef\nghijklm")
    'bcdf\nghjklm'
    >>> remove_vowels('abcdef')
    'bcdf'
    >>> remove_vowels('aaaaa')
    ''
    >>> remove_vowels('aaBAA')
    'B'
    >>> remove_vowels('zbcd')
    'zbcd'
    """

def correct_bracketing(brackets: str):
    """problem 56
    brackets is a string of "<" and ">".
    return True if every opening bracket has a corresponding closing bracket.

    >>> correct_bracketing("<")
    False
    >>> correct_bracketing("<>")
    True
    >>> correct_bracketing("<<><>>")
    True
    >>> correct_bracketing("><<>")
    False
    """

def correct_bracketing(brackets: str):
    """problem 61
    brackets is a string of "(" and ")".
    return True if every opening bracket has a corresponding closing bracket.

    >>> correct_bracketing("(")
    False
    >>> correct_bracketing("()")
    True
    >>> correct_bracketing("(()())")
    True
    >>> correct_bracketing(")(()")
    False
    """

def compare_problem_generations(problem_id, models_to_compare, n_responses=None):
    """
    Compare generations for a specific problem across different models.
    
    Args:
        problem_id (int): ID of the problem to analyze
        models_to_compare (list): List of model names to compare
        n_responses (int, optional): Limit to first n responses. Defaults to None (all responses).
        
    Returns:
        str: Formatted string containing the comparison results
    """
    output = []  # List to store output strings
    task = get_task("humanevalplus")
    dataset = task.get_dataset()
    
    output.append(f"\nProblem {problem_id}:")
    output.append("="*80)
    output.append(dataset['prompt'][problem_id])
    output.append("="*80)
    
    for model in models_to_compare:
        output.append(f"\nModel: {model}")
        output.append("-"*40)
        
        # Load generations and results
        generations_path = os.path.join(results_root, model, "humanevalplus_generations.json")
        results_path = os.path.join(results_root, model, "evaluation_results.json")
        
        if not os.path.exists(generations_path) or not os.path.exists(results_path):
            output.append(f"Missing files for {model}")
            continue
            
        with open(generations_path, 'r') as f:
            generations = json.load(f)
            
        with open(results_path, 'r') as f:
            loaded_results = json.load(f)
            results = {str(k): v for k, v in loaded_results.items()}
        
        # Get generations for this problem
        problem_generations = generations[problem_id]
        problem_results = results[str(problem_id)]
        
        # Limit responses if requested
        if n_responses is not None:
            problem_generations = problem_generations[:n_responses]
            problem_results = problem_results[:n_responses]
        
        # Format each generation and its result
        for i, (generation, result) in enumerate(zip(problem_generations, problem_results)):
            output.append(f"\nGeneration {i+1}:")
            output.append(f"Passed: {result[1]['passed']}")
            if not result[1]['passed']:
                output.append(f"Result: {result[1]['result']}")
            output.append(generation)
            output.append("-"*40)
    
    return "\n".join(output)

# Example usage for problem 56 (one of the correct_bracketing problems)
print(compare_problem_generations(
    56,  # problem ID instead of name
    ["allenai/OLMo2-7B-1124", "allenai/OLMo-2-1124-7B-DPO"],
    n_responses=2
))

#%%

from pathlib import Path

def save_str(output, filename):
    with open(filename, 'w') as f:
        f.write(output)

for model_name in ["allenai/OLMo2-7B-1124", "allenai/OLMo-2-1124-7B-DPO"]:
    output = compare_problem_generations(
        56,  # problem ID instead of name
        [model_name],
        n_responses=20
    )
    save_str(output, str(Path(results_root) / model_name / "generations_comparison.txt"))

#%%

