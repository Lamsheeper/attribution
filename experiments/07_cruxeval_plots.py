# Creates visualizations specifically for CruxEval benchmark results across different models.
# Generates plots showing correctness patterns, comparative model performance, and 
# problem-specific analytics to identify strengths and weaknesses in code generation.

from pathlib import Path
import json
from bigcode_eval.tasks import get_task
import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
# Plotting code - reads from the saved JSON file
# Load the saved results
results_file = Path("/home/lofty/code_llm/attribution/data/cruxeval_true_pass_results.json")
with open(results_file, 'r') as f:
    all_data = json.load(f)

# Create a directory for plots
plots_dir = Path("/home/lofty/code_llm/attribution/data/cruxeval_plots")
plots_dir.mkdir(exist_ok=True)

# Extract model data
models_data = all_data["models"]
model_names = list(models_data.keys())
true_passes = [data["true_pass"] for data in models_data.values()]
pass_rates = [data["true_pass_rate"] for data in models_data.values()]

# Sort by performance
sorted_indices = np.argsort(true_passes)[::-1]  # Descending order
model_names = [model_names[i] for i in sorted_indices]
true_passes = [true_passes[i] for i in sorted_indices]
pass_rates = [pass_rates[i] for i in sorted_indices]

# Plot 1: Total problems solved by each model
plt.figure(figsize=(12, 6))
bars = plt.bar(model_names, true_passes)
plt.title('Number of CruxEval Problems Solved by Each Model')
plt.xlabel('Model')
plt.ylabel('Problems Solved')
plt.xticks(rotation=45, ha='right')

# Set y-axis to start at 0 with enough headroom
max_value = max(true_passes)
plt.ylim(0, max_value * 1.15)  # Add 15% headroom

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + (max_value * 0.02),
            f'{height}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(plots_dir / "cruxeval_model_comparison.png")
plt.show()

# Plot 2: Pass rates
plt.figure(figsize=(12, 6))
bars = plt.bar(model_names, pass_rates)
plt.title('CruxEval Pass Rates by Model')
plt.xlabel('Model')
plt.ylabel('Pass Rate (%)')
plt.xticks(rotation=45, ha='right')

# Set y-axis to start at 0 with enough headroom
max_value = max(pass_rates)
plt.ylim(0, max_value * 1.15)  # Add 15% headroom

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + (max_value * 0.02),
            f'{height:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(plots_dir / "cruxeval_pass_rates.png")
plt.show()

# Calculate unique problems solved by each model
all_problem_ids = {}
for model, data in models_data.items():
    all_problem_ids[model] = set(data["solved_problem_ids"])

# Count how many models solved each problem
problem_solve_count = Counter()
for model_problems in all_problem_ids.values():
    problem_solve_count.update(model_problems)

# Find unique problems for each model
unique_counts = []
for model, problem_set in all_problem_ids.items():
    unique_problems = [p for p in problem_set if problem_solve_count[p] == 1]
    unique_counts.append(len(unique_problems))

# Reorder to match the sorted models
unique_counts = [unique_counts[model_names.index(model)] for model in model_names]

# Plot 3: Unique problems solved by each model
plt.figure(figsize=(12, 6))
bars = plt.bar(model_names, unique_counts)
plt.title('Number of CruxEval Problems Uniquely Solved by Each Model')
plt.xlabel('Model')
plt.ylabel('Unique Problems Solved')
plt.xticks(rotation=45, ha='right')

# Set y-axis to start at 0 with enough headroom
max_value = max(unique_counts) if unique_counts else 1  # Handle empty list
plt.ylim(0, max_value * 1.15)  # Add 15% headroom

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + (max_value * 0.02),
            f'{height}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(plots_dir / "cruxeval_unique_problems.png")
plt.show()

# Plot 4: Stacked bar chart (total and unique)
plt.figure(figsize=(12, 6))
non_unique = [total - unique for total, unique in zip(true_passes, unique_counts)]

plt.bar(model_names, non_unique, label='Solved by Multiple Models')
plt.bar(model_names, unique_counts, bottom=non_unique, label='Uniquely Solved')

plt.title('CruxEval Problems Solved by Each Model')
plt.xlabel('Model')
plt.ylabel('Problems Solved')
plt.legend()
plt.xticks(rotation=45, ha='right')

# Set y-axis to start at 0 with enough headroom
max_value = max(true_passes)
plt.ylim(0, max_value * 1.15)  # Add 15% headroom

plt.tight_layout()
plt.savefig(plots_dir / "cruxeval_stacked_comparison.png")
plt.show()

# Print some additional statistics
print("\nUnique Problems Solved by Each Model:")
for model, count in zip(model_names, unique_counts):
    print(f"{model}: {count} unique problems")

# Find problems solved by all models
common_problems = set.intersection(*[set(data["solved_problem_ids"]) for data in models_data.values()])
print(f"\nNumber of problems solved by ALL models: {len(common_problems)}")

# Find problems solved by at least one model
any_solved = set.union(*[set(data["solved_problem_ids"]) for data in models_data.values()])
print(f"Number of problems solved by AT LEAST ONE model: {len(any_solved)}")
print(f"Number of problems not solved by any model: {800 - len(any_solved)}")

#%%
