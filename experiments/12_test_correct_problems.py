# Analyzes the overlap of correctly solved programming problems between different OLMo checkpoints.
# Creates visualizations to compare how different model ingredients solve problems, measuring
# intersection-over-union metrics and identifying patterns in problem-solving capabilities.
# %% Now that we know which cases are in the 6% - which training iteration causes those cases to flip to working?  If you look at the specific training documents, do they also contain easy programming exercises?

from fastcore.all import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json


def json_to_df(results_file):
    """Convert a JSON file containing problem evaluation results to a pandas DataFrame."""
    results = Path(results_file).read_json()
    df = pd.DataFrame(results)
    df['problem_id'] = df['problem_id'].astype(int)
    return df


# Path to results directory
results_dir = Path("/share/u/lofty/code_llm/attribution/data/cruxeval_results_base_revisions/cruxeval_results_allenai/13B")

# Original reference file
reference_file = results_dir / "OLMo-2-1124-13B_stage2-ingredient1-step11931-tokens100B.json"
reference_df = json_to_df(reference_file)
reference_correct = set(reference_df[reference_df["is_correct"]]['problem_id'])
print(f"Reference file has {len(reference_correct)} correct problems")

# Find and analyze all stage2 files
stage2_files = [f for f in results_dir.glob("*stage2*.json") if f.name != reference_file.name]

# Calculate overlap metrics
results = []
for file_path in stage2_files:
    try:
        # Extract metadata
        pattern = r'OLMo-2-1124-13B_stage2-ingredient(\d+)-step(\d+)'
        match = re.search(pattern, file_path.name)
        if not match:
            continue
            
        ingredient, step = match.groups()
        
        # Get correct problem IDs
        df = json_to_df(file_path)
        correct_ids = set(df[df['is_correct']]['problem_id'])
        
        # Calculate metrics
        intersection = len(reference_correct.intersection(correct_ids))
        union = len(reference_correct.union(correct_ids))
        iou = intersection / union if union > 0 else 0
        
        results.append({
            'file': file_path.name,
            'ingredient': int(ingredient),
            'step': int(step),
            'num_correct': len(correct_ids),
            'intersection': intersection,
            'union': union,
            'iou': iou
        })
        
        print(f"Processed {file_path.name}: {len(correct_ids)} correct problems, IoU: {iou:.2f}")
            
    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")

# Create DataFrame for plotting
# Create DataFrame for plotting
df_results = pd.DataFrame(results)
df_results["label"] = df_results.apply(
    lambda x: f"ing{x['ingredient']}-step{x['step']}", axis=1
)
df_results = df_results.sort_values("iou", ascending=False)
#%%
import seaborn as sns
# Setup plot style - Tufte-inspired
plt.figure(figsize=(12, 8))
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1: Bar plot of union sizes
sns.barplot(x="label", y="union", data=df_results, palette="Blues_d", ax=ax1)
ax1.set_title("Union of Correct Problems", fontsize=14)
ax1.set_xlabel("Checkpoint", fontsize=12)
ax1.set_ylabel("Number of Unique Correct Problems", fontsize=12)
ax1.tick_params(axis="x", rotation=45)

# Plot 2: Bar plot of IoU values
sns.barplot(x="label", y="iou", data=df_results, palette="YlGnBu", ax=ax2)
ax2.set_title("IoU with Reference Checkpoint", fontsize=14)
ax2.set_xlabel("Checkpoint", fontsize=12)
ax2.set_ylabel("Intersection over Union (IoU)", fontsize=12)
ax2.tick_params(axis="x", rotation=45)

# Add overall title
plt.suptitle("Overlap of Correctly Solved Problems with reference checkpoint. \nReference checkpoint ing1-step11931. 800 problems total.", fontsize=16, y=1.02)
plt.tight_layout()

#%%

# Also create a single consolidated plot for a cleaner view
plt.figure(figsize=(12, 6))
df_plot = df_results.copy()
df_plot = df_plot.sort_values(["ingredient", "step"])  # Sort by ingredient and step

# Set up a color palette that distinguishes ingredients
palette = sns.color_palette("viridis", n_colors=len(df_plot["ingredient"].unique()))
ingredient_colors = {
    ing: palette[i] for i, ing in enumerate(sorted(df_plot["ingredient"].unique()))
}
colors = [ingredient_colors[ing] for ing in df_plot["ingredient"]]

# Create the IoU bar plot with colored bars by ingredient
ax = sns.barplot(x="label", y="iou", data=df_plot, palette=colors)

# Annotate bars with union values
for i, row in enumerate(df_plot.itertuples()):
    ax.text(
        i,
        row.iou + 0.01,  # Position just above the bar
        f"U:{row.union}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
    )

plt.title("IoU with Reference Checkpoint and Union Sizes", fontsize=14)
plt.xlabel("Checkpoint", fontsize=12)
plt.ylabel("Intersection over Union (IoU)", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Add a legend for ingredients
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color=ingredient_colors[ing], lw=4, label=f"Ingredient {ing}")
    for ing in sorted(ingredient_colors.keys())
]
plt.legend(handles=legend_elements, title="Ingredient", loc="best")
#%%

# Get a dataframe of the union of correct problems across all ingredients
# First, create dictionaries to store the correct problems for each ingredient
ingredient_correct_problems = {}

# Group the results by ingredient
for file_path in stage2_files:
    try:
        # Extract metadata
        pattern = r'OLMo-2-1124-13B_stage2-ingredient(\d+)-step(\d+)'
        match = re.search(pattern, file_path.name)
        if not match:
            continue
            
        ingredient, step = match.groups()
        ingredient = int(ingredient)
        
        # Get correct problem IDs
        df = json_to_df(file_path)
        correct_ids = set(df[df['is_correct']]['problem_id'])
        
        # Add to the ingredient dictionary (we'll use the latest step for each ingredient)
        if ingredient not in ingredient_correct_problems:
            ingredient_correct_problems[ingredient] = correct_ids
        else:
            # If this is a later step for the same ingredient, replace the set
            if int(step) > max([int(s) for i, s in ingredient_correct_problems.keys() if i == ingredient]):
                ingredient_correct_problems[ingredient] = correct_ids
    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")

# Get the union and intersection of correct problems across all ingredients
all_ingredients = sorted(ingredient_correct_problems.keys())
union_all = set()
for ingredient in all_ingredients:
    union_all = union_all.union(ingredient_correct_problems[ingredient])

# Create a DataFrame to show which problems are solved by which ingredients
problem_matrix = []
for problem_id in sorted(union_all):
    row = {'problem_id': problem_id}
    
    # For each ingredient, check if this problem is solved
    for ingredient in all_ingredients:
        row[f'ing{ingredient}_correct'] = problem_id in ingredient_correct_problems[ingredient]
    
    # Add a count of how many ingredients solved this problem
    row['num_ingredients_correct'] = sum([1 for ing in all_ingredients if problem_id in ingredient_correct_problems[ing]])
    problem_matrix.append(row)

# Convert to DataFrame
union_df = pd.DataFrame(problem_matrix)

# Sort by number of ingredients that solve the problem (descending)
union_df = union_df.sort_values('num_ingredients_correct', ascending=False)

# Display summary statistics
print(f"\nTotal unique correctly solved problems across all ingredients: {len(union_all)}")
for n in range(1, len(all_ingredients) + 1):
    count = len(union_df[union_df['num_ingredients_correct'] == n])
    print(f"Problems solved by exactly {n} ingredient(s): {count} ({count/len(union_all):.1%})")

# Look at the distribution
plt.figure(figsize=(10, 6))
counts = union_df['num_ingredients_correct'].value_counts().sort_index()
sns.barplot(x=counts.index, y=counts.values, palette='viridis')
plt.title('Distribution of Problems by Number of Ingredients that Solve Them', fontsize=14)
plt.xlabel('Number of Ingredients', fontsize=12)
plt.ylabel('Count of Problems', fontsize=12)
plt.xticks(range(len(all_ingredients) + 1))
plt.tight_layout()


# Display the top of the DataFrame
union_df.head(10)
#%%

all_correct_df = union_df.query("num_ingredients_correct == 3")
all_correct_df.to_csv(results_dir / "union_df_3_problems_correct.csv", index=False)

#%%
l = all_correct_df.problem_id.to_list()
print(l)
