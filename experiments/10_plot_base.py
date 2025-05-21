# Generates plots of model performance metrics across different OLMo model versions.
# Visualizes accuracy trends over training steps, comparing results across different model 
# stages, ingredients, and token counts.

#%%
import json
import os
import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

output_dir = Path("/home/lofty/code_llm/attribution/data/cruxeval_results_base_stage2/cruxeval_results_allenai")
model = "allenai/OLMo-2-1124-13B"
 
def extract_revision_from_filename(filename: str | Path) -> str:
    """Extract the revision information from a filename by removing prefix and suffix.
    
    Args:
        filename (str | Path): The filename to process
    
    Returns:
        str: The extracted revision
    """
    filename_path = Path(filename)
    return filename_path.stem.replace("OLMo-2-1124-13B_", "")

results = []
for result_file in output_dir.glob("*.json"):
    revision = extract_revision_from_filename(result_file)
    with result_file.open("r") as f:
        data = json.load(f)
    
    correct_count = sum(1 for r in data if r["is_correct"])
    total_count = len(data)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    results.append({
        "model": model,
        "revision": revision,
        "correct": correct_count,
        "total": total_count,
        "accuracy": accuracy
    })

results
#%%
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
df = pd.DataFrame(results)
df["accuracy_pct"] = df["accuracy"] * 100

# Extract stage and tokens for sorting
def extract_stage_and_tokens(revision):
    stage_match = re.search(r'stage(\d+)', revision)
    tokens_match = re.search(r'tokens(\d+)B', revision)
    ingredient_match = re.search(r'ingredient(\d+)', revision)
    
    stage = int(stage_match.group(1)) if stage_match else 0
    tokens = int(tokens_match.group(1)) if tokens_match else 0
    ingredient = int(ingredient_match.group(1)) if ingredient_match else 0
    
    return stage, ingredient, tokens

# Add sorting columns
df[["stage", "ingredient", "tokens"]] = pd.DataFrame(
    [extract_stage_and_tokens(r) for r in df["revision"]], 
    index=df.index
)

stage1_max_tokens = df[df["stage"] == 1]["tokens"].max()
df["total_tokens"] = df.apply(
    lambda row: row["tokens"]
    if row["stage"] == 1
    else stage1_max_tokens + row["tokens"],
    axis=1,
)


# Sort hierarchically
df = df.sort_values(["stage", "total_tokens", "ingredient"])

# Save as CSV
csv_path = os.path.join(output_dir, "summary.csv")
df.to_csv(csv_path, index=False)

# Print summary
print("\nSummary of Results:")
summary_cols = ["model", "revision", "correct", "total", "accuracy_pct"]
print(df[summary_cols].to_string(index=False))
print(f"\nSummary saved to {csv_path}")


# Create output directory for plots
plots_dir = output_dir / "plots"
plots_dir.mkdir(exist_ok=True)

# 1. Stage 1 Training Progress
plt.figure(figsize=(10, 6))

# Plot the continuous line first without markers
sns.lineplot(data=df, x='total_tokens', y='accuracy_pct', marker=None)

# Add scatter points colored by stage
colors = ['#1f77b4', '#ff7f0e']  # Blue for stage 1, Orange for stage 2
for stage, color in enumerate(colors, start=1):
    stage_data = df[df['stage'] == stage]
    plt.scatter(stage_data['total_tokens'], stage_data['accuracy_pct'], 
                color=color, label=f'Stage {stage}',  zorder=5)

# Add vertical line at stage transition
plt.axvline(x=stage1_max_tokens, color='red', linestyle='--', alpha=0.7, 
            label='Stage Transition', zorder=4)

plt.title(f'Training Progress Across Stages, model: {model}')
plt.xlabel('Tokens Trained (Billions)')
plt.ylabel('Accuracy (%)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(plots_dir / 'stage1_progress.png', dpi=300, bbox_inches='tight')
#%%
# 2. Stage 2 Training Progress by Ingredient
# plt.figure()
# stage2_df = df[df['stage'] == 2].copy()
# sns.lineplot(
#     data=stage2_df,
#     x="total_tokens",
#     y="accuracy_pct",
#     hue="ingredient",
#     marker="o",
#     markersize=8,
#     markeredgewidth=1,
#     markeredgecolor="black",
#     errorbar=None
# )
# # Get unique x values for ticks
# x_ticks = sorted(stage2_df["total_tokens"].unique())

# plt.xticks(x_ticks, rotation=45)
# plt.grid(True, alpha=0.3)
# plt.title("Stage 2 Training Progress")
# plt.xlabel("Total Tokens Trained (Billions)")
# plt.ylabel("Accuracy (%)")
# plt.legend(title="Ingredient")
# plt.savefig(plots_dir / 'stage2_progress.png', dpi=300, bbox_inches='tight')

# # 3. Box Plot of Accuracy Distribution by Stage
# plt.figure()
# sns.boxplot(data=df, x='stage', y='accuracy_pct')
# plt.title('Accuracy Distribution by Stage')
# plt.xlabel('Training Stage')
# plt.ylabel('Accuracy (%)')
# plt.grid(True, alpha=0.3)
# plt.savefig(plots_dir / 'accuracy_by_stage.png', dpi=300, bbox_inches='tight')
# plt.close()

# # 4. Combined Training Progress with Shared Y-axis
# fig = plt.figure(figsize=(12, 10))
# gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)

# # Stage 1
# ax1 = fig.add_subplot(gs[0])
# sns.lineplot(data=stage1_df, x='total_tokens', y='accuracy_pct', marker='o', ax=ax1)
# ax1.set_title('Stage 1 Training Progress')
# ax1.set_xlabel('Tokens Trained (Billions)')
# ax1.set_ylabel('Accuracy (%)')
# ax1.grid(True, alpha=0.3)

# # Stage 2
# ax2 = fig.add_subplot(gs[1])
# sns.lineplot(data=stage2_df, x='total_tokens', y='accuracy_pct', hue='ingredient', marker='o', ax=ax2)
# ax2.set_title('Stage 2 Training Progress by Ingredient')
# ax2.set_xlabel('Tokens Trained (Billions)')
# ax2.set_ylabel('Accuracy (%)')
# ax2.grid(True, alpha=0.3)
# ax2.legend(title='Ingredient')

# # Adjust layout and save
# plt.suptitle('Complete Training Progress', y=1.02, fontsize=16)
# plt.savefig(plots_dir / 'combined_progress.png', dpi=300, bbox_inches='tight')
# plt.close()

# # 5. Violin Plot for Detailed Distribution Analysis
# plt.figure(figsize=(10, 6))
# sns.violinplot(data=df, x='stage', y='accuracy_pct')
# plt.title('Detailed Accuracy Distribution by Stage')
# plt.xlabel('Training Stage')
# plt.ylabel('Accuracy (%)')
# plt.grid(True, alpha=0.3)
# plt.savefig(plots_dir / 'accuracy_distribution.png', dpi=300, bbox_inches='tight')
# plt.close()

# # 6. Stage 2 Ingredient Comparison (Box Plot)
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=stage2_df, x='ingredient', y='accuracy_pct')
# plt.title('Stage 2: Accuracy Distribution by Ingredient')
# plt.xlabel('Ingredient')
# plt.ylabel('Accuracy (%)')
# plt.grid(True, alpha=0.3)
# plt.savefig(plots_dir / 'stage2_ingredient_comparison.png', dpi=300, bbox_inches='tight')
# plt.close()

# # Print summary statistics
# print("\nSummary Statistics by Stage:")
# print(df.groupby('stage')['accuracy_pct'].agg(['mean', 'std', 'min', 'max']).round(2))

# if df['stage'].eq(2).any():
#     print("\nStage 2 Summary Statistics by Ingredient:")
#     print(df[df['stage'] == 2].groupby('ingredient')['accuracy_pct']
#           .agg(['mean', 'std', 'min', 'max']).round(2))

# print(f"\nPlots have been saved to: {plots_dir}")

    
# %%
