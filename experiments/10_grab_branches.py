# Retrieves metadata about different branches and versions of OLMo models from Hugging Face.
# Creates a structured DataFrame of model versions, extracting information about steps,
# tokens, and training stages to facilitate analysis and evaluation.

# IMPORTANT: Always run this script using `uv run` instead of `python3`
# Example: uv run 10_instruct_benchmarks_eval.py
#%%
from transformers import AutoModelForCausalLM, AutoTokenizer

from OLMo.hf_olmo.configuration_olmo import OLMoConfig
MODEL = "allenai/OLMo-2-1124-13B"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)
#%%
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig



# Create config object
MODEL = "allenai/OLMo-2-1124-13B"
REVISION = "stage1-step596057-tokens5001B"
CONFIG = AutoConfig.from_pretrained(MODEL)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, config=CONFIG)

#%%
from datasets import load_dataset

ds = load_dataset("cruxeval-org/cruxeval")
#%%
from attribution.utils import CruxEvalUtil
ce = CruxEvalUtil()
#%%

#%%
import re
from huggingface_hub import list_repo_refs


def print_branches(model_name, printing=True):
    out = list_repo_refs(model_name)
    branches = [b.name for b in out.branches]
    if printing:
        print("\n".join(branches))
    return branches


branches = print_branches("allenai/OLMo-2-1124-13B", printing=False)

# Get all stage2 branches and the five largest token counts from stage1
stage2_branches = [b for b in branches if "stage2" in b]
stage1_branches = [b for b in branches if "stage1" in b]

# Sort stage1 branches by token count (descending) and take top 5
def extract_tokens(branch):
    tokens_match = re.search(r"tokens(\d+)B", branch)
    return int(tokens_match.group(1)) if tokens_match else 0

top_stage1_branches = sorted(stage1_branches, key=extract_tokens, reverse=True)[:5]

# Print all stage2 branches and top 5 stage1 branches
print("\n".join(stage2_branches + top_stage1_branches))
#%%
from huggingface_hub import list_repo_refs


def print_branches(model_name, printing=True):
    out = list_repo_refs(model_name)
    branches = [b.name for b in out.branches]
    if printing:
        print("\n".join(branches))
    return branches


branches = print_branches("allenai/OLMo-2-1124-13B", printing=False)
print("\n".join([b for b in branches if "stage2" in b]))
#%%
LOAD_FROM = "/share/u/yu.stev/.cache/huggingface/hub/models--allenai--OLMo-2-1124-13B/snapshots/"

#%%
#%%
from urllib.parse import urlparse
from olmo.train import Trainer

load_path = "https://olmo-checkpoints.org/ai2-llm/peteish13/stage2/olmo-13b-1124_stage2_ingredient1/step1000-unsharded/"
parsed = urlparse(str(dir))

trainer = Trainer(re

trainer.restore_checkpoint(load_path)

# trainer.restore_checkpoint(
# cfg.load_path,
# load_optimizer_state=not cfg.reset_optimizer_state,
# load_trainer_state=not cfg.reset_trainer_state,
# sharded_checkpointer=cfg.load_path_sharded_checkpointer,
# )

#%%
def extract_stage_and_tokens(revision):
    stage_match = re.search(r"stage(\d+)", revision)
    tokens_match = re.search(r"tokens(\d+)B", revision)
    step_match = re.search(r"step(\d+)", revision)

    step = int(step_match.group(1)) if step_match else 0
    stage = int(stage_match.group(1)) if stage_match else 0
    tokens = int(tokens_match.group(1)) if tokens_match else 0

    return revision, stage, step, tokens

model = models_13b[0]
branches = print_branches(model, printing=False)

# Extract data for each branch
data = [extract_stage_and_tokens(branch) for branch in branches]

# Create DataFrame
df = pd.DataFrame(data, columns=['revision', 'stage', 'step', 'tokens'])
#%%
df = df.sort_values(by=["stage", "step"], ascending=True)
df
#%%
# Get index where stage transitions from 1 to 2
df.reset_index(drop=True, inplace=True)
transition_idx = df[df['stage'] == 2].index[0]

# Get 10 rows before and 10 rows after the transition (20 total)
window_start = int(max(0, transition_idx - 10))
window_end = int(min(len(df), transition_idx + 10))

# Display the window around the transition
print("\nWindow of 20 branches around stage 1 to 2 transition:")
window_df = df.loc[window_start:window_end]
print(window_df)

# Print in a format suitable for shell script array
print("\nBranches for shell script:")
branches_for_script = window_df['revision'].tolist()
print("REVISIONS=(")
for branch in branches_for_script:
    print(f'    "{branch}"')
print(")")

#%%
# for m in models_13b:
#     b = print_branches(m, printing=False)
#     print(f"{m} branches total: {len(b)}")
#     print(f"  stage1: {len([b for b in b if b.startswith('stage1')])}")
#     print(f"  stage2: {len([b for b in b if b.startswith('stage2')])}", "\n")

# %
# from huggingface_hub import list_repo_refs
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# revision = "stage2-ingredient3-step2000-tokens9B"

# tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B", revision=revision)
# model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-7B", revision=revision)

#%%
