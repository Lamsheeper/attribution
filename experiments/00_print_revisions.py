from huggingface_hub import list_repo_refs
import re


def print_branches(model_name, printing=True):
    out = list_repo_refs(model_name)
    branches = [b.name for b in out.branches]
    if printing:
        print("\n".join(branches))
    return branches


# Extract tokens or steps from branch name
def extract_tokens(branch):
    tokens_match = re.search(r"tokens(\d+)B", branch)
    return int(tokens_match.group(1)) if tokens_match else 0

def extract_steps(branch):
    step_match = re.search(r"step(\d+)", branch)
    return int(step_match.group(1)) if step_match else 0


branches = print_branches("allenai/OLMo-2-1124-7B", printing=False)
stage1_branches = [b for b in branches if "stage1" in b]

# Sort by step number (descending)
stage1_by_tokens = sorted(stage1_branches, key=extract_tokens, reverse=True)
print("\nStage 1 branches sorted by tokens (descending):")
print("\n".join(stage1_by_tokens))