# Evaluates language model performance across multiple code generation benchmarks.
# Implements a comprehensive evaluation framework that runs models through various
# coding tasks, collecting and analyzing performance metrics like pass@k scores.

#%%
import json
import os
import warnings

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

import bigcode_eval
import argparse
# ------------------------------------------------------------------------------------
# 1) Parse your command-line arguments (this snippet assumes you have them defined).
# ------------------------------------------------------------------------------------
from bigcode_eval.arguments import EvalArguments  # Example import (adjust if needed)
from bigcode_eval.evaluator import Evaluator
from pathlib import Path

# After imports, modify the argument parsing section
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model name or path")
parser.add_argument("--generate_new", action="store_true", help="Force new generation even if files exist")
args = parser.parse_args()

## DEBUGGING
# class Args:
#     def __init__(self):
#         self.model = "allenai/OLMo-2-1124-7B-DPO"  # Default model, change as needed
#         self.generate_new = False  # Default value, change as needed

# args = Args()

# Update the variables that were hardcoded
model_name = args.model
generate_new_generations = args.generate_new
base_dir = Path("/home/lofty/code_llm/attribution/results") / model_name 

# Create the directory if it doesn't exist
base_dir.mkdir(parents=True, exist_ok=True)

# Create base eval arguments
eval_args = EvalArguments(
    prefix="",
    do_sample=True,
    temperature=0.2,
    top_k=0,
    top_p=0.95,
    n_samples=1,
    eos="<|endoftext|>",
    seed=0,
)

# Additional arguments from command line parser
additional_args = {
    "model": model_name,  # Use command line arg
    "modeltype": "causal",
    "peft_model": None,
    "revision": None,
    "use_auth_token": False,
    "trust_remote_code": False,
    "tasks": "cruxeval",
    "instruction_tokens": None,
    "batch_size": 8,
    "max_length_generation": 512,
    "precision": "bf16",
    "load_in_8bit": True,
    "load_in_4bit": False,
    "left_padding": False,
    "limit": None, 
    "limit_start": 0,
    "save_every_k_tasks": -1,
    "postprocess": True,
    "allow_code_execution": True,
    "generation_only": False,
    "load_generations_path": None,
    "load_data_path": None,
    "metric_output_path": base_dir / "cruxeval_evaluation_results.json",  # Use command line base_dir
    "save_generations": True,
    "load_generations_intermediate_paths": None,
    "save_generations_path": base_dir / "cruxeval_generations.json",  # Use command line base_dir
    "save_references": True,
    "save_references_path": base_dir / "cruxeval_references.json",  # Use command line base_dir
    "prompt": "prompt",
    "max_memory_per_gpu": None,
    "check_references": False,
}

# Combine both into a namespace
combined_args = {**vars(eval_args), **additional_args}
args_namespace = argparse.Namespace(**combined_args)
#%%
accelerator = Accelerator()
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
evaluator = Evaluator(accelerator=accelerator, model=model, tokenizer=tokenizer, args=args_namespace)
# Example: you might fetch the model name from args as well
# e.g. args.model_name_or_path = "bigcode/santacoder"

model.eval()

# Prepare for distributed or GPU environment (if any)
model = accelerator.prepare(model)

# Evaluate the "humaneval" task. You can pick other tasks if desired.
# The evaluate() method will call generate_text(...) internally
# and then call the appropriate Task.process_results(...)
task_name = "cruxeval"
# results = evaluator.evaluate(task_name)
#%%
from bigcode_eval.tasks import get_task
from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval
from pathlib import Path
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
# results

task = get_task(task_name)
# generations_path = args_namespace.save_generations_path.replace('.json', '_humanevalplus.json')
generations_path = Path(args_namespace.save_generations_path).parent / 'generations.json'
results_path = Path(args_namespace.save_generations_path).parent / 'results.json'

cruxeval_generations_path = args_namespace.save_generations_path
cruxeval_results_path = args_namespace.metric_output_path.parent / (args_namespace.metric_output_path.stem + "_detailed.json")

'''# messy code, but the names are different beacuse I messed up the saving and I don't wanna rerun
# can delete later
if not Path(generations_path).exists():
    if (generations_path.parent / 'humanevalplus_generations.json').exists():
        print('using humanevalplus_generations.json')
        generations_path = generations_path.parent / 'humanevalplus_generations.json'
if not Path(results_path).exists():
    if (results_path.parent / 'evaluation_results.json').exists():
        print('using evaluation_results.json')
        results_path = results_path.parent / 'evaluation_results.json'
'''

# Force new generations for CruxEval
if not Path(cruxeval_generations_path).exists() or generate_new_generations:
    print('Creating new CruxEval generations and references...')
    generations, references = evaluator.generate_text(task_name)
    pass_at_k, results = compute_code_eval(
        references=references, 
        predictions=generations, 
        k=task.k, 
        num_workers=task.num_workers, 
        timeout=task.timeout
    )
    
    # Save immediately after generation
    print(f"Saving generations to {cruxeval_generations_path}")
    with open(cruxeval_generations_path, 'w') as f:
        json.dump(generations, f)
    
    print(f"Saving results to {cruxeval_results_path}")
    with open(cruxeval_results_path, 'w') as f:
        json.dump(results, f)
else:
    print(f'Loading existing generations from {cruxeval_generations_path}')
    with open(cruxeval_generations_path, 'r') as f:
        generations = json.load(f)
    if Path(cruxeval_results_path).exists():
        with open(cruxeval_results_path, 'r') as f:
            results = json.load(f)
    else:
        # Compute results if they don't exist
        dataset = task.get_dataset()
        references = [task.get_reference(d) for d in dataset]
        pass_at_k, results = compute_code_eval(references=references, predictions=generations, k=task.k, num_workers = task.num_workers, timeout=task.timeout)

#%%
from bigcode_eval.tasks.custom_metrics.code_eval import estimate_pass_at_k
import numpy as np
generate_new_generations = False
if Path(cruxeval_generations_path).exists() and not generate_new_generations:
  if Path(cruxeval_results_path).exists():
    with open(cruxeval_results_path, 'r') as f:
        results = json.load(f)
    with open(cruxeval_generations_path, 'r') as f:
        generations = json.load(f)
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = 10
    if not isinstance(ks, (list, tuple)):
        ks = [ks]
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}
    print(f"pass_at_k: {pass_at_k}")
    
  else:
    with open(cruxeval_generations_path, 'r') as f:
        generations = json.load(f)
        dataset = task.get_dataset()
        n_tasks = min(args_namespace.limit, len(dataset) - args_namespace.limit_start) if args_namespace.limit else len(dataset)
        if not args_namespace.limit:
            n_tasks -= args_namespace.limit_start
        references = [task.get_reference(dataset[i]) for i in range(args_namespace.limit_start, args_namespace.limit_start+n_tasks)]
        pass_at_k, results = compute_code_eval(references=references, predictions=generations, k=task.k, num_workers = task.num_workers, timeout=task.timeout)
else:
    print('creating generations and references')
    generations, references = evaluator.generate_text(task_name)
    pass_at_k, results = compute_code_eval(references=references, predictions=generations, k=task.k, num_workers = task.num_workers, timeout=task.timeout)


# switch to strs for results keys
#%%
from pprint import pprint

def print_generations(generations, results, problem_index=None, prompt_only=False):
    # Convert results keys to strings for consistent access
    results = {str(k): v for k, v in results.items()}
    
    # Get dataset for prompts
    dataset = task.get_dataset()
    
    # Determine which problems to process
    problems = [generations[problem_index]] if problem_index is not None else generations
    problem_indices = [str(problem_index)] if problem_index is not None else [str(i) for i in range(len(generations))]
    
    # Process each problem
    for prob_idx, problem in zip(problem_indices, problems):
        if prompt_only:
            # Get prompt directly from dataset
            prompt = dataset['prompt'][int(prob_idx)]
            print(f'Problem {prob_idx} Prompt:')
            print(prompt)
            print('\n' + '='*100 + '\n')
            continue
            
        # Print full generation details
        for i, result in enumerate(problem):
            print(f'# Result {i+1}:')
            test_result = results[prob_idx][i][1]
            print(f'# Test result: {test_result["result"]}')
            print(f'# Passed: {test_result["passed"]}')
            
            # Print code with original formatting
            print(result)
            print('\n' + '='*100 + '\n')

# print_results(results)
# print_generations(generations, results)
#%%
def get_pass_rates(results):
    """Calculate pass rates for each problem in results"""
    pass_rates = []
    # Convert string keys to integers
    results = {int(k): v for k, v in results.items()}
    problem_ids = sorted(results.keys())
    for problem_id in problem_ids:
        problem_results = results[problem_id]
        # Count number of passed tests
        passed = sum(1 for _, result_dict in problem_results if result_dict['passed'])
        total = len(problem_results)
        pass_rate = (passed / total) * 100
        pass_rates.append(pass_rate)
    return problem_ids, pass_rates

def plot_pass_rates(results, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np
    
    problem_ids, pass_rates = get_pass_rates(results)
    
    # Create the plot
    plt.figure(figsize=(35, 5))
    # Use range(len(pass_rates)) for x-axis positioning
    plt.bar(range(len(pass_rates)), pass_rates)
    
    # Set integer ticks on x-axis with problem IDs as labels
    plt.xticks(range(len(pass_rates)), problem_ids)
    plt.margins(x=0.01)
    
    # Customize the plot
    plt.xlabel('Problem ID')
    plt.ylabel('Pass Rate (%)')
    plt.title(f'Pass Rate by Problem (model = {args_namespace.model})')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for i, rate in enumerate(pass_rates):
        plt.text(i, rate + 1, f'{rate:.1f}%', 
                ha='center', va='bottom')
    
    # Set y-axis limits to go slightly above 100%
    plt.ylim(0, max(105, max(pass_rates) + 5))
    
    # Adjust layout to remove empty space
    plt.tight_layout()
    
    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
    
    return plt

def save_pass_rates(problem_ids, pass_rates, save_path):
    data = {
        'problem_ids': problem_ids,
        'pass_rates': pass_rates
    }
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_pass_rates(load_path):
    with open(load_path) as f:
        data = json.load(f)
    return data['problem_ids'], data['pass_rates']

problem_ids, pass_rates = get_pass_rates(results)
# plot_pass_rates(results)


def plot_pass_rates_heatmap(results, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np
    
    problem_ids, pass_rates = get_pass_rates(results)
    
    # Reshape into a roughly square grid
    grid_size = int(np.ceil(np.sqrt(len(pass_rates))))
    grid = np.full((grid_size, grid_size), np.nan)
    for i, rate in enumerate(pass_rates):
        if i < len(grid.flat):
            grid.flat[i] = rate
    
    plt.figure(figsize=(12, 10))
    im = plt.imshow(grid, cmap='RdYlGn')
    plt.colorbar(im, label='Pass Rate (%)')
    
    # Add problem IDs as text in cells
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < len(problem_ids):
                plt.text(j, i, f'{problem_ids[idx]}\n{grid[i,j]:.1f}%', 
                        ha='center', va='center', fontsize=8)
    
    plt.title(f'Pass Rates Heatmap (model = {args_namespace.model})')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    return plt

# plot_pass_rates_heatmap(results)

def plot_pass_rates_line(results, window=10, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np
    
    problem_ids, pass_rates = get_pass_rates(results)
    
    plt.figure(figsize=(24, 10))
    
    # Plot individual points with reduced opacity
    plt.scatter(problem_ids, pass_rates, label='HumanEval+ Problems')
    
    # Add text labels above each point
    for x, y in zip(problem_ids, pass_rates):
        plt.text(x, y + 1, f'{y:.0f}%', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Problem ID')
    plt.ylabel('Pass Rate (%)')
    plt.title(f'Pass Rates Over Problems (model = {args_namespace.model})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set major ticks every 5th problem
    major_ticks = problem_ids[::5]
    # Set minor ticks for every problem
    minor_ticks = problem_ids[::1]
    
    ax = plt.gca()
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(np.arange(0, 101, 5))
    
    # Format the ticks
    plt.xticks(rotation=45)
    
    # Add grid for minor ticks
    # ax.grid(True, which='minor', alpha=0.5)
    ax.grid(True, which='major', alpha=1)
    
    # Adjust y-axis limits to accommodate labels
    plt.ylim(0, max(pass_rates) + 8)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    return plt

# plot_pass_rates_line(results)
#%%

# Save generations
cruxeval_generations_path = args_namespace.save_generations_path
with open(cruxeval_generations_path, 'w') as f:
    json.dump(generations, f)

# Save results
cruxeval_results_path = args_namespace.metric_output_path.parent / (args_namespace.metric_output_path.stem + "_detailed.json")
with open(cruxeval_results_path, 'w') as f:
    json.dump(results, f)


# Save plot with task-specific names
task_prefix = "cruxeval"  # Use this to differentiate from HumanEval+
plot_base = args_namespace.save_generations_path.parent / f'{task_prefix}_pass_rates'
plot_pass_rates(results, save_path=str(plot_base) + '_bar.jpg')
plot_pass_rates_heatmap(results, save_path=str(plot_base) + '_heatmap.jpg')
plot_pass_rates_line(results, save_path=str(plot_base) + '_line.jpg')

# Also update the pass rates file name
pass_rates_path = args_namespace.save_generations_path.parent / f'{task_prefix}_pass_rates.json'
save_pass_rates(problem_ids, pass_rates, save_path=pass_rates_path)

# And the config file
config_path = args_namespace.save_generations_path.parent / f'{task_prefix}_config.json'
config = {}
config[task_name] = pass_at_k
config['config'] = vars(args_namespace).copy()
# turn any Path objects in 'config' into strings
for key, value in config['config'].items():
    if isinstance(value, Path):
        config['config'][key] = str(value)
with open(config_path, 'w') as f:
    json.dump(config, f)

print(f"Saved all outputs to {args_namespace.save_generations_path.parent}:")
print(f"- Generations: {cruxeval_generations_path}")
print(f"- Results: {cruxeval_results_path}")
print(f"- Plot: {plot_base}_(bar, heatmap, line).jpg")
print(f"- Pass Rates: {pass_rates_path}")
print(f"- Config: {config_path}")
#%%
