# Evaluates language models on the CruxEval benchmark for code generation capabilities.
# Supports various evaluation modes including standard inference, causal tracing, and
# token-by-token analysis to attribute model performance.

# Run causal tracing on {input, function, output} triplets.
# Use olmo's DPO model. 
# First, I need to find a function in CruxEval that follows the following two properties:
#  1. Olmo2's DPO model gets the correct output given the input. I will use the `guidance` library to build code that checks this. Filter down to all functions for which this is true.
#  2. There is an obvious output token that I can trace on with causal tracing. Given the filtering in 1, further filter here, and then run causal tracing.

# For the causal tracing, here are the steps:
#  1. grab a (prompt, input, output) triplet from the CruxEvalUtil class
#  2. run a forward pass on a CruxEval function that gets the correct output given the input. Save the activations.
#  3. find the set of input tokens and the set of output tokens


# 800 problems total in CruxEval. Examples of input/
#%%
# %load_ext autoreload
# %autoreload
from dataclasses import dataclass
import polars as pl
import os
import json
import torch
import einops
import plotly.express as px
from pathlib import Path
from nnsight import LanguageModel
from bigcode_eval.tasks import get_task
from bigcode_eval.tasks.cruxeval import CruxEval
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import time
from tqdm import tqdm
import argparse
import sys

from attribution.utils import CruxEvalUtil, CausalTracingInput, causal_trace, format_template

from guidance import models, gen, guidance
import sys
import os

print("Python executable:", sys.executable)
print("Python path:", sys.path)
print("HF_HOME:", os.environ.get("HF_HOME"))
print("HF_TOKEN:", os.environ.get("HF_TOKEN"))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluate CruxEval problems with a specified model')
parser.add_argument(
    "--model",
    type=str,
    default="allenai/OLMo-2-1124-7B",
    help="Model to use for evaluation (default: allenai/OLMo-2-1124-7B)",
)
parser.add_argument('--num_problems', type=int, default=800,
                    help='Number of problems to evaluate (default: 800)')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch size for evaluation (default: 8)')
parser.add_argument('--output_dir', type=str, default="../data",
                    help='Directory to save results (default: ../data)')
parser.add_argument('--revision', type=str, default="main",
                    help='Model revision to use (default: main)')

# Check if running as script or in interactive mode
if not sys.argv[0].endswith('ipykernel_launcher.py'):
    args = parser.parse_args()
else:
    # Default values for interactive mode
    args = parser.parse_args([])


# Initialize CruxEvalUtil and model
ce = CruxEvalUtil()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
# Clear CUDA memory if it's in use
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Additional forceful memory cleanup
    with torch.no_grad():
        torch.cuda.synchronize()
    
    # Print available memory for debugging
    free_memory: float = torch.cuda.mem_get_info()[0] / 1024**3  # Convert to GB
    total_memory: float = torch.cuda.mem_get_info()[1] / 1024**3  # Convert to GB
    print(f"GPU memory: {free_memory:.2f}GB free / {total_memory:.2f}GB total")


# Model paths
MODELS = {
    "base": "allenai/OLMo-2-1124-7B",
    "sft": "allenai/OLMo-2-1124-7B-SFT",
    "dpo": "allenai/OLMo-2-1124-7B-DPO",
    "instruct": "allenai/OLMo-2-1124-7B-Instruct",
    "rm": "allenai/OLMo-2-1124-7B-RM",
    "base13b": "allenai/OLMo-2-1124-13B",
    "dpo13b": "allenai/OLMo-2-1124-13B-DPO",
    "instruct13b": "allenai/OLMo-2-1124-13B-Instruct",
    "base32b": "allenai/OLMo-2-0325-32B"
}

# Get model name from command line or use default
model_name = args.model
if model_name in MODELS:
    model_name = MODELS[model_name]

# Extract model short name for file naming
if '/' in model_name:
    model_short_name = model_name.split('/')[-1]
else:
    # If it's already a short name, use it directly
    model_short_name = model_name

# For predefined models, use the key for simplicity
for key, value in MODELS.items():
    if model_name == value:
        model_short_name = key
        break


# Set up cache paths
PERMANENT_CACHE = Path("/share/datasets/huggingface_hub")
HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))

revision = args.revision
print(f"Loading model: {model_name}")
print(f"Loading revision: {revision}")

# Check permanent cache first
model_cache_path = PERMANENT_CACHE / f"models--{model_name.replace('/', '-')}/snapshots/{revision}"
if model_cache_path.exists():
    print(f"Found model in permanent cache at {model_cache_path}")
    # Copy to HF cache if needed
    hf_model_path = HF_CACHE / f"models--{model_name.replace('/', '-')}/snapshots/{revision}"
    if not hf_model_path.exists():
        print("Copying model to HF cache...")
        hf_model_path.parent.mkdir(parents=True, exist_ok=True)
        os.system(f"cp -r {model_cache_path} {hf_model_path}")

# Load tokenizer with optimized settings
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side='left',
    use_fast=True,
    trust_remote_code=True
)

CONFIG = AutoConfig.from_pretrained(model_name)
# Initialize model with optimized settings
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically distribute across available GPUs
    torch_dtype=torch.bfloat16,  # Use bfloat16 for faster computation
    low_cpu_mem_usage=True,  # Optimize CPU memory usage
    # use_cache=True,  # Enable KV cache for faster generation
    revision=revision,
    config=CONFIG,
    trust_remote_code=True
)

# Set model to evaluation mode
model.eval()

# Print GPU memory usage after loading model
if torch.cuda.is_available():
    free_memory: float = torch.cuda.mem_get_info()[0] / 1024**3  # Convert to GB
    total_memory: float = torch.cuda.mem_get_info()[1] / 1024**3  # Convert to GB
    used_memory: float = total_memory - free_memory
    print(f"GPU memory after loading model: {used_memory:.2f}GB used / {total_memory:.2f}GB total ({free_memory:.2f}GB free)")
#%%
prompt, true_in, true_out = ce.output_full(2)
formatted_prompt = prompt.replace('{input}', true_in).replace('{output}', '')
print(formatted_prompt)

#%%
# print("SANITY CHECK", "\n---------------")
# prompt, true_in, true_out = ce.output_full(0)
# print(prompt, '\n----')

# formatted_prompt = prompt.replace('{input}', true_in).replace('{output}', '')
# print(f"Formatted prompt: {formatted_prompt}", '\n----')

# tokens = tokenizer.encode(formatted_prompt)
# print(f"Tokens: {tokens}", '\n----')

# input_id_length = len(tokens)

# inputs = tokenizer(formatted_prompt, return_tensors='pt', padding=True, truncation=True, max_length=2048).to(model.device)
# with torch.no_grad():
#     outputs = model.generate(
#         inputs.input_ids, 
#         attention_mask=inputs.attention_mask,
#         max_new_tokens=100,
#         do_sample=True,
#         temperature=0.2,
#         pad_token_id=tokenizer.eos_token_id,
#         use_cache=True,
#         early_stopping=True
#     )
# print(outputs, '\n----')

# input_length = input_id_length - 1
# print(f"Input length: {input_length}", '\n----')

# generated_tokens = outputs[0]
# print(f"Generated tokens: {generated_tokens}", '\n----')
# print(f"generated text: {tokenizer.decode(generated_tokens, skip_special_tokens=True)}", '\n----')

# clipped = generated_tokens[input_length:]
# print(f"clipped: {clipped}", '\n----')
# print(f"clipped text: {tokenizer.decode(clipped, skip_special_tokens=True)}", '\n----')


#%%
# Function to evaluate a single problem using standard Transformers
def evaluate_problem(model, tokenizer, problem_id: int, max_tokens: int = 100) -> tuple[bool, str, str]:
    """
    Evaluate a single CruxEval problem using standard Transformers.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        problem_id: The problem ID to evaluate
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Tuple of (is_correct, generated_output, true_output)
    """
    try:
        # Get problem
        prompt, true_in, true_out = ce.output_full(problem_id)
        
        # Format prompt with input but leave output empty by replacing placeholders directly
        # This avoids issues with inputs that contain format specifiers
        formatted_prompt = prompt.replace('{input}', true_in).replace('{output}', '')
        
        # Tokenize the prompt
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Generate output with optimized settings
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,  
                temperature=0.2,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,   # Use KV cache for faster generation
                early_stopping=True
            )
        
        # Get only the newly generated tokens
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        generated = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Check if output matches
        is_correct = generated.strip() == true_out.strip()
        
        # Only print detailed info for correct outputs or every 50th problem
        if is_correct or problem_id % 50 == 0:
            print(f"\nProblem ID: {problem_id}")
            print(f"Input: {repr(true_in)}")
            print(f"Generated: {generated}")
            print(f"Expected: {true_out}")
            print(f"Correct: {is_correct}")
        
        return is_correct, generated, true_out
    except Exception as e:
        print(f"Error evaluating problem {problem_id}: {str(e)}")
        return False, str(e), f"Error: {str(e)}"

# Function to evaluate a batch of problems using a single forward pass
def evaluate_batch(model, tokenizer, problem_ids: list[int], max_tokens: int = 100) -> list[tuple[bool, str, str]]:
    """
    Evaluate a batch of CruxEval problems in a single forward pass.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        problem_ids: List of problem IDs to evaluate in this batch
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        List of tuples (is_correct, generated_output, true_output) for each problem
    """
    try:
        # Prepare all prompts
        batch_prompts = []
        true_outputs = []
        problem_inputs = []
        
        for problem_id in problem_ids:
            prompt, true_in, true_out = ce.output_full(problem_id)
            # Format prompt with input but leave output empty
            formatted_prompt = prompt.replace('{input}', true_in).replace('{output}', '')
            batch_prompts.append(formatted_prompt)
            true_outputs.append(true_out)
            problem_inputs.append(true_in)
        
        # Calculate input lengths before tokenization (for token extraction later)
        input_ids_lengths = []
        for prompt in batch_prompts:
            tokens = tokenizer.encode(prompt)
            input_ids_lengths.append(len(tokens))
        
        # Tokenize as a batch
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=2048  # Set an appropriate max length
        ).to(model.device)
        
        # Generate outputs for the entire batch
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.2,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                early_stopping=True
            )
        
        # Process results
        batch_results = []
        for i, (output, true_out, problem_id, true_in) in enumerate(zip(outputs, true_outputs, problem_ids, problem_inputs)):
            # Get the original input length for this example (subtract 1 for correct slicing)
            input_length = input_ids_lengths[i] - 1  # -1 because we want to start extracting from the next token
            
            # Extract only the newly generated tokens for this example
            generated_tokens = output
            full_generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Extract only the part after "output:" if it exists
            if "output:" in full_generated_text.lower():
                # Find the position of "output:" (case insensitive)
                output_pos = full_generated_text.lower().find("output:")
                # Extract everything after "output:" (including the 8 characters for "output:")
                generated = full_generated_text[output_pos + 7:].strip()
            else:
                # If "output:" is not found, use the full generated text
                generated = full_generated_text.strip()
            
            # Check if output matches
            is_correct = generated.strip() == true_out.strip()
            
            # Print detailed info for correct outputs or sample problems
            if is_correct or problem_id % 50 == 0:
                print(f"\nProblem ID: {problem_id}")
                print(f"Input: {repr(true_in)}")
                print(f"Full Generated: {full_generated_text}", '\n')
                print(f"Extracted Output: {generated}")
                print(f"Ground truth: {true_out}")
                print(f"Correct: {is_correct}")
            
            batch_results.append((is_correct, full_generated_text, generated, true_out))
        
        return batch_results
    
    except Exception as e:
        print(f"Error evaluating batch {problem_ids}: {str(e)}")
        # Return failed results for each problem in the batch
        return [(False, str(e), f"Error: {str(e)}") for _ in problem_ids]

#%%
# Evaluate all problems with batch processing for better performance
num_problems = args.num_problems
batch_size = args.batch_size  # Process multiple problems at once
results = []
checkpoint_file = os.path.join(args.output_dir, f"cruxeval_checkpoint_{model_name}_{revision}.json")

# Check if checkpoint exists to resume from previous run
try:
    with open(checkpoint_file, "r") as f:
        checkpoint_data = json.load(f)
        results = checkpoint_data["results"]
        start_idx = len(results)
        print(f"Resuming from checkpoint with {start_idx} problems already processed")
except FileNotFoundError:
    start_idx = 0
    print("Starting fresh evaluation")

# Adjust batch size based on model and available GPU memory
# Use a smaller batch size initially to be safer
adjusted_batch_size = min(batch_size, 64)  # Start with max 64 examples per batch 
print(f"Evaluating {num_problems} problems in batches of {adjusted_batch_size}...")
start_time = time.time()

# Process problems in true batches
for batch_start in tqdm(range(start_idx, num_problems, adjusted_batch_size)):
    batch_end = min(batch_start + adjusted_batch_size, num_problems)
    problem_ids = list(range(batch_start, batch_end))
    
    # Process the entire batch at once
    batch_results = evaluate_batch(model, tokenizer, problem_ids, max_tokens=100)
    
    # Format results for storage
    formatted_results = []
    for i, (is_correct, full_generated_text, generated, true_out) in enumerate(batch_results):
        formatted_results.append({
            "problem_id": problem_ids[i],
            "is_correct": is_correct,
            "full_generated_text": full_generated_text,
            "generated": generated,
            "true_output": true_out
        })
    
    # Add batch results to overall results
    results.extend(formatted_results)
    
    # Save checkpoint after each batch
    # make checkpoint file directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, "w") as f:
        json.dump({"results": results}, f)
    
    # Print progress after each batch
    processed_count = len(results)
    correct_count = sum(r["is_correct"] for r in results)
    print(f"Progress: {processed_count}/{num_problems}, Correct: {correct_count}/{processed_count} ({correct_count/processed_count*100:.2f}%)\n")
    
    # Calculate and print time estimates
    elapsed_time = time.time() - start_time
    problems_per_second = processed_count / elapsed_time
    remaining_problems = num_problems - processed_count
    estimated_remaining_time = remaining_problems / problems_per_second
    
    print(f"Speed: {problems_per_second:.2f} problems/second")
    print(f"Estimated time remaining: {estimated_remaining_time/60:.1f} minutes ({estimated_remaining_time/3600:.1f} hours)\n")

end_time = time.time()
total_time = end_time - start_time
print(f"\n\nEvaluation completed in {total_time:.2f} seconds ({total_time/60:.1f} minutes)")

# Calculate final accuracy
correct_count = sum(r["is_correct"] for r in results)
accuracy = correct_count / num_problems
print(f"Final accuracy: {correct_count}/{num_problems} ({accuracy*100:.2f}%)")

# Save final results
results_file = os.path.join(args.output_dir, f"cruxeval_results_{model_name}_{revision}.json")
# make results file directory if it doesn't exist
os.makedirs(os.path.dirname(results_file), exist_ok=True)
with open(results_file, "w") as f:
    json.dump(results, f)
print(f"Results saved to {results_file}")

# Find problems where the model gets the correct output
correct_problems = [r["problem_id"] for r in results if r["is_correct"]]
print(f"Found {len(correct_problems)} problems with correct outputs")

# These problems can be used for causal tracing
print("First 10 problems with correct outputs:")
for i in correct_problems[:10]:
    print(f"Problem {i}")

# Clean up checkpoint file after successful completion
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)
    print(f"Removed checkpoint file: {checkpoint_file}")

# If running as a script, exit here
if not sys.argv[0].endswith('ipykernel_launcher.py'):
    sys.exit(0)
