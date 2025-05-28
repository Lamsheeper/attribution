#%% [markdown]

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
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, torch.nn.functional as F
import time
from tqdm import tqdm
import argparse
import sys
from typing import List
#from accelerate import Accelerator

from attribution.utils import CruxEvalUtil, CausalTracingInput, causal_trace, format_template

from guidance import models, gen, guidance

# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluate CruxEval problems with a specified model')
parser.add_argument('--model', type=str, default="allenai/OLMo-2-1124-7B-DPO", 
                    help='Model to use for evaluation (default: allenai/OLMo-2-1124-7B-DPO)')
parser.add_argument('--num_problems', type=int, default=800,
                    help='Number of problems to evaluate (default: 800)')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch size for evaluation (default: 8)')
parser.add_argument('--output_dir', type=str, default="/share/u/yu.stev/attribution/data/cruxeval_results/k_sample",
                    help='Directory to save results (default: /share/u/yu.stev/attribution/data/cruxeval_results/k_sample)')
parser.add_argument('--revision', type=str, default="main",)
parser.add_argument('--persona', type=bool, default=False,
                    help='Test personas, default is False')
parser.add_argument('--multi_trial', type=int, default=0)
parser.add_argument('--num_samples', type=int, default=1,
                    help='Number of samples to generate per problem (default: 1)')
parser.add_argument('--temperature', type=float, default=0.7,
                    help='Temperature for sampling when num_samples > 1 (default: 0.7)')
parser.add_argument('--top_p', type=float, default=0.95,
                    help='Top-p (nucleus sampling) parameter when num_samples > 1 (default: 0.95)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')

# Check if running as script or in interactive mode
if not sys.argv[0].endswith('ipykernel_launcher.py'):
    args = parser.parse_args()
else:
    # Default values for interactive mode
    args = parser.parse_args([])

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
    "base": "allenai/OLMo2-7B-1124",
    "sft": "allenai/OLMo-2-1124-7B-SFT",
    "dpo": "allenai/OLMo-2-1124-7B-DPO",
    "instruct": "allenai/OLMo-2-1124-7B-Instruct",
    "rm": "allenai/OLMo-2-1124-7B-RM",
    "base13b": "allenai/OLMo-2-1124-13B",
    "dpo13b": "allenai/OLMo-2-1124-13B-DPO",
    "instruct13b": "allenai/OLMo-2-1124-13B-Instruct",
    "base32b": "allenai/OLMo-2-0325-32B",
    "instruct32b": "allenai/OLMo-2-0325-32B-Instruct",
}

PERSONAS = {
    #"software_engineer": "You are a seasoned software engineer who values clean architecture, logical precision, and efficient problem-solving.",
    #"high_school_teacher": "You are a high school teacher passionate about helping students develop critical thinking through clear explanations and thoughtful guidance.",
    #"journalist": "You are a journalist writing for a broad audience, focused on clarity, relevance, and the ability to communicate complex ideas simply.",
    #"fiction_writer": "You are a fiction writer who cares deeply about narrative logic, character motivation, and emotional resonance.",
    #"philosophy_student": "You are a university philosophy student trained to analyze arguments, identify logical flaws, and construct coherent counterpoints.",
    #"competitive_programmer": "You are a competitive programmer who excels at solving algorithmic challenges under time pressure and writing concise, optimized code.",
    #"backend_developer": "You are a backend developer who designs scalable APIs, writes clean code, and debugs complex systems with precision.",
    #"systems_programmer": "You are a systems programmer who writes low-level, performance-critical code and understands memory management and concurrency.",
    #"open_source_maintainer": "You are an open source maintainer who values readability, documentation, and clean pull requests from contributors.",
    #"ai_engineer": "You are an AI engineer who prototypes and deploys machine learning models, often debugging training loops and optimizing inference code."
    "codeforces_user": "You are a Codeforces user who regularly competes in programming contests and thinks in terms of time complexity, edge cases, and tight loops.",
    "github_contributor": "You are an active GitHub contributor who cares about clean pull requests, code readability, and writing maintainable, well-documented code.",
    "stackoverflow_helper": "You are a Stack Overflow contributor known for writing concise, high-quality answers to coding questions, often including code snippets and edge case explanations.",
    "": ""
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

# Initialize CruxEvalUtil and model
ce = CruxEvalUtil()

#%%
print(f"Loading model: {model_name}")
revision = args.revision

# Load tokenizer with optimized settings
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True,  # Use fast tokenizer
    padding_side="left"  # Pad on the left for more efficient generation
)

# Ensure pad token is set properly
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set pad_token to eos_token because it was None")

# Check padding settings
print(f"Tokenizer padding settings:")
print(f"  pad_token: {tokenizer.pad_token}")
print(f"  padding_side: {tokenizer.padding_side}")
print(f"  model_max_length: {tokenizer.model_max_length}")
print(f"  is_fast: {tokenizer.is_fast}")

# Use device_map instead of accelerator
print("Loading model with device_map to utilize multiple GPUs...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # This distributes the model across available GPUs
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    revision=revision,
    temperature=0
    #load_in_8bit=True
)

# Set model to evaluation mode
model.eval()

# Set PyTorch to deterministic mode
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Ensure model generation config is set for deterministic outputs
'''if hasattr(model, 'generation_config'):
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.num_beams = 1
    model.generation_config.top_p = 1.0
    print("Set model generation config to deterministic settings")'''

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
print("SANITY CHECK", "\n---------------")
prompt, true_in, true_out = ce.output_full(0)
print(prompt, '\n----')

formatted_prompt = prompt.replace('{input}', true_in).replace('{output}', '')
print(f"Formatted prompt: {formatted_prompt}", '\n----')

tokens = tokenizer.encode(formatted_prompt)
print(f"Tokens: {tokens}", '\n----')

input_id_length = len(tokens)

inputs = tokenizer(formatted_prompt, return_tensors='pt', padding=True, truncation=True, max_length=2048).to(model.device)
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids, 
        attention_mask=inputs.attention_mask,
        max_new_tokens=100,
        do_sample=False,
        num_beams=1,
        temperature=0.2,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        early_stopping=True
    )
print(outputs, '\n----')

input_length = input_id_length - 1
print(f"Input length: {input_length}", '\n----')

generated_tokens = outputs[0]
print(f"Generated tokens: {generated_tokens}", '\n----')
print(f"generated text: {tokenizer.decode(generated_tokens, skip_special_tokens=True)}", '\n----')

clipped = generated_tokens[input_length:]
print(f"clipped: {clipped}", '\n----')
print(f"clipped text: {tokenizer.decode(clipped, skip_special_tokens=True)}", '\n----')

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
        
        # Generate output with fully deterministic settings
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
                do_sample=False,         # Deterministic generation
                num_beams=1,             # No beam search
                temperature=None,        # No temperature for deterministic generation
                top_p=1.0,               # No nucleus sampling
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                early_stopping=False,    # No early stopping with num_beams=1
                output_scores=True,      # Get logprobs
                return_dict_in_generate=True  # Return full generation info
            )
        
        # Process results
        batch_results = []
        for i, (output, true_out, problem_id, true_in) in enumerate(zip(outputs.sequences, true_outputs, problem_ids, problem_inputs)):
            # Get the original input length for this example (subtract 1 for correct slicing)
            input_length = input_ids_lengths[i]
            
            # Extract only the newly generated tokens for this example
            generated_tokens = output[input_length:]
            full_generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Get logprobs for generated tokens
            token_logprobs = []
            if hasattr(outputs, 'scores'):
                # Get logprobs for each generated token
                for j in range(len(generated_tokens)):
                    if j < len(outputs.scores):
                        # Get logprobs for this token
                        logprob = outputs.scores[j][i][generated_tokens[j]].item()
                        # Only include non-padding tokens (filter out -infinity)
                        if logprob != float('-inf'):
                            token_logprobs.append(logprob)
            
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
                print(f"Full Generated: {full_generated_text}")
                print(f"Extracted Output: {generated}")
                print(f"Expected: {true_out}")
                print(f"Correct: {is_correct}")
                if token_logprobs:
                    print(f"Token logprobs: {token_logprobs}")
                    print(f"Mean logprob: {sum(token_logprobs)/len(token_logprobs):.4f}")
            
            batch_results.append((is_correct, full_generated_text, generated, true_out, token_logprobs))
        
        return batch_results
    except Exception as e:
        print(f"Error evaluating problem {problem_id}: {str(e)}")
        return False, str(e), f"Error: {str(e)}"

# Function to evaluate a batch of problems using a single forward pass
def evaluate_batch(model, tokenizer, problem_ids: list[int], max_tokens: int = 100, persona: str = "") -> list[tuple[bool, str, str, str, list]]:
    """
    Evaluate a batch of CruxEval problems in a single forward pass.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        problem_ids: List of problem IDs to evaluate in this batch
        max_tokens: Maximum number of tokens to generate
        persona: Persona to use for evaluation
    Returns:
        List of tuples (is_correct, generated_output, true_output, token_logprobs) for each problem
    """
    try:
        # Prepare all prompts
        batch_prompts = []
        true_outputs = []
        problem_inputs = []
        
        for problem_id in problem_ids:
            prompt, true_in, true_out = ce.output_full(problem_id)
            if persona != "":
                prompt = f"{persona}\n\n{prompt}"
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
            max_length=2048,  # Set an appropriate max length
            padding_side=tokenizer.padding_side  # Explicitly use the tokenizer's padding_side setting
        ).to(model.device)
        
        # Generate outputs for the entire batch with fully deterministic settings
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
                do_sample=False,         # Deterministic generation
                num_beams=1,             # No beam search
                temperature=None,        # No temperature for deterministic generation
                top_p=1.0,               # No nucleus sampling
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                early_stopping=False,    # No early stopping with num_beams=1
                output_scores=True,      # Get the logits
                return_dict_in_generate=True  # Return full generation info including scores
            )
        
        # Extract scores from generation
        sequences = outputs.sequences
        
        # Compute transition scores using the built-in HuggingFace method
        try:
            transition_scores = model.compute_transition_scores(
                sequences=outputs.sequences,
                scores=outputs.scores,
                normalize_logits=True  # Get log probabilities
            )
        except (AttributeError, RuntimeError) as e:
            # Some models don't support compute_transition_scores
            print(f"Warning: compute_transition_scores failed: {e}")
            # Fall back to our custom method
            transition_scores = None
        
        # Process results
        batch_results = []
        for i, (output, true_out, problem_id, true_in) in enumerate(zip(sequences, true_outputs, problem_ids, problem_inputs)):
            # Get the original input length for this example (subtract 1 for correct slicing)
            input_length = input_ids_lengths[i] - 1
            
            # Extract only the newly generated tokens for this example
            generated_tokens = output[input_length:]
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
            
            # Extract logprobs for this sample using transition scores if available
            token_logprobs = []
            
            if transition_scores is not None:
                # Get the transition scores for this sample (sequence)
                sample_scores = transition_scores[i]
                
                # Only consider scores for tokens after the input
                # The +1 offset is because transition_scores has one less element than generated tokens
                # (the first token has no transition)
                relevant_scores = sample_scores[max(0, input_length - output.shape[0] + 1):]
                
                # Filter out padding tokens and special tokens at the end
                token_list = output[input_length:].tolist()
                filtered_scores = []
                
                # Make sure token_list and relevant_scores align in length
                min_len = min(len(token_list), len(relevant_scores))
                for idx in range(min_len):
                    score = relevant_scores[idx]
                    token_id = token_list[idx]
                    
                    # Skip padding tokens (usually eos_token_id) and tokens at the end that might cause -inf
                    if token_id == tokenizer.pad_token_id or token_id == tokenizer.eos_token_id:
                        continue
                        
                    # Once we encounter a really low score (likely padding or end), stop
                    if score.item() < -15 and idx > min_len // 2:  # Only apply in the latter half
                        break
                        
                    # Only include valid scores
                    if not (torch.isinf(score) or torch.isnan(score)):
                        filtered_scores.append(score.item())
                    else:
                        filtered_scores.append(-20.0)  # Use fallback
                
                token_logprobs = filtered_scores
            else:
                # Extract token logprobs more carefully using our fallback method
                try:
                    # Check that we have scores
                    if hasattr(outputs, 'scores') and len(outputs.scores) > 0:
                        # Map from score index to sequence index
                        for score_idx, score_tensor in enumerate(outputs.scores):
                            # Make sure this sample index is valid for this score tensor
                            if i < score_tensor.size(0):
                                # Get the token id that was actually generated at this step
                                seq_idx = input_length + score_idx + 1
                                
                                # Only process if the index is within bounds
                                if seq_idx < output.size(0):
                                    token = output[seq_idx].item()
                                    
                                    # Get the log probability for this token
                                    log_softmax = F.log_softmax(score_tensor[i], dim=-1)
                                    
                                    # Only add valid logprobs
                                    if token < log_softmax.size(-1):
                                        logprob = log_softmax[token].item()
                                        # Add only non-infinite logprobs
                                        if not torch.isinf(torch.tensor(logprob)) and not torch.isnan(torch.tensor(logprob)):
                                            token_logprobs.append(logprob)
                                        else:
                                            # Use a small negative value instead of -Infinity
                                            token_logprobs.append(-20.0)
                                    else:
                                        # Token out of vocabulary range, use fallback value
                                        token_logprobs.append(-20.0)
                except Exception as e:
                    print(f"Error extracting logprobs for problem {problem_id}: {e}")
            
            # Set a minimum logprob value to avoid -infinity
            token_logprobs = [-20.0 if torch.isinf(torch.tensor(lp)) or torch.isnan(torch.tensor(lp)) else lp for lp in token_logprobs]
            
            # Calculate mean logprob safely
            if token_logprobs:
                mean_logprob = sum(token_logprobs) / len(token_logprobs)
            else:
                mean_logprob = -20.0  # Use a default negative value
            
            # Calculate logprob statistics
            if token_logprobs:
                # Replace any remaining infinity values with our fallback
                safe_logprobs = [-20.0 if torch.isinf(torch.tensor(lp)) or torch.isnan(torch.tensor(lp)) else lp for lp in token_logprobs]
                
                logprob_stats = {
                    "count": len(safe_logprobs),
                    "mean": sum(safe_logprobs) / len(safe_logprobs) if safe_logprobs else -20.0,
                    "min": min(safe_logprobs) if safe_logprobs else -20.0,
                    "max": max(safe_logprobs) if safe_logprobs else -20.0,
                    "median": sorted(safe_logprobs)[len(safe_logprobs) // 2] if safe_logprobs else -20.0,
                    "high_confidence_ratio": sum(1 for lp in safe_logprobs if lp > -0.1) / len(safe_logprobs) if safe_logprobs else 0.0,
                    "low_confidence_ratio": sum(1 for lp in safe_logprobs if lp < -1.0) / len(safe_logprobs) if safe_logprobs else 1.0
                }
            else:
                logprob_stats = {
                    "count": 0,
                    "mean": -20.0,
                    "min": -20.0,
                    "max": -20.0,
                    "median": -20.0,
                    "high_confidence_ratio": 0.0,
                    "low_confidence_ratio": 1.0
                }
            
            # Print detailed info for correct outputs or sample problems
            if is_correct or problem_id % 50 == 0:
                print(f"\nProblem ID: {problem_id}")
                print(f"Input: {repr(true_in)}")
                print(f"Full Generated: {full_generated_text}")
                print(f"Extracted Output: {generated}")
                print(f"Expected: {true_out}")
                print(f"Correct: {is_correct}")
                if token_logprobs:
                    print(f"Mean logprob: {mean_logprob:.4f}")
                    print(f"High confidence token ratio: {logprob_stats['high_confidence_ratio']:.2f}")
            
            batch_results.append((is_correct, full_generated_text, generated, true_out, token_logprobs, logprob_stats))
        
        return batch_results
    
    except Exception as e:
        print(f"Error evaluating batch {problem_ids}: {str(e)}")
        # Return failed results for each problem in the batch
        return [(False, str(e), f"Error: {str(e)}", "", [], {}) for _ in problem_ids]

# Function to evaluate a batch of problems with multiple samples per problem
def evaluate_batch_with_sampling(model, tokenizer, problem_ids: list[int], 
                                max_tokens: int = 100, num_samples: int = 5,
                                temperature: float = 0.7, top_p: float = 0.95,
                                persona: str = "") -> list[dict]:
    """
    Evaluate a batch of CruxEval problems with multiple samples per problem.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        problem_ids: List of problem IDs to evaluate in this batch
        max_tokens: Maximum number of tokens to generate
        num_samples: Number of samples to generate per problem
        temperature: Temperature for sampling
        top_p: Top-p nucleus sampling parameter
        persona: Persona to use for evaluation
        
    Returns:
        List of dictionaries containing results for each problem, including all samples
    """
    try:
        # Prepare all prompts
        batch_prompts = []
        true_outputs = []
        problem_inputs = []
        
        for problem_id in problem_ids:
            prompt, true_in, true_out = ce.output_full(problem_id)
            if persona != "":
                prompt = f"{persona}\n\n{prompt}"
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
            max_length=2048,
            padding_side=tokenizer.padding_side
        ).to(model.device)
        
        # Save original generation parameters (if they exist)
        original_do_sample = None
        original_temperature = None
        original_top_p = None
        original_num_beams = None
        original_num_return_sequences = None
        original_early_stopping = None
        
        if hasattr(model, 'generation_config'):
            original_do_sample = model.generation_config.do_sample
            original_temperature = model.generation_config.temperature
            original_top_p = model.generation_config.top_p
            original_num_beams = model.generation_config.num_beams
            original_num_return_sequences = model.generation_config.num_return_sequences
            original_early_stopping = model.generation_config.early_stopping
            
            # Configure for sampling
            model.generation_config.do_sample = True
            model.generation_config.temperature = temperature
            model.generation_config.top_p = top_p
            model.generation_config.num_beams = 1
            model.generation_config.num_return_sequences = num_samples
            model.generation_config.early_stopping = False
        
        # Print generation parameters for diagnostic purposes
        print(f"\nGeneration parameters:")
        print(f"  num_samples: {num_samples}")
        print(f"  temperature: {temperature}")
        print(f"  top_p: {top_p}")
        print(f"  max_tokens: {max_tokens}")
        
        # Generate outputs for the entire batch with sampling
        batch_results = []
        
        # Process each problem individually for sampling
        for i, problem_id in enumerate(problem_ids):
            try:
                # Create input tensor for this specific problem
                problem_input_ids = inputs.input_ids[i:i+1]  # Keep batch dimension
                problem_attention_mask = inputs.attention_mask[i:i+1] if inputs.attention_mask is not None else None
                
                # Get input length for this problem
                input_length = input_ids_lengths[i] - 1
                
                # Generate multiple samples with logprobs
                with torch.no_grad():
                    generated_outputs = model.generate(
                        problem_input_ids,
                        attention_mask=problem_attention_mask,
                        max_new_tokens=max_tokens,
                        do_sample=True,  # Enable sampling
                        num_return_sequences=num_samples,  # Return multiple sequences
                        num_beams=1,  # No beam search
                        temperature=temperature,
                        top_p=top_p,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                        early_stopping=False,  # Not needed with num_beams=1
                        output_scores=True,    # Get the logits
                        return_dict_in_generate=True  # Return full generation info including scores
                    )
                
                # Process each sample
                sample_results = []
                any_correct = False
                best_sample_idx = -1
                highest_logprob_idx = -1
                correct_count = 0
                highest_mean_logprob = float('-inf')
                all_mean_logprobs = []
                
                # Get sequences and compute transition scores
                sequences = generated_outputs.sequences
                
                # Compute transition scores using the built-in HuggingFace method
                # This handles the token probability calculations correctly
                try:
                    transition_scores = model.compute_transition_scores(
                        sequences=generated_outputs.sequences,
                        scores=generated_outputs.scores,
                        normalize_logits=True  # Get log probabilities
                    )
                except (AttributeError, RuntimeError) as e:
                    # Some models don't support compute_transition_scores
                    print(f"Warning: compute_transition_scores failed: {e}")
                    # Fall back to our custom method
                    transition_scores = None
                
                for j in range(num_samples):
                    # Get generated tokens for this sample
                    sample_output = sequences[j]
                    
                    # Extract only newly generated tokens (after input)
                    generated_tokens = sample_output[input_length:]
                    full_generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    # Extract only the part after "output:" if it exists
                    if "output:" in full_generated_text.lower():
                        output_pos = full_generated_text.lower().find("output:")
                        generated = full_generated_text[output_pos + 7:].strip()
                    else:
                        generated = full_generated_text.strip()
                    
                    # Check if output matches
                    is_correct = generated.strip() == true_outputs[i].strip()
                    
                    # Track if any sample is correct and which one
                    if is_correct:
                        correct_count += 1
                        if not any_correct:  # Only update for the first correct sample
                            any_correct = True
                            best_sample_idx = j

                    # Extract logprobs for this sample using transition scores if available
                    token_logprobs = []
                    
                    if transition_scores is not None:
                        # Get the transition scores for this sample (sequence)
                        sample_scores = transition_scores[j]
                        
                        # Only consider scores for tokens after the input
                        # The +1 offset is because transition_scores has one less element than generated tokens
                        # (the first token has no transition)
                        relevant_scores = sample_scores[max(0, input_length - sample_output.shape[0] + 1):]
                        
                        # Filter out padding tokens and special tokens at the end
                        token_list = sample_output[input_length:].tolist()
                        filtered_scores = []
                        
                        # Make sure token_list and relevant_scores align in length
                        min_len = min(len(token_list), len(relevant_scores))
                        for idx in range(min_len):
                            score = relevant_scores[idx]
                            token_id = token_list[idx]
                            
                            # Skip padding tokens (usually eos_token_id) and tokens at the end that might cause -inf
                            if token_id == tokenizer.pad_token_id or token_id == tokenizer.eos_token_id:
                                continue
                                
                            # Once we encounter a really low score (likely padding or end), stop
                            if score.item() < -15 and idx > min_len // 2:  # Only apply in the latter half
                                break
                                
                            # Only include valid scores
                            if not (torch.isinf(score) or torch.isnan(score)):
                                filtered_scores.append(score.item())
                            else:
                                filtered_scores.append(-20.0)  # Use fallback
                        
                        token_logprobs = filtered_scores
                    else:
                        # Fall back to our custom method for models that don't support compute_transition_scores
                        # Extract token logprobs more carefully
                        try:
                            # Check that we have scores
                            if hasattr(generated_outputs, 'scores') and len(generated_outputs.scores) > 0:
                                # Map from score index to sequence index
                                for score_idx, score_tensor in enumerate(generated_outputs.scores):
                                    # Make sure this sample index is valid for this score tensor
                                    if j < score_tensor.size(0):
                                        # Get the token id that was actually generated at this step
                                        seq_idx = input_length + score_idx + 1
                                        
                                        # Only process if the index is within bounds
                                        if seq_idx < sample_output.size(0):
                                            token = sample_output[seq_idx].item()
                                            
                                            # Get the log probability for this token
                                            log_softmax = F.log_softmax(score_tensor[j], dim=-1)
                                            
                                            # Only add valid logprobs
                                            if token < log_softmax.size(-1):
                                                logprob = log_softmax[token].item()
                                                # Add only non-infinite logprobs
                                                if not torch.isinf(torch.tensor(logprob)) and not torch.isnan(torch.tensor(logprob)):
                                                    token_logprobs.append(logprob)
                                                else:
                                                    # Use a small negative value instead of -Infinity
                                                    token_logprobs.append(-20.0)
                                            else:
                                                # Token out of vocabulary range, use fallback value
                                                token_logprobs.append(-20.0)
                        except Exception as e:
                            print(f"Error extracting logprobs for problem {problem_id}, sample {j}: {e}")
                    
                    # Set a minimum logprob value to avoid -infinity
                    token_logprobs = [-20.0 if torch.isinf(torch.tensor(lp)) or torch.isnan(torch.tensor(lp)) else lp for lp in token_logprobs]
                    
                    # Calculate mean logprob more safely
                    if token_logprobs:
                        mean_logprob = sum(token_logprobs) / len(token_logprobs)
                    else:
                        mean_logprob = -20.0  # Use a default negative value
                    all_mean_logprobs.append(mean_logprob)
                    
                    # Track sample with highest logprob
                    if mean_logprob > highest_mean_logprob:
                        highest_mean_logprob = mean_logprob
                        highest_logprob_idx = j
                    
                    # Calculate logprob statistics
                    if token_logprobs:
                        # Replace any remaining infinity values with our fallback
                        safe_logprobs = [-20.0 if torch.isinf(torch.tensor(lp)) or torch.isnan(torch.tensor(lp)) else lp for lp in token_logprobs]
                        
                        logprob_stats = {
                            "count": len(safe_logprobs),
                            "mean": sum(safe_logprobs) / len(safe_logprobs) if safe_logprobs else -20.0,
                            "min": min(safe_logprobs) if safe_logprobs else -20.0,
                            "max": max(safe_logprobs) if safe_logprobs else -20.0,
                            "median": sorted(safe_logprobs)[len(safe_logprobs) // 2] if safe_logprobs else -20.0,
                            "high_confidence_ratio": sum(1 for lp in safe_logprobs if lp > -0.1) / len(safe_logprobs) if safe_logprobs else 0.0,
                            "low_confidence_ratio": sum(1 for lp in safe_logprobs if lp < -1.0) / len(safe_logprobs) if safe_logprobs else 1.0
                        }
                    else:
                        logprob_stats = {
                            "count": 0,
                            "mean": -20.0,
                            "min": -20.0,
                            "max": -20.0,
                            "median": -20.0,
                            "high_confidence_ratio": 0.0,
                            "low_confidence_ratio": 1.0
                        }
                    
                    # Add to sample results
                    sample_results.append({
                        "sample_idx": j,
                        "generated": generated,
                        "full_generated_text": full_generated_text,
                        "is_correct": is_correct,
                        "token_logprobs": token_logprobs,
                        "mean_logprob": mean_logprob,
                        "num_tokens": len(generated_tokens),
                        "logprob_stats": logprob_stats
                    })
                
                # Calculate correct rate
                correct_rate = correct_count / num_samples if num_samples > 0 else 0.0
                
                # Calculate average logprob across all samples
                avg_logprob = sum(all_mean_logprobs) / len(all_mean_logprobs) if all_mean_logprobs else -20.0
                
                # Add result for this problem
                batch_results.append({
                    "problem_id": problem_id,
                    "any_correct": any_correct,
                    "best_sample_idx": best_sample_idx,
                    "highest_logprob_idx": highest_logprob_idx,
                    "correct_count": correct_count,
                    "correct_rate": correct_rate,
                    "avg_logprob": avg_logprob,
                    "true_output": true_outputs[i],
                    "samples": sample_results
                })
                
            except Exception as problem_error:
                print(f"Error processing problem {problem_id}: {str(problem_error)}")
                batch_results.append({
                    "problem_id": problem_id,
                    "error": str(problem_error),
                    "any_correct": False,
                    "best_sample_idx": -1,
                    "highest_logprob_idx": -1,
                    "correct_count": 0,
                    "correct_rate": 0.0,
                    "avg_logprob": -20.0,
                    "samples": []
                })
        
        # Restore original generation parameters if they were set
        if hasattr(model, 'generation_config'):
            if original_do_sample is not None:
                model.generation_config.do_sample = original_do_sample
            if original_temperature is not None:
                model.generation_config.temperature = original_temperature
            if original_top_p is not None:
                model.generation_config.top_p = original_top_p
            if original_num_beams is not None:
                model.generation_config.num_beams = original_num_beams
            if original_num_return_sequences is not None:
                model.generation_config.num_return_sequences = original_num_return_sequences
            if original_early_stopping is not None:
                model.generation_config.early_stopping = original_early_stopping
        
        return batch_results
        
    except Exception as batch_error:
        # Restore original generation parameters if there was an error
        if hasattr(model, 'generation_config'):
            if original_do_sample is not None:
                model.generation_config.do_sample = original_do_sample
            if original_temperature is not None:
                model.generation_config.temperature = original_temperature
            if original_top_p is not None:
                model.generation_config.top_p = original_top_p
            if original_num_beams is not None:
                model.generation_config.num_beams = original_num_beams
            if original_num_return_sequences is not None:
                model.generation_config.num_return_sequences = original_num_return_sequences
            if original_early_stopping is not None:
                model.generation_config.early_stopping = original_early_stopping
                
        print(f"Error evaluating batch {problem_ids}: {str(batch_error)}")
        # Return failed results for each problem in the batch
        return [{
            "problem_id": pid,
            "error": str(batch_error),
            "any_correct": False,
            "best_sample_idx": -1,
            "highest_logprob_idx": -1,
            "correct_count": 0,
            "correct_rate": 0.0,
            "avg_logprob": -20.0,
            "samples": []
        } for pid in problem_ids]

#%%
eval_persona = args.persona
output_dir = args.output_dir
if eval_persona:
    output_dir = os.path.join(output_dir, "multiprompt_results",model_short_name)

# Evaluate all problems with batch processing for better performance
num_problems = args.num_problems
batch_size = args.batch_size  # Process multiple problems at once
multi_trial = args.multi_trial

def evaluate_cruxeval(persona: str = "", trial_num: int | None = None):
    # Process problems in true batches
    persona_description = ""

    results = []
    
    # --- Filename Generation --- 
    num_samples = args.num_samples
    temperature = args.temperature
    top_p = args.top_p
    
    # Create the base filename in the new format
    if num_samples > 1:
        base_filename = f"cruxeval_k{num_samples}_temp{temperature}_topp{top_p}_{model_short_name}"
    else:
        base_filename = f"cruxeval_{model_short_name}"
    
    # Add persona if applicable
    if eval_persona and persona != "":
        base_filename += f"_{persona}"
    
    # Add revision if not main
    if revision != "main":
        base_filename += f"_{revision}"

    # Add trial number if applicable
    if trial_num is not None:
        base_filename += f"_trial{trial_num}"
        
    # Add file extension
    results_filename = f"{base_filename}.json"
    checkpoint_filename = f"{base_filename}_checkpoint.json"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct full file paths
    results_file = os.path.join(output_dir, results_filename)
    checkpoint_file = os.path.join(output_dir, checkpoint_filename)
    # --- End Filename Generation ---

    print(f"\n--- Starting Evaluation Trial {trial_num if trial_num is not None else 'Single'} ---")
    print(f"Persona: '{persona}'")
    print(f"Number of samples per problem: {num_samples}")
    if num_samples > 1:
        print(f"Sampling temperature: {temperature}, Top-p: {top_p}")
    print(f"Results will be saved to: {results_file}")
    print(f"Checkpoint file: {checkpoint_file}")

    # Check if checkpoint exists to resume from previous run for THIS trial
    try:
        with open(checkpoint_file, "r") as f:
            checkpoint_data = json.load(f)
            results = checkpoint_data["results"]
            start_idx = len(results)
            print(f"Resuming from checkpoint with {start_idx} problems already processed for this trial")
    except FileNotFoundError:
        start_idx = 0
        print("Starting fresh evaluation for this trial")
    except json.JSONDecodeError:
         print(f"Warning: Checkpoint file {checkpoint_file} is corrupted. Starting fresh.")
         start_idx = 0
         results = [] # Reset results if checkpoint is bad

    # Adjust batch size based on model and available GPU memory
    # Use a smaller batch size initially to be safer
    adjusted_batch_size = min(batch_size, 64)  # Start with max 64 examples per batch 
    print(f"Evaluating {num_problems - start_idx} remaining problems in batches of {adjusted_batch_size}...")
    
    current_start_time = time.time() # Use a start time specific to this trial/resumption
    if persona != "":
        persona_description = PERSONAS[persona]
        print(f"Evaluating with persona description: {persona_description[:50]}...") # Print start of description

    for batch_start in tqdm(range(start_idx, num_problems, adjusted_batch_size), desc=f"Trial {trial_num if trial_num is not None else 'Single'}"):
        batch_end = min(batch_start + adjusted_batch_size, num_problems)
        problem_ids = list(range(batch_start, batch_end))
        
        # Use the appropriate evaluation function based on whether we're doing sampling
        try:
            if num_samples > 1:
                batch_results = evaluate_batch_with_sampling(
                    model, tokenizer, problem_ids, 
                    max_tokens=100, 
                    num_samples=num_samples,
                    temperature=temperature,
                    top_p=top_p,
                    persona=persona_description
                )
            else:
                # For single samples, use the original evaluation function
                batch_results = evaluate_batch(model, tokenizer, problem_ids, max_tokens=100, persona=persona_description)
                
                # Convert to the new format for consistency
                formatted_batch_results = []
                for i, result_tuple in enumerate(batch_results):
                    # Ensure the tuple has the expected number of elements
                    if len(result_tuple) >= 5:  # Updated to expect at least 5 elements
                        is_correct, full_generated_text, generated, true_out, token_logprobs, logprob_stats = result_tuple
                        mean_logprob = logprob_stats.get("mean", 0.0) if logprob_stats else 0.0
                        
                        formatted_batch_results.append({
                            "problem_id": problem_ids[i],
                            "any_correct": is_correct,
                            "best_sample_idx": 0 if is_correct else -1,
                            "highest_logprob_idx": 0,  # Only one sample, so it's the highest by default
                            "correct_count": 1 if is_correct else 0,
                            "correct_rate": 1.0 if is_correct else 0.0,
                            "avg_logprob": mean_logprob,
                            "true_output": true_out,
                            "samples": [{
                                "sample_idx": 0,
                                "generated": generated,
                                "full_generated_text": full_generated_text,
                                "is_correct": is_correct,
                                "token_logprobs": token_logprobs,
                                "mean_logprob": mean_logprob,
                                "num_tokens": len(tokenizer.encode(generated)),
                                "logprob_stats": logprob_stats
                            }]
                        })
                    else:
                        print(f"Warning: Unexpected result format for problem {problem_ids[i]}: {result_tuple}")
                        formatted_batch_results.append({
                            "problem_id": problem_ids[i],
                            "any_correct": False,
                            "best_sample_idx": -1,
                            "highest_logprob_idx": -1,
                            "correct_count": 0,
                            "correct_rate": 0.0,
                            "avg_logprob": 0.0,
                            "error": f"FormatError: {result_tuple}",
                            "samples": []
                        })
                batch_results = formatted_batch_results
        except Exception as batch_error:
            print(f"\nERROR evaluating batch {problem_ids}: {batch_error}. Saving partial results.")
            # Create error results for the failed batch
            batch_results = [{
                "problem_id": pid,
                "error": f"ERROR: {batch_error}",
                "any_correct": False,
                "best_sample_idx": -1,
                "correct_count": 0,
                "correct_rate": 0.0,
                "samples": []
            } for pid in problem_ids]

        # Add batch results to overall results
        results.extend(batch_results)
        
        # Save checkpoint after each batch
        try:
            with open(checkpoint_file, "w") as f:
                json.dump({"results": results}, f)
        except Exception as cp_error:
            print(f"\nWarning: Could not save checkpoint file {checkpoint_file}: {cp_error}")
        
        # Print progress after each batch
        processed_count = len(results)
        if processed_count > 0: # Avoid division by zero
            if num_samples > 1:
                correct_count = sum(1 for r in results if r.get("any_correct", False))
                avg_correct_rate = sum(r.get("correct_rate", 0.0) for r in results) / processed_count
                current_accuracy = correct_count / processed_count * 100
                
                # Calculate logprob statistics
                avg_logprob = sum(r.get("avg_logprob", 0.0) for r in results) / processed_count
                
                # Count how often highest logprob sample is correct
                logprob_match_count = sum(1 for r in results 
                                          if r.get("highest_logprob_idx", -1) != -1 and 
                                          r.get("samples") and 
                                          r.get("samples")[r.get("highest_logprob_idx", 0)].get("is_correct", False))
                logprob_match_rate = logprob_match_count / processed_count * 100
                
                print(f"Progress: {processed_count}/{num_problems}, Any Correct: {correct_count}/{processed_count} ({current_accuracy:.2f}%)")
                print(f"Average correct rate across all problems: {avg_correct_rate:.4f} ({avg_correct_rate*100:.2f}%)")
                print(f"Average logprob: {avg_logprob:.4f}")
                print(f"Highest logprob sample is correct: {logprob_match_count}/{processed_count} ({logprob_match_rate:.2f}%)\n")
            else:
                correct_count = sum(1 for r in results if r.get("any_correct", False))
                current_accuracy = correct_count / processed_count * 100
                print(f"Progress: {processed_count}/{num_problems}, Correct: {correct_count}/{processed_count} ({current_accuracy:.2f}%)\n")
        else:
             print(f"Progress: {processed_count}/{num_problems}\n")

        # Calculate and print time estimates based on current trial's progress
        elapsed_time = time.time() - current_start_time
        if processed_count > start_idx: # Avoid division by zero if no new problems processed
            problems_processed_this_run = processed_count - start_idx
            time_this_run = elapsed_time
            problems_per_second = problems_processed_this_run / time_this_run if time_this_run > 0 else 0
            
            remaining_problems = num_problems - processed_count
            estimated_remaining_time = remaining_problems / problems_per_second if problems_per_second > 0 else float('inf')
            
            print(f"Speed (this run): {problems_per_second:.2f} problems/second")
            if estimated_remaining_time != float('inf'):
                print(f"Estimated time remaining: {estimated_remaining_time/60:.1f} minutes ({estimated_remaining_time/3600:.1f} hours)\n")
            else:
                 print("Estimating time remaining...\n")

    end_time = time.time()
    total_time = end_time - current_start_time # Use trial-specific start time
    print(f"\nEvaluation Trial {trial_num if trial_num is not None else 'Single'} completed in {total_time:.2f} seconds ({total_time/60:.1f} minutes)")

    # Calculate final accuracy for this trial
    if num_samples > 1:
        correct_count = sum(1 for r in results if r.get("any_correct", False))
        accuracy = correct_count / num_problems if num_problems > 0 else 0
        
        # Calculate average correct rate across all problems
        avg_correct_rate = sum(r.get("correct_rate", 0.0) for r in results) / num_problems if num_problems > 0 else 0.0
        
        # Calculate problems with high consistency (>= 50% correct rate)
        high_consistency_count = sum(1 for r in results if r.get("correct_rate", 0.0) >= 0.5)
        
        # Calculate logprob statistics
        avg_logprob = sum(r.get("avg_logprob", 0.0) for r in results) / num_problems if num_problems > 0 else 0.0
        
        # Count how often highest logprob sample is correct
        logprob_match_count = sum(1 for r in results 
                                  if r.get("highest_logprob_idx", -1) != -1 and 
                                  r.get("samples") and 
                                  r.get("samples")[r.get("highest_logprob_idx", 0)].get("is_correct", False))
        logprob_match_rate = logprob_match_count / num_problems * 100 if num_problems > 0 else 0.0
        
        # Calculate correlation between logprob and correctness
        correct_logprobs = [r.get("avg_logprob", 0.0) for r in results if r.get("any_correct", False)]
        incorrect_logprobs = [r.get("avg_logprob", 0.0) for r in results if not r.get("any_correct", False)]
        avg_correct_logprob = sum(correct_logprobs) / len(correct_logprobs) if correct_logprobs else 0.0
        avg_incorrect_logprob = sum(incorrect_logprobs) / len(incorrect_logprobs) if incorrect_logprobs else 0.0
        
        print(f"Final accuracy with {num_samples} samples: {correct_count}/{num_problems} ({accuracy*100:.2f}%)")
        print(f"Average correct rate across all problems: {avg_correct_rate:.4f} ({avg_correct_rate*100:.2f}%)")
        print(f"Problems with ≥50% correct rate: {high_consistency_count}/{num_problems} ({high_consistency_count/num_problems*100:.2f}%)")
        print(f"Average logprob: {avg_logprob:.4f}")
        print(f"Highest logprob sample is correct: {logprob_match_count}/{num_problems} ({logprob_match_rate:.2f}%)")
        
        if correct_logprobs and incorrect_logprobs:
            print(f"Average logprob for correct problems: {avg_correct_logprob:.4f}")
            print(f"Average logprob for incorrect problems: {avg_incorrect_logprob:.4f}")
            print(f"Logprob gap (correct - incorrect): {avg_correct_logprob - avg_incorrect_logprob:.4f}")
        
        # Calculate greedy accuracy (first sample only)
        greedy_correct = sum(1 for r in results if r.get("samples") and len(r.get("samples", [])) > 0 and r.get("samples")[0].get("is_correct", False))
        greedy_accuracy = greedy_correct / num_problems if num_problems > 0 else 0
        print(f"Greedy accuracy (first sample only): {greedy_correct}/{num_problems} ({greedy_accuracy*100:.2f}%)")
        
        # Calculate improvement from sampling
        improvement = correct_count - greedy_correct
        print(f"Improvement from sampling: +{improvement} problems ({improvement/num_problems*100:.2f}%)")
    else:
        correct_count = sum(1 for r in results if r.get("any_correct", False))
        accuracy = correct_count / num_problems if num_problems > 0 else 0
        print(f"Final accuracy: {correct_count}/{num_problems} ({accuracy*100:.2f}%)")

    # Save the final results for this trial
    try:
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2) # Add indent for readability
        print(f"Results saved to {results_file}")
    except Exception as save_error:
        print(f"\nERROR: Could not save final results file {results_file}: {save_error}")


    # Find problems where the model gets the correct output (for this trial)
    correct_problems = [r["problem_id"] for r in results if r.get("any_correct", False)]
    print(f"Found {len(correct_problems)} problems with correct outputs in this trial.")
    
    # Print most consistent problems (highest correct rates)
    if num_samples > 1:
        consistent_problems = sorted(
            [(r["problem_id"], r.get("correct_rate", 0.0)) for r in results if r.get("correct_rate", 0.0) > 0], 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        if consistent_problems:
            print("\nMost consistent problems (highest correct rates):")
            for problem_id, rate in consistent_problems:
                print(f"  Problem {problem_id}: {rate:.2%} correct rate")

    # These problems can be used for causal tracing
    if correct_problems:
        print("\nFirst 10 problems with correct outputs (this trial):")
        print(correct_problems[:10])
    else:
        print("No problems solved correctly in this trial.")


    # Clean up checkpoint file after successful completion FOR THIS trial
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            print(f"Removed checkpoint file: {checkpoint_file}")
        except OSError as rm_error:
            print(f"Warning: Could not remove checkpoint file {checkpoint_file}: {rm_error}")


# --- Main Execution Logic --- 
if multi_trial > 0:
    print(f"\n===== Starting Multi-Trial Evaluation ({multi_trial} trials) =====")
    if eval_persona:
        for k in range(multi_trial):
            print(f"\n===== TRIAL {k} =====")
            for persona in PERSONAS:
                evaluate_cruxeval(persona=persona, trial_num=k)
    else:
        for k in range(multi_trial):
            print(f"\n===== TRIAL {k} =====")
            evaluate_cruxeval(persona="", trial_num=k)
    print(f"\n===== Multi-Trial Evaluation Completed =====")
else:
    print("\n===== Starting Single Evaluation Run =====")
    if eval_persona:
        for persona in PERSONAS:
            evaluate_cruxeval(persona=persona, trial_num=None) # Pass None for single run
    else:
        evaluate_cruxeval(persona="", trial_num=None) # Pass None for single run
    print("\n===== Single Evaluation Run Completed =====")

# If running as a script, exit here
if not sys.argv[0].endswith('ipykernel_launcher.py'):
    sys.exit(0)
