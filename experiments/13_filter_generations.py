#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generation Filter Script

This script filters out gibberish from model generations and re-evaluates
the filtered generations for correctness, writing results to a new file.
"""

import json
import os
import re
import argparse
from pathlib import Path
import ast
import numpy as np
from copy import deepcopy


def load_results(filepath):
    """
    Load results from a JSON file.
    
    Args:
        filepath: Path to the result JSON file
        
    Returns:
        List of problem results
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []


def filter_generation(text):
    """
    Filter out gibberish from a generated text.
    Handles both cases where text has "output:" prefix or directly contains the answer.
    
    Args:
        text: The generated text to filter
        
    Returns:
        Filtered text
    """
    if not text:
        return text
    
    # Case 1: Text contains "output:" - extract content after it up to first newline
    output_pattern = re.search(r'output:\s*(.*?)(\n|$)', text, re.IGNORECASE | re.DOTALL)
    if output_pattern:
        return output_pattern.group(1).strip()
    
    # Case 2: Text directly contains the answer - split at common markers
    common_markers = ['\n```', '\n###', '\n---', '\n***', "\n'''", '\n"""']
    result = text
    for marker in common_markers:
        if marker in result:
            result = result.split(marker, 1)[0]
    
    # Also handle repetitive patterns like "2.2.2.2.2..."
    pattern = r'(\n\d\.){3,}'  # Match newline followed by digit+dot repeated 3+ times
    match = re.search(pattern, result)
    if match:
        result = result[:match.start()]
    
    # If the answer contains multiple lines, take only the first line
    # (unless it looks like a Python data structure that can span multiple lines)
    if '\n' in result and not (result.strip().startswith('[') or result.strip().startswith('{')):
        result = result.split('\n', 1)[0]
    
    return result.strip()


def evaluate_output_correctness(generated, true_output):
    """
    Evaluate if the generated output matches the true output.
    
    Args:
        generated: The generated output
        true_output: The true expected output
        
    Returns:
        Boolean indicating correctness
    """
    # Clean up the outputs
    generated = generated.strip()
    true_output = true_output.strip()
    
    # Direct string comparison
    if generated == true_output:
        return True
    
    # Special handling for boolean values - prevent True/False from matching 1/0
    if true_output.lower() == "true" or true_output.lower() == "false":
        return generated.lower() == true_output.lower()
    
    if generated.lower() == "true" or generated.lower() == "false":
        return generated.lower() == true_output.lower()
    
    # Try to parse as Python objects and compare
    try:
        # Handle tuple/list/dict representations
        gen_obj = ast.literal_eval(generated)
        true_obj = ast.literal_eval(true_output)
        
        # Special handling to prevent True/1 and False/0 equivalence
        if isinstance(gen_obj, bool) or isinstance(true_obj, bool):
            return type(gen_obj) == type(true_obj) and gen_obj == true_obj
        
        # Convert numpy arrays to lists for comparison
        if isinstance(gen_obj, np.ndarray):
            gen_obj = gen_obj.tolist()
        if isinstance(true_obj, np.ndarray):
            true_obj = true_obj.tolist()
            
        return gen_obj == true_obj
    except:
        # If parsing fails, fall back to direct comparison
        return False


def create_filtered_results(results):
    """
    Create filtered results with gibberish removed and re-evaluate correctness.
    
    Args:
        results: List of original evaluation results
        
    Returns:
        New list with filtered results
    """
    filtered_results = []
    
    correct_before = sum(1 for result in results if result.get("is_correct", False))
    correct_after = 0
    improved_count = 0
    
    for result in results:
        # Create a new result object with only the necessary fields
        new_result = {
            "problem_id": result.get("problem_id")
        }
        
        # Copy any other fields that should be preserved
        for key in ["problem_description", "function", "input", "true_output"]:
            if key in result:
                new_result[key] = result[key]
        
        # Get the text to filter - prefer full_generated_text if available
        source_text = result.get("full_generated_text", result.get("generated", ""))
        
        # Always store the full generated text for reference if available
        if "full_generated_text" in result:
            new_result["full_generated_text"] = result["full_generated_text"]
        
        # Apply filtering to get just the answer
        filtered_generation = filter_generation(source_text)
        new_result["generated"] = filtered_generation
        
        # Re-evaluate correctness
        original_correct = result.get("is_correct", False)
        if "true_output" in result:
            is_correct = evaluate_output_correctness(filtered_generation, result["true_output"])
            new_result["is_correct"] = is_correct
            if is_correct:
                correct_after += 1
                if not original_correct:
                    improved_count += 1
        
        filtered_results.append(new_result)
    
    print(f"Filtering improved correctness from {correct_before}/{len(results)} to {correct_after}/{len(filtered_results)} instances.")
    print(f"Number of problems that changed from incorrect to correct: {improved_count}")
    return filtered_results


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Filter out gibberish from model generations")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input results JSON")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save filtered results JSON")
    parser.add_argument("--examples", type=int, default=3, help="Number of examples to print (default: 3)")
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.input_file}")
    results = load_results(args.input_file)
    
    if not results:
        print("Error: No results loaded.")
        return
    
    print(f"Loaded {len(results)} results. Creating filtered outputs...")
    
    # Create filtered results
    filtered_results = create_filtered_results(results)
    
    # Save filtered results to the specified output file
    print(f"Saving filtered results to {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(filtered_results, f, indent=2)
    
    # Print examples of filtered results
    if filtered_results:
        # Find examples where filtering changed the correctness status
        original_correct = [r.get("is_correct", False) for r in results]
        new_correct = [r.get("is_correct", False) for r in filtered_results]
        
        different_indices = [i for i, (orig, new) in enumerate(zip(original_correct, new_correct)) if orig != new]
        
        print(f"\nFound {len(different_indices)} examples where filtering changed correctness.")
        num_examples = min(args.examples, len(different_indices)) if different_indices else min(args.examples, len(filtered_results))
        
        # Print examples
        if different_indices:
            example_indices = different_indices[:num_examples]
            print(f"\nShowing {len(example_indices)} examples where filtering improved correctness:")
        else:
            example_indices = list(range(min(num_examples, len(filtered_results))))
            print(f"\nShowing {len(example_indices)} filtered examples:")
        
        for i, idx in enumerate(example_indices):
            print(f"\nExample {i+1}:")
            print(f"Problem ID: {filtered_results[idx].get('problem_id')}")
            print(f"Original generated: {results[idx].get('generated')}")
            print(f"Filtered generated: {filtered_results[idx].get('generated')}")
            print(f"True output: {filtered_results[idx].get('true_output')}")
            print(f"Original correctness: {results[idx].get('is_correct')}")
            print(f"New correctness: {filtered_results[idx].get('is_correct')}")


if __name__ == "__main__":
    main()
