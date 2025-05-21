#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Log Probability Sanity Check Script

This script provides various functions to validate and analyze log probability values
from CruxEval model evaluations, ensuring they are meaningful and correctly calculated.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def validate_logprob_ranges(summary):
    """
    Check that log probabilities fall within expected ranges.
    
    Args:
        summary: Comparison summary dictionary
    """
    problems = summary["problems"]
    
    # Log probabilities should be <= 0 (probability <= 1)
    max_std_logprob = max(problem["standard"]["avg_logprob"] for problem in problems)
    max_quant_logprob = max(problem["quantized"]["avg_logprob"] for problem in problems)
    
    # Log probabilities shouldn't be extremely negative (probability > 0)
    min_std_logprob = min(problem["standard"]["avg_logprob"] for problem in problems)
    min_quant_logprob = min(problem["quantized"]["avg_logprob"] for problem in problems)
    
    print("\nLog Probability Range Check:")
    print(f"  Standard model: {min_std_logprob:.4f} to {max_std_logprob:.4f}")
    print(f"  Quantized model: {min_quant_logprob:.4f} to {max_quant_logprob:.4f}")
    
    # Flag potential issues
    if max_std_logprob > 0.01 or max_quant_logprob > 0.01:
        print("  WARNING: Some log probabilities are > 0, which is unexpected")
    
    if min_std_logprob < -30 or min_quant_logprob < -30:
        print("  WARNING: Some log probabilities are extremely negative, indicating possible underflow")
    
    # Check the distribution with some percentiles
    std_logprobs = [problem["standard"]["avg_logprob"] for problem in problems]
    quant_logprobs = [problem["quantized"]["avg_logprob"] for problem in problems]
    
    std_percentiles = np.percentile(std_logprobs, [5, 25, 50, 75, 95])
    quant_percentiles = np.percentile(quant_logprobs, [5, 25, 50, 75, 95])
    
    print("\nPercentiles (5th, 25th, 50th, 75th, 95th):")
    print(f"  Standard model: {std_percentiles}")
    print(f"  Quantized model: {quant_percentiles}")


def check_token_level_logprobs(results_file, problem_ids_to_check=5):
    """
    Verify token-level log probabilities for a sample of problems.
    
    Args:
        results_file: Path to the full results JSON file
        problem_ids_to_check: Number of problem IDs to examine
    """
    try:
        with open(results_file) as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error loading {results_file}: {e}")
        return
    
    # Sample some problems to check
    problem_sample = results[:min(problem_ids_to_check, len(results))]
    
    print("\nToken-Level Log Probability Check:")
    for problem in problem_sample:
        print(f"\nProblem {problem['problem_id']}:")
        
        # Check a sample from each problem
        if problem.get("samples") and len(problem["samples"]) > 0:
            sample = problem["samples"][0]
            token_logprobs = sample.get("token_logprobs", [])
            
            if token_logprobs:
                print(f"  Token count: {len(token_logprobs)}")
                print(f"  Token logprobs: {token_logprobs}")
                print(f"  Token logprobs sum: {sum(token_logprobs):.4f}")
                print(f"  Mean logprob: {sum(token_logprobs)/len(token_logprobs):.4f}")
                print(f"  Reported mean logprob: {sample.get('mean_logprob', 'N/A')}")
                
                # Check conversion to probabilities
                probs = [np.exp(lp) for lp in token_logprobs]
                print(f"  Min probability: {min(probs):.6f}")
                print(f"  Max probability: {max(probs):.6f}")
                
                # Verify no infinities
                if any(lp == float('-inf') for lp in token_logprobs):
                    print("  WARNING: Contains -infinity values")
                
                # Verify consistency with reported mean
                reported_mean = sample.get('mean_logprob', 0)
                calculated_mean = sum(token_logprobs)/len(token_logprobs)
                if abs(reported_mean - calculated_mean) > 0.0001:
                    print(f"  WARNING: Reported mean ({reported_mean:.4f}) doesn't match calculated mean ({calculated_mean:.4f})")
            else:
                print("  No token-level log probabilities found")
        else:
            print("  No samples found")


def compare_model_logprobs(summary):
    """
    Compare log probability distributions between models.
    
    Args:
        summary: Comparison summary dictionary
    """
    std_logprobs = [problem["standard"]["avg_logprob"] for problem in summary["problems"]]
    quant_logprobs = [problem["quantized"]["avg_logprob"] for problem in summary["problems"]]
    
    # Calculate statistics
    std_mean = np.mean(std_logprobs)
    quant_mean = np.mean(quant_logprobs)
    std_std = np.std(std_logprobs)
    quant_std = np.std(quant_logprobs)
    
    print("\nCross-Model Log Probability Comparison:")
    print(f"  Standard model: mean={std_mean:.4f}, std={std_std:.4f}")
    print(f"  Quantized model: mean={quant_mean:.4f}, std={quant_std:.4f}")
    
    # Calculate correlation
    correlation = np.corrcoef(std_logprobs, quant_logprobs)[0, 1]
    print(f"  Correlation between models: {correlation:.4f}")
    
    # Check for systematic bias
    logprob_diffs = [std - quant for std, quant in zip(std_logprobs, quant_logprobs)]
    mean_diff = np.mean(logprob_diffs)
    print(f"  Mean difference (Standard - Quantized): {mean_diff:.4f}")
    
    if abs(mean_diff) > 0.1:
        print("  NOTE: There appears to be a systematic difference between models")
    
    # Visualize the relationship
    plt.figure(figsize=(10, 6))
    plt.scatter(std_logprobs, quant_logprobs, alpha=0.5)
    plt.plot([-10, 0], [-10, 0], 'k--')  # Identity line
    plt.xlabel('Standard Model Log Probability')
    plt.ylabel('Quantized Model Log Probability')
    plt.title('Relationship Between Model Log Probabilities')
    plt.grid(linestyle='--', alpha=0.7)
    
    # Add regression line
    z = np.polyfit(std_logprobs, quant_logprobs, 1)
    p = np.poly1d(z)
    plt.plot(sorted(std_logprobs), p(sorted(std_logprobs)), "r--", 
             label=f"Fit: y={z[0]:.4f}x+{z[1]:.4f}")
    plt.legend()
    
    # Save the plot
    output_dir = os.path.join('/share/u/yu.stev/attribution/data/cruxeval_results/k_sample', 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'logprob_model_comparison.png'))
    plt.close()


def check_logprob_correctness_correlation(summary):
    """
    Check if higher log probabilities correlate with correctness.
    
    Args:
        summary: Comparison summary dictionary
    """
    problems = summary["problems"]
    
    # Standard model
    correct_logprobs = [p["standard"]["avg_logprob"] for p in problems if p["standard"]["any_correct"]]
    incorrect_logprobs = [p["standard"]["avg_logprob"] for p in problems if not p["standard"]["any_correct"]]
    
    std_correct_mean = np.mean(correct_logprobs) if correct_logprobs else float('nan')
    std_incorrect_mean = np.mean(incorrect_logprobs) if incorrect_logprobs else float('nan')
    
    # Quantized model
    q_correct_logprobs = [p["quantized"]["avg_logprob"] for p in problems if p["quantized"]["any_correct"]]
    q_incorrect_logprobs = [p["quantized"]["avg_logprob"] for p in problems if not p["quantized"]["any_correct"]]
    
    q_correct_mean = np.mean(q_correct_logprobs) if q_correct_logprobs else float('nan')
    q_incorrect_mean = np.mean(q_incorrect_logprobs) if q_incorrect_logprobs else float('nan')
    
    print("\nLog Probability vs. Correctness:")
    print(f"  Standard model: correct={std_correct_mean:.4f}, incorrect={std_incorrect_mean:.4f}, diff={std_correct_mean-std_incorrect_mean:.4f}")
    print(f"  Quantized model: correct={q_correct_mean:.4f}, incorrect={q_incorrect_mean:.4f}, diff={q_correct_mean-q_incorrect_mean:.4f}")
    
    if std_correct_mean <= std_incorrect_mean:
        print("  WARNING: Standard model's correct answers don't have higher confidence")
    if q_correct_mean <= q_incorrect_mean:
        print("  WARNING: Quantized model's correct answers don't have higher confidence")
    
    # Visualize the distributions
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    if correct_logprobs and incorrect_logprobs:
        sns.histplot(correct_logprobs, color='green', label='Correct', alpha=0.6, kde=True)
        sns.histplot(incorrect_logprobs, color='red', label='Incorrect', alpha=0.6, kde=True)
        plt.axvline(x=std_correct_mean, color='green', linestyle='--')
        plt.axvline(x=std_incorrect_mean, color='red', linestyle='--')
        plt.title('Standard Model: Log Probability by Correctness')
        plt.xlabel('Log Probability')
        plt.legend()
    
    plt.subplot(1, 2, 2)
    if q_correct_logprobs and q_incorrect_logprobs:
        sns.histplot(q_correct_logprobs, color='green', label='Correct', alpha=0.6, kde=True)
        sns.histplot(q_incorrect_logprobs, color='red', label='Incorrect', alpha=0.6, kde=True)
        plt.axvline(x=q_correct_mean, color='green', linestyle='--')
        plt.axvline(x=q_incorrect_mean, color='red', linestyle='--')
        plt.title('Quantized Model: Log Probability by Correctness')
        plt.xlabel('Log Probability')
        plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join('/share/u/yu.stev/attribution/data/cruxeval_results/k_sample', 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'logprob_correctness_correlation.png'))
    plt.close()


def check_sample_consistency(results_file):
    """
    Check consistency of log probabilities across samples within problems.
    
    Args:
        results_file: Path to the full results JSON file
    """
    try:
        with open(results_file) as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error loading {results_file}: {e}")
        return
    
    print("\nSample Consistency Check:")
    
    inconsistent_problems = 0
    problem_stats = []
    
    for problem in results:
        samples = problem.get("samples", [])
        if len(samples) < 2:
            continue
            
        # Get all sample log probabilities
        sample_logprobs = [s.get("mean_logprob", 0) for s in samples]
        
        # Calculate statistics
        mean_logprob = np.mean(sample_logprobs)
        std_logprob = np.std(sample_logprobs)
        range_logprob = max(sample_logprobs) - min(sample_logprobs)
        
        problem_stats.append({
            "problem_id": problem["problem_id"],
            "mean": mean_logprob,
            "std": std_logprob,
            "range": range_logprob
        })
        
        # Check if any sample has very extreme values compared to others
        if range_logprob > 1.0:
            inconsistent_problems += 1
            if inconsistent_problems <= 5:  # Limit output to avoid excessive printing
                print(f"  Problem {problem['problem_id']} has variable logprobs: range={range_logprob:.4f}, mean={mean_logprob:.4f}, std={std_logprob:.4f}")
                print(f"    Sample logprobs: {sample_logprobs}")
    
    if inconsistent_problems > 5:
        print(f"  ... and {inconsistent_problems - 5} more problems with high variability")
    
    print(f"  Found {inconsistent_problems} out of {len(results)} problems with highly variable logprobs across samples")
    
    # Calculate aggregate statistics
    if problem_stats:
        avg_std = np.mean([s["std"] for s in problem_stats])
        avg_range = np.mean([s["range"] for s in problem_stats])
        print(f"  Average standard deviation across samples: {avg_std:.4f}")
        print(f"  Average range across samples: {avg_range:.4f}")
    
    # Create visualization of the variability
    if problem_stats:
        plt.figure(figsize=(10, 6))
        ranges = [s["range"] for s in problem_stats]
        stds = [s["std"] for s in problem_stats]
        
        plt.subplot(2, 1, 1)
        plt.hist(ranges, bins=30, alpha=0.7)
        plt.title('Distribution of Log Probability Ranges Within Problems')
        plt.xlabel('Range (Max - Min) of Log Probabilities')
        plt.ylabel('Count')
        
        plt.subplot(2, 1, 2)
        plt.hist(stds, bins=30, alpha=0.7)
        plt.title('Distribution of Log Probability Standard Deviations Within Problems')
        plt.xlabel('Standard Deviation of Log Probabilities')
        plt.ylabel('Count')
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = os.path.join('/share/u/yu.stev/attribution/data/cruxeval_results/k_sample', 'visualizations')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'logprob_sample_consistency.png'))
        plt.close()


def analyze_logprob_thresholds(summary):
    """
    Analyze if there are threshold patterns for correctness.
    
    Args:
        summary: Comparison summary dictionary
    """
    problems = summary["problems"]
    
    # Try different thresholds to find a potential cutoff
    thresholds = np.linspace(-2.0, 0, 20)
    
    print("\nLog Probability Threshold Analysis:")
    
    std_results = []
    quant_results = []
    
    print("  Standard Model:")
    for threshold in thresholds:
        correct_above = sum(1 for p in problems 
                           if p["standard"]["avg_logprob"] >= threshold and p["standard"]["any_correct"])
        incorrect_above = sum(1 for p in problems 
                             if p["standard"]["avg_logprob"] >= threshold and not p["standard"]["any_correct"])
        
        if correct_above + incorrect_above > 0:
            precision = correct_above / (correct_above + incorrect_above)
            print(f"    Threshold {threshold:.2f}: precision={precision:.2f} (correct={correct_above}, incorrect={incorrect_above})")
            std_results.append((threshold, precision, correct_above, incorrect_above))
    
    print("  Quantized Model:")
    for threshold in thresholds:
        correct_above = sum(1 for p in problems 
                           if p["quantized"]["avg_logprob"] >= threshold and p["quantized"]["any_correct"])
        incorrect_above = sum(1 for p in problems 
                             if p["quantized"]["avg_logprob"] >= threshold and not p["quantized"]["any_correct"])
        
        if correct_above + incorrect_above > 0:
            precision = correct_above / (correct_above + incorrect_above)
            print(f"    Threshold {threshold:.2f}: precision={precision:.2f} (correct={correct_above}, incorrect={incorrect_above})")
            quant_results.append((threshold, precision, correct_above, incorrect_above))
    
    # Create visualization of the thresholds
    if std_results and quant_results:
        plt.figure(figsize=(10, 6))
        
        std_thresholds = [r[0] for r in std_results]
        std_precisions = [r[1] for r in std_results]
        quant_thresholds = [r[0] for r in quant_results]
        quant_precisions = [r[1] for r in quant_results]
        
        plt.plot(std_thresholds, std_precisions, 'b-', label='Standard Model')
        plt.plot(quant_thresholds, quant_precisions, 'r-', label='Quantized Model')
        
        plt.title('Precision by Log Probability Threshold')
        plt.xlabel('Log Probability Threshold')
        plt.ylabel('Precision (Correct / Total)')
        plt.grid(linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save the plot
        output_dir = os.path.join('/share/u/yu.stev/attribution/data/cruxeval_results/k_sample', 'visualizations')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'logprob_threshold_analysis.png'))
        plt.close()


def analyze_extreme_logprobs(results_file, threshold=-5.0, max_problems=10, token_threshold=-10.0):
    """
    Identify and analyze problems with extremely negative log probabilities.
    
    Args:
        results_file: Path to the full results JSON file
        threshold: Log probability threshold to consider as "extremely negative"
        max_problems: Maximum number of extreme problems to analyze in detail
        token_threshold: Threshold to identify individual tokens with extremely low probability
    """
    try:
        with open(results_file) as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error loading {results_file}: {e}")
        return
    
    print(f"\nExtreme Log Probability Analysis (threshold = {threshold}):")
    
    # Find problems with extremely negative log probabilities
    extreme_problems = []
    
    for problem in results:
        problem_id = problem["problem_id"]
        samples = problem.get("samples", [])
        
        if not samples:
            continue
        
        # Get the mean logprob for each sample
        sample_logprobs = [s.get("mean_logprob", 0) for s in samples]
        
        # Sort samples by log probability (most negative first)
        sorted_samples = sorted(zip(sample_logprobs, samples), key=lambda x: x[0])
        
        # If any sample is below the threshold, add to extreme problems
        if sorted_samples and sorted_samples[0][0] < threshold:
            worst_logprob, worst_sample = sorted_samples[0]
            output_text = worst_sample.get("full_generated_text", "")
            token_logprobs = worst_sample.get("token_logprobs", [])
            token_count = len(token_logprobs)
            
            # Find problematic tokens (with extremely low probabilities)
            problematic_tokens = []
            if "token_ids" in worst_sample and "token_logprobs" in worst_sample:
                try:
                    # If tokenized text is available
                    if "tokenized_text" in worst_sample and len(worst_sample["tokenized_text"]) == len(token_logprobs):
                        tokens = worst_sample["tokenized_text"]
                        for i, (token, logprob) in enumerate(zip(tokens, token_logprobs)):
                            if logprob < token_threshold:
                                problematic_tokens.append({
                                    "index": i,
                                    "token": token,
                                    "logprob": logprob
                                })
                    else:
                        # Just track indices for problematic tokens
                        for i, logprob in enumerate(token_logprobs):
                            if logprob < token_threshold:
                                problematic_tokens.append({
                                    "index": i,
                                    "token": f"token_{i}",
                                    "logprob": logprob
                                })
                except Exception as e:
                    print(f"  Error analyzing tokens for problem {problem_id}: {e}")
            
            # Create summary for this problem
            extreme_problems.append({
                "problem_id": problem_id,
                "logprob": worst_logprob,
                "token_count": token_count,
                "output_length": len(output_text),
                "output_preview": output_text[:100] + "..." if len(output_text) > 100 else output_text,
                "min_token_logprob": min(token_logprobs) if token_logprobs else None,
                "full_output": output_text,
                "true_output": problem.get("true_output", ""),
                "problematic_tokens": problematic_tokens
            })
    
    # Sort extreme problems by log probability (most extreme first)
    extreme_problems.sort(key=lambda x: x["logprob"])
    
    # Report findings
    print(f"  Found {len(extreme_problems)} problems with log probability below {threshold}")
    
    if extreme_problems:
        # Calculate statistics on token counts
        token_counts = [p["token_count"] for p in extreme_problems]
        output_lengths = [p["output_length"] for p in extreme_problems]
        
        print(f"  Average token count: {np.mean(token_counts):.1f}")
        print(f"  Average output length: {np.mean(output_lengths):.1f} characters")
        
        # Print details for top N most extreme cases
        print(f"\n  Top {min(max_problems, len(extreme_problems))} most extreme cases:")
        
        for i, problem in enumerate(extreme_problems[:max_problems]):
            print(f"\n    [{i+1}] Problem {problem['problem_id']} (logprob: {problem['logprob']:.4f}, tokens: {problem['token_count']})")
            print(f"      Min token logprob: {problem['min_token_logprob']:.4f}")
            print(f"      True output: '{problem['true_output']}'")
            print(f"      Generated output: '{problem['output_preview']}'")
            
            # Report problematic tokens
            if problem["problematic_tokens"]:
                print(f"      Found {len(problem['problematic_tokens'])} problematic tokens (logprob < {token_threshold}):")
                for t in problem["problematic_tokens"][:5]:  # Show up to 5 problematic tokens
                    print(f"        - Token at position {t['index']}: '{t['token']}', logprob: {t['logprob']:.4f}")
                if len(problem["problematic_tokens"]) > 5:
                    print(f"        ... and {len(problem['problematic_tokens']) - 5} more problematic tokens")
            
            # Calculate token-to-char ratio (higher might indicate lots of special tokens or non-English text)
            char_count = len(problem["full_output"])
            token_count = problem["token_count"]
            if char_count > 0:
                token_char_ratio = token_count / char_count
                if token_char_ratio > 0.5:  # Heuristic threshold
                    print(f"      WARNING: High token-to-char ratio ({token_char_ratio:.2f}), possible special token usage")
            
            # Check if the output is much longer than the true output
            true_length = len(problem["true_output"])
            if true_length > 0 and problem["output_length"] / true_length > 2:
                print(f"      WARNING: Output is {problem['output_length'] / true_length:.1f}x longer than expected")
        
        # Create visualization of extreme log probabilities vs output length
        plt.figure(figsize=(10, 6))
        plt.scatter(
            [p["logprob"] for p in extreme_problems],
            [p["output_length"] for p in extreme_problems],
            alpha=0.7
        )
        plt.xlabel('Mean Log Probability')
        plt.ylabel('Output Length (characters)')
        plt.title('Output Length vs Log Probability for Extreme Cases')
        plt.grid(linestyle='--', alpha=0.7)
        
        # Add a best fit line
        x = [p["logprob"] for p in extreme_problems]
        y = [p["output_length"] for p in extreme_problems]
        if len(x) > 1:  # Need at least 2 points for a fit
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(sorted(x), p(sorted(x)), "r--", 
                    label=f"Trend: length = {z[0]:.1f}*logprob + {z[1]:.1f}")
            plt.legend()
        
        # Create another plot for token counts
        plt.figure(figsize=(10, 6))
        plt.scatter(
            [p["logprob"] for p in extreme_problems],
            [len(p["problematic_tokens"]) for p in extreme_problems],
            alpha=0.7,
            c='red'
        )
        plt.xlabel('Mean Log Probability')
        plt.ylabel('Number of Problematic Tokens')
        plt.title('Problematic Tokens vs Log Probability')
        plt.grid(linestyle='--', alpha=0.7)
        
        # Save the plots
        output_dir = os.path.join('/share/u/yu.stev/attribution/data/cruxeval_results/k_sample', 'visualizations')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'extreme_logprob_problematic_tokens.png'))
        plt.close()
        
        # Original length vs logprob plot
        plt.figure(figsize=(10, 6))
        plt.scatter(
            [p["logprob"] for p in extreme_problems],
            [p["output_length"] for p in extreme_problems],
            alpha=0.7
        )
        plt.xlabel('Mean Log Probability')
        plt.ylabel('Output Length (characters)')
        plt.title('Output Length vs Log Probability for Extreme Cases')
        plt.grid(linestyle='--', alpha=0.7)
        
        # Add a best fit line if we have enough data points
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(sorted(x), p(sorted(x)), "r--", 
                    label=f"Trend: length = {z[0]:.1f}*logprob + {z[1]:.1f}")
            plt.legend()
            
        plt.savefig(os.path.join(output_dir, 'extreme_logprob_analysis.png'))
        plt.close()
    
    return extreme_problems


def main():
    """Run all sanity checks on the log probabilities"""
    # Set paths
    base_dir = '/share/u/yu.stev/attribution/data/cruxeval_results/k_sample'
    comparison_file = os.path.join(base_dir, 'comparison_summary.json')
    standard_file = os.path.join(base_dir, 'cruxeval_k4_temp0.3_topp0.95_base13b.json')
    quantized_file = os.path.join(base_dir, 'cruxeval_k4_temp0.3_topp0.95_base13b_8bit.json')
    
    # Load comparison summary
    print(f"Loading comparison summary from {comparison_file}")
    try:
        with open(comparison_file, 'r') as f:
            summary = json.load(f)
    except Exception as e:
        print(f"Error loading comparison summary from {comparison_file}: {e}")
        return
    
    # Run all sanity checks
    validate_logprob_ranges(summary)
    check_token_level_logprobs(standard_file, problem_ids_to_check=3)
    check_token_level_logprobs(quantized_file, problem_ids_to_check=3)
    compare_model_logprobs(summary)
    check_logprob_correctness_correlation(summary)
    check_sample_consistency(standard_file)
    analyze_logprob_thresholds(summary)
    
    # Add extreme log probability analysis
    print("\n--- Extreme Log Probability Analysis ---")
    print("Standard model:")
    std_extreme = analyze_extreme_logprobs(standard_file, threshold=-2.0)
    print("\nQuantized model:")
    quant_extreme = analyze_extreme_logprobs(quantized_file, threshold=-2.0)
    
    print("\nLog probability sanity checks complete!")


if __name__ == "__main__":
    main() 