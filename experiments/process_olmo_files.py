#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OLMo File Processing Script

This script finds all files matching the OLMo pattern in a directory,
processes them using the 13_filter_generations.py script, and saves
the filtered results in a designated output folder.
"""

import os
import re
import argparse
import subprocess
from pathlib import Path
import glob


def find_olmo_files(input_dir):
    """
    Find all files matching the OLMo pattern in the given directory.
    
    Args:
        input_dir: Directory to search in
        
    Returns:
        List of file paths matching the pattern
    """
    pattern = "OLMo-2-1124-7B_stage1-step*-tokens*B.json"
    search_path = os.path.join(input_dir, pattern)
    
    files = glob.glob(search_path)
    return sorted(files)


def process_files(input_files, output_dir, filter_script):
    """
    Process each input file with the filter script and save to output directory.
    
    Args:
        input_files: List of input file paths
        output_dir: Directory to save filtered results
        filter_script: Path to the filter script
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    for input_file in input_files:
        # Get the base filename
        base_name = os.path.basename(input_file)
        
        # Extract the stage1-tokensXXXXB.json part
        match = re.search(r'(stage1-tokens\d+B\.json)', base_name)
        if match:
            output_filename = match.group(1)
        else:
            # Fallback if pattern not found
            output_filename = f"filtered_{base_name}"
        
        # Create output path
        output_file = os.path.join(output_dir, output_filename)
        
        print(f"Processing {base_name}...")
        print(f"Output will be saved as: {output_filename}")
        
        # Run the filter script
        cmd = [
            "python", filter_script,
            "--input_file", input_file,
            "--output_file", output_file
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully processed {base_name} -> {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {base_name}: {e}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process OLMo generation files with filter script")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing OLMo files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save filtered results")
    parser.add_argument("--filter_script", type=str, default="13_filter_generations.py", 
                        help="Path to filter script (default: 13_filter_generations.py)")
    args = parser.parse_args()
    
    # Find all matching files
    input_files = find_olmo_files(args.input_dir)
    
    if not input_files:
        print(f"No files matching the OLMo pattern found in {args.input_dir}")
        return
    
    print(f"Found {len(input_files)} files to process.")
    
    # Process the files
    process_files(input_files, args.output_dir, args.filter_script)
    
    print("Processing complete.")


if __name__ == "__main__":
    main() 