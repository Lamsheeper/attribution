# Downloads OLMo model checkpoints from a CSV file containing checkpoint URLs.
# Provides command-line interface for downloading specific checkpoints by step number and ingredient,
# with functionality to list available checkpoints or download selected ones.

import argparse
import csv
import os
from pathlib import Path
import requests
from tqdm import tqdm
from urllib.parse import urljoin


def download_file(url, save_path, chunk_size=8192):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc=save_path.name
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def try_get_directory_listing(url):
    common_files = [
        "config.yaml",
        "train.pt",
        "model.safetensors",
        "optim.safetensors",
    ]
    found_files = []
    for pattern in common_files:
        try:
            test_url = urljoin(url.rstrip("/") + "/", pattern)
            response = requests.head(test_url)
            response.raise_for_status()
            found_files.append(pattern)
        except requests.exceptions.HTTPError as e:
            if response.status_code != 404:
                raise
        except requests.exceptions.RequestException:
            raise
    if len(found_files) <= 0:
        raise ValueError(f"No checkpoint files found at {url}")

    return found_files


def download_checkpoint(url, save_dir):
    base_path = Path(save_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {base_path}")
    available_files = try_get_directory_listing(url)

    if not available_files:
        raise ValueError("Matching files not found in directory")

    failed_files = []
    for file in available_files:
        file_url = urljoin(url.rstrip("/") + "/", file)
        file_path = base_path / file
        try:
            print(f"\nDownloading: {file}")
            download_file(file_url, file_path)
        except requests.exceptions.Timeout:
            print(f"Timeout error for {file}, retryin...")
            try:
                download_file(file_url, file_path)
            except requests.exceptions.RequestException as e:
                failed_files.append(file)
                print(f"Failed to download {file}: {e}")
        except requests.exceptions.RequestException as e:
            failed_files.append(file)
            print(f"Failed to download {file}: {e}")
    if failed_files:
        print(f"\nFAILED to download these files: {failed_files}")


def main():
    parser = argparse.ArgumentParser(description="Download OLMo checkpoints")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    download_parser = subparsers.add_parser(
        "download", help="Download checkpoints from CSV file"
    )
    download_parser.add_argument(
        "csv_file", type=str, help="Path to the CSV file containing checkpoint URLs"
    )
    download_parser.add_argument(
        "--step", type=str, required=True, help="Specific step number to download"
    )
    download_parser.add_argument(
        "--ingredient",
        type=str,
        default=None,
        help="Optional ingredient number to filter by",
    )
    download_parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints",
        help="Base directory to save downloaded checkpoints",
    )
    list_parser = subparsers.add_parser("list", help="List available checkpoint steps")
    list_parser.add_argument(
        "csv_file", type=str, help="Path to the CSV file containing checkpoint URLs"
    )
    args = parser.parse_args()

    print(f"Reading CSV file: {args.csv_file}")

    with open(args.csv_file, "r") as f:
        reader = csv.DictReader(f)
        rows = [
            (row["Ingredient"], row["Step"], row["Checkpoint Directory"])
            for row in reader
        ]

    if args.command == "list":
        print("Available steps (grouped by Ingredient):")
        grouped_steps = {}
        for ingredient, step, _ in rows:
            if ingredient not in grouped_steps:
                grouped_steps[ingredient] = []
            grouped_steps[ingredient].append(step)

        for ingredient, steps in sorted(grouped_steps.items()):
            print(f"  Ingredient {ingredient}:")
            print(f"    Steps: {', '.join(sorted(steps, key=int))}")
        return
    elif args.command == "download":
        filtered_rows = rows

        if args.step:
            filtered_rows = [
                (ing, step, url)
                for ing, step, url in filtered_rows
                if step == args.step
            ]
            if not filtered_rows:
                print(f"Error: Step {args.step} not found in the CSV file.")
                print("Use the 'list' command to see available steps.")
                return

        if args.ingredient is not None:
            filtered_rows = [
                (ing, step, url)
                for ing, step, url in filtered_rows
                if ing == args.ingredient
            ]
            if not filtered_rows:
                if args.step:
                    print(
                        f"Error: No checkpoints found for Ingredient {args.ingredient} at Step {args.step}."
                    )
                else:
                    print(
                        f"Error: Ingredient {args.ingredient} not found in the CSV file."
                    )
                print("Use the 'list' command to see available ingredients and steps.")
                return

        urls_to_download = [(step, url) for _, step, url in filtered_rows]

        print(f"Saving checkpoints to: {args.save_dir}")
        if not urls_to_download:
            print("No checkpoints match the specified criteria.")
            return

        for step, url in urls_to_download:
            print(f"\nProcessing Step {step}:")
            print(f"URL: {url}")
            save_subdir = f"step{step}"
            if args.ingredient is not None:
                save_subdir = f"ingredient{args.ingredient}_step{step}"
            save_path = os.path.join(args.save_dir, save_subdir)
            download_checkpoint(url, save_path)


if __name__ == "__main__":
    main()
