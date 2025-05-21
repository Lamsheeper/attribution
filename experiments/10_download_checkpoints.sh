#!/bin/bash

# Define variables
CSV_URL="https://raw.githubusercontent.com/allenai/OLMo/main/configs/official-1124/OLMo-2-1124-13B-stage2.csv"
LOCAL_CSV_FILE="olmo_13b_stage2_checkpoints.csv"
TARGET_STEP="2000" # <<< Replace with the desired step number
INGREDIENT="1"
SAVE_DIR="$HOME/.cache/huggingface/hub/models--allenai--OLMo-2-1124-13B"

# Ensure the target directory exists
mkdir -p "$SAVE_DIR"

# Download the CSV file if it doesn't exist or is empty
if [ ! -s "$LOCAL_CSV_FILE" ]; then
  echo "Downloading checkpoint list from $CSV_URL..."
  curl -L -o "$LOCAL_CSV_FILE" "$CSV_URL"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to download CSV file."
    exit 1
  fi
else
  echo "Using existing local CSV file: $LOCAL_CSV_FILE"
fi


# Check if the CSV file was downloaded successfully
if [ ! -f "$LOCAL_CSV_FILE" ]; then
    echo "Error: Local CSV file not found at $LOCAL_CSV_FILE"
    exit 1
fi

# Run the python script with corrected arguments
echo "Running checkpoint download script for step $TARGET_STEP..."
uv run 10_download_checkpoints.py download "$LOCAL_CSV_FILE" --step "$TARGET_STEP" --ingredient "$INGREDIENT" --save-dir "$SAVE_DIR"

echo "Script finished."

# Optional: Clean up the downloaded CSV file
# rm "$LOCAL_CSV_FILE"
# echo "Removed temporary CSV file: $LOCAL_CSV_FILE" 