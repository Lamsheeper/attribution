#!/bin/bash

# Create output directory if it doesn't exist
OUTPUT_DIR="../data/cruxeval_results_base_stage2"
mkdir -p $OUTPUT_DIR

MODEL="allenai/OLMo-2-1124-13B"
# LOAD_FROM="/share/u/yu.stev/.cache/huggingface/hub/models--allenai--OLMo-2-1124-13B/ snapshots/"
PERMANENT_CACHE="/share/datasets/huggingface_hub"
HF_CACHE="${HOME}/.cache/huggingface"

# Function to clear HuggingFace cache
clear_hf_cache() {
    echo "Clearing HuggingFace cache..."
    # Only remove revision-specific files from the default cache
    find "${HF_CACHE}" -type d -name "*stage*" -exec rm -rf {} +
    echo "HuggingFace cache cleared"
}

# Function to backup model to permanent storage
backup_to_permanent() {
    local revision=$1
    echo "Backing up model revision $revision to permanent storage..."
    
    # Create model-specific directory in permanent storage
    local model_dir="${PERMANENT_CACHE}/models--${MODEL//\//-}/snapshots"
    mkdir -p "$model_dir"
    
    # Copy revision files to permanent storage if they exist in cache
    if [ -d "${HF_CACHE}/models--${MODEL//\//-}/snapshots/$revision" ]; then
        cp -r "${HF_CACHE}/models--${MODEL//\//-}/snapshots/$revision" "$model_dir/"
        echo "Backed up revision $revision to permanent storage"
    fi
}

# Function to check if a revision has already been evaluated
has_results() {
    local revision=$1
    local result_file="$OUTPUT_DIR/cruxeval_results_${MODEL//\//_}_${revision}.json"
    if [ -f "$result_file" ]; then
        echo "Found existing results for revision $revision at $result_file"
        return 0
    else
        return 1
    fi
}

# Function to process a batch of revisions
process_batch() {
    local start=$1
    local end=$2
    local batch_num=$3
    
    echo "Processing batch $batch_num (revisions $start to $end)"
    
    # Process each revision in the batch
    for ((i=start; i<=end && i<${#REVISIONS[@]}; i++)); do
        revision="${REVISIONS[$i]}"
        echo "====================================================="
        echo "Evaluating revision: $revision ($(($i + 1))/${#REVISIONS[@]})"
        echo "====================================================="
        
        # Skip if results already exist
        if has_results "$revision"; then
            echo "Skipping revision $revision - results already exist"
            continue
        fi
        
        # Run the evaluation script
        TRANSFORMERS_CACHE="${HF_CACHE}" 
        uv run --active 07_evaluate_cruxeval.py \
            --model "$MODEL" \
            --revision "$revision" \
            --num_problems 800 \
            --batch_size 512 \
            --output_dir "$OUTPUT_DIR"
        
        # Check if evaluation was successful
        if [ $? -eq 0 ]; then
            echo "Completed evaluation for $MODEL, revision $revision"
            # Backup successful evaluation to permanent storage
            backup_to_permanent "$revision"
        else
            echo "Error evaluating $MODEL, revision $revision"
        fi
        
        echo "====================================================="
        echo ""
        
        # Clear CUDA cache
        uv run --active python -c "import torch; torch.cuda.empty_cache()"
    done
    
    # Clear HuggingFace cache after batch
    clear_hf_cache
}

# NOTE: Revisions ordered by ingredient > stage > tokens
REVISIONS=(
    # Ingredient 1
    # Stage 1 (ascending by tokens)
    "stage1-step593000-tokens4975B"
    "stage1-step594000-tokens4983B"
    "stage1-step595000-tokens4992B"
    "stage1-step596000-tokens5000B"
    "stage1-step596057-tokens5001B"
    # Stage 2 (ascending by tokens/steps)
    "stage2-ingredient1-step1000-tokens9B"
    "stage2-ingredient1-step2000-tokens17B"
    "stage2-ingredient1-step3000-tokens26B"
    "stage2-ingredient1-step4000-tokens34B"
    "stage2-ingredient1-step5000-tokens42B"
    "stage2-ingredient1-step6000-tokens51B"
    "stage2-ingredient1-step7000-tokens59B"
    "stage2-ingredient1-step8000-tokens68B"
    "stage2-ingredient1-step9000-tokens76B"
    "stage2-ingredient1-step10000-tokens84B"
    "stage2-ingredient1-step11000-tokens93B"
    "stage2-ingredient1-step11931-tokens100B"
    "stage2-ingredient1-step11931-tokens101B"

    # # Ingredient 2
    # # Stage 2 (ascending by tokens/steps)
    # "stage2-ingredient2-step2000-tokens17B"
    # "stage2-ingredient2-step3000-tokens26B"
    # "stage2-ingredient2-step4000-tokens34B"
    # "stage2-ingredient2-step6000-tokens51B"
    # "stage2-ingredient2-step7000-tokens59B"
    # "stage2-ingredient2-step8000-tokens68B"
    # "stage2-ingredient2-step9000-tokens76B"
    # "stage2-ingredient2-step11000-tokens93B"
    # "stage2-ingredient2-step11931-tokens100B"
    # "stage2-ingredient2-step11931-tokens101B"

    # # Ingredient 3
    # # Stage 2 (ascending by tokens/steps)
    # "stage2-ingredient3-step1000-tokens9B"
    # "stage2-ingredient3-step2000-tokens17B"
    # "stage2-ingredient3-step3000-tokens26B"
    # "stage2-ingredient3-step4000-tokens34B"
    # "stage2-ingredient3-step6000-tokens51B"
    # "stage2-ingredient3-step7000-tokens59B"
    # "stage2-ingredient3-step8000-tokens68B"
    # "stage2-ingredient3-step9000-tokens76B"
    # "stage2-ingredient3-step10000-tokens84B"
    # "stage2-ingredient3-step11000-tokens93B"
    # "stage2-ingredient3-step11931-tokens100B"
    # "stage2-ingredient3-step11931-tokens101B"

    # # Ingredient 4
    # # Stage 2 (ascending by tokens/steps)
    # "stage2-ingredient4-step1000-tokens9B"
    # "stage2-ingredient4-step2000-tokens17B"
    # "stage2-ingredient4-step3000-tokens26B"
    # "stage2-ingredient4-step4000-tokens34B"
    # "stage2-ingredient4-step5000-tokens42B"
    # "stage2-ingredient4-step6000-tokens51B"
    # "stage2-ingredient4-step7000-tokens59B"
    # "stage2-ingredient4-step8000-tokens68B"
    # "stage2-ingredient4-step9000-tokens76B"
    # "stage2-ingredient4-step35773-tokens300B"
)

# Batch size (number of revisions per batch)
BATCH_SIZE=5  # Reduced batch size for 13B model due to memory constraints

# Checkpoint file to track progress
CHECKPOINT_FILE="$OUTPUT_DIR/batch_checkpoint.txt"

# Get the starting batch from checkpoint if it exists
START_BATCH=0
if [ -f "$CHECKPOINT_FILE" ]; then
    START_BATCH=$(cat "$CHECKPOINT_FILE")
    echo "Resuming from batch $START_BATCH"
fi

# Calculate total number of batches
TOTAL_REVISIONS=${#REVISIONS[@]}
TOTAL_BATCHES=$(( (TOTAL_REVISIONS + BATCH_SIZE - 1) / BATCH_SIZE ))

# Loop through each batch
for ((batch=START_BATCH; batch<TOTAL_BATCHES; batch++)); do
    start=$((batch * BATCH_SIZE))
    end=$((start + BATCH_SIZE - 1))
    
    echo "Processing batch $((batch + 1)) of $TOTAL_BATCHES"
    process_batch $start $end $((batch + 1))
    
    # Save checkpoint
    echo $((batch + 1)) > "$CHECKPOINT_FILE"
done

# Remove checkpoint file after successful completion
rm -f "$CHECKPOINT_FILE"

echo "All evaluations complete!"
echo "Results are saved in $OUTPUT_DIR"
echo "Done!" 