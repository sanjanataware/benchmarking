#!/bin/bash

# ==============================================================================
# Script to evaluate both RNABERT checkpoints on SSP benchmark
#
# This script runs zero-shot evaluation (no fine-tuning) on the BEACON
# Secondary Structure Prediction benchmark for both RNABERT models.
#
# Usage:
#   bash run_rnabert_ssp_eval.sh
#
# Or run individual checkpoints:
#   bash run_rnabert_ssp_eval.sh checkpoint1
#   bash run_rnabert_ssp_eval.sh checkpoint2
# ==============================================================================

# Checkpoint paths
CHECKPOINT1="/large_storage/goodarzilab/public/model_checkpoints/RNABERT/RNABERT_L2048_l20_a16_b6_r1e-05_mlm0.2_wd0.01_g4-rjiphp2y/checkpoint-494241"
CHECKPOINT2="/large_storage/goodarzilab/public/model_checkpoints/RNABERT/RNABERT_L2048_l20_a8_b8_r1e-05_mlm0.2_wd0.01_g4-g7p6hpqj/checkpoint-670000"

# Data path
DATA_PATH="/home/ali/DMS/DMS-FM/Downstream_Tasks/RNABenchmark/data/Secondary_structure_prediction"

# Evaluation parameters
BATCH_SIZE=8
NUM_WORKERS=4
MAX_LENGTH=2048
SEED=42

# Output directory
OUTPUT_DIR="./rnabert_ssp_results"

# ==============================================================================
# Function to run evaluation
# ==============================================================================
run_evaluation() {
    local checkpoint_path=$1
    local run_name=$2

    echo "========================================================================"
    echo "Starting evaluation: $run_name"
    echo "Checkpoint: $checkpoint_path"
    echo "========================================================================"

    python rnabert_ssp_eval.py \
        --model_name_or_path "$checkpoint_path" \
        --data_path "$DATA_PATH" \
        --model_type rnabert \
        --token_type single \
        --model_max_length $MAX_LENGTH \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --seed $SEED \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$run_name"

    echo ""
    echo "========================================================================"
    echo "Completed: $run_name"
    echo "========================================================================"
    echo ""
}

# ==============================================================================
# Main execution
# ==============================================================================

# Check which checkpoint(s) to run
MODE=${1:-"all"}

case $MODE in
    "checkpoint1")
        echo "Running evaluation for Checkpoint 1 only..."
        run_evaluation "$CHECKPOINT1" "rnabert_a16_b6_checkpoint494241"
        ;;
    "checkpoint2")
        echo "Running evaluation for Checkpoint 2 only..."
        run_evaluation "$CHECKPOINT2" "rnabert_a8_b8_checkpoint670000"
        ;;
    "all")
        echo "Running evaluation for both checkpoints..."
        echo ""

        # Checkpoint 1: 16 attention heads, 6 layers
        run_evaluation "$CHECKPOINT1" "rnabert_a16_b6_checkpoint494241"

        # Checkpoint 2: 8 attention heads, 8 layers
        run_evaluation "$CHECKPOINT2" "rnabert_a8_b8_checkpoint670000"

        echo "========================================================================"
        echo "ALL EVALUATIONS COMPLETE"
        echo "========================================================================"
        echo "Results saved in: $OUTPUT_DIR/results/"
        echo ""
        echo "To view results:"
        echo "  cat $OUTPUT_DIR/results/rnabert_a16_b6_checkpoint494241/zero_shot_results.json"
        echo "  cat $OUTPUT_DIR/results/rnabert_a8_b8_checkpoint670000/zero_shot_results.json"
        echo "========================================================================"
        ;;
    *)
        echo "Unknown option: $MODE"
        echo "Usage: $0 [checkpoint1|checkpoint2|all]"
        exit 1
        ;;
esac
