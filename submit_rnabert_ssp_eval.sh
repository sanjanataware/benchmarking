#!/bin/bash
#SBATCH --job-name=rnabert_ssp_eval
#SBATCH --output=logs/rnabert_ssp_eval_%j.out
#SBATCH --error=logs/rnabert_ssp_eval_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# ==============================================================================
# SLURM Job Script for RNABERT SSP Evaluation on Chimera
#
# This script evaluates RNABERT checkpoints on the Secondary Structure
# Prediction benchmark without fine-tuning (zero-shot with linear probe).
#
# Usage:
#   sbatch submit_rnabert_ssp_eval.sh
#
# To run specific checkpoint:
#   sbatch --export=CHECKPOINT=checkpoint1 submit_rnabert_ssp_eval.sh
#   sbatch --export=CHECKPOINT=checkpoint2 submit_rnabert_ssp_eval.sh
# ==============================================================================

# Print job information
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "======================================================================"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules (adjust based on chimera's module system)
# module load cuda/11.8
# module load python/3.9

# Activate conda environment (adjust to your environment name)
# source activate rna_benchmark
# OR
# conda activate rna_benchmark

# Print Python and CUDA info
echo "Python version:"
python --version
echo ""
echo "CUDA devices:"
nvidia-smi
echo ""

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Checkpoint paths
CHECKPOINT1="/large_storage/goodarzilab/public/model_checkpoints/RNABERT/RNABERT_L2048_l20_a16_b6_r1e-05_mlm0.2_wd0.01_g4-rjiphp2y/checkpoint-494241"
CHECKPOINT2="/large_storage/goodarzilab/public/model_checkpoints/RNABERT/RNABERT_L2048_l20_a8_b8_r1e-05_mlm0.2_wd0.01_g4-g7p6hpqj/checkpoint-670000"

# Data path
DATA_PATH="/home/ali/DMS/DMS-FM/Downstream_Tasks/RNABenchmark/data/Secondary_structure_prediction"

# Evaluation parameters
BATCH_SIZE=16  # Increase batch size for GPU
NUM_WORKERS=8
MAX_LENGTH=2048
SEED=42

# Output directory
OUTPUT_DIR="./rnabert_ssp_results"

# Determine which checkpoint to run
CHECKPOINT_MODE=${CHECKPOINT:-"all"}

# ==============================================================================
# Function to run evaluation
# ==============================================================================
run_evaluation() {
    local checkpoint_path=$1
    local run_name=$2

    echo "======================================================================"
    echo "Starting evaluation: $run_name"
    echo "Checkpoint: $checkpoint_path"
    echo "======================================================================"

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

    local exit_code=$?

    echo ""
    echo "======================================================================"
    if [ $exit_code -eq 0 ]; then
        echo "COMPLETED SUCCESSFULLY: $run_name"
    else
        echo "FAILED: $run_name (exit code: $exit_code)"
    fi
    echo "======================================================================"
    echo ""

    return $exit_code
}

# ==============================================================================
# Main execution
# ==============================================================================

case $CHECKPOINT_MODE in
    "checkpoint1")
        echo "Running evaluation for Checkpoint 1 only..."
        run_evaluation "$CHECKPOINT1" "rnabert_a16_b6_checkpoint494241"
        FINAL_EXIT_CODE=$?
        ;;
    "checkpoint2")
        echo "Running evaluation for Checkpoint 2 only..."
        run_evaluation "$CHECKPOINT2" "rnabert_a8_b8_checkpoint670000"
        FINAL_EXIT_CODE=$?
        ;;
    "all")
        echo "Running evaluation for both checkpoints..."
        echo ""

        # Checkpoint 1: 16 attention heads, 6 layers
        run_evaluation "$CHECKPOINT1" "rnabert_a16_b6_checkpoint494241"
        EXIT_CODE1=$?

        # Checkpoint 2: 8 attention heads, 8 layers
        run_evaluation "$CHECKPOINT2" "rnabert_a8_b8_checkpoint670000"
        EXIT_CODE2=$?

        echo "======================================================================"
        echo "ALL EVALUATIONS COMPLETE"
        echo "======================================================================"
        echo "Checkpoint 1 exit code: $EXIT_CODE1"
        echo "Checkpoint 2 exit code: $EXIT_CODE2"
        echo ""
        echo "Results saved in: $OUTPUT_DIR/results/"
        echo ""
        echo "To view results:"
        echo "  cat $OUTPUT_DIR/results/rnabert_a16_b6_checkpoint494241/zero_shot_results.json"
        echo "  cat $OUTPUT_DIR/results/rnabert_a8_b8_checkpoint670000/zero_shot_results.json"
        echo "======================================================================"

        # Return non-zero if any evaluation failed
        if [ $EXIT_CODE1 -ne 0 ] || [ $EXIT_CODE2 -ne 0 ]; then
            FINAL_EXIT_CODE=1
        else
            FINAL_EXIT_CODE=0
        fi
        ;;
    *)
        echo "Unknown option: $CHECKPOINT_MODE"
        echo "Usage: sbatch [--export=CHECKPOINT=checkpoint1|checkpoint2|all] $0"
        FINAL_EXIT_CODE=1
        ;;
esac

# Print job completion info
echo ""
echo "======================================================================"
echo "Job completed at: $(date)"
echo "Exit code: $FINAL_EXIT_CODE"
echo "======================================================================"

exit $FINAL_EXIT_CODE
