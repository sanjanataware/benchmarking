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
# IMPROVEMENTS:
# - Runs both linear probe and attention baseline methods
# - Better batch size (20) for GPU utilization
# - Enhanced metrics (F1, MCC, AUPRC)
# - Structural constraints enforced
# - Memory-efficient processing
#
# Usage:
#   sbatch submit_rnabert_ssp_eval.sh
#
# To run specific checkpoint or method:
#   sbatch --export=CHECKPOINT=checkpoint1 submit_rnabert_ssp_eval.sh
#   sbatch --export=CHECKPOINT=checkpoint2 submit_rnabert_ssp_eval.sh
#   sbatch --export=METHOD=attention submit_rnabert_ssp_eval.sh
#   sbatch --export=METHOD=both submit_rnabert_ssp_eval.sh
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
# Uncomment and modify as needed:
# module load cuda/11.8
# module load python/3.9

# Activate conda environment (adjust to your environment name)
# Uncomment and modify as needed:
# source activate rna_benchmark
# OR
# conda activate rna_benchmark

# Print Python and CUDA info
echo "======================================================================"
echo "Environment Information"
echo "======================================================================"
echo "Python version:"
python --version
echo ""
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""
echo "======================================================================"
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
BATCH_SIZE=20
NUM_WORKERS=8
MAX_LENGTH=2048
SEED=42
MIN_PAIR_DISTANCE=4

# Output directory
OUTPUT_DIR="./rnabert_ssp_results"

# Determine which checkpoint(s) and method(s) to run
CHECKPOINT_MODE="${CHECKPOINT:-all}"
METHOD_MODE="${METHOD:-both}"

# ==============================================================================
# Function to run evaluation
# ==============================================================================
run_evaluation() {
    local checkpoint_path="$1"
    local run_name="$2"
    local use_attention="$3"

    if [ "$use_attention" = "true" ]; then
        local method_flag="--use_attention_baseline"
        local method_name="attention"
        echo "======================================================================"
        echo "Starting evaluation: $run_name (ATTENTION BASELINE)"
        echo "Checkpoint: $checkpoint_path"
        echo "Method: Attention weights from pretrained model"
        echo "======================================================================"
    else
        local method_flag=""
        local method_name="probe"
        echo "======================================================================"
        echo "Starting evaluation: $run_name (LINEAR PROBE)"
        echo "Checkpoint: $checkpoint_path"
        echo "Method: Random linear probe on frozen embeddings"
        echo "======================================================================"
    fi

    python rnabert_ssp_eval.py \
        --model_name_or_path "$checkpoint_path" \
        --data_path "$DATA_PATH" \
        --model_type rnabert \
        --token_type single \
        --model_max_length "$MAX_LENGTH" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --seed "$SEED" \
        --min_pair_distance "$MIN_PAIR_DISTANCE" \
        --output_dir "$OUTPUT_DIR" \
        --run_name "${run_name}_${method_name}" \
        $method_flag

    local exit_code=$?

    echo ""
    echo "======================================================================"
    if [ $exit_code -eq 0 ]; then
        echo "COMPLETED SUCCESSFULLY: ${run_name}_${method_name}"
    else
        echo "FAILED: ${run_name}_${method_name} (exit code: $exit_code)"
    fi
    echo "======================================================================"
    echo ""

    return $exit_code
}

# ==============================================================================
# Function to run both methods for a checkpoint
# ==============================================================================
run_both_methods() {
    local checkpoint_path="$1"
    local base_name="$2"

    echo ""
    echo "======================================================================"
    echo "Running BOTH methods for: $base_name"
    echo "======================================================================"
    echo ""

    # Run linear probe
    run_evaluation "$checkpoint_path" "$base_name" "false"
    local exit_probe=$?

    # Run attention baseline
    run_evaluation "$checkpoint_path" "$base_name" "true"
    local exit_attn=$?

    echo "======================================================================"
    echo "Both methods completed for: $base_name"
    echo "  Linear probe exit code: $exit_probe"
    echo "  Attention baseline exit code: $exit_attn"
    echo "======================================================================"
    echo ""

    # Return 0 only if both succeeded
    if [ $exit_probe -eq 0 ] && [ $exit_attn -eq 0 ]; then
        return 0
    else
        return 1
    fi
}

# ==============================================================================
# Main execution
# ==============================================================================

echo "======================================================================"
echo "Execution Configuration"
echo "======================================================================"
echo "Checkpoint mode: $CHECKPOINT_MODE"
echo "Method mode: $METHOD_MODE"
echo "Batch size: $BATCH_SIZE"
echo "Min pair distance: $MIN_PAIR_DISTANCE"
echo "======================================================================"
echo ""

EXIT_CODE1=0
EXIT_CODE2=0
FINAL_EXIT_CODE=0

if [ "$CHECKPOINT_MODE" = "checkpoint1" ]; then
    echo "Running evaluation for Checkpoint 1 only..."
    if [ "$METHOD_MODE" = "probe" ]; then
        run_evaluation "$CHECKPOINT1" "rnabert_a16_b6_checkpoint494241" "false"
        FINAL_EXIT_CODE=$?
    elif [ "$METHOD_MODE" = "attention" ]; then
        run_evaluation "$CHECKPOINT1" "rnabert_a16_b6_checkpoint494241" "true"
        FINAL_EXIT_CODE=$?
    else
        run_both_methods "$CHECKPOINT1" "rnabert_a16_b6_checkpoint494241"
        FINAL_EXIT_CODE=$?
    fi

elif [ "$CHECKPOINT_MODE" = "checkpoint2" ]; then
    echo "Running evaluation for Checkpoint 2 only..."
    if [ "$METHOD_MODE" = "probe" ]; then
        run_evaluation "$CHECKPOINT2" "rnabert_a8_b8_checkpoint670000" "false"
        FINAL_EXIT_CODE=$?
    elif [ "$METHOD_MODE" = "attention" ]; then
        run_evaluation "$CHECKPOINT2" "rnabert_a8_b8_checkpoint670000" "true"
        FINAL_EXIT_CODE=$?
    else
        run_both_methods "$CHECKPOINT2" "rnabert_a8_b8_checkpoint670000"
        FINAL_EXIT_CODE=$?
    fi

elif [ "$CHECKPOINT_MODE" = "all" ]; then
    echo "Running evaluation for BOTH checkpoints..."
    echo ""

    # Checkpoint 1: 16 attention heads, 6 layers
    if [ "$METHOD_MODE" = "probe" ]; then
        run_evaluation "$CHECKPOINT1" "rnabert_a16_b6_checkpoint494241" "false"
        EXIT_CODE1=$?
    elif [ "$METHOD_MODE" = "attention" ]; then
        run_evaluation "$CHECKPOINT1" "rnabert_a16_b6_checkpoint494241" "true"
        EXIT_CODE1=$?
    else
        run_both_methods "$CHECKPOINT1" "rnabert_a16_b6_checkpoint494241"
        EXIT_CODE1=$?
    fi

    # Checkpoint 2: 8 attention heads, 8 layers
    if [ "$METHOD_MODE" = "probe" ]; then
        run_evaluation "$CHECKPOINT2" "rnabert_a8_b8_checkpoint670000" "false"
        EXIT_CODE2=$?
    elif [ "$METHOD_MODE" = "attention" ]; then
        run_evaluation "$CHECKPOINT2" "rnabert_a8_b8_checkpoint670000" "true"
        EXIT_CODE2=$?
    else
        run_both_methods "$CHECKPOINT2" "rnabert_a8_b8_checkpoint670000"
        EXIT_CODE2=$?
    fi

    echo "======================================================================"
    echo "ALL EVALUATIONS COMPLETE"
    echo "======================================================================"
    echo "Checkpoint 1 exit code: $EXIT_CODE1"
    echo "Checkpoint 2 exit code: $EXIT_CODE2"
    echo ""
    echo "Results saved in: $OUTPUT_DIR/results/"
    echo ""
    echo "To view results:"
    echo "  # Linear probe results:"
    echo "  cat $OUTPUT_DIR/results/rnabert_a16_b6_checkpoint494241_probe/zero_shot_results.json"
    echo "  cat $OUTPUT_DIR/results/rnabert_a8_b8_checkpoint670000_probe/zero_shot_results.json"
    echo ""
    echo "  # Attention baseline results:"
    echo "  cat $OUTPUT_DIR/results/rnabert_a16_b6_checkpoint494241_attention/zero_shot_results.json"
    echo "  cat $OUTPUT_DIR/results/rnabert_a8_b8_checkpoint670000_attention/zero_shot_results.json"
    echo "======================================================================"

    # Return non-zero if any evaluation failed
    if [ $EXIT_CODE1 -ne 0 ] || [ $EXIT_CODE2 -ne 0 ]; then
        FINAL_EXIT_CODE=1
    else
        FINAL_EXIT_CODE=0
    fi

else
    echo "ERROR: Unknown checkpoint option: $CHECKPOINT_MODE"
    echo "Valid options: checkpoint1, checkpoint2, all"
    echo ""
    echo "Usage examples:"
    echo "  sbatch submit_rnabert_ssp_eval.sh"
    echo "  sbatch --export=CHECKPOINT=checkpoint1 submit_rnabert_ssp_eval.sh"
    echo "  sbatch --export=CHECKPOINT=checkpoint2 submit_rnabert_ssp_eval.sh"
    echo "  sbatch --export=METHOD=probe submit_rnabert_ssp_eval.sh"
    echo "  sbatch --export=METHOD=attention submit_rnabert_ssp_eval.sh"
    echo "  sbatch --export=CHECKPOINT=checkpoint1,METHOD=probe submit_rnabert_ssp_eval.sh"
    FINAL_EXIT_CODE=1
fi

# Print job completion info
echo ""
echo "======================================================================"
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "Exit code: $FINAL_EXIT_CODE"
echo "======================================================================"

exit $FINAL_EXIT_CODE