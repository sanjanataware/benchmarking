#!/bin/bash


# Checkpoint paths
CHECKPOINT1="/large_storage/goodarzilab/public/model_checkpoints/RNABERT/RNABERT_L2048_l20_a16_b6_r1e-05_mlm0.2_wd0.01_g4-rjiphp2y/checkpoint-494241"
CHECKPOINT2="/large_storage/goodarzilab/public/model_checkpoints/RNABERT/RNABERT_L2048_l20_a8_b8_r1e-05_mlm0.2_wd0.01_g4-g7p6hpqj/checkpoint-670000"


# Use checkpoint 1's tokenizer for both (they have same vocab)
TOKENIZER_PATH="/large_storage/goodarzilab/public/model_checkpoints/RNABERT/RNABERT_L2048_l20_a16_b6_r1e-05_mlm0.2_wd0.01_g4-rjiphp2y/checkpoint-494241"


# Data path - FIXED: Added /bpRNA subdirectory
DATA_PATH="/home/ali/DMS/DMS-FM/Downstream_Tasks/RNABenchmark/data/Secondary_structure_prediction/bpRNA"


# Training parameters
EPOCHS=5
LEARNING_RATE=1e-3
WEIGHT_DECAY=0.01
PATIENCE=5


# Evaluation parameters
BATCH_SIZE=8
NUM_WORKERS=4
MAX_LENGTH=2048
SEED=42


# Output directory
OUTPUT_DIR="./rnabert_ssp_results"


run_training() {
   local checkpoint_path=$1
   local run_name=$2


   echo "========================================================================"
   echo "Starting training and evaluation: $run_name"
   echo "Checkpoint: $checkpoint_path"
   echo "Tokenizer: $TOKENIZER_PATH"
   echo "========================================================================"


   python rnabert_ssp_eval.py \
       --model_name_or_path "$checkpoint_path" \
       --pretrained_lm_dir "$TOKENIZER_PATH" \
       --data_path "$DATA_PATH" \
       --model_type rnabert \
       --token_type single \
       --model_max_length $MAX_LENGTH \
       --batch_size $BATCH_SIZE \
       --num_workers $NUM_WORKERS \
       --epochs $EPOCHS \
       --learning_rate $LEARNING_RATE \
       --weight_decay $WEIGHT_DECAY \
       --patience $PATIENCE \
       --seed $SEED \
       --output_dir "$OUTPUT_DIR" \
       --run_name "$run_name"


   echo ""
   echo "========================================================================"
   echo "Completed: $run_name"
   echo "========================================================================"
   echo ""
}


# Main execution
MODE=${1:-"all"}


case $MODE in
   "checkpoint1")
       echo "Training with Checkpoint 1 only..."
       run_training "$CHECKPOINT1" "rnabert_a16_b6_checkpoint494241"
       ;;
   "checkpoint2")
       echo "Training with Checkpoint 2 only..."
       run_training "$CHECKPOINT2" "rnabert_a8_b8_checkpoint670000"
       ;;
   "all")
       echo "Training with both checkpoints..."
       echo ""
      
       run_training "$CHECKPOINT1" "rnabert_a16_b6_checkpoint494241"
       run_training "$CHECKPOINT2" "rnabert_a8_b8_checkpoint670000"


       echo "========================================================================"
       echo "ALL TRAINING AND EVALUATIONS COMPLETE"
       echo "========================================================================"
       ;;
   *)
       echo "Unknown option: $MODE"
       echo "Usage: $0 [checkpoint1|checkpoint2|all]"
       exit 1
       ;;
esac
