# RNABERT Secondary Structure Prediction Evaluation

This directory contains scripts for evaluating RNABERT models on the BEACON Secondary Structure Prediction (SSP) benchmark **without fine-tuning** (zero-shot evaluation).

## Overview

**Task:** Secondary Structure Prediction (SSP)
**Benchmark:** BEACON bpRNA dataset
**Evaluation Method:** Zero-shot with linear probe (randomly initialized, not trained)
**Models:** Two RNABERT checkpoints with different architectures

## Files

- `rnabert_ssp_eval.py` - Main evaluation script
- `run_rnabert_ssp_eval.sh` - Bash script to run evaluations locally
- `submit_rnabert_ssp_eval.sh` - SLURM job script for chimera cluster

## RNABERT Checkpoints on Chimera

### Checkpoint 1: RNABERT (16 heads, 6 layers)
```
/large_storage/goodarzilab/public/model_checkpoints/RNABERT/RNABERT_L2048_l20_a16_b6_r1e-05_mlm0.2_wd0.01_g4-rjiphp2y/checkpoint-494241
```
- Max length: 2048
- Layers: 20
- Attention heads: 16
- Hidden dim per head: 6
- Training step: 494,241

### Checkpoint 2: RNABERT (8 heads, 8 layers)
```
/large_storage/goodarzilab/public/model_checkpoints/RNABERT/RNABERT_L2048_l20_a8_b8_r1e-05_mlm0.2_wd0.01_g4-g7p6hpqj/checkpoint-670000
```
- Max length: 2048
- Layers: 20
- Attention heads: 8
- Hidden dim per head: 8
- Training step: 670,000

## BEACON Data Location

```
/home/ali/DMS/DMS-FM/Downstream_Tasks/RNABenchmark/data/Secondary_structure_prediction
```

**Data structure:**
```
Secondary_structure_prediction/
├── bpRNA.csv          # Metadata (sequence IDs, splits)
├── TR0/               # Training set structure matrices (.npy files)
├── VL0/               # Validation set structure matrices (.npy files)
└── TS0/               # Test set structure matrices (.npy files)
```

## Usage

### Option 1: Submit SLURM Job on Chimera (Recommended)

```bash
# Make the script executable
chmod +x submit_rnabert_ssp_eval.sh

# Create logs directory
mkdir -p logs

# Submit job to evaluate both checkpoints
sbatch submit_rnabert_ssp_eval.sh

# OR evaluate specific checkpoint only
sbatch --export=CHECKPOINT=checkpoint1 submit_rnabert_ssp_eval.sh
sbatch --export=CHECKPOINT=checkpoint2 submit_rnabert_ssp_eval.sh
```

Check job status:
```bash
squeue -u $USER
```

View output:
```bash
tail -f logs/rnabert_ssp_eval_<JOB_ID>.out
tail -f logs/rnabert_ssp_eval_<JOB_ID>.err
```

### Option 2: Run Locally/Interactively

```bash
# Make the script executable
chmod +x run_rnabert_ssp_eval.sh

# Run both checkpoints
bash run_rnabert_ssp_eval.sh

# OR run specific checkpoint
bash run_rnabert_ssp_eval.sh checkpoint1
bash run_rnabert_ssp_eval.sh checkpoint2
```

### Option 3: Run Python Script Directly

```bash
# Checkpoint 1
python rnabert_ssp_eval.py \
    --model_name_or_path /large_storage/goodarzilab/public/model_checkpoints/RNABERT/RNABERT_L2048_l20_a16_b6_r1e-05_mlm0.2_wd0.01_g4-rjiphp2y/checkpoint-494241 \
    --data_path /home/ali/DMS/DMS-FM/Downstream_Tasks/RNABenchmark/data/Secondary_structure_prediction \
    --batch_size 8 \
    --run_name rnabert_a16_b6_checkpoint494241

# Checkpoint 2
python rnabert_ssp_eval.py \
    --model_name_or_path /large_storage/goodarzilab/public/model_checkpoints/RNABERT/RNABERT_L2048_l20_a8_b8_r1e-05_mlm0.2_wd0.01_g4-g7p6hpqj/checkpoint-670000 \
    --data_path /home/ali/DMS/DMS-FM/Downstream_Tasks/RNABenchmark/data/Secondary_structure_prediction \
    --batch_size 8 \
    --run_name rnabert_a8_b8_checkpoint670000
```

## Configuration Options

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name_or_path` | checkpoint1 | Path to RNABERT checkpoint |
| `--data_path` | BEACON data | Path to bpRNA dataset |
| `--model_type` | `rnabert` | Model type |
| `--token_type` | `single` | Tokenization (single nucleotide) |
| `--model_max_length` | `2048` | Maximum sequence length |
| `--batch_size` | `8` | Batch size for evaluation |
| `--num_workers` | `4` | Data loading workers |
| `--seed` | `42` | Random seed |
| `--output_dir` | `./rnabert_ssp_results` | Output directory |
| `--run_name` | auto-generated | Name for this run |

### Advanced Options

```bash
python rnabert_ssp_eval.py \
    --model_name_or_path <path> \
    --data_path <path> \
    --batch_size 16 \          # Larger batch size for GPU
    --num_workers 8 \           # More workers for faster loading
    --model_max_length 2048 \   # Adjust if needed
    --seed 123 \                # Different random seed
    --output_dir ./custom_results
```

## Output

### Results Directory Structure

```
rnabert_ssp_results/
└── results/
    ├── rnabert_a16_b6_checkpoint494241/
    │   └── zero_shot_results.json
    └── rnabert_a8_b8_checkpoint670000/
        └── zero_shot_results.json
```

### Results JSON Format

```json
{
    "task": "secondary_structure_prediction",
    "evaluation_method": "zero_shot_linear_probe",
    "model_name_or_path": "/path/to/checkpoint",
    "model_type": "rnabert",
    "token_type": "single",
    "hidden_size": 120,
    "num_layers": 20,
    "num_heads": 16,
    "max_length": 2048,
    "batch_size": 8,
    "seed": 42,
    "timestamp": "2025-01-15T10:30:00",
    "validation": {
        "precision": 0.xxxx,
        "recall": 0.xxxx,
        "f1": 0.xxxx
    },
    "test": {
        "precision": 0.xxxx,
        "recall": 0.xxxx,
        "f1": 0.xxxx
    }
}
```

### View Results

```bash
# Pretty-print JSON results
cat rnabert_ssp_results/results/rnabert_a16_b6_checkpoint494241/zero_shot_results.json | python -m json.tool

cat rnabert_ssp_results/results/rnabert_a8_b8_checkpoint670000/zero_shot_results.json | python -m json.tool
```

## Evaluation Metrics

The script reports three metrics for both validation and test sets:

- **Precision:** Of predicted base pairs, how many are correct?
- **Recall:** Of true base pairs, how many were predicted?
- **F1 Score:** Harmonic mean of precision and recall (primary metric)

All metrics are calculated on base pair predictions with a threshold of 0.5.

## How It Works

### Zero-Shot Evaluation Pipeline

1. **Load RNABERT:** Load pretrained RNABERT model (frozen, no updates)
2. **Extract Embeddings:** Get contextualized embeddings for each nucleotide
3. **Linear Probe:** Use a randomly initialized linear layer to predict base pairing
4. **Evaluate:** Calculate metrics on validation and test sets

### Why Zero-Shot?

This evaluation tests the **quality of pretrained representations** without any task-specific training. Good zero-shot performance indicates that RNABERT has learned useful structural information during pretraining.

### Linear Probe Details

```
Input:  [embedding_i, embedding_j] (concatenated) - size: 2 * hidden_size
Output: pairing_probability - size: 1
Status: Randomly initialized (NOT trained)
```

For each pair of positions (i, j):
- Concatenate their RNABERT embeddings
- Pass through linear layer
- Output: probability that positions i and j form a base pair

## Expected Runtime

- **Per checkpoint:** ~30-60 minutes on a single GPU (depends on GPU type)
- **Both checkpoints:** ~1-2 hours total

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   ```bash
   # Reduce batch size
   python rnabert_ssp_eval.py --batch_size 4
   ```

2. **Data path not found**
   ```bash
   # Check if data exists
   ls /home/ali/DMS/DMS-FM/Downstream_Tasks/RNABenchmark/data/Secondary_structure_prediction
   ```

3. **Model checkpoint not found**
   ```bash
   # Verify checkpoint exists
   ls /large_storage/goodarzilab/public/model_checkpoints/RNABERT/
   ```

4. **Import errors**
   ```bash
   # Make sure you're in the correct directory
   cd /path/to/RNABenchmark

   # Check if required modules exist
   ls downstream/structure/
   ```

### Dependencies

Required Python packages:
- `torch`
- `transformers`
- `accelerate`
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `tqdm`

Install via:
```bash
pip install torch transformers accelerate numpy pandas scikit-learn scipy tqdm
```

## Comparing Results

After running both checkpoints, compare performance:

```bash
# Extract F1 scores
echo "Checkpoint 1 (16 heads, 6 layers):"
cat rnabert_ssp_results/results/rnabert_a16_b6_checkpoint494241/zero_shot_results.json | grep -A 3 '"test"'

echo "Checkpoint 2 (8 heads, 8 layers):"
cat rnabert_ssp_results/results/rnabert_a8_b8_checkpoint670000/zero_shot_results.json | grep -A 3 '"test"'
```

## References

- **RNABERT:** [Original paper/repository]
- **BEACON Benchmark:** Benchmark for evaluating RNA language models on downstream tasks
- **bpRNA Dataset:** Secondary structure prediction benchmark dataset

## Contact

For questions or issues:
- Check SLURM logs in `logs/` directory
- Review error messages in `.err` files
- Verify all paths are correct for chimera environment

## Notes

- The evaluation is **completely zero-shot** - no gradient updates occur
- The linear probe is **randomly initialized** each run (controlled by seed)
- Results may vary slightly due to random initialization of the probe
- GPU is recommended but not required (will be slower on CPU)
- Multi-GPU is supported via Accelerate (automatically detected)
