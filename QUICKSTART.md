# Quick Start Guide - RNABERT SSP Evaluation

## TL;DR

Evaluate both RNABERT checkpoints on the Secondary Structure Prediction benchmark with zero-shot (no fine-tuning):

```bash
# On chimera cluster
sbatch submit_rnabert_ssp_eval.sh

# Check results
python compare_rnabert_results.py
```

## Step-by-Step Instructions

### 1. Set Up Environment

```bash
# Navigate to RNABenchmark directory
cd /path/to/RNABenchmark

# (Optional) Activate conda environment if needed
# conda activate your_rna_env
```

### 2. Run Evaluation

**Option A: SLURM (Recommended for chimera)**
```bash
# Create logs directory
mkdir -p logs

# Submit job
sbatch submit_rnabert_ssp_eval.sh

# Monitor job
squeue -u $USER
tail -f logs/rnabert_ssp_eval_*.out
```

**Option B: Interactive/Local**
```bash
bash run_rnabert_ssp_eval.sh
```

**Option C: Single Checkpoint**
```bash
# Just checkpoint 1
bash run_rnabert_ssp_eval.sh checkpoint1

# Just checkpoint 2
bash run_rnabert_ssp_eval.sh checkpoint2
```

### 3. Check Results

```bash
# Compare both checkpoints
python compare_rnabert_results.py

# View individual results
cat rnabert_ssp_results/results/rnabert_a16_b6_checkpoint494241/zero_shot_results.json
cat rnabert_ssp_results/results/rnabert_a8_b8_checkpoint670000/zero_shot_results.json
```

## What Gets Evaluated?

| Checkpoint | Architecture | Location |
|------------|-------------|----------|
| Checkpoint 1 | 20 layers, 16 heads | `/large_storage/.../checkpoint-494241` |
| Checkpoint 2 | 20 layers, 8 heads | `/large_storage/.../checkpoint-670000` |

**Dataset:** BEACON bpRNA (Secondary Structure Prediction)
- Validation set: ~400 sequences
- Test set: ~400 sequences

**Method:** Zero-shot with linear probe
- RNABERT is frozen (no training)
- Linear probe is randomly initialized
- Tests quality of pretrained representations

## Expected Output

```
====================================================================
RNABERT ZERO-SHOT SECONDARY STRUCTURE PREDICTION - RESULTS COMPARISON
====================================================================

Checkpoint                               Architecture         Split      Precision    Recall       F1
----------------------------------------------------------------------------------------------------
RNABERT_a16_b6_checkpoint494241          20L-16H-120D         Val        0.xxxx       0.xxxx       0.xxxx
                                                              Test       0.xxxx       0.xxxx       0.xxxx
----------------------------------------------------------------------------------------------------
RNABERT_a8_b8_checkpoint670000           20L-8H-64D           Val        0.xxxx       0.xxxx       0.xxxx
                                                              Test       0.xxxx       0.xxxx       0.xxxx
====================================================================
```

## Customization

### Adjust Batch Size
```bash
# In run_rnabert_ssp_eval.sh or submit_rnabert_ssp_eval.sh
BATCH_SIZE=16  # Increase for faster evaluation
```

### Different Random Seed
```bash
# In the scripts
SEED=123
```

### Custom Output Directory
```bash
python rnabert_ssp_eval.py \
    --model_name_or_path /path/to/checkpoint \
    --data_path /path/to/data \
    --output_dir ./my_custom_results
```

## Troubleshooting

### Job Failed?
```bash
# Check error log
cat logs/rnabert_ssp_eval_<JOB_ID>.err

# Common fixes:
# 1. Out of memory -> reduce batch size
# 2. Module not found -> activate correct conda environment
# 3. Path not found -> verify checkpoint and data paths
```

### Can't Find Results?
```bash
# Check if evaluation completed
ls rnabert_ssp_results/results/

# Should see:
# rnabert_a16_b6_checkpoint494241/zero_shot_results.json
# rnabert_a8_b8_checkpoint670000/zero_shot_results.json
```

## Files Created

```
RNABenchmark/
├── rnabert_ssp_eval.py              # Main evaluation script
├── run_rnabert_ssp_eval.sh          # Bash runner script
├── submit_rnabert_ssp_eval.sh       # SLURM job script
├── compare_rnabert_results.py       # Results comparison tool
├── RNABERT_SSP_EVAL_README.md       # Detailed documentation
├── QUICKSTART.md                     # This file
├── logs/                             # SLURM job logs
│   ├── rnabert_ssp_eval_*.out
│   └── rnabert_ssp_eval_*.err
└── rnabert_ssp_results/             # Evaluation results
    └── results/
        ├── rnabert_a16_b6_checkpoint494241/
        │   └── zero_shot_results.json
        └── rnabert_a8_b8_checkpoint670000/
            └── zero_shot_results.json
```

## Next Steps

After running evaluation:

1. **Compare results** using `compare_rnabert_results.py`
2. **Analyze performance** - which architecture performs better?
3. **Check validation vs test** - is there overfitting?
4. **Try different seeds** - how stable are the results?

## Questions?

- See `RNABERT_SSP_EVAL_README.md` for detailed documentation
- Check SLURM logs in `logs/` directory
- Verify paths match your chimera setup

## Summary

This evaluation tests whether RNABERT learned useful structural information during pretraining, by seeing how well a simple linear model can predict secondary structure using only the pretrained embeddings.

**Key point:** No training happens - the linear probe is randomly initialized, so performance depends entirely on the quality of RNABERT's representations.
