# Minimal Setup for rnabert_ssp_eval.py

This guide explains how to set up a minimal environment to run **only** `rnabert_ssp_eval.py` for benchmarking RNABERT on Secondary Structure Prediction.

## Quick Start

### 1. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv_ssp

# Activate it
source venv_ssp/bin/activate  # On Linux/Mac
# OR
venv_ssp\Scripts\activate  # On Windows
```

### 2. Install Minimal Dependencies

```bash
pip install -r requirements_minimal_ssp.txt
```

**Note**: If you need GPU support for PyTorch with CUDA 11.7, use:
```bash
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements_minimal_ssp.txt
```

For other CUDA versions or CPU-only, modify the torch version accordingly.

### 3. Verify Installation

```bash
python3 test_minimal_imports.py
```

This script will test all required imports and confirm your environment is ready.

### 4. Run the Evaluation

```bash
python3 rnabert_ssp_eval.py \
    --model_name_or_path /path/to/your/rnabert/checkpoint \
    --data_path /path/to/Secondary_structure_prediction/data \
    --batch_size 20 \
    --output_dir ./rnabert_ssp_results
```

## What's Included

The `requirements_minimal_ssp.txt` file includes **only** the dependencies needed for:

1. **Core Deep Learning**: PyTorch, Transformers, Accelerate
2. **Scientific Computing**: NumPy, SciPy, Scikit-learn (for metrics)
3. **Data Handling**: Pandas (for reading CSV files)
4. **Utilities**: TQDM (progress bars), SafeTensors (model loading)

## What's NOT Included

The minimal requirements **exclude** packages from the full `requirements.txt` that are not needed for `rnabert_ssp_eval.py`:

- Training-specific packages (optuna, wandb, tensorboard, etc.)
- Cloud/storage packages (awscli, boto3, petrel-oss-sdk, etc.)
- Other model frameworks (not needed for RNABERT evaluation)
- Development tools (grad-cam, graphviz, etc.)
- Biological databases (mygene, biothings-client, etc.)

## Comparison

| Package Count | File |
|--------------|------|
| **14** packages | `requirements_minimal_ssp.txt` (minimal for SSP evaluation) |
| **147** packages | `requirements.txt` (full project) |

**Space savings**: ~90% fewer dependencies!

## Dependencies Summary

### Required Packages (14 total)

```
torch==1.13.1
transformers==4.38.1
accelerate==0.27.2
numpy==1.24.4
scipy==1.10.1
scikit-learn==1.3.0
pandas==2.0.3
tqdm==4.66.1
safetensors==0.4.2
huggingface-hub==0.21.1
PyYAML==6.0.1
regex==2023.8.8
filelock==3.12.4
packaging==24.0
```

### What Each Package Does

- **torch**: Deep learning framework for running RNABERT
- **transformers**: Hugging Face library for BERT-based models
- **accelerate**: Multi-GPU and distributed training support
- **numpy**: Array operations and numerical computing
- **scipy**: Statistical functions (sigmoid for probabilities)
- **scikit-learn**: Evaluation metrics (precision, recall, F1, MCC, AUPRC)
- **pandas**: Reading CSV datasets
- **tqdm**: Progress bars during evaluation
- **safetensors**: Safe model checkpoint loading
- **huggingface-hub**: Downloading models from Hugging Face
- **PyYAML**: Configuration file parsing
- **regex**: Regular expressions for tokenization
- **filelock**: File locking for concurrent access
- **packaging**: Version parsing utilities

## Troubleshooting

### Import Errors

If you get import errors related to local modules:
```python
from downstream.structure.data import SSDataset
from downstream.structure.lm import get_extractor
```

Make sure you're running the script **from the project root directory**:
```bash
cd /path/to/RNABenchmark
python3 rnabert_ssp_eval.py [args]
```

### CUDA/GPU Issues

If you encounter CUDA version mismatches:

1. Check your CUDA version: `nvcc --version`
2. Install the matching PyTorch version from [pytorch.org](https://pytorch.org)

For CPU-only (no GPU):
```bash
pip install torch==1.13.1 --index-url https://download.pytorch.org/whl/cpu
```

### Model Loading Issues

If the model fails to load, ensure:
1. The checkpoint path exists and is correct
2. The model files include `config.json`, `pytorch_model.bin` (or `.safetensors`)
3. You have read permissions for the checkpoint directory

## Additional Notes

- This minimal setup is **only** for running `rnabert_ssp_eval.py`
- For training, fine-tuning, or other benchmarks, use the full `requirements.txt`
- The local codebase structure (`model/`, `downstream/`, `tokenizer/`) is still required
- Only external package dependencies are minimized

## Support

If you encounter issues:

1. Verify your Python version: `python3 --version` (3.8+ recommended)
2. Run the test script: `python3 test_minimal_imports.py`
3. Check that you're in the correct directory
4. Ensure all paths in your command are correct

For package version conflicts, you may need to adjust versions based on your system.
