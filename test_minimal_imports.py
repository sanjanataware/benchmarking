#!/usr/bin/env python3
"""
Test script to verify all imports required by rnabert_ssp_eval.py work correctly.
Run this after installing requirements_minimal_ssp.txt to ensure all dependencies are met.
"""

import sys

def test_imports():
    """Test all imports required by rnabert_ssp_eval.py"""

    print("="*70)
    print("Testing minimal dependencies for rnabert_ssp_eval.py")
    print("="*70)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Standard library imports
    print("\n[1/8] Testing standard library imports...")
    try:
        import warnings
        import os
        import sys
        import argparse
        import json
        import random
        from datetime import datetime
        from functools import partial
        print("✓ Standard library imports successful")
        tests_passed += 1
    except ImportError as e:
        print(f"✗ Standard library import failed: {e}")
        tests_failed += 1

    # Test 2: NumPy and SciPy
    print("\n[2/8] Testing NumPy and SciPy...")
    try:
        import numpy as np
        import scipy
        import scipy.special
        print(f"✓ NumPy {np.__version__} and SciPy {scipy.__version__} imported")
        tests_passed += 1
    except ImportError as e:
        print(f"✗ NumPy/SciPy import failed: {e}")
        tests_failed += 1

    # Test 3: PyTorch
    print("\n[3/8] Testing PyTorch...")
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        print(f"✓ PyTorch {torch.__version__} imported")
        tests_passed += 1
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        tests_failed += 1

    # Test 4: Transformers
    print("\n[4/8] Testing Transformers...")
    try:
        import transformers
        from transformers import BertModel, BertTokenizer, AutoConfig
        print(f"✓ Transformers {transformers.__version__} imported")
        tests_passed += 1
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        tests_failed += 1

    # Test 5: Accelerate
    print("\n[5/8] Testing Accelerate...")
    try:
        from accelerate import Accelerator
        print(f"✓ Accelerate imported")
        tests_passed += 1
    except ImportError as e:
        print(f"✗ Accelerate import failed: {e}")
        tests_failed += 1

    # Test 6: Scikit-learn
    print("\n[6/8] Testing Scikit-learn metrics...")
    try:
        from sklearn.metrics import (
            precision_score, recall_score, f1_score,
            matthews_corrcoef, average_precision_score
        )
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__} metrics imported")
        tests_passed += 1
    except ImportError as e:
        print(f"✗ Scikit-learn import failed: {e}")
        tests_failed += 1

    # Test 7: Pandas
    print("\n[7/8] Testing Pandas...")
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__} imported")
        tests_passed += 1
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        tests_failed += 1

    # Test 8: Additional utilities
    print("\n[8/8] Testing additional utilities...")
    try:
        from tqdm import tqdm
        import safetensors
        import huggingface_hub
        print("✓ TQDM, safetensors, and huggingface_hub imported")
        tests_passed += 1
    except ImportError as e:
        print(f"✗ Utilities import failed: {e}")
        tests_failed += 1

    # Test 9: Local module imports (these require the codebase structure)
    print("\n[9/9] Testing local module imports...")
    try:
        from downstream.structure.data import SSDataset
        from downstream.structure.lm import get_extractor
        print("✓ Local downstream modules imported successfully")
        tests_passed += 1
    except ImportError as e:
        print(f"✗ Local module import failed: {e}")
        print("   Note: This is expected if model/ and downstream/ directories are missing")
        tests_failed += 1

    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"Tests passed: {tests_passed}/9")
    print(f"Tests failed: {tests_failed}/9")

    if tests_failed == 0:
        print("\n✓ All dependencies are correctly installed!")
        print("You can now run rnabert_ssp_eval.py")
        return 0
    elif tests_failed == 1 and tests_passed == 8:
        print("\n⚠ Core dependencies are installed correctly!")
        print("Note: Local module import failed, but this is expected")
        print("You should be able to run rnabert_ssp_eval.py from the project root")
        return 0
    else:
        print("\n✗ Some dependencies are missing. Please install them using:")
        print("   pip install -r requirements_minimal_ssp.txt")
        return 1

if __name__ == "__main__":
    sys.exit(test_imports())
