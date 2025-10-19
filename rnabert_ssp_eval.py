#!/usr/bin/env python3
"""
Zero-shot evaluation of RNABERT on Secondary Structure Prediction (SSP)

This script evaluates pretrained RNABERT models on the BEACON SSP benchmark
without any fine-tuning. It uses a randomly initialized linear probe on top
of frozen RNABERT embeddings.

Checkpoints to evaluate:
1. RNABERT_L2048_l20_a16_b6 (16 attention heads, 6 layers)
2. RNABERT_L2048_l20_a8_b8 (8 attention heads, 8 layers)

Data: BEACON Secondary Structure Prediction benchmark (bpRNA)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from accelerate import Accelerator
import scipy
import random
import json
from datetime import datetime

# Import from BEACON framework
from downstream.structure.data import SSDataset
from downstream.structure.lm import get_extractor


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")


class SimpleLinearProbe(nn.Module):
    """
    Simple linear layer for zero-shot structure prediction.
    Takes RNABERT embeddings and predicts base pairing probabilities.

    This probe is NOT trained - it uses random initialization to test
    the quality of the pretrained RNABERT representations.
    """
    def __init__(self, hidden_size):
        super().__init__()
        # Linear layer takes concatenated embeddings from two positions
        # and outputs a single pairing probability
        self.probe = nn.Linear(hidden_size * 2, 1)

    def forward(self, embeddings):
        """
        Predict base pairing for all position pairs.

        Args:
            embeddings: [batch_size, seq_len, hidden_size]
        Returns:
            logits: [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, hidden_size = embeddings.shape

        # Create pairwise representations by broadcasting
        # left[i,j] = embedding at position i
        # right[i,j] = embedding at position j
        left = embeddings.unsqueeze(2).expand(batch_size, seq_len, seq_len, hidden_size)
        right = embeddings.unsqueeze(1).expand(batch_size, seq_len, seq_len, hidden_size)

        # Concatenate to get [emb_i, emb_j] for all pairs
        pairs = torch.cat([left, right], dim=-1)

        # Predict pairing probability for each pair
        logits = self.probe(pairs).squeeze(-1)

        return logits


def calculate_metrics(logits, labels):
    """
    Calculate precision, recall, and F1 score.

    Args:
        logits: Model predictions (before sigmoid), shape: [N, 1]
        labels: Ground truth (0/1 for paired/not paired, -1 for padding), shape: [N, 1]
    Returns:
        dict: precision, recall, f1
    """
    labels = labels.squeeze().astype(int)
    logits = logits.squeeze()

    # Convert logits to probabilities using sigmoid
    probs = scipy.special.expit(logits)

    # Filter out padding positions (marked as -1)
    mask = labels != -1
    labels_filtered = labels[mask]
    probs_filtered = probs[mask]

    # Binary predictions (threshold at 0.5)
    predictions = (probs_filtered > 0.5).astype(int)

    # Calculate metrics
    precision = precision_score(labels_filtered, predictions, average='binary', zero_division=0)
    recall = recall_score(labels_filtered, predictions, average='binary', zero_division=0)
    f1 = f1_score(labels_filtered, predictions, average='binary', zero_division=0)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def evaluate(probe, extractor, data_loader, accelerator, split_name="Test"):
    """
    Perform zero-shot evaluation on a dataset split.

    Args:
        probe: Linear probe model (randomly initialized)
        extractor: RNABERT model (pretrained, frozen)
        data_loader: DataLoader for the split
        accelerator: Accelerate accelerator
        split_name: Name of the split (for logging)
    Returns:
        dict: Evaluation metrics
    """
    probe.eval()
    extractor.eval()

    all_logits = []
    all_labels = []

    print(f"\nEvaluating {split_name} set...")

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Processing {split_name}"):
            # Move batch to device
            input_ids = batch['input_ids'].to(accelerator.device)
            attention_mask = batch['attention_mask'].to(accelerator.device)
            labels = batch['struct'].to(accelerator.device)

            # Get RNABERT embeddings
            outputs = extractor(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            embeddings = outputs.last_hidden_state

            # Remove [CLS] and [SEP] special tokens
            # Original: [batch, seq_len+2, hidden_size]
            # After:    [batch, seq_len, hidden_size]
            embeddings_no_special = embeddings[:, 1:-1, :]

            # Predict structure using linear probe
            logits = probe(embeddings_no_special)

            # Collect predictions and labels
            all_logits.append(logits.detach().cpu().numpy().reshape(-1, 1))
            all_labels.append(labels.detach().cpu().numpy().reshape(-1, 1))

    # Concatenate all batches
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate metrics
    metrics = calculate_metrics(all_logits, all_labels)

    # Print results
    print(f'\n{"="*60}')
    print(f'{split_name} Results:')
    print(f'{"="*60}')
    print(f'  Precision: {metrics["precision"]:.4f}')
    print(f'  Recall:    {metrics["recall"]:.4f}')
    print(f'  F1 Score:  {metrics["f1"]:.4f}')
    print(f'{"="*60}\n')

    return metrics


def collate_fn(batch, tokenizer, max_length):
    """
    Collate function for batching sequences of different lengths.

    Args:
        batch: List of samples from dataset (each has 'seq' and 'struct')
        tokenizer: Tokenizer for encoding sequences
        max_length: Maximum sequence length
    Returns:
        dict: Batched and padded data
    """
    seqs = [x['seq'] for x in batch]
    structs = [x['struct'] for x in batch]

    # Tokenize sequences with padding
    encoded = tokenizer(
        seqs,
        padding='longest',
        max_length=max_length,
        truncation=True,
        return_tensors='pt'
    )

    # Pad structure matrices to the same size
    # All matrices should be max_len x max_len
    max_len = max([len(seq) for seq in seqs])
    structs_padded = np.array([
        np.pad(
            s,
            ((0, max_len - s.shape[0]), (0, max_len - s.shape[1])),
            'constant',
            constant_values=-1  # Use -1 for padding (will be masked out)
        )
        for s in structs
    ])

    encoded['struct'] = torch.tensor(structs_padded)
    return encoded


def main(args):
    """Main evaluation function."""

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Initialize accelerator (handles multi-GPU, mixed precision, etc.)
    accelerator = Accelerator()

    # Print evaluation configuration
    print("\n" + "="*70)
    print("ZERO-SHOT RNA SECONDARY STRUCTURE PREDICTION EVALUATION")
    print("Using RNABERT with Linear Probe (No Fine-tuning)")
    print("="*70)
    print(f"Model Name/Path: {args.model_name_or_path}")
    print(f"Data Path:       {args.data_path}")
    print(f"Model Type:      {args.model_type}")
    print(f"Token Type:      {args.token_type}")
    print(f"Max Length:      {args.model_max_length}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Random Seed:     {args.seed}")
    print(f"Device:          {accelerator.device}")
    print(f"Num GPUs:        {accelerator.num_processes}")
    print("="*70 + "\n")

    # Load RNABERT model and tokenizer
    print("Loading RNABERT model and tokenizer...")
    extractor, tokenizer = get_extractor(args)

    print(f"Model loaded successfully!")
    print(f"  Hidden size:     {extractor.config.hidden_size}")
    print(f"  Num layers:      {extractor.config.num_hidden_layers}")
    print(f"  Num heads:       {extractor.config.num_attention_heads}")
    print(f"  Vocab size:      {extractor.config.vocab_size}")
    print(f"  Max positions:   {extractor.config.max_position_embeddings}")

    # Create linear probe
    hidden_size = extractor.config.hidden_size
    probe = SimpleLinearProbe(hidden_size)

    total_params = sum(p.numel() for p in probe.parameters())
    print(f"\nLinear probe created:")
    print(f"  Input:           {hidden_size * 2} (concatenated pair embeddings)")
    print(f"  Output:          1 (pairing probability)")
    print(f"  Total params:    {total_params:,}")
    print(f"  Status:          Randomly initialized (zero-shot)")

    # Load datasets
    print(f"\nLoading datasets from: {args.data_path}")

    val_dataset = SSDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        args=args,
        mode='val'
    )

    test_dataset = SSDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        args=args,
        mode='test'
    )

    print(f"Validation set:  {len(val_dataset):,} samples")
    print(f"Test set:        {len(test_dataset):,} samples")

    # Create data loaders with custom collate function
    from functools import partial
    collate = partial(
        collate_fn,
        tokenizer=tokenizer,
        max_length=args.model_max_length
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate
    )

    # Prepare models with accelerator
    extractor, probe = accelerator.prepare(extractor, probe)

    # Run evaluation
    print("\n" + "="*70)
    print("STARTING ZERO-SHOT EVALUATION")
    print("="*70)

    # Evaluate on validation set
    val_metrics = evaluate(
        probe, extractor, val_loader, accelerator, "Validation"
    )

    # Evaluate on test set
    test_metrics = evaluate(
        probe, extractor, test_loader, accelerator, "Test"
    )

    # Save results
    results_dir = os.path.join(args.output_dir, "results", args.run_name)
    os.makedirs(results_dir, exist_ok=True)

    results = {
        'task': 'secondary_structure_prediction',
        'evaluation_method': 'zero_shot_linear_probe',
        'model_name_or_path': args.model_name_or_path,
        'model_type': args.model_type,
        'token_type': args.token_type,
        'hidden_size': hidden_size,
        'num_layers': extractor.config.num_hidden_layers,
        'num_heads': extractor.config.num_attention_heads,
        'max_length': args.model_max_length,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'timestamp': datetime.now().isoformat(),
        'validation': val_metrics,
        'test': test_metrics
    }

    results_file = os.path.join(results_dir, "zero_shot_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    # Print final summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Task:            Secondary Structure Prediction (SSP)")
    print(f"Model:           {args.model_type}")
    print(f"Results saved:   {results_file}")
    print(f"\nFinal Scores:")
    print(f"  Validation F1: {val_metrics['f1']:.4f}")
    print(f"  Test F1:       {test_metrics['f1']:.4f}")
    print("="*70 + "\n")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Zero-shot evaluation of RNABERT on Secondary Structure Prediction'
    )

    # Model paths - default to chimera checkpoints
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='/large_storage/goodarzilab/public/model_checkpoints/RNABERT/RNABERT_L2048_l20_a16_b6_r1e-05_mlm0.2_wd0.01_g4-rjiphp2y/checkpoint-494241',
        help='Path to RNABERT checkpoint directory'
    )

    # Data path - default to BEACON data on chimera
    parser.add_argument(
        '--data_path',
        type=str,
        default='/home/ali/DMS/DMS-FM/Downstream_Tasks/RNABenchmark/data/Secondary_structure_prediction',
        help='Path to bpRNA dataset directory (for SSP task)'
    )

    # Model configuration
    parser.add_argument(
        '--model_type',
        type=str,
        default='rnabert',
        help='Model type (default: rnabert)'
    )
    parser.add_argument(
        '--token_type',
        type=str,
        default='single',
        help='Tokenization type: single, 6mer, etc. (default: single)'
    )
    parser.add_argument(
        '--model_max_length',
        type=int,
        default=2048,
        help='Maximum sequence length (default: 2048)'
    )

    # Evaluation parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for evaluation (default: 8)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers (default: 4)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    # Output parameters
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./rnabert_ssp_results',
        help='Directory to save results (default: ./rnabert_ssp_results)'
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help='Name for this evaluation run (auto-generated if not provided)'
    )

    # Legacy parameters for compatibility with BEACON framework
    parser.add_argument('--mode', type=str, default='bprna')
    parser.add_argument('--pretrained_lm_dir', type=str, default='')
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--model_scale', type=str, default='')
    parser.add_argument('--attn_implementation', type=str, default='eager')

    args = parser.parse_args()

    # Auto-generate run name if not provided
    if args.run_name is None:
        model_name = os.path.basename(os.path.dirname(args.model_name_or_path))
        args.run_name = f"rnabert_ssp_zeroshot_{model_name}"

    # Validate inputs
    if not os.path.exists(args.model_name_or_path):
        print(f"ERROR: Model path not found: {args.model_name_or_path}")
        print("\nAvailable RNABERT checkpoints on chimera:")
        print("  1. /large_storage/goodarzilab/public/model_checkpoints/RNABERT/RNABERT_L2048_l20_a16_b6_r1e-05_mlm0.2_wd0.01_g4-rjiphp2y/checkpoint-494241")
        print("  2. /large_storage/goodarzilab/public/model_checkpoints/RNABERT/RNABERT_L2048_l20_a8_b8_r1e-05_mlm0.2_wd0.01_g4-g7p6hpqj/checkpoint-670000")
        sys.exit(1)

    if not os.path.exists(args.data_path):
        print(f"ERROR: Data path not found: {args.data_path}")
        print("\nExpected data location on chimera:")
        print("  /home/ali/DMS/DMS-FM/Downstream_Tasks/RNABenchmark/data/Secondary_structure_prediction")
        sys.exit(1)

    # Run evaluation
    main(args)
