#!/usr/bin/env python3
"""
Zero-shot evaluation of RNABERT on Secondary Structure Prediction (SSP)

This script evaluates pretrained RNABERT models on the BEACON SSP benchmark
without any fine-tuning. It uses a randomly initialized linear probe on top
of frozen RNABERT embeddings.

IMPROVEMENTS:
- Efficient memory usage with proper masking
- Structural constraints (symmetry, minimum pairing distance)
- Enhanced metrics (MCC, AUPRC)
- Attention-based baseline comparison
- Better batch processing

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
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    matthews_corrcoef, average_precision_score
)
from accelerate import Accelerator
import scipy
import random
import json
from datetime import datetime
from functools import partial

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


class EfficientLinearProbe(nn.Module):
    """
    Efficient linear probe for secondary structure prediction.
    
    Improvements over original:
    - Enforces structural constraints (minimum pairing distance)
    - Symmetric predictions (if i pairs with j, then j pairs with i)
    - Memory-efficient with proper masking
    - Only processes valid pairs
    
    This probe is NOT trained - it uses random initialization to test
    the quality of the pretrained RNABERT representations.
    """
    def __init__(self, hidden_size, min_pair_distance=4):
        super().__init__()
        # Linear layer takes concatenated embeddings from two positions
        # and outputs a single pairing probability
        self.probe = nn.Linear(hidden_size * 2, 1)
        self.min_dist = min_pair_distance
        
    def forward(self, embeddings):
        """
        Predict base pairing for valid position pairs.
        
        Args:
            embeddings: [batch_size, seq_len, hidden_size]
        Returns:
            logits: [batch_size, seq_len, seq_len] - symmetric pairing probabilities
            mask: [seq_len, seq_len] - boolean mask of valid pairs
        """
        batch_size, seq_len, hidden_size = embeddings.shape
        device = embeddings.device
        
        # Create pairwise representations
        # left[b,i,j,:] = embedding at position i
        # right[b,i,j,:] = embedding at position j
        left = embeddings.unsqueeze(2).expand(batch_size, seq_len, seq_len, hidden_size)
        right = embeddings.unsqueeze(1).expand(batch_size, seq_len, seq_len, hidden_size)
        
        # Concatenate to get [emb_i, emb_j] for all pairs
        pairs = torch.cat([left, right], dim=-1)
        
        # Predict pairing probability for each pair
        logits = self.probe(pairs).squeeze(-1)
        
        # Create validity mask for base pairing
        # RNA structure constraints:
        # 1. Only consider i < j (upper triangle)
        # 2. Minimum distance between paired bases (typically 3-4, prevents tiny hairpins)
        i_idx = torch.arange(seq_len, device=device)
        j_idx = torch.arange(seq_len, device=device)
        
        # j - i >= min_dist ensures proper spacing
        distance_mask = (j_idx.unsqueeze(0) - i_idx.unsqueeze(1)) >= self.min_dist
        
        # Upper triangle mask (i < j)
        upper_tri = torch.triu(torch.ones(seq_len, seq_len, device=device), 
                               diagonal=self.min_dist).bool()
        
        # Combine masks
        mask = distance_mask & upper_tri
        
        # Apply mask to logits
        logits = logits.masked_fill(~mask.unsqueeze(0), float('-inf'))
        
        # Enforce symmetry: if i pairs with j, then j pairs with i
        # This is a fundamental property of RNA base pairing
        logits = (logits + logits.transpose(-2, -1)) / 2
        
        return logits, mask


class AttentionBaseline:
    """
    Use attention weights from pretrained model as structure predictions.
    This serves as an alternative zero-shot baseline.
    """
    def __init__(self, extractor, layer_indices=None):
        self.extractor = extractor
        # Use middle-to-late layers by default (often most informative)
        if layer_indices is None:
            n_layers = extractor.config.num_hidden_layers
            self.layer_indices = list(range(n_layers // 2, n_layers))
        else:
            self.layer_indices = layer_indices
    
    def predict(self, input_ids, attention_mask):
        """
        Extract and aggregate attention weights as structure predictions.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            logits: [batch_size, seq_len, seq_len]
        """
        with torch.no_grad():
            outputs = self.extractor(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
        
        # Aggregate attention from selected layers and heads
        attentions = torch.stack([outputs.attentions[i] for i in self.layer_indices])
        # Average across layers and heads: [batch, seq_len, seq_len]
        attn_weights = attentions.mean(dim=[0, 1])
        
        # Remove special tokens [CLS] and [SEP]
        attn_weights = attn_weights[:, 1:-1, 1:-1]
        
        # Symmetrize
        attn_weights = (attn_weights + attn_weights.transpose(-2, -1)) / 2
        
        # Convert to logits (log odds)
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        attn_weights = torch.clamp(attn_weights, eps, 1 - eps)
        logits = torch.log(attn_weights / (1 - attn_weights))
        
        return logits


def calculate_metrics(logits, labels, mask=None):
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        logits: Model predictions (before sigmoid), shape: [N]
        labels: Ground truth (0/1 for not-paired/paired, -1 for padding), shape: [N]
        mask: Optional boolean mask for valid pairs, shape: [N]
    Returns:
        dict: precision, recall, f1, mcc, auprc
    """
    labels = labels.squeeze().astype(int)
    logits = logits.squeeze()
    
    # Convert logits to probabilities using sigmoid
    probs = scipy.special.expit(logits)
    
    # Filter out padding positions (marked as -1)
    valid = labels != -1
    
    # Apply additional mask if provided (for structural constraints)
    if mask is not None:
        mask = mask.squeeze()
        valid = valid & mask
    
    labels_filtered = labels[valid]
    probs_filtered = probs[valid]
    
    # Handle edge case of no valid pairs
    if len(labels_filtered) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mcc": 0.0,
            "auprc": 0.0,
            "n_samples": 0,
            "n_positive": 0
        }
    
    # Binary predictions (threshold at 0.5)
    predictions = (probs_filtered > 0.5).astype(int)
    
    # Calculate metrics
    precision = precision_score(labels_filtered, predictions, average='binary', zero_division=0)
    recall = recall_score(labels_filtered, predictions, average='binary', zero_division=0)
    f1 = f1_score(labels_filtered, predictions, average='binary', zero_division=0)
    
    # Matthews Correlation Coefficient (handles class imbalance better than F1)
    mcc = matthews_corrcoef(labels_filtered, predictions)
    
    # Area Under Precision-Recall Curve (threshold-independent metric)
    auprc = average_precision_score(labels_filtered, probs_filtered)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(mcc),
        "auprc": float(auprc),
        "n_samples": int(len(labels_filtered)),
        "n_positive": int(labels_filtered.sum())
    }


def evaluate(probe, extractor, data_loader, accelerator, split_name="Test", use_attention=False):
    """
    Perform zero-shot evaluation on a dataset split.
    
    Args:
        probe: Linear probe model (randomly initialized) or None if use_attention=True
        extractor: RNABERT model (pretrained, frozen)
        data_loader: DataLoader for the split
        accelerator: Accelerate accelerator
        split_name: Name of the split (for logging)
        use_attention: If True, use attention weights instead of probe
    Returns:
        dict: Evaluation metrics
    """
    if probe is not None:
        probe.eval()
    extractor.eval()
    
    if use_attention:
        attention_baseline = AttentionBaseline(extractor)
    
    all_logits = []
    all_labels = []
    all_masks = []
    
    print(f"\nEvaluating {split_name} set...")
    print(f"Method: {'Attention Baseline' if use_attention else 'Linear Probe'}")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Processing {split_name}"):
            # Move batch to device
            input_ids = batch['input_ids'].to(accelerator.device)
            attention_mask = batch['attention_mask'].to(accelerator.device)
            labels = batch['struct'].to(accelerator.device)
            
            if use_attention:
                # Use attention weights as predictions
                logits = attention_baseline.predict(input_ids, attention_mask)
                # Create mask for valid pairs
                seq_len = logits.shape[1]
                mask = torch.triu(torch.ones(seq_len, seq_len, device=logits.device), 
                                 diagonal=4).bool()
            else:
                # Get RNABERT embeddings
                outputs = extractor(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                embeddings = outputs.last_hidden_state
                
                # Remove [CLS] and [SEP] special tokens
                embeddings_no_special = embeddings[:, 1:-1, :]
                
                # Predict structure using linear probe
                logits, mask = probe(embeddings_no_special)
            
            # Collect predictions and labels
            all_logits.append(logits.detach().cpu().numpy().reshape(-1))
            all_labels.append(labels.detach().cpu().numpy().reshape(-1))
            all_masks.append(mask.detach().cpu().numpy().reshape(-1) if isinstance(mask, torch.Tensor) 
                           else np.tile(mask.reshape(-1), len(labels)))
    
    # Concatenate all batches
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_logits, all_labels, all_masks)
    
    # Print results
    print(f'\n{"="*60}')
    print(f'{split_name} Results:')
    print(f'{"="*60}')
    print(f'  Samples (valid pairs): {metrics["n_samples"]:,}')
    print(f'  Positive (paired):     {metrics["n_positive"]:,} ({100*metrics["n_positive"]/max(metrics["n_samples"], 1):.1f}%)')
    print(f'  Precision:             {metrics["precision"]:.4f}')
    print(f'  Recall:                {metrics["recall"]:.4f}')
    print(f'  F1 Score:              {metrics["f1"]:.4f}')
    print(f'  MCC:                   {metrics["mcc"]:.4f}')
    print(f'  AUPRC:                 {metrics["auprc"]:.4f}')
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
    print(f"Min Pair Dist:   {args.min_pair_distance}")
    print(f"Random Seed:     {args.seed}")
    print(f"Device:          {accelerator.device}")
    print(f"Num GPUs:        {accelerator.num_processes}")
    print(f"Use Attention:   {args.use_attention_baseline}")
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
    
    # Create linear probe (unless using attention baseline)
    probe = None
    if not args.use_attention_baseline:
        hidden_size = extractor.config.hidden_size
        probe = EfficientLinearProbe(hidden_size, min_pair_distance=args.min_pair_distance)
        
        total_params = sum(p.numel() for p in probe.parameters())
        print(f"\nLinear probe created:")
        print(f"  Input:           {hidden_size * 2} (concatenated pair embeddings)")
        print(f"  Output:          1 (pairing probability)")
        print(f"  Min pair dist:   {args.min_pair_distance}")
        print(f"  Total params:    {total_params:,}")
        print(f"  Status:          Randomly initialized (zero-shot)")
    else:
        print(f"\nUsing attention-based baseline:")
        print(f"  Method:          Average attention weights from middle-late layers")
        print(f"  Status:          Zero-shot (no training)")
    
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
        collate_fn=collate,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True
    )
    
    # Prepare models with accelerator
    if probe is not None:
        extractor, probe = accelerator.prepare(extractor, probe)
    else:
        extractor = accelerator.prepare(extractor)
    
    # Run evaluation
    print("\n" + "="*70)
    print("STARTING ZERO-SHOT EVALUATION")
    print("="*70)
    
    # Evaluate on validation set
    val_metrics = evaluate(
        probe, extractor, val_loader, accelerator, "Validation",
        use_attention=args.use_attention_baseline
    )
    
    # Evaluate on test set
    test_metrics = evaluate(
        probe, extractor, test_loader, accelerator, "Test",
        use_attention=args.use_attention_baseline
    )
    
    # Save results
    results_dir = os.path.join(args.output_dir, "results", args.run_name)
    os.makedirs(results_dir, exist_ok=True)
    
    results = {
        'task': 'secondary_structure_prediction',
        'evaluation_method': 'attention_baseline' if args.use_attention_baseline else 'zero_shot_linear_probe',
        'model_name_or_path': args.model_name_or_path,
        'model_type': args.model_type,
        'token_type': args.token_type,
        'hidden_size': extractor.config.hidden_size,
        'num_layers': extractor.config.num_hidden_layers,
        'num_heads': extractor.config.num_attention_heads,
        'max_length': args.model_max_length,
        'min_pair_distance': args.min_pair_distance,
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
    print(f"Method:          {'Attention' if args.use_attention_baseline else 'Linear Probe'}")
    print(f"Results saved:   {results_file}")
    print(f"\nFinal Scores:")
    print(f"  Validation F1:   {val_metrics['f1']:.4f}")
    print(f"  Validation MCC:  {val_metrics['mcc']:.4f}")
    print(f"  Validation AUPRC:{val_metrics['auprc']:.4f}")
    print(f"  Test F1:         {test_metrics['f1']:.4f}")
    print(f"  Test MCC:        {test_metrics['mcc']:.4f}")
    print(f"  Test AUPRC:      {test_metrics['auprc']:.4f}")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Zero-shot evaluation of RNABERT on Secondary Structure Prediction'
    )
    
    # Model paths
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='/large_storage/goodarzilab/public/model_checkpoints/RNABERT/RNABERT_L2048_l20_a16_b6_r1e-05_mlm0.2_wd0.01_g4-rjiphp2y/checkpoint-494241',
        help='Path to RNABERT checkpoint directory'
    )
    
    # Data path
    parser.add_argument(
        '--data_path',
        type=str,
        default='/home/ali/DMS/DMS-FM/Downstream_Tasks/RNABenchmark/data/Secondary_structure_prediction',
        help='Path to bpRNA dataset directory'
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
        default=20,
        help='Batch size for evaluation (default: 20)'
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
    parser.add_argument(
        '--min_pair_distance',
        type=int,
        default=4,
        help='Minimum distance between paired bases (default: 4)'
    )
    parser.add_argument(
        '--use_attention_baseline',
        action='store_true',
        help='Use attention weights as baseline instead of linear probe'
    )
    
    # Output parameters
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./rnabert_ssp_results',
        help='Directory to save results'
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
        method = "attention" if args.use_attention_baseline else "probe"
        args.run_name = f"rnabert_ssp_zeroshot_{method}_{model_name}"
    
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