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
    print(f"✓ Random seed set to: {seed}")


class SimpleLinearProbe(nn.Module):
    """
    Simple linear layer for zero-shot structure prediction.
    Takes RNABERT embeddings and predicts base pairing probabilities.
    """
    def __init__(self, hidden_size):
        super().__init__()
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
        
        # Create pairwise representations
        left = embeddings.unsqueeze(2).expand(batch_size, seq_len, seq_len, hidden_size)
        right = embeddings.unsqueeze(1).expand(batch_size, seq_len, seq_len, hidden_size)
        
        # Concatenate pairs
        pairs = torch.cat([left, right], dim=-1)
        
        # Predict pairing probability
        logits = self.probe(pairs).squeeze(-1)
        
        return logits


def calculate_metrics(logits, labels):
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        logits: Model predictions (before sigmoid)
        labels: Ground truth (0/1 for paired/not paired, -1 for padding)
    Returns:
        dict: precision, recall, f1
    """
    labels = labels.squeeze().astype(int)
    logits = logits.squeeze()
    
    # Convert logits to probabilities
    probs = scipy.special.expit(logits)
    
    # Filter out padding
    mask = labels != -1
    labels_filtered = labels[mask]
    probs_filtered = probs[mask]
    
    # Binary predictions
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
        probe: Linear probe model
        extractor: RNABERT model
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
        batch: List of samples from dataset
        tokenizer: Tokenizer for encoding sequences
        max_length: Maximum sequence length
    Returns:
        dict: Batched and padded data
    """
    seqs = [x['seq'] for x in batch]
    structs = [x['struct'] for x in batch]
    
    # Tokenize sequences
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
            constant_values=-1
        )
        for s in structs
    ])
    
    encoded['struct'] = torch.tensor(structs_padded)
    return encoded


def main(args):
    """Main evaluation function."""
    
    # Set random seed
    set_seed(args.seed)
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Print header
    print("\n" + "="*70)
    print("ZERO-SHOT RNA SECONDARY STRUCTURE PREDICTION EVALUATION")
    print("Linear Probe on RNABERT Embeddings")
    print("="*70)
    print(f"Checkpoint:      {args.checkpoint_path}")
    print(f"Data:            {args.data_path}")
    print(f"Model Type:      {args.model_type}")
    print(f"Token Type:      {args.token_type}")
    print(f"Max Length:      {args.model_max_length}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Random Seed:     {args.seed}")
    print("="*70 + "\n")
    
    print("Loading RNABERT model and tokenizer...")
    extractor, tokenizer = get_extractor(args)
    
    print(f"✓ Model loaded")
    print(f"  Hidden size: {extractor.config.hidden_size}")
    print(f"  Num layers:  {extractor.config.num_hidden_layers}")
    print(f"  Num heads:   {extractor.config.num_attention_heads}")
    
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"\nLoading checkpoint weights...")
        print(f"  Path: {args.checkpoint_path}")
        
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            extractor.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("  ✓ Loaded from 'model_state_dict' key")
        elif 'state_dict' in checkpoint:
            extractor.load_state_dict(checkpoint['state_dict'], strict=False)
            print("  ✓ Loaded from 'state_dict' key")
        else:
            extractor.load_state_dict(checkpoint, strict=False)
            print("  ✓ Loaded as direct state dict")
    else:
        print(f"\n⚠ WARNING: Checkpoint not found at {args.checkpoint_path}")
        print("  Using randomly initialized weights!\n")
    
    hidden_size = extractor.config.hidden_size
    probe = SimpleLinearProbe(hidden_size)
    
    total_params = sum(p.numel() for p in probe.parameters())
    print(f"\n✓ Linear probe created:")
    print(f"  Input:  {hidden_size * 2} (concatenated embeddings)")
    print(f"  Output: 1 (pairing probability)")
    print(f"  Params: {total_params:,}")
    print(f"  Status: Randomly initialized (zero-shot)")
    
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
    
    print(f"✓ Validation set: {len(val_dataset):,} samples")
    print(f"✓ Test set:       {len(test_dataset):,} samples")
    
    # Create data loaders
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
    
    extractor, probe = accelerator.prepare(extractor, probe)
    
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
    
    results_dir = os.path.join(args.output_dir, "results", args.run_name)
    os.makedirs(results_dir, exist_ok=True)
    
    results = {
        'task': 'secondary_structure_prediction',
        'evaluation_method': 'zero_shot_linear_probe',
        'checkpoint_path': args.checkpoint_path,
        'model_type': args.model_type,
        'token_type': args.token_type,
        'hidden_size': hidden_size,
        'max_length': args.model_max_length,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'validation': val_metrics,
        'test': test_metrics
    }
    
    results_file = os.path.join(results_dir, "zero_shot_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Task: Secondary Structure Prediction (SSP)")
    print(f"Results saved to: {results_file}")
    print(f"\nFinal Scores:")
    print(f"  Validation F1: {val_metrics['f1']:.4f}")
    print(f"  Test F1:       {test_metrics['f1']:.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Zero-shot evaluation of RNABERT on Secondary Structure Prediction (SSP)'
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Path to RNABERT checkpoint file'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
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
        default='./zero_shot_results',
        help='Directory to save results (default: ./zero_shot_results)'
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default='rnabert_ssp_zeroshot',
        help='Name for this evaluation run (default: rnabert_ssp_zeroshot)'
    )
    
    # Legacy parameters for compatibility with BEACON code
    parser.add_argument('--mode', type=str, default='bprna')
    parser.add_argument('--pretrained_lm_dir', type=str, default='')
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--model_scale', type=str, default='')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path not found: {args.data_path}")
    
    # Run evaluation
    main(args)