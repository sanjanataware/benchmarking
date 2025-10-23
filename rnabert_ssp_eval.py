#!/usr/bin/env python3
"""
Training and evaluation of RNABERT on Secondary Structure Prediction (SSP)


This script trains a linear probe on top of frozen RNABERT embeddings,
then evaluates on the BEACON SSP benchmark.


MODIFICATIONS FROM ORIGINAL:
- Added training loop for linear probe
- Keeps RNABERT frozen during training
- Uses validation set for early stopping
- Saves best model checkpoint
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

       # Enforce symmetry: if i pairs with j, then j pairs with i
       # This is a fundamental property of RNA base pairing
       # We need to copy upper triangle to lower triangle BEFORE masking
       logits = (logits + logits.transpose(-2, -1)) / 2

       # Now apply the full symmetric mask (both upper and lower triangle)
       # The mask should only keep pairs that satisfy distance constraint
       symmetric_mask = mask | mask.transpose(-2, -1)

       # Apply mask to logits - set invalid pairs to -inf
       logits = logits.masked_fill(~symmetric_mask.unsqueeze(0), float('-inf'))

       # Return symmetric mask for proper filtering during training/evaluation
       return logits, symmetric_mask




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




def evaluate(probe, extractor, data_loader, accelerator, split_name="Test"):
   """
   Evaluate the model on a dataset split.
  
   Args:
       probe: Linear probe model
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
   all_masks = []
  
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
          
           # Also trim labels to match (remove first and last positions)
           labels_trimmed = labels[:, 1:-1, 1:-1]
          
           # Predict structure using linear probe
           logits, mask = probe(embeddings_no_special)
          
           # Collect predictions and labels
           batch_size = logits.shape[0]
           all_logits.append(logits.detach().cpu().numpy().reshape(-1))
           all_labels.append(labels_trimmed.detach().cpu().numpy().reshape(-1))


           # Handle mask - must match batch size
           if isinstance(mask, torch.Tensor):
               mask_np = mask.detach().cpu().numpy()
               # Check if mask needs to be broadcast for batch
               if mask_np.ndim == 2:  # Shape [seq_len, seq_len] - same mask for all in batch
                   # Expand mask to batch: [seq_len, seq_len] -> [batch_size, seq_len, seq_len] -> [batch_size * seq_len * seq_len]
                   mask_np = np.broadcast_to(mask_np, (batch_size, mask_np.shape[0], mask_np.shape[1]))
                   mask_np = mask_np.reshape(-1)
               elif mask_np.ndim == 3:  # Already has batch dimension
                   mask_np = mask_np.reshape(-1)
               else:
                   mask_np = mask_np.reshape(-1)
           else:
               # Mask is same for all samples - need to repeat for entire batch
               mask_np = np.broadcast_to(mask, (batch_size, mask.shape[0], mask.shape[1]))
               mask_np = mask_np.reshape(-1)
           all_masks.append(mask_np)
  
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




def train_probe(probe, extractor, train_loader, val_loader, accelerator, args):
   """
   Train the linear probe while keeping RNABERT frozen.
  
   Args:
       probe: Linear probe model (randomly initialized)
       extractor: RNABERT model (pretrained, frozen)
       train_loader: DataLoader for training data
       val_loader: DataLoader for validation data
       accelerator: Accelerate accelerator
       args: Training arguments (epochs, lr, etc.)
   Returns:
       probe: Trained linear probe
   """
   print("\n" + "="*70)
   print("TRAINING LINEAR PROBE")
   print("="*70)
   print(f"Epochs:          {args.epochs}")
   print(f"Learning Rate:   {args.learning_rate}")
   print(f"Weight Decay:    {args.weight_decay}")
   print(f"Batch Size:      {args.batch_size}")
   print("="*70 + "\n")
  
   # Freeze RNABERT parameters
   for param in extractor.parameters():
       param.requires_grad = False
  
   # Only train the probe
   optimizer = torch.optim.AdamW(
       probe.parameters(),
       lr=args.learning_rate,
       weight_decay=args.weight_decay
   )
  
   # Use weighted BCE loss to handle class imbalance
   # RNA secondary structure is sparse (most positions don't pair)
   criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(accelerator.device))
  
   best_val_f1 = -1.0  # Start with -1 so any score will be better
   best_probe_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}  # Initialize with current state
   patience_counter = 0
  
   for epoch in range(args.epochs):
       probe.train()
       extractor.eval()
      
       total_loss = 0
       num_batches = 0
      
       pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
       for batch_idx, batch in enumerate(pbar):
           # Move batch to device
           input_ids = batch['input_ids'].to(accelerator.device)
           attention_mask = batch['attention_mask'].to(accelerator.device)
           labels = batch['struct'].to(accelerator.device)
          
           # Get embeddings from frozen RNABERT
           with torch.no_grad():
               outputs = extractor(
                   input_ids=input_ids,
                   attention_mask=attention_mask
               )
               embeddings = outputs.last_hidden_state
               # Remove [CLS] and [SEP] tokens
               embeddings = embeddings[:, 1:-1, :]
          
           # IMPORTANT: Labels should also have special tokens removed
           # Labels shape: [batch, max_len, max_len]
           # We need to remove first and last positions to match embeddings
           labels_trimmed = labels[:, 1:-1, 1:-1]
          
           # Forward through probe 
           logits, mask = probe(embeddings)
          
           # Calculate loss only on valid pairs (not padding)
           valid_mask = (labels_trimmed != -1)
          
           # Combine with structural mask
           if isinstance(mask, torch.Tensor):
               if mask.ndim == 2:
                   # Get actual dimensions
                   trimmed_seq_len = labels_trimmed.shape[1]
                   mask_seq_len = mask.shape[0]
                  
                   # If there's a mismatch, take the smaller dimension
                   if mask_seq_len != trimmed_seq_len:
                       min_len = min(mask_seq_len, trimmed_seq_len)
                       mask = mask[:min_len, :min_len]
                       labels_trimmed = labels_trimmed[:, :min_len, :min_len]
                       logits = logits[:, :min_len, :min_len]
                       valid_mask = (labels_trimmed != -1)
                  
                   # Broadcast to batch
                   mask = mask.unsqueeze(0).expand(labels_trimmed.shape[0], -1, -1)
               valid_mask = valid_mask & mask
          
           # Flatten everything
           logits_flat = logits.reshape(-1)[valid_mask.reshape(-1)]
           labels_flat = labels_trimmed.reshape(-1)[valid_mask.reshape(-1)].float()
          
           # Debug first batch
           if batch_idx == 0 and epoch == 0:
               print(f"\nDEBUG - First batch info:")
               print(f"  Input shape: {input_ids.shape}")
               print(f"  Embeddings shape (after special token removal): {embeddings.shape}")
               print(f"  Labels shape (original): {labels.shape}")
               print(f"  Labels unique values (original): {torch.unique(labels).cpu().numpy()}")
               print(f"  Labels shape (trimmed): {labels_trimmed.shape}")
               print(f"  Labels unique values (trimmed): {torch.unique(labels_trimmed).cpu().numpy()}")
               print(f"  Logits shape: {logits.shape}")
               print(f"  Logits unique values: {torch.unique(logits).cpu().numpy()[:10]}...")  # First 10
               print(f"  Logits min/max: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
               print(f"  Mask shape: {mask.shape if isinstance(mask, torch.Tensor) else 'not tensor'}")
               print(f"  Mask sum (True values): {mask.sum().item() if isinstance(mask, torch.Tensor) else 'N/A'}")
               print(f"  Valid mask (labels != -1) sum: {(labels_trimmed != -1).sum().item()}")
               print(f"  Structural mask sum: {valid_mask.sum().item()}")
               print(f"  Valid pairs: {len(logits_flat)}")
               if len(logits_flat) > 0:
                   print(f"  Positive pairs: {(labels_flat == 1).sum().item()}")
                   print(f"  Logits range: [{logits_flat.min().item():.4f}, {logits_flat.max().item():.4f}]")
               else:
                   print(f"  WARNING: No valid pairs found!")
                   print(f"  Min pair distance: {args.min_pair_distance}")
                   print(f"  Sequence length after tokens: {embeddings.shape[1]}")
                   print(f"  First sample labels corner (5x5):")
                   print(labels_trimmed[0, :5, :5].cpu().numpy())
          
           if len(logits_flat) > 0:
               loss = criterion(logits_flat, labels_flat)
              
               # Backward pass
               accelerator.backward(loss)
               optimizer.step()
               optimizer.zero_grad()
              
               total_loss += loss.item()
               num_batches += 1
              
               pbar.set_postfix({'loss': f'{loss.item():.4f}', 'valid': len(logits_flat)})
           else:
               pbar.set_postfix({'warning': 'no valid pairs'})
      
       avg_loss = total_loss / max(num_batches, 1)
      
       if num_batches == 0:
           print(f"\nWARNING: No valid batches processed in epoch {epoch+1}!")
           print("This might indicate a data loading or masking issue.")
      
       print(f"\nEpoch {epoch+1} Training Summary:")
       print(f"  Batches processed: {num_batches}/{len(train_loader)}")
       print(f"  Average loss: {avg_loss:.4f}")
      
       # Validation
       val_metrics = evaluate(probe, extractor, val_loader, accelerator, f"Validation (Epoch {epoch+1})")
      
       print(f"\nEpoch {epoch+1} Validation Summary:")
       print(f"  Train Loss:  {avg_loss:.4f}")
       print(f"  Val F1:      {val_metrics['f1']:.4f}")
       print(f"  Val MCC:     {val_metrics['mcc']:.4f}")
       print(f"  Val AUPRC:   {val_metrics['auprc']:.4f}")
      
       # Early stopping and checkpoint saving
       if val_metrics['f1'] > best_val_f1:
           best_val_f1 = val_metrics['f1']
           best_probe_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
           patience_counter = 0
           print(f"  ✓ New best F1: {best_val_f1:.4f} - saving checkpoint")
          
           # Save checkpoint
           checkpoint_dir = os.path.join(args.output_dir, "checkpoints", args.run_name)
           os.makedirs(checkpoint_dir, exist_ok=True)
           torch.save(best_probe_state, os.path.join(checkpoint_dir, "best_probe.pt"))
       else:
           patience_counter += 1
           print(f"  No improvement ({patience_counter}/{args.patience})")
          
           if patience_counter >= args.patience:
               print(f"\nEarly stopping triggered after {epoch+1} epochs")
               break
      
       print()
  
   # Load best probe
   print(f"\nLoading best probe (F1: {best_val_f1:.4f})")
   probe.load_state_dict(best_probe_state)
  
   return probe




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
  
   # Pad structure matrices to the same size AS THE TOKENIZED SEQUENCES
   # Note: tokenized sequences include special tokens [CLS] and [SEP]
   # so we need to add 2 to the max sequence length
   max_seq_len = max([len(seq) for seq in seqs])
   max_tokenized_len = max_seq_len + 2  # +2 for [CLS] and [SEP] tokens

   structs_padded = []
   for s in structs:
       # Add one row/column at the beginning and end for special tokens
       # These positions will be filled with -1 (padding/masking value)
       padded = np.full((max_tokenized_len, max_tokenized_len), -1, dtype=np.float32)
       # Place the original structure matrix in the middle (skip first and last positions)
       padded[1:s.shape[0]+1, 1:s.shape[1]+1] = s
       structs_padded.append(padded)

   structs_padded = np.array(structs_padded)

   encoded['struct'] = torch.tensor(structs_padded)
   return encoded




def main(args):
   """Main training and evaluation function."""
  
   # Set random seed for reproducibility
   set_seed(args.seed)
  
   # Initialize accelerator (handles multi-GPU, mixed precision, etc.)
   accelerator = Accelerator()
  
   # Print configuration
   print("\n" + "="*70)
   print("RNA SECONDARY STRUCTURE PREDICTION")
   print("Training Linear Probe on Frozen RNABERT Embeddings")
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
   probe = EfficientLinearProbe(hidden_size, min_pair_distance=args.min_pair_distance)
  
   total_params = sum(p.numel() for p in probe.parameters())
   print(f"\nLinear probe created:")
   print(f"  Input:           {hidden_size * 2} (concatenated pair embeddings)")
   print(f"  Output:          1 (pairing probability)")
   print(f"  Min pair dist:   {args.min_pair_distance}")
   print(f"  Total params:    {total_params:,}")
   print(f"  Status:          Randomly initialized")
  
   # Load datasets
   print(f"\nLoading datasets from: {args.data_path}")
  
   # Create collate function
   collate = partial(
       collate_fn,
       tokenizer=tokenizer,
       max_length=args.model_max_length
   )
  
   # Training dataset
   train_dataset = SSDataset(
       data_path=args.data_path,
       tokenizer=tokenizer,
       args=args,
       mode='train'
   )
  
   train_loader = DataLoader(
       train_dataset,
       batch_size=args.batch_size,
       shuffle=True,
       num_workers=args.num_workers,
       collate_fn=collate,
       pin_memory=True
   )
  
   # Validation dataset
   val_dataset = SSDataset(
       data_path=args.data_path,
       tokenizer=tokenizer,
       args=args,
       mode='val'
   )
  
   val_loader = DataLoader(
       val_dataset,
       batch_size=args.batch_size,
       shuffle=False,
       num_workers=args.num_workers,
       collate_fn=collate,
       pin_memory=True
   )
  
   # Test dataset
   test_dataset = SSDataset(
       data_path=args.data_path,
       tokenizer=tokenizer,
       args=args,
       mode='test'
   )
  
   test_loader = DataLoader(
       test_dataset,
       batch_size=args.batch_size,
       shuffle=False,
       num_workers=args.num_workers,
       collate_fn=collate,
       pin_memory=True
   )
  
   print(f"Training set:    {len(train_dataset):,} samples")
   print(f"Validation set:  {len(val_dataset):,} samples")
   print(f"Test set:        {len(test_dataset):,} samples")
  
   # Prepare models with accelerator
   extractor, probe = accelerator.prepare(extractor, probe)
  
   # Train the probe
   probe = train_probe(probe, extractor, train_loader, val_loader, accelerator, args)
  
   # Final evaluation on test set
   print("\n" + "="*70)
   print("FINAL TEST SET EVALUATION")
   print("="*70)
  
   test_metrics = evaluate(probe, extractor, test_loader, accelerator, "Test")
  
   # Also re-evaluate on validation for comparison
   val_metrics = evaluate(probe, extractor, val_loader, accelerator, "Validation (Final)")
  
   # Save results
   results_dir = os.path.join(args.output_dir, "results", args.run_name)
   os.makedirs(results_dir, exist_ok=True)
  
   results = {
       'task': 'secondary_structure_prediction',
       'evaluation_method': 'trained_linear_probe',
       'model_name_or_path': args.model_name_or_path,
       'model_type': args.model_type,
       'token_type': args.token_type,
       'hidden_size': extractor.config.hidden_size,
       'num_layers': extractor.config.num_hidden_layers,
       'num_heads': extractor.config.num_attention_heads,
       'max_length': args.model_max_length,
       'min_pair_distance': args.min_pair_distance,
       'batch_size': args.batch_size,
       'epochs': args.epochs,
       'learning_rate': args.learning_rate,
       'weight_decay': args.weight_decay,
       'seed': args.seed,
       'timestamp': datetime.now().isoformat(),
       'validation': val_metrics,
       'test': test_metrics
   }
  
   results_file = os.path.join(results_dir, "trained_results.json")
   with open(results_file, "w") as f:
       json.dump(results, f, indent=4)
  
   # Print final summary
   print("\n" + "="*70)
   print("TRAINING AND EVALUATION COMPLETE")
   print("="*70)
   print(f"Task:            Secondary Structure Prediction (SSP)")
   print(f"Model:           {args.model_type}")
   print(f"Method:          Trained Linear Probe (RNABERT frozen)")
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
       description='Train and evaluate linear probe on RNABERT for Secondary Structure Prediction'
   )
  
   # Model paths
   parser.add_argument(
       '--model_name_or_path',
       type=str,
       required=True,
       help='Path to RNABERT checkpoint directory'
   )
  
   # Data path
   parser.add_argument(
       '--data_path',
       type=str,
       required=True,
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
  
   # Training parameters
   parser.add_argument(
       '--epochs',
       type=int,
       default=10,
       help='Number of training epochs (default: 10)'
   )
   parser.add_argument(
       '--learning_rate',
       type=float,
       default=1e-3,
       help='Learning rate for linear probe (default: 1e-3)'
   )
   parser.add_argument(
       '--weight_decay',
       type=float,
       default=0.01,
       help='Weight decay for regularization (default: 0.01)'
   )
   parser.add_argument(
       '--patience',
       type=int,
       default=5,
       help='Early stopping patience (default: 5)'
   )
  
   # Evaluation parameters
   parser.add_argument(
       '--batch_size',
       type=int,
       default=8,
       help='Batch size for training and evaluation (default: 8)'
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
       help='Name for this training run (auto-generated if not provided)'
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
       args.run_name = f"rnabert_ssp_trained_{model_name}"
  
   # Validate inputs
   if not os.path.exists(args.model_name_or_path):
       print(f"ERROR: Model path not found: {args.model_name_or_path}")
       sys.exit(1)
  
   if not os.path.exists(args.data_path):
       print(f"ERROR: Data path not found: {args.data_path}")
       sys.exit(1)
  
   # Run training and evaluation
   main(args)

