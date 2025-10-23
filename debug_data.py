#!/usr/bin/env python3
"""Debug script to inspect data loading issues."""

import numpy as np
import os
import pandas as pd

# Check structure matrices
data_path = '/home/ali/DMS/DMS-FM/Downstream_Tasks/RNABenchmark/data/Secondary_structure_prediction/bpRNA'
tr_path = f"{data_path}/TR0"

# Load CSV
df = pd.read_csv(f"{data_path}/bpRNA.csv")
train_df = df[df["data_name"] == "TR0"].reset_index(drop=True)

print(f"Total training samples in CSV: {len(train_df)}")

# Check first valid file
for idx in range(min(5, len(train_df))):
    row = train_df.iloc[idx]
    file_path = os.path.join(tr_path, row["file_name"] + ".npy")

    if os.path.exists(file_path):
        struct = np.load(file_path)
        seq = row["seq"]

        print(f"\n{'='*60}")
        print(f"Sample {idx}: {row['file_name']}")
        print(f"{'='*60}")
        print(f"Sequence length: {len(seq)}")
        print(f"Structure shape: {struct.shape}")
        print(f"Structure dtype: {struct.dtype}")
        print(f"Unique values: {np.unique(struct)}")
        print(f"Value counts:")
        unique, counts = np.unique(struct, return_counts=True)
        for val, count in zip(unique, counts):
            print(f"  {val}: {count} ({100*count/struct.size:.2f}%)")

        # Check if it's a binary pairing matrix
        print(f"\nIs binary (0/1)? {set(np.unique(struct)).issubset({0, 1, -1})}")
        print(f"Is symmetric? {np.allclose(struct, struct.T, equal_nan=True)}")

        # Check diagonal
        print(f"Diagonal sum: {np.diag(struct).sum()}")

        # Show a sample region
        print(f"\nFirst 15x15 corner:")
        print(struct[:15, :15].astype(int))

        break
