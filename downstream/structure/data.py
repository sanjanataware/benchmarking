import warnings
warnings.filterwarnings("ignore")


import os
import re
import csv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset




def generate_kmer_str(sequence: str, k: int) -> str:
   """Generate k-mer string from a DNA sequence."""
   return " ".join([sequence[i:i + k] for i in range(len(sequence) - k + 1)])




class SSDataset(Dataset):
   """
   Dataset for RNA secondary structure prediction (SSP) tasks.
   Loads sequences and structure (.npy) matrices for train/val/test splits.
   """


   def __init__(self, data_path, tokenizer, args, mode):
       df = pd.read_csv(f"{data_path}/bpRNA.csv")


       # Choose correct data subset
       mode_map = {"train": "TR0", "val": "VL0", "test": "TS0"}
       if mode not in mode_map:
           raise ValueError("Mode must be one of: 'train', 'val', 'test'")


       subset = mode_map[mode]
       df = df[df["data_name"] == subset].reset_index(drop=True)
       data_path = f"{data_path}/{subset}"


       print(f"Original dataset size: {len(df)}")


       # Filter only valid files
       valid_mask = df["file_name"].apply(lambda f: os.path.isfile(os.path.join(data_path, f + ".npy")))
       df = df[valid_mask].reset_index(drop=True)
       print(f"Filtered dataset size (files exist): {len(df)}")


       self.df = df
       self.data_path = data_path
       self.tokenizer = tokenizer
       self.args = args
       self.num_labels = 1


       # Tokenizer test
       token_test = df.iloc[0]["seq"].upper().replace("U", "T")
       if "mer" in self.args.token_type:
           k = int(re.findall(r"\d+", self.args.token_type)[0])
           token_test = generate_kmer_str(token_test, k)


       if getattr(args, "debug", False):
           print(token_test)
           print(tokenizer.tokenize(token_test))
           print(tokenizer(token_test))


       print(f"Dataset ready. Total samples: {len(self.df)}")


   def __len__(self):
       return len(self.df)


   def __getitem__(self, idx):
       row = self.df.iloc[idx]
       seq = row["seq"].upper().replace("U", "T")
       file_path = os.path.join(self.data_path, row["file_name"] + ".npy")


       struct = np.load(file_path).astype(np.float32)
       max_len = self.tokenizer.model_max_length - 2


       if len(seq) > max_len:
           seq = seq[:max_len]
           struct = struct[:max_len, :max_len]


       return {"seq": seq, "struct": struct, "id": row["file_name"]}




class ContactMapDataset(Dataset):
   """
   Dataset for RNA contact map prediction.
   Each entry contains a sequence (tokenized) and its corresponding contact map (.npy).
   """


   def __init__(self, data_path, tokenizer, args):
       with open(data_path, "r") as f:
           data = list(csv.reader(f))[1:]  # skip header


       if len(data[0]) == 2:
           texts = [d[1].upper().replace("U", "T") for d in data]
           ids = [d[0] for d in data]
       else:
           raise ValueError("Data format not supported — expected [id, sequence].")


       self.tokenizer = tokenizer
       self.args = args
       self.ids = ids
       self.texts = texts
       self.num_labels = 1
       self.data_path = data_path


       parent_dir = os.path.dirname(data_path)
       self.target_path = os.path.join(parent_dir, "contact_map")


       if getattr(args, "debug", False):
           print(texts[0])
           print(tokenizer.tokenize(texts[0]))
           print(len(tokenizer.tokenize(texts[0])))
           print(tokenizer(texts[0]))


   def __len__(self):
       return len(self.texts)


   def __getitem__(self, idx):
       seq_id = self.ids[idx]
       target_file = os.path.join(self.target_path, seq_id + ".npy")


       struct = np.load(target_file).astype(np.float32)
       seq = self.texts[idx]


       max_len = self.tokenizer.model_max_length - 2
       if len(seq) > max_len:
           seq = seq[:max_len]
           struct = struct[:max_len, :max_len]


       return {"seq": seq, "struct": struct, "id": seq_id}




class DistanceMapDataset(Dataset):
   """
   Dataset for RNA distance map prediction.
   Each entry contains a sequence (tokenized) and its corresponding distance map (.npy).
   """


   def __init__(self, data_path, tokenizer, args):
       with open(data_path, "r") as f:
           data = list(csv.reader(f))[1:]  # skip header


       if len(data[0]) == 2:
           texts = [d[1].upper().replace("U", "T") for d in data]
           ids = [d[0] for d in data]
       else:
           raise ValueError("Data format not supported — expected [id, sequence].")


       self.tokenizer = tokenizer
       self.args = args
       self.ids = ids
       self.texts = texts
       self.num_labels = 1
       self.data_path = data_path


       parent_dir = os.path.dirname(data_path)
       self.target_path = os.path.join(parent_dir, "distance_map")


       if getattr(args, "debug", False):
           print(texts[0])
           print(tokenizer.tokenize(texts[0]))
           print(len(tokenizer.tokenize(texts[0])))
           print(tokenizer(texts[0]))


   def __len__(self):
       return len(self.texts)


   def __getitem__(self, idx):
       seq_id = self.ids[idx]
       target_file = os.path.join(self.target_path, seq_id + ".npy")


       struct = np.load(target_file).astype(np.float32)
       seq = self.texts[idx]


       max_len = self.tokenizer.model_max_length - 2
       if len(seq) > max_len:
           seq = seq[:max_len]
           struct = struct[:max_len, :max_len]


       return {"seq": seq, "struct": struct, "id": seq_id}




