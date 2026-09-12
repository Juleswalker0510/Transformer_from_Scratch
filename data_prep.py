"""
data_prep.py
----------------------------------
Downloads the TinyStories dataset, trains a small byte-level BPE tokenizer on it,
tokenizes corpus, streams token ids to disk as uint16 binary files ready for
next-token-prediction training.

Run once:
   pip install datasets tokenizers numpy torch
   python data_prep.py

Outputs (in ./data/tinystories/):
   tokenizer.json | the trained BPE tokenizer (load using Tokenizer.from_file)
   train.bin      | training token ids, flat uint16 array
   val.bin        | validation token ids, flat uint16 array
   meta.pkl       | vocab_size, eot_id, token_counts

Import get_batch() in training script later to pull (x, y) minibatches.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset 
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# --------------------------
# CONFIG 
# --------------------------
DATA_DIR = Path("data/tinystories")
DATASET_NAME = "roneneldan/TinyStories"

VOCAB_SIZE = 8192 # small vocab size for a smaller embedding table
DTYPE = np.uint16 # vocab < 65536 fits in uint16
EOT_TOKEN = "<|endoftext|>"

TOKENIZER_TRAIN_SAMPLE = 200_000 # stories to learn the merges 
MAX_TRAIN_STORIES = None # how many stories to train on, change for faster iteration
ENCODE_BATCH = 1000

# ---------------------------
# DATASET
# ---------------------------
def load_splits():
    ds = load_dataset(DATASET_NAME)
    train, val = ds['train'], ds['validation']
    if MAX_TRAIN_STORIES is not None:
        train = train.select(range(min(MAX_TRAIN_STORIES, len(train))))
    print(f"train stories: {len(train):,} val stories: {len(val):,}")
    return train, val

# --------------------------------
# BUILD TOKENIZER
# --------------------------------
def build_tokenizer(train_ds):
    tok_path = DATA_DIR / "tokenizer.json"
    if tok_path.exists():
        print(f"loading existing tokenizer from {tok_path}")
        return Tokenizer.from_file(str(tok_path))

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[EOT_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True
    )

    n = min(TOKENIZER_TRAIN_SAMPLE, len(train_ds))
    print(f"training tokenizer on {n:,} stories...")

    def text_iter():
        for i in range(n):
            yield train_ds[i]["text"]

    tokenizer.train_from_iterator(text_iter(), trainer=trainer, length=n)
    tokenizer.save(str(tok_path))
    print(f"tokenizer saved to {tok_path} (vocab={tokenizer.get_vocab_size()})")
    return tokenizer

# --------------------------------------------------
# TOKENIZE A SPLIT AND STREAM TO FLAT .BIN FILE
# --------------------------------------------------
def encode_split(ds, tokenizer, out_path, eot_id):
    """
    Encode all stories, append EOT after each, write ids as uint16 to disk.

    Writing incrementally keeps only ENCODE_BATCH stories in memory at one time.
    This allows smooth tokenization on personal laptop for larger dataset.
    """
    n = len(ds)
    total = 0
    with open(out_path, 'wb') as f:
        for start in range(0, n, ENCODE_BATCH):
            texts = ds[start:start + ENCODE_BATCH]["text"]
            encodings = tokenizer.encode_batch(texts)
            ids = []
            for enc in encodings:
                ids.extend(enc.ids)
                ids.append(eot_id)
            np.array(ids, dtype=DTYPE).tofile(f)
            total += len(ids)
            if start % (ENCODE_BATCH * 50) == 0:
                print(f"  {out_path.name}: {start:,}/{n:,} stories"
                      f"({total:,} tokens)")
    print(f"{out_path.name}: {total:,} tokens written")
    return total

# -------------------------------------------------
# BATCH LOADER 
# -------------------------------------------------
def get_batch(split, block_size, batch_size, device='cpu'):
    """
    Return (x,y) for next-token prediction.

    x, y are LongTensors of shape (batch_size, block_size): y is x shifted by one.
    The memmap is re-opened each call on purpose to avoid a slow memory leak

    """
    path = DATA_DIR / ('train.bin' if split == 'train' else 'val.bin')
    data = np.memmap(path, dtype=DTYPE, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# ------------------------------------------------------------------
# PUTTING EVERYTHING TOGETHER 
# ------------------------------------------------------------------
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = load_splits()
    tokenizer = build_tokenizer(train_ds)
    eot_id = tokenizer.token_to_id(EOT_TOKEN)

    train_tokens = encode_split(train_ds, tokenizer, DATA_DIR / "train.bin", eot_id)
    val_tokens = encode_split(val_ds, tokenizer, DATA_DIR / "val.bin", eot_id)

    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "eot_id": eot_id,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "tokenizer_path": str(DATA_DIR / "tokenizer.json")
    }

    with open(DATA_DIR / "meta.pkl", 'wb') as f:
        pickle.dump(meta, f)
    print(f'\nmeta: {meta}')

    # sanity check - pull single batch and round-trip a sequence through tokenizer
    x, y = get_batch('val', block_size=64, batch_size=4, device='cpu')
    print(f"\nsample batch x={tuple(x.shape)} y={tuple(y.shape)} dtype={x.dtype}")
    print(f"decoded x[0]:")
    print(f"   {tokenizer.decode(x[0].tolist())}")


if __name__ == '__main__':
    main()







