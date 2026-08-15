"""
data_prep.py
------------------
Download the TinyStories dataset, 
tokenize with BPE tokenizer.

Write token stream to disk as flat uint16 binary 
files (train.bin / val.bin) using np.memmap.

Run once for preprocessing:
    python data_prep.py

Downstream (train.py) reads the .bin files directly off disk,
This prevents the full corpus from sitting in RAM.

"""

import os
import numpy as np
import tiktoken
from tqdm import tqdm
from datasets import load_dataset

#---------------------------------
# config
#---------------------------------
DATA_DIR = 'data'
DATASET_NAME = 'roneneldan/TinyStories'
NUM_PROC = 6 # number of processes for tokenization
NUM_PROC_LOAD = 6 # processes for intial download/prep
SHARDS = 1024 # how many chunks to stream to disk

# GPT-2 BPE. 
enc = tiktoken.get_encoding('gpt2')

def process(story):
    """
    Tokenize one story and append end-of-text token as seperator.

    """
    ids = enc.encode_ordinary(story['text']) # ignores special characters
    ids.append(enc.eot_token) # appends end-of-text token to mark story boundary
    return {'ids': ids, 'len': len(ids)}

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    dataset = load_dataset(DATASET_NAME, num_proc=NUM_PROC_LOAD)

    tokenized = dataset.map(
        process,
        remove_columns=['text'],
        desc='tokenizing the splits',
        num_proc=NUM_PROC
    )

    # write each split to its own flat binary file
    for split, dset in tokenized.items():
        filename = 'val.bin' if split == 'validation' else f'{split}.bin'
        path = os.path.join(DATA_DIR, filename)

        # preallocate memmap to right length
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        arr = np.memmap(path, dtype=np.uint16, mode='w+', shape=(int(arr_len)))

        # stream tokens to disk in shards
        idx = 0
        for batch_idx in tqdm(range(SHARDS), desc=f"writing {path}"):
            batch = dset.shard(
                num_shards=SHARDS, index=batch_idx, contiguous=True
            ).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

        print(f"{path}: {arr_len:,} tokens")

def test_print(s: str) -> str:
    print(s)

if __name__ == '__main__':
    test_print('Hello World')

                              
        




