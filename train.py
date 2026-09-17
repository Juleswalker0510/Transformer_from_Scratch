"""
train.py
---------------------------
trains the transformer in model.py on the token files created in data_prep.py

simple by design: plain fp32, plain AdamW, one straight loop. It reads vocab_size
meta.pkl and pulls batches through the shared get_batch(), so the data 
pipeline and model stay in sync.

run: python train.py
"""

# imports
import pickle
from contextlib import nullcontext
import matplotlib.pyplot as plt

import torch
from tokenizers import Tokenizer

from model import Transformer, Config
from data_prep import get_batch, DATA_DIR

# ------------------------------------------
# hyperparameters
# ------------------------------------------
block_size    = 256   # MUST match what is passed to get_batch
batch_size    = 32    
max_iters     = 7500  # total training steps
eval_interval = 250   # evaluate + print every this many steps
eval_iters    = 50    # batches averaged per evaluation
learning_rate = 3e-4

# ---------------------------------------------
# setup
# ---------------------------------------------
torch.manual_seed(1337) # seed for reproducability 
# device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:',device)

# speed up nvidia gpu
torch.set_float32_matmul_precision("high")
if device == "cuda":
    autocast = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
else:
    autocast = nullcontext()

# vocab_size taken from data pipeline
with open(DATA_DIR / 'meta.pkl', 'rb') as f:
    meta = pickle.load(f)

config = Config(
    vocab_size=meta['vocab_size'], block_size=block_size,
    n_layer=6, n_head=6, n_embd=384, dropout=0.1
)

model = Transformer(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    """
    Average loss over a few batches of train and val.
    """
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, block_size, batch_size, device)
            with autocast:
                _, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# -----------------------------------------
# training loop
# -----------------------------------------
history = [] # store losses
for it in range(max_iters + 1):
    if it % eval_interval == 0:
        losses = estimate_loss()
        history.append((it, losses['train'], losses['val']))
        print(f"iter {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    X, Y = get_batch('train', block_size, batch_size, device)
    with autocast:
        _, loss = model(X,Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# --------------------------------------------
# save + sample + loss plot 
# --------------------------------------------
its, tr, va = zip(*history[1:]) # skip it=0 baseline
plt.plot(its, tr, label='train')
plt.plot(its, va, label='val')
plt.xlabel('iteration')
plt.ylabel('cross-entropy loss')
plt.legend()
plt.title('Training loss')
plt.savefig('assets/loss.png', dpi=150, bbox_inches='tight')

torch.save({"model": model.state_dict(), "config": config.__dict__, "meta": meta}, "ckpt.pt")
print('saved checkpoint -> ckpt.pt')
tok = Tokenizer.from_file(meta['tokenizer_path'])
context = torch.tensor([tok.encode('Once upon a time').ids], dtype=torch.long, device=device)
sample = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=200)
print("\n--- sample ---")
print(tok.decode(sample[0].tolist()))