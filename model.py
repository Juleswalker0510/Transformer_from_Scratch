"""
model.py
--------------------
Small decoder-only transformer for next-token prediction,
written using separated classes for the following:

    PositionalEncoding      adds fixed sinusoidal position info to embeddings
    MultiHeadAttention      lets tokens attend to earlier tokens (causal)
    PosWiseFeedForward      a per-position MLP
    Block                   one layer = attention + feed-forward
    Transformer             embeddings + N Blocks + output head

shape convention: B = batch, T = sequence length, C = n_embd, V = vocab_size

is compatible with data_prep.py: forward() takes (B, T) token ids
exactly as get_batch returns them, and config.vocab_size / config.block_size 
are the only values that must match the data pipeline.
"""

# imports 
import math 
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F  

# data class for architecture configs
@dataclass 
class Config:
    vocab_size: int        # size of token vocab (tokenizer.get_vocab_size())
    block_size: int = 256  # max context length for model to view
    n_layer: int = 6       # number of stacked transformer blocks
    n_head: int = 6        # attention heads per block (n_embd must divide)
    n_embd: int = 384      # width of the residual stream
    dropout: float = 0.1   # dropout proportion 


class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (not learned) as used by the 2017 paper 
    "Attention is all you need". 

    Precomputes a (block_size, n_embd) table of sine/cosine values at 
    geometrically spaced frequencies and adds it to each token embedding,
    giving each position a unique, order-aware signal.
    """
    def __init__(self, n_embd: int, block_size: int, dropout: float=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(block_size, n_embd) # (block_size, C)
        position = torch.arange(0, block_size).unsqueeze(1).float() # (block_size, 1)
        # frequencies: geometric spacing from 1 down to ~1/10000
        div_term = torch.exp(torch.arange(0, n_embd, 2).float()
                             * (-math.log(10000.0) / n_embd))
        pe[:,0::2] = torch.sin(position * div_term) # even -> sine
        pe[:,1::2] = torch.cos(position * div_term)  # odd  -> cosine
        self.register_buffer('pe', pe) # (block_size, C)

    def forward(self, x):
        # x = (B, T, C). add encoding for the first T positions and broadcast over
        # the batch dims.
        x = x + self.pe[: x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Masked multi-head self-attention.

    Projects the input into query/key/value, splits them across n_head heads,
    computes scaled dot-product attention with lower-triangular mask ensuring
    each position attends only itself and earlier ones, recombines heads and
    projects back to n_embd.

    Input and Output are both (B, T, n_embd)
    """
    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, \
            'n_embd must be divisible by n_head'
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # query/key/value projections
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key   = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.proj  = nn.Linear(config.n_embd, config.n_embd) # combines heads
        self.dropout = nn.Dropout(config.dropout)

        # lower-triangluar mask
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer('mask', mask.view(1,1,config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape

        # project, then reshape into heads: (B,T,C) -> (B, n_head, T, head_dim)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)

        # attention scores scaled by 1/sqrt(head_dim)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        # block out the future then normalize into probabilities
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v # (B, n_head, T, head_dim)
        y = y.transpose(1,2).contiguous().view(B,T,C) # recombine heads
        return self.proj(y)


class PosWiseFeedForward(nn.Module):
    """
    Position-wise feed-forward layer.

    Two-layer MLP (n_embd -> 4*n_embd -> n_embd) with GELU activation function.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4*config.n_embd),
            nn.GELU(),
            nn.Linear(4*config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Single pre-norm transformer block.

    Applies LayerNorm then masked self-attention, and LayerNorm 
    then the feed-forward network, each wrapped in a residual connection.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = PosWiseFeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class Transformer(nn.Module):
    """
    Decoder-only (GPT-style) transformer language model.

    Embeds token IDs, adds positional encoding, runs the sequence through
    a stack of n_layer blocks, applies final LayerNorm, projects to per-position
    vocab logits.

    forward() optionally returns next-token cross-entropy loss.
    generate() samples autogregressively.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_encoding = PositionalEncoding(config.n_embd, config.block_size, config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size)

        self.apply(self._init_weights)
        print(f"model parameters: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, \
        f"sequence length {T} exceeds block_size {self.config.block_size}"

        x = self.token_embedding(idx) # (B, T, C)
        x = self.pos_encoding(x) # add position information
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x) # (B, T, V)

        loss = None
        if targets is not None:
            # compare preds at every position to the true next token
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Sample tokens one at a time, feeding each back in. idx: (B, T0)."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]   # keep the last block_size tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature       # focus on the final position
            if top_k is not None:                          # optionally keep only top-k
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

