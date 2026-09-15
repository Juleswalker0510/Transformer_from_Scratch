# Decoder-Only Transformer Model from Scratch (PyTorch)
Building a minimal **decoder-only transformer** using PyTorch, a ground up implementation. Core components: sinusoidal positional encoding, masked multi-head self-attention, position-wise feed-forward layers, and pre-norm residual blocks. All core components built by hand using classes that extend `nn.Module` rather than using `torch.nn.Transformer`, with the goal of developing a deeper more intuitive understanding of underlying logic and foundational components that make up the architecture. 


## Highlights
- Pure PyTorch, no high-level transformer abstractions.
- Explicit attention math, causal masking, and residual wiring.
- Use of **pre-norm** blocks, **scaled dot-product attention**, GELU feed-forward, and GPT-2 style weight initialization.
- Trains on a standard **next-token prediction** objective, every position in a sequence contributes to the loss in a single forward pass.
- Autoregressive text generation with **temperature** and **top-k** sampling.
- Configuration-driven architecture via a single `Config` dataclass.


## Architecture
Data flows from token IDs to probability distribution over the vocabulary:
```mermaid
flowchart TD
    A["token ids &nbsp;(B, T)"] --> EMB["Token Embedding"]
    POS["Sinusoidal Positional Encoding"] --> ADD0(("+"))
    EMB --> ADD0
    ADD0 --> BIN["block input"]

    subgraph BLK["Transformer Block &nbsp;× N"]
        direction TB
        BIN --> LN1["LayerNorm"]
        LN1 --> MHA["Masked Multi-Head Attention"]
        MHA --> R1(("+"))
        BIN -. residual .-> R1

        R1 --> LN2["LayerNorm"]
        LN2 --> FF["FeedForward"]
        FF --> R2(("+"))
        R1 -. residual .-> R2
    end

    R2 --> FLN["Final LayerNorm"]
    FLN --> HEAD["Linear Head"]
    HEAD --> LOG["logits &nbsp;(B, T, vocab_size)"]
    LOG --> SM["softmax"]
    SM --> OUT["sample next token"]

    classDef io fill:#1f6feb,stroke:#0d3a8f,color:#fff;
    classDef add fill:#2ea043,stroke:#136227,color:#fff;
    class A,LOG,OUT io;
    class ADD0,R1,R2 add;
```

| Module | Responsibility |
| ------ | -------------- |
| `PositionalEncoding` | Fixed sinusoidal position signal added to the token embeddings so the attention layers can reason about order. |
| `MultiHeadAttention` | Masked causal multi-head self-attention, each position attends to only itself and previous positions. |
| `PosWiseFeedForward` | Two-Layer MLP (`n_embd -> 4*n_embd -> n_embd`) with GELU, applied identically at every position. |
| `Block` | One pre-norm transformer block: `x = x + attn(ln1(x))` then `x = x + ffn(ln2(x))`. |
| `Transformer` | Assembles the embeddings, block stack, final norm, and output head. Computes the training loss and defines the generation loop. |


## Model configuration
Architecture is defined by the Config dataclass. 

| Field | Meaning | Example |
| -- | -- | -- |
| `vocab_size` | Size of token vocabulary | Taken from meta with meta['vocab_size'] |
| `block_size` | Maximum context length | 256 |
| `n_embd` | Residual-stream width | 384 |
| `n_head` | Number of attention heads | 6 |
| `n_layer` | Number of transformer blocks | 6 |
| `dropout` | Dropout probability | 0.1 |

## Installation


## Usage


## Implementation notes
Some design choices
- **Pre-norm blocks**. LayerNorm is applied before each sub-layer, with a residual add outside the norm. This leaves a clean unnormalized gradient from input to output and trains stably at depth. This is the current standard in contrast to post-norm used in the original 2017 paper.
- **Causal masking**. A lower triangular mask blocks each position from attending to future tokens. This makes the model autoregressive and lets all positions be trained as independent next-token predictions in one forward pass.
- **Scaled dot-product attention**. Attention scores are divided by `sqrt(head_dim)` to keep the softmax out of its saturated, low-gradient regime.
- **Weight initialization**. Linear and embedding weights are drawn from `N(0, 0.02)` with zero biases, following GPT-2 convention.

## Potential next steps and extensions

## Results 





