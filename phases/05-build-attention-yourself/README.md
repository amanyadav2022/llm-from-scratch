# Phase 5 — Build Attention Yourself

*Reading about attention (Phase 4) is not the same as building it. This phase has no new theory — it's you, writing the code, with nothing hidden behind a library shortcut.*

---

## Step 1: Why You Cannot Skip This Phase

`nn.MultiheadAttention` in PyTorch is a black box — it works, but it hides every shape transformation and every step behind one function call. If you only ever call that function, you'll never truly know what's happening inside vLLM's attention kernels later (Phase 9), because those kernels are hand-written implementations of exactly what you're about to build.

The rule for this phase: **no `nn.MultiheadAttention`, no `torch.nn.functional.scaled_dot_product_attention`.** Only basic building blocks — `nn.Linear`, matrix multiplication, softmax, reshape.

---

## Step 2: Build Plan — Increasing Difficulty

Build these four things, in order, each one reusing the previous:

```mermaid
flowchart LR
    A["1. Single-head\nattention"] --> B["2. Multi-head\nattention"]
    B --> C["3. Causal\nmasking"]
    C --> D["4. Full attention\nlayer as a class"]
```

### 1. Single-head attention (one function)

Write a function that takes `Q`, `K`, `V` tensors and returns the attention output, using only the formula from Phase 4 Part 2:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

```python
import torch
import torch.nn.functional as F

def single_head_attention(Q, K, V):
    # Q, K, V: shape (seq_len, dim)
    d_k = Q.shape[-1]
    scores = Q @ K.T / (d_k ** 0.5)      # shape (seq_len, seq_len)
    weights = F.softmax(scores, dim=-1)  # shape (seq_len, seq_len)
    output = weights @ V                  # shape (seq_len, dim)
    return output

# Test it
Q = torch.randn(4, 8)
K = torch.randn(4, 8)
V = torch.randn(4, 8)
out = single_head_attention(Q, K, V)
print(out.shape)   # should print torch.Size([4, 8])
```

**Do this yourself first**, without copying — then compare against the snippet above.

### 2. Multi-head attention (a class)

Now wrap this in a full module: project the input into Q, K, V using learned `nn.Linear` layers, split into heads, run attention per head, concatenate, and project back out.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, dim=64, n_heads=8):
        super().__init__()
        assert dim % n_heads == 0, "dim must divide evenly into heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: shape (batch, seq_len, dim)
        B, T, D = x.shape

        Q = self.q_proj(x)   # (B, T, D)
        K = self.k_proj(x)   # (B, T, D)
        V = self.v_proj(x)   # (B, T, D)

        # split into heads: (B, T, D) -> (B, n_heads, T, head_dim)
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # attention per head, all heads computed in parallel
        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)   # (B, n_heads, T, T)
        weights = F.softmax(scores, dim=-1)                          # (B, n_heads, T, T)
        out = weights @ V                                             # (B, n_heads, T, head_dim)

        # merge heads back: (B, n_heads, T, head_dim) -> (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        return self.out_proj(out)   # (B, T, D) -- same shape as input

# Test it
mha = MultiHeadAttention(dim=64, n_heads=8)
x = torch.randn(2, 10, 64)   # (batch=2, seq_len=10, dim=64)
out = mha(x)
print(out.shape)   # should print torch.Size([2, 10, 64])
```

**Trace every shape yourself on paper before running it.** This is the exercise from Phase 2, Step 1 — narrate what shape each line produces before checking.

### 3. Add causal masking

Modify the class so it can optionally prevent attending to future tokens (Phase 4 Part 2, Step 5):

```python
def forward(self, x, causal=False):
    B, T, D = x.shape
    Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    K = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    V = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)   # (B, n_heads, T, T)

    if causal:
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()   # upper triangle = future
        scores = scores.masked_fill(mask, float('-inf'))          # block the future

    weights = F.softmax(scores, dim=-1)
    out = weights @ V
    out = out.transpose(1, 2).contiguous().view(B, T, D)
    return self.out_proj(out)
```

`torch.triu(..., diagonal=1)` builds exactly the "future positions" grid from Phase 4 Part 2 — everywhere `col > row` becomes `True`, and `masked_fill` sets those scores to negative infinity so softmax turns them into exactly 0.

**Verify it worked:** print `weights[0, 0]` (the attention weights for head 0, batch 0) with `causal=True` — you should see a lower-triangular pattern, all zeros above the diagonal.

### 4. Sanity-check against PyTorch's own implementation

Once your version works, compare its output shape (not necessarily exact values, since weight initialization differs) against the built-in version, to build confidence you built it correctly:

```python
built_in = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)
x = torch.randn(2, 10, 64)
out, _ = built_in(x, x, x)
print(out.shape)   # torch.Size([2, 10, 64]) -- same shape your version produces
```

---

## Step 3: Practical Tips While Building

- **Print shapes obsessively.** After every line that changes a tensor's shape, add a `print(x.shape)` while developing, then remove them once it works.
- **Test with tiny numbers first.** Use `seq_len=4`, `dim=8`, `n_heads=2` while debugging — small enough to reason about by hand, large enough to catch real bugs.
- **The most common bug** is forgetting `.contiguous()` before a `.view()` call after a `.transpose()` — PyTorch will throw a clear error when this happens; read it, it's usually telling you exactly this.
- **Don't move on until `causal=True` produces the lower-triangular zero pattern you expect.** This is the single most important thing to verify by hand.

---

## Checkpoint

1. Without looking at your code, write the attention formula from memory.
2. Explain what `.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)` is doing to the tensor, step by step.
3. Why does causal masking use negative infinity instead of just zero?
4. What would happen if you forgot the `/ sqrt(head_dim)` scaling step?

## Resources

- Andrej Karpathy - Let's build GPT from scratch: https://www.youtube.com/watch?v=kCc8FmEb1nY (codes exactly this, line by line)
- PyTorch - nn.MultiheadAttention docs: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
- PyTorch - scaled_dot_product_attention docs: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
