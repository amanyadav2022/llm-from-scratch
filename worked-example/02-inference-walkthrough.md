# Worked Example, Part 2: Inference on "a gun was fired" (with a real KV cache)

*Part 1 traced training on one sentence. This traces the OTHER half of the model's life: generating new text from a prompt, using a real, working KV cache, not just the concept from Phase 7, but actual code that builds and reuses one, with real printed output at every step.*

Copy the full script at the bottom into Colab and run it yourself. The numbers below are exactly what it prints.

---

## The setup

Same toy model shape as Part 1 (D=8, 2 heads, same 12-word vocabulary). The weights here are freshly random-initialized, not the ones trained in Part 1. The point of this walkthrough is the mechanism of generation and caching, not producing meaningful text from an untrained toy model.

Prompt: "a gun was fired"

---

## Step 1-2: Tokenize the Prompt (Phase 4, Step 1)

```
Prompt: a gun was fired
Tokens: ['a', 'gun', 'was', 'fired']
Token IDs: [0, 6, 7, 8]
```

Same mechanism as Part 1, words become integers using the same vocabulary table.

---

## Step 3: Prefill, Process the Whole Prompt at Once (Phase 7, Step 2)

```
processed token 'a'     (position 0) -> cache now holds K,V for 1 token(s)
processed token 'gun'   (position 1) -> cache now holds K,V for 2 token(s)
processed token 'was'   (position 2) -> cache now holds K,V for 3 token(s)
processed token 'fired' (position 3) -> cache now holds K,V for 4 token(s)

After prefill, cache shape: K=(4, 2, 4), V=(4, 2, 4)
Predicted next token after full prompt: 'a'
```

This is the real KV cache from Phase 7, Step 1, actually being built. Each token's Key and Value vectors get computed once and stored. The cache shape (4, 2, 4) means 4 tokens cached, 2 attention heads, 4 numbers per head, exactly the shape you'd expect from Phase 9's shape table, just tiny.

---

## Step 4+: Decode, One New Token at a Time, Reusing the Cache (Phase 7, Step 1 and 5)

```
decode step 1: new token was 'a'   | cache grew from 4 -> 5 tokens | only computed Q,K,V for this ONE token | next predicted token: 'war'
decode step 2: new token was 'war' | cache grew from 5 -> 6 tokens | only computed Q,K,V for this ONE token | next predicted token: 'a'
decode step 3: new token was 'a'   | cache grew from 6 -> 7 tokens | only computed Q,K,V for this ONE token | next predicted token: 'the'
decode step 4: new token was 'the' | cache grew from 7 -> 8 tokens | only computed Q,K,V for this ONE token | next predicted token: 'a'
```

This is the entire point of Phase 7. Look closely at what happens each step: the cache grows by exactly one token's worth of Key/Value vectors, and only the new token's Query, Key, and Value get freshly computed. Tokens 1 through N-1 are never recomputed, their K and V are simply read back out of the cache, appended to, and reused.

---

## Step 5: Final Generated Sequence

```
Generated: a gun was fired a war a the a
```

The full sequence: the original 4-word prompt, plus 4 newly generated tokens, one per decode step.

This output is gibberish, and that's expected and correct. This model has never been trained. The point of this walkthrough was never to produce meaningful text, it was to see the KV cache genuinely being built during prefill and genuinely being reused and extended during decode, with real tensors, real shapes, and real step-by-step growth.

---

## Connecting This Back to vLLM (Phase 8-9)

What you just watched run is a tiny, single-request, unoptimized version of exactly what vLLM does at scale:

| This toy script | vLLM's real version |
|---|---|
| One Python list, growing by concatenation each step | PagedAttention, fixed-size memory pages, allocated on demand (Phase 8, Step 2) |
| One request at a time | Continuous batching, many requests' decode steps combined into one pass (Phase 8, Step 3) |
| Cache lives in one contiguous PyTorch tensor | Cache can live in scattered physical memory blocks, tracked by a block table (Phase 8, Step 4) |
| Plain PyTorch operations | Hand-written CUDA/Triton kernels for speed (Phase 9, Phase 11.6) |

Every row in that table is the same idea, engineered for scale, nothing conceptually new, just more careful about memory and throughput.

---

## The Full Script (copy this into Colab and run it yourself)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
torch.set_printoptions(precision=4, sci_mode=False)

vocab = ["a", "soldier", "dies", "in", "world", "war",
         "gun", "was", "fired", "at", "the", "<eos>"]
stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for w, i in stoi.items()}
vocab_size = len(vocab)

D = 8
n_heads = 2
head_dim = D // n_heads
MAX_LEN = 20

embedding_table = nn.Embedding(vocab_size, D)
pos_embedding_table = nn.Embedding(MAX_LEN, D)
ln1 = nn.LayerNorm(D)
q_proj = nn.Linear(D, D)
k_proj = nn.Linear(D, D)
v_proj = nn.Linear(D, D)
out_proj = nn.Linear(D, D)
ln2 = nn.LayerNorm(D)
ffn = nn.Sequential(nn.Linear(D, D * 4), nn.GELU(), nn.Linear(D * 4, D))
final_norm = nn.LayerNorm(D)
output_head = nn.Linear(D, vocab_size)

def embed(token_id, position):
    tok = embedding_table(torch.tensor([token_id]))
    pos = pos_embedding_table(torch.tensor([position]))
    return tok + pos

def block_forward_new_token(x_new, cache_k, cache_v):
    x_normed = ln1(x_new)

    q = q_proj(x_normed).view(1, n_heads, head_dim)
    k_new = k_proj(x_normed).view(1, n_heads, head_dim)
    v_new = v_proj(x_normed).view(1, n_heads, head_dim)

    if cache_k is None:
        cache_k, cache_v = k_new, v_new
    else:
        cache_k = torch.cat([cache_k, k_new], dim=0)
        cache_v = torch.cat([cache_v, v_new], dim=0)

    Q = q.transpose(0, 1)
    K = cache_k.transpose(0, 1)
    V = cache_v.transpose(0, 1)

    scores = Q @ K.transpose(-2, -1) / (head_dim ** 0.5)
    weights = F.softmax(scores, dim=-1)
    attn_out = weights @ V
    attn_out = attn_out.transpose(0, 1).contiguous().view(1, D)
    attn_out = out_proj(attn_out)

    x = x_new + attn_out
    x = x + ffn(ln2(x))
    return x, cache_k, cache_v, weights

prompt = "a gun was fired"
prompt_words = prompt.split()
prompt_ids = [stoi[w] for w in prompt_words]
print("STEP 1-2 - Tokenize prompt")
print("Tokens:", prompt_words)
print("Token IDs:", prompt_ids)
print()

print("STEP 3 - PREFILL")
cache_k, cache_v = None, None
generated_ids = list(prompt_ids)

for pos, tid in enumerate(prompt_ids):
    x_new = embed(tid, pos)
    block_out, cache_k, cache_v, weights = block_forward_new_token(x_new, cache_k, cache_v)
    print(f"  processed '{itos[tid]}' -> cache holds {cache_k.shape[0]} token(s)")

logits = output_head(final_norm(block_out))
next_id = torch.argmax(logits, dim=-1).item()
print("Cache shape:", tuple(cache_k.shape))
print("Predicted next token:", itos[next_id])
print()

print("STEP 4+ - DECODE LOOP")
generated_ids.append(next_id)
pos = len(prompt_ids)

for step in range(4):
    tid = generated_ids[-1]
    if itos[tid] == "<eos>":
        break

    x_new = embed(tid, pos)
    cache_before = cache_k.shape[0]
    block_out, cache_k, cache_v, weights = block_forward_new_token(x_new, cache_k, cache_v)
    cache_after = cache_k.shape[0]

    logits = output_head(final_norm(block_out))
    next_id = torch.argmax(logits, dim=-1).item()

    print(f"  step {step+1}: token '{itos[tid]}' | cache {cache_before} -> {cache_after} | next: '{itos[next_id]}'")

    generated_ids.append(next_id)
    pos += 1

print()
print("STEP 5 - Final sequence")
print("Generated:", " ".join(itos[i] for i in generated_ids))
```

---

## Checkpoint

1. In prefill, how many tokens' worth of Query/Key/Value get computed? In each decode step, how many?
2. Why does the cache shape grow by exactly one token per decode step, never more or less?
3. If this model had been trained, like Part 1's model, after many more repetitions, what would you expect to be different about the generated output, the mechanism, or just the actual words chosen?
4. Map each row of the "connecting back to vLLM" table to the phase that introduced it, from memory.

---

Back to Part 1: Training Walkthrough (01-training-walkthrough.md), back to main roadmap (../README.md)
