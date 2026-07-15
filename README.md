# 🧠 LLMs From Scratch — A Visual, Step-by-Step Learning Path

> 👋 **New here? Read [START-HERE.md](START-HERE.md) first** — covers setup (no installation needed), what background you need (almost none), and how long this takes.

> From "what is a neuron" to "how does vLLM serve attention on a live server" — one continuous, visual roadmap. No step skipped, no black boxes.

This repo is for people who want to understand **how LLMs actually work internally**: the neural network math, the transformer architecture, and the inference engines (like vLLM) that serve models like GPT and Claude in production.

Every phase is a folder with its own `README.md` containing:
- 📖 Plain-English intuition before any math
- 🧮 The actual math, worked through, not hand-waved
- 📊 Diagrams (Mermaid — render natively on GitHub)
- 💻 Code with tensor shapes annotated at every step
- ✅ Checkpoint questions before you move on
- 🔗 Curated resources (videos, papers, docs)

## The Roadmap

| # | Phase | You'll be able to... | Status |
|---|-------|----------------------|--------|
| 1 | [AI/ML/DL/LLM Foundations](phases/01-foundations/) | Explain AI ⊃ ML ⊃ DL ⊃ LLM, training vs inference | 🚧 |
| 2 | [Math & Tensor Foundations](phases/02-math-tensor-foundations/) | Predict tensor shapes, write softmax from scratch | 🚧 |
| 3 | [Why Transformers Were Invented](phases/03-why-transformers/) | Explain what RNNs got wrong, and the transformer trade-off | 🚧 |
| 4 | [Transformers Deep Dive](phases/04-transformers-deep-dive/) | Trace tokenization → embeddings → attention → MLP | ✅ |
| 5 | [Build Attention Yourself](phases/05-build-attention-yourself/) | Implement multi-head attention in raw PyTorch | ✅ |
| 6 | [GPT & LLM Architecture](phases/06-gpt-llm-architecture/) | Assemble a full decoder-only GPT | ✅ |
| 7 | [LLM Inference Internals](phases/07-inference-internals/) | Explain KV cache, batching, sampling | ✅ |
| 8 | [vLLM Architecture](phases/08-vllm-architecture/) | Describe PagedAttention & continuous batching | ✅ |
| 9 | [Attention in vLLM (CPU)](phases/09-attention-in-vllm/) | Read vLLM's actual attention kernel | ✅ |
| 10 | [Research Papers](phases/10-research-papers/) | Read the foundational papers with guided notes | ✅ |
| 11 | [Advanced Topics](phases/11-advanced-topics/) | Navigate MoE, quantization, speculative decoding | ✅ |

## Worked Example

Want to see everything from Phases 1-9 traced through one real sentence, with real numbers? See **[worked-example/](worked-example/)** — tokenization through training (with real gradients and a real weight update), plus a separate inference walkthrough with an actual working KV cache.

## How to use this repo

Go in order. Each phase assumes the one before it. Do the checkpoint questions before moving on — if you can't answer them on paper, re-read.
