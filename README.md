# Qwen3-0.6B from scratch

A from-scratch PyTorch implementation of the **Qwen3-0.6B** architecture, written piece by piece to understand how a modern LLM actually works, then **verified against the official Hugging Face implementation** using Qwen's real pretrained weights.

No `transformers` modelling code is used in the model itself. `transformers` is only used to load the tokenizer and the reference model for comparison.

## Result

Same weights, same input, two implementations:

```
Prompt          : "I need 500 wedding cards"
Max logit diff  : 2.29e-05   (over 151,936 vocabulary scores)
torch.allclose  : True  (atol=1e-4)
HF next token   : " for"
My next token   : " for"
Parameters      : 596,049,920  (exactly Qwen3-0.6B)
```

The remaining difference is floating-point rounding across 28 layers, not a difference in the maths.

## What's implemented

| Component | Notes |
|---|---|
| **RMSNorm** | Computed in fp32, cast back, then scaled. Used at two sizes: 1024 (layer norms) and 128 (q/k norms) |
| **RoPE** (rotary position embeddings) | Half-split pairing (column `i` with `i + 64`), `rope_theta = 1,000,000`, built with `outer` + `cat` + `rotate_half` |
| **Grouped-Query Attention** | 16 query heads share 8 key/value heads (`kv_head = q_head // 2`), `head_dim = 128` |
| **QK-Norm** | RMSNorm on each head's queries and keys *before* RoPE (new in Qwen3) |
| **Causal mask** | Upper triangle set to `-inf` before softmax |
| **SwiGLU MLP** | `down(silu(gate(x)) * up(x))`, 1024 → 3072 → 1024 |
| **Decoder layer** | Pre-norm with residuals: `x + attn(norm(x))`, then `x + mlp(norm(x))` |
| **Full model** | Embedding → 28 decoder layers → final norm → LM head |
| **Tied embeddings** | `lm_head` shares its weight with `embed_tokens` (saves ~155M parameters) |

Parameter and module names match the official checkpoint, so the real weights load with `load_state_dict(..., strict=True)`.

## Architecture at a glance

```
text → tokenizer → ids [B, T]
     → embed_tokens                         [B, T, 1024]
     → 28 × DecoderLayer
          x = x + Attention(RMSNorm(x))      # q/k/v proj → q/k norm → RoPE → GQA → o_proj
          x = x + MLP(RMSNorm(x))            # SwiGLU
     → final RMSNorm
     → lm_head (tied)                       [B, T, 151936] logits
```

| Config | Value |
|---|---|
| hidden_size | 1024 |
| layers | 28 |
| query heads / KV heads | 16 / 8 |
| head_dim | 128 |
| intermediate_size | 3072 |
| vocab_size | 151,936 |
| rope_theta | 1,000,000 |

## Run it

**1. Install**

```bash
pip install torch transformers huggingface_hub
```

**2. Download the weights** (~1.5 GB) to a local folder:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen3-0.6B", local_dir="path/to/qwen3_0_6b")
```

Or download the files manually from [huggingface.co/Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/main): `config.json`, `generation_config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`.

**3. Point the script at it and run**

Set `MODEL_DIR` in `qwen3_imp.py`, then:

```bash
python qwen3_imp.py
```

It loads the weights into both this implementation and the Hugging Face reference (fp32, eager attention), runs the same prompt through both, and prints the difference and the predicted next token.

## Design choices (kept simple on purpose)

This is written for understanding, not speed:

- **Attention loops over heads** instead of using `repeat_kv` and one batched matmul, so each head's work is easy to follow.
- **RoPE tables are rebuilt inside every attention layer** rather than once per forward pass.
- **No KV cache yet**, so generation recomputes the whole sequence each step.

All three give identical results to the optimised versions. They're just slower.

## Roadmap

- [x] RMSNorm, SwiGLU MLP, RoPE, GQA attention with QK-norm
- [x] Decoder layer, full model, tied LM head
- [x] Load real Qwen3-0.6B weights and match Hugging Face's output
- [ ] Token-by-token text generation
- [ ] KV cache
- [ ] Norm-placement experiments on the pretrained weights (remove or move norms, measure the damage)
- [ ] LoRA fine-tuning
- [ ] Tiny models trained from scratch to compare pre-norm / post-norm / sandwich-norm

## Credits

- Model architecture and weights: [Qwen team, Alibaba](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache-2.0)
- Reference implementation used for verification: [Hugging Face `transformers`](https://github.com/huggingface/transformers)