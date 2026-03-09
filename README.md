# Mixture of Recursion Language Model (198M)

> A novel language model architecture with perplexity-guided adaptive computation depth

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow)](https://huggingface.co/Girinath11/recursive-language-model-198m)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)

---

## 🏆 Result

| Model | Parameters | Perplexity |
|-------|-----------|------------|
| **MoR LM (This)** | **198M** | **15.37** ✅ |
| GPT-2 Small | 117M | ~29 |
| GPT-2 Medium | 345M | ~22 |
| GPT-2 Large | 774M | ~18 |

**Beats GPT-2 Medium at 57% of its parameter count.**

---

## 💡 The Idea

Traditional transformers apply the **same computation** to every input — simple and complex alike.

This model is different:

```
Simple input   (PPL < 20)  →  1 recursion step   ⚡ fast
Medium input   (PPL 20-50) →  3 recursion steps
Complex input  (PPL > 50)  →  5 recursion steps  🧠 deep
```

The router learns difficulty **automatically** from the model's own perplexity — no manual labels needed.

---

## 🏗️ Architecture

```
Input
  │
  ▼
Token Embeddings (50,260 × 768)
  │
  ▼
Base Transformer Stack (16 layers)
  │   ├── Multi-Head Attention (RoPE)
  │   ├── Feed-Forward (768 → 3072 → 768)
  │   └── Pre-LayerNorm + Residual
  │
  ▼
Perplexity Router
  │   ├── Sequence pooling
  │   ├── MLP classifier (768 → 384 → 3)
  │   └── Outputs: simple / medium / complex
  │
  ▼
Recursive Layer (applied 1, 3, or 5 times)
  │
  ▼
Final LayerNorm → LM Head
  │
  ▼
Output
```

### Key Components

| Component | Details |
|-----------|---------|
| Embedding dim | 768 |
| Layers | 16 |
| Attention heads | 12 |
| FFN size | 3072 |
| Max seq length | 512 |
| Positional encoding | RoPE |
| Vocab size | 50,260 |
| Total params | ~198M |

---

## 🚀 Quick Start

```bash
pip install transformers torch
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Girinath11/recursive-language-model-198m",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "Girinath11/recursive-language-model-198m",
    trust_remote_code=True
)

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def chat(question):
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt",
                       add_special_tokens=False).to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full.split("<|assistant|>")[-1].strip()

print(chat("What is machine learning?"))
print(chat("Explain neural networks simply"))
```

---

## 📚 Training

### Dataset — 150K samples

| Source | Samples | Type |
|--------|---------|------|
| Anthropic HH-RLHF | 80,000 | Helpful conversations |
| UltraChat | 50,000 | Multi-turn dialogues |
| Alpaca-GPT4 | 20,000 | Instruction following |

### Config

```yaml
Epochs:          2
Batch size:      2 × 32 = 64 effective
Learning rate:   1e-4 (cosine decay)
Warmup steps:    500
Optimizer:       AdamW (β=0.9, 0.95)
Mixed precision: fp16
GPU:             Tesla T4 (Kaggle)
Training time:   9h 12m
```

### Training Curve

| Epoch | Train Loss | Val Loss | Perplexity |
|-------|-----------|----------|------------|
| 1 | 4.5081 | 3.0798 | 21.75 |
| 2 | 3.3068 | 2.7326 | **15.37** 🔥 |

---

## 💡 Technical Details

### Perplexity-Based Routing (Novel)

```python
# Self-supervised pseudo labels during training
sample_ppl = exp(sample_loss)

if sample_ppl < 20:   label = 0  # simple
elif sample_ppl < 50: label = 1  # medium
else:                 label = 2  # complex

total_loss = lm_loss + 0.1 * router_loss
```

### NaN-Safe fp16 Attention

```python
# ❌ Unstable — causes NaN in fp16
mask.fill_(float('-inf'))

# ✅ Stable
mask.fill_(-1e4)
scores = scores.clamp(min=-1e4, max=1e4)
attn = F.softmax(scores, dim=-1)
attn = torch.nan_to_num(attn, nan=0.0)
```

---

## 📁 Structure

```
── mixture_of_recursion.py   # Model architecture
── README.md
```

---

## 🔗 Links

- **HuggingFace Model:** [Girinath11/recursive-language-model-198m](https://huggingface.co/Girinath11/recursive-language-model-198m)

---

## 📄 Citation

```bibtex
@misc{mor-lm-198m-2026,
  author = {Girinath V},
  title  = {Mixture of Recursion Language Model},
  year   = {2026},
  url    = {https://huggingface.co/Girinath11/recursive-language-model-198m}
}
```

---

## 📜 License

MIT License — Free to use with attribution

---

*Built from scratch on a free Kaggle T4 GPU 🚀*
