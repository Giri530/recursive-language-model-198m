# Mixture of Recursion Language Model (MoR LM) — 198M

A 198M parameter conversational language model with **adaptive recursive computation** via self-supervised perplexity-based routing. Built from scratch — no pretrained base, no fine-tuning — trained on 150K conversational samples on a single Kaggle T4 GPU.

> **HuggingFace Model**: [`Girinath11/recursive-language-model-198m`](https://huggingface.co/Girinath11/recursive-language-model-198m)

---

## What is Mixture of Recursion?

Standard transformers apply the same depth to every token — simple and complex inputs both go through all N layers. This wastes compute on easy inputs and under-processes hard ones.

MoR fixes this with **self-supervised perplexity-guided routing**: the model measures its own uncertainty at inference and allocates recursion steps accordingly.

```
High PPL (> 50)  →  Model struggling  →  5 recursive steps
Mid  PPL (20–50) →  Uncertain         →  3 recursive steps
Low  PPL (< 20)  →  Confident         →  1 recursive step
```

No manual labels. The router learns difficulty directly from the model's own perplexity signal during training.

---

## Architecture

| Component | Description | Params |
|---|---|---|
| Token Embedding | GPT-2 BPE vocab (50,260 tokens) | ~39M |
| Base Transformer | 16 layers, RoPE, pre-norm | ~149M |
| Perplexity Router | 2-layer MLP (768 → 384 → 3), self-supervised | ~1.2M |
| Recursive Layer | Shared transformer block, reused 1/3/5× | ~7M |
| **Total** | | **~198M** |

### NaN-Safe FP16 Training

Standard `-inf` masking causes NaN during fp16 mixed-precision. MoR uses `-1e4` masking + pre-softmax clamping for completely stable training:

```python
scores = scores.clamp(min=-1e4, max=1e4)
attn   = F.softmax(scores, dim=-1)
attn   = torch.nan_to_num(attn, nan=0.0)
```

Result: **0 NaN batches** across all 150K training steps.

---

## Results

### In-Distribution Validation (same conversational distribution as training)

| Epoch | Train Loss | Val Loss | Val PPL |
|---|---|---|---|
| 1 | 4.5081 | 3.0798 | 21.75 |
| 2 | 3.3068 | 2.7326 | **15.37** |

### Out-of-Distribution Test (fresh HH-RLHF test samples, never seen during training)

| Model | Params | Test PPL |
|---|---|---|
| GPT-2 Medium | 345M | 26.89 |
| **MoR 198M** | **198M** | **43.21** |

**Honest assessment:**

- The 15.37 val PPL was measured on in-distribution data — do not compare it directly to GPT-2 Medium's standard benchmark numbers, they use different datasets
- The generalization gap (val 15.37 → test 43.21) indicates overfitting to the training distribution
- 150K samples is a small training set for a 198M model; more diverse data would help significantly
- The architecture and routing mechanism are validated — training is stable, loss decreases, router signals are correct

---

## Quick Start

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

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = model.to(device).eval()
```

The model was trained with a specific chat format — always wrap your input like this:

```python
def chat(question, max_new_tokens=150, temperature=0.7, top_p=0.9):
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full.split("<|assistant|>")[-1].strip() if "<|assistant|>" in full else full

print(chat("What is machine learning?"))
print(chat("Explain neural networks simply"))
```

---

## Training Details

### Dataset — 150K Conversational Samples

| Dataset | Samples | Share | Description |
|---|---|---|---|
| Anthropic HH-RLHF | 80,000 | 53% | Helpful & harmless human feedback |
| UltraChat | 50,000 | 33% | GPT-4 multi-turn dialogues |
| Alpaca-GPT4 | 20,000 | 14% | Instruction following |

### Config

```
GPU             : NVIDIA Tesla T4 (15.6 GB VRAM) — Kaggle, single GPU
Epochs          : 2
Total steps     : 150,000
Training time   : ~9h 12m

Batch size      : 2
Grad accum      : 32  →  effective batch = 64
Max seq len     : 512
Learning rate   : 1e-4
LR schedule     : Linear warmup (500 steps) + Cosine decay
Optimizer       : AdamW  (β = 0.9, 0.95,  ε = 1e-8)
Weight decay    : 0.01
Grad clip       : 1.0
Mixed precision : FP16 (AMP)

Loss            : LM loss + 0.1 × Router loss
```

---

## Self-Supervised Perplexity Routing — How It Works

```python
# No manual labels — router trains on the model's own uncertainty
sample_ppl = exp(per_sample_ce_loss)

if sample_ppl < 20:   pseudo_label = 0  # simple
elif sample_ppl < 50: pseudo_label = 1  # medium
else:                 pseudo_label = 2  # complex

router_loss = CrossEntropyLoss(router_logits, pseudo_label)
total_loss  = lm_loss + 0.1 * router_loss
```

As training progresses, more samples naturally fall into the "simple" bucket — an emergent curriculum with no annotation cost.

---

## Limitations

- **Context**: 512 tokens max
- **Language**: English only
- **Generalization**: Overfits to conversational training distribution
- **Repetition**: May loop on generations longer than ~200 tokens
- **Factual accuracy**: Will hallucinate — not suitable for factual Q&A

---

## Intended Use

This model is intended for research and learning purposes.

| Use case | Suitable? |
|---|---|
| Research on adaptive computation | Yes |
| Learning how LLMs are built from scratch | Yes |
| Prototyping conversational AI | With caveats |
| Production chatbots | No |
| Factual / medical / legal / financial use | No |
---

## Citation

```bibtex
@misc{girinath2026mor,
  author       = {Girinath V},
  title        = {Mixture of Recursion: Self-Supervised Perplexity-Guided
                  Adaptive Computation for Language Models},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/Girinath11/recursive-language-model-198m}},
  note         = {198M parameter LM trained from scratch with adaptive
                  recursive computation via perplexity-based routing}
}
```

---

## Acknowledgments

- Anthropic — HH-RLHF dataset
- Tsinghua University — UltraChat dataset
- Vicgalle — Alpaca-GPT4 dataset
- HuggingFace — Transformers library
- Kaggle — Free GPU access

---

**License**: MIT &nbsp;|&nbsp; **Status**: Research / Educational &nbsp;|&nbsp; **Last updated**: March 2026
