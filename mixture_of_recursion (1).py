import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import math
class RecursiveLanguageModelConfig(PretrainedConfig):
    model_type = "recursive_language_model"
    def __init__(
        self,
        vocab_size=50260,
        embedding_dim=768,
        num_layers=16,
        num_attention_heads=12,
        max_recursion_steps=5,
        max_position_embeddings=512,
        hidden_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        intermediate_size=3072,
        layer_norm_eps=1e-5,
        pad_token_id=50257,
        bos_token_id=50256,
        eos_token_id=50256,
        simple_recursion_steps=1,
        medium_recursion_steps=3,
        complex_recursion_steps=5,
        initializer_range=0.02,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.max_recursion_steps = max_recursion_steps
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.simple_recursion_steps = simple_recursion_steps
        self.medium_recursion_steps = medium_recursion_steps
        self.complex_recursion_steps = complex_recursion_steps
        self.initializer_range = initializer_range
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)
def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim  = config.embedding_dim // config.num_attention_heads
        self.embed_dim = config.embedding_dim
        assert self.embed_dim % self.num_heads == 0
        self.q_proj   = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj   = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj   = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.attn_drop = nn.Dropout(config.attention_dropout_prob)
        self.rope = RotaryPositionalEmbedding(self.head_dim, config.max_position_embeddings)
    def forward(self, x, causal_mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(T, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale 
        if causal_mask is not None:
            scores = scores + causal_mask
        scores = scores.clamp(min=-1e4, max=1e4)
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)                             
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1  = nn.Linear(config.embedding_dim, config.intermediate_size, bias=False)
        self.fc2  = nn.Linear(config.intermediate_size, config.embedding_dim, bias=False)
        self.drop = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ff   = FeedForward(config)
        self.ln1  = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.ln2  = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x
class SequenceLevelRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooler = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.act    = nn.Tanh()
        self.head   = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.embedding_dim // 2, 3)
        )
        self.register_buffer('steps_map', torch.tensor([
            config.simple_recursion_steps,
            config.medium_recursion_steps,
            config.complex_recursion_steps,
        ], dtype=torch.long))
    def forward(self, x, valid_mask=None):
        if valid_mask is not None:
            m = valid_mask.unsqueeze(-1).float()
            pooled = (x * m).sum(1) / m.sum(1).clamp(min=1e-9)
        else:
            pooled = x.mean(1)
        pooled  = self.act(self.pooler(pooled))
        logits  = self.head(pooled)
        cls     = logits.argmax(dim=-1)
        return logits, cls, self.steps_map[cls]
class RecursionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block = TransformerBlock(config)
    def forward(self, x, mask=None):
        return self.block(x, mask)
class RecursiveLanguageModel(PreTrainedModel):
    config_class = RecursiveLanguageModelConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embed    = nn.Embedding(config.vocab_size, config.embedding_dim,
                                     padding_idx=config.pad_token_id)
        self.layers   = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.router   = SequenceLevelRouter(config)
        self.rec_layer = RecursionLayer(config)
        self.ln_f     = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.lm_head  = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.post_init()
    def get_input_embeddings(self): return self.embed
    def set_input_embeddings(self, v): self.embed = v
    def get_output_embeddings(self): return self.lm_head
    def set_output_embeddings(self, v): self.lm_head = v
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    def _make_causal_mask(self, input_ids):
        B, T   = input_ids.shape
        device = input_ids.device
        mask = torch.zeros(T, T, device=device)
        mask = mask.masked_fill(
            torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1),
            -1e4
        )
        mask = mask.unsqueeze(0).unsqueeze(0)
        pad_mask = (input_ids == self.config.pad_token_id)  # [B, T]
        valid_mask = ~pad_mask
        if pad_mask.any():
            pad_key_mask = pad_mask.unsqueeze(1).unsqueeze(2).float() * -1e4
            mask = mask + pad_key_mask
        return mask, valid_mask
    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        B, T = input_ids.shape
        x = self.embed(input_ids)
        causal_mask, valid_mask = self._make_causal_mask(input_ids)
        for layer in self.layers:
            x = layer(x, causal_mask)
        router_logits, cls, steps = self.router(x, valid_mask)
        max_steps = int(steps.max().item())
        for s in range(max_steps):
            gate = (steps > s).float().view(B, 1, 1)
            x    = gate * self.rec_layer(x, causal_mask) + (1 - gate) * x
        x      = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            with torch.no_grad():
                per_tok   = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100, reduction='none'
                ).view(B, -1)
                valid_tok = (shift_labels != -100).sum(1).clamp(min=1).float()
                ppl       = torch.exp((per_tok.sum(1) / valid_tok).clamp(max=20))
                pseudo    = torch.zeros(B, dtype=torch.long, device=input_ids.device)
                pseudo[(ppl >= 20) & (ppl < 50)] = 1
                pseudo[ppl >= 50] = 2
            router_loss = F.cross_entropy(router_logits, pseudo)
            loss = lm_loss + 0.1 * router_loss
        return CausalLMOutputWithPast(loss=loss, logits=logits)
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.8,
                 top_p=0.9, do_sample=True, **kwargs):
        self.eval()
        gen = input_ids
        for _ in range(max_new_tokens):
            ctx = gen[:, -self.config.max_position_embeddings:]
            logits = self.forward(ctx).logits[:, -1, :]
            if temperature != 1.0:
                logits = logits / temperature
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                remove = cum_probs - sorted_probs > top_p
                sorted_probs[remove] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_tok = torch.gather(sorted_idx, -1,
                                        torch.multinomial(sorted_probs, 1))
            else:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            gen = torch.cat([gen, next_tok], dim=-1)
            if (next_tok == self.config.eos_token_id).all():
                break
        return gen