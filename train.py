import os, gc, json, random, time, math, warnings
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings('ignore')
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime
import numpy as np
from mixture_of_recursion import RecursiveLanguageModel, RecursiveLanguageModelConfig
# i just kept all the config in one place so its easier to tweak
CFG = {
    "embedding_dim":768,
    "num_layers":16,
    "num_attention_heads":12,
    "intermediate_size":3072,
    "max_position_embeddings":512,
    "simple_recursion_steps":1,
    "medium_recursion_steps":3,
    "complex_recursion_steps":5,
    "batch_size":2,
    "grad_accum":32,    # effective batch size = 64
    "lr":1e-4,
    "weight_decay":0.01,
    "epochs":2,
    "warmup_steps":500,
    "max_grad_norm":1.0,
    "max_length":512,
    "target_samples":150000,
    "save_every_hours": 2,
    "ckpt_dir":"/kaggle/working/checkpoints",
}
def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
def fmt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"
def print_gpu():
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU: {used:.1f}/{total:.1f} GB")
class ChatDataset(Dataset):
    def __init__(self, tokenizer, max_len=512, split='train'):
        self.tok = tokenizer
        self.maxlen = max_len
        self.data   = []
        print(f"\nloading {split} data...")
        if split == 'train':
            self._load_hh(80000)
            self._load_ultrachat(50000)
            self._load_alpaca(20000)
        else:
            self._load_hh(2000)
        random.shuffle(self.data)
        print(f"total samples: {len(self.data):,}\n")
    def _load_hh(self, n):
        print(f"  hh-rlhf ({n:,})...")
        try:
            ds = load_dataset("Anthropic/hh-rlhf", split='train')
            before = len(self.data)
            for item in ds:
                if len(self.data) - before >= n:
                    break
                text = self._parse_hh(item.get('chosen', ''))
                if text and self._is_valid(text):
                    self.data.append(text)
            print(f"  got: {len(self.data) - before:,}")
        except Exception as e:
            print(f"  failed: {e}")
    def _load_ultrachat(self, n):
        print(f"  ultrachat ({n:,})...")
        try:
            ds = load_dataset("stingning/ultrachat", split='train')
            before=len(self.data)
            for item in ds:
                if len(self.data) - before >= n:
                    break
                if 'data' in item and isinstance(item['data'], list):
                    text = self._parse_ultrachat(item['data'])
                    if text and self._is_valid(text):
                        self.data.append(text)
            print(f"  got: {len(self.data) - before:,}")
        except Exception as e:
            print(f"  failed: {e}")
    def _load_alpaca(self, n):
        print(f"  alpaca-gpt4 ({n:,})...")
        try:
            ds = load_dataset("vicgalle/alpaca-gpt4", split='train')
            before = len(self.data)
            for item in ds:
                if len(self.data) - before >= n:
                    break
                instruction = item.get('instruction', '').strip()
                inp = item.get('input', '').strip()
                output = item.get('output', '').strip()
                if not instruction or not output or len(output) < 20:
                    continue
                user_msg = f"{instruction}\n\n{inp}" if inp else instruction
                text = f"<|user|>\n{user_msg}\n<|assistant|>\n{output}\n<|endoftext|>"
                if self._is_valid(text):
                    self.data.append(text)
            print(f"  got: {len(self.data) - before:,}")
        except Exception as e:
            print(f"  failed: {e}")
    def _parse_hh(self, text):
        try:
            if 'Human:' not in text or 'Assistant:' not in text:
                return None
            result = ""
            for part in text.split('\n\n'):
                part = part.strip()
                if part.startswith('Human:'):
                    content = part[6:].strip()
                    if content:
                        result += f"<|user|>\n{content}\n"
                elif part.startswith('Assistant:'):
                    content = part[10:].strip()
                    if content:
                        result += f"<|assistant|>\n{content}\n"
            return result + "<|endoftext|>" if result else None
        except:
            return None
    def _parse_ultrachat(self, turns):
        try:
            result = ""
            for i, turn in enumerate(turns[:6]):
                role    = 'user' if i % 2 == 0 else 'assistant'
                content = str(turn).strip()
                for prefix in ['User:', 'Assistant:', 'Human:', 'AI:']:
                    content = content.replace(prefix, '')
                content = content.strip()
                if content:
                    result += f"<|{role}|>\n{content}\n"
            return result + "<|endoftext|>" if result else None
        except:
            return None
    def _is_valid(self, text):
        if not text or len(text) < 50:
            return False
        if '<|user|>' not in text or '<|assistant|>' not in text:
            return False
        try:
            response    = text.split('<|assistant|>')[1].split('<|endoftext|>')[0].strip()
            if len(response) < 15 or len(response.split()) < 4:
                return False
            alpha_ratio = sum(c.isalpha() for c in response) / len(response)
            if alpha_ratio < 0.5:
                return False
        except:
            return False
        return True
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        text = self.data[idx]
        enc  = self.tok(
            text,
            truncation=True,
            max_length=self.maxlen,
            padding='max_length',
            return_tensors='pt'
        )
        ids  = enc['input_ids'].squeeze(0)
        attn = enc['attention_mask'].squeeze(0)
        lbl  = ids.clone()
        assistant_ids = self.tok.encode("<|assistant|>", add_special_tokens=False)
        found = False
        for i in range(len(ids) - len(assistant_ids)):
            if ids[i:i+len(assistant_ids)].tolist() == assistant_ids:
                lbl[:i + len(assistant_ids)] = -100
                found = True
                break
        if not found:
            lbl[:] = -100
        lbl[attn == 0] = -100
        if (lbl != -100).sum() == 0:
            lbl = ids.clone()
            lbl[attn == 0] = -100
        return {'input_ids': ids, 'labels': lbl}
def train():
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    start_time = time.time()
    os.makedirs(CFG['ckpt_dir'], exist_ok=True)
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.add_special_tokens({
        'pad_token': '<|pad|>',
        'additional_special_tokens': ['<|user|>', '<|assistant|>']
    })
    vocab_size = len(tok)
    assert tok.pad_token_id != tok.eos_token_id, "pad and eos must be different!"
    print(f"\nvocab={vocab_size} | pad={tok.pad_token_id} | eos={tok.eos_token_id}")
    config = RecursiveLanguageModelConfig(
        vocab_size=vocab_size,
        embedding_dim=CFG['embedding_dim'],
        num_layers=CFG['num_layers'],
        num_attention_heads=CFG['num_attention_heads'],
        intermediate_size=CFG['intermediate_size'],
        max_position_embeddings=CFG['max_position_embeddings'],
        simple_recursion_steps=CFG['simple_recursion_steps'],
        medium_recursion_steps=CFG['medium_recursion_steps'],
        complex_recursion_steps=CFG['complex_recursion_steps'],
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        bos_token_id=tok.bos_token_id or tok.eos_token_id,
    )
    model = RecursiveLanguageModel(config)
    model.resize_token_embeddings(vocab_size)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {total_params/1e6:.0f}M")
    print_gpu()
    print("\npre-training NaN check...")
    model.train()
    dummy_ids = torch.randint(1, vocab_size - 1, (2, 32)).to(device)
    dummy_lbl = dummy_ids.clone()
    dummy_lbl[:, :8] = -100
    with torch.no_grad():
        out = model(dummy_ids, labels=dummy_lbl)
        loss_val = out.loss.item()
    if math.isnan(loss_val):
        print("NaN loss before training — check the model architecture")
        return
    print(f"pre-check ok, loss={loss_val:.4f}")
    train_ds = ChatDataset(tok, CFG['max_length'], 'train')
    val_ds   = ChatDataset(tok, CFG['max_length'], 'validation')
    train_dl = DataLoader(train_ds, batch_size=CFG['batch_size'],
                          shuffle=True,  num_workers=2, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=CFG['batch_size'],
                          shuffle=False, num_workers=2, pin_memory=True)
    opt = AdamW(
        model.parameters(),
        lr=CFG['lr'],
        weight_decay=CFG['weight_decay'],
        betas=(0.9, 0.95),
        eps=1e-8
    )
    total_steps  = len(train_dl) * CFG['epochs']
    warmup_steps = CFG['warmup_steps']
    sched = SequentialLR(opt, schedulers=[
        LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
        CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6)
    ], milestones=[warmup_steps])
    scaler = torch.amp.GradScaler('cuda')
    print(f"\nlr={CFG['lr']} | warmup={warmup_steps} steps")
    print(f"effective batch = {CFG['batch_size']} x {CFG['grad_accum']} = {CFG['batch_size']*CFG['grad_accum']}")
    print(f"total steps={total_steps:,} | epochs={CFG['epochs']}")
    print(f"note: using -1e4 instead of -inf in attention mask (fp16 NaN fix)\n")
    best_ppl = float('inf')
    last_save = time.time()
    global_step = 0
    for epoch in range(CFG['epochs']):
        print(f"\n--- epoch {epoch+1}/{CFG['epochs']} ---\n")
        model.train()
        epoch_start = time.time()
        total_loss  = 0.0
        num_batches = 0
        nan_count   = 0
        opt.zero_grad()
        pbar = tqdm(train_dl, desc=f"epoch {epoch+1}")
        for step, batch in enumerate(pbar):
            ids = batch['input_ids'].to(device, non_blocking=True)
            lbl = batch['labels'].to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                out  = model(input_ids=ids, labels=lbl)
                loss = out.loss / CFG['grad_accum']
            if torch.isnan(loss):
                nan_count += 1
                opt.zero_grad()
                del ids, lbl, out, loss
                torch.cuda.empty_cache()
                continue
            scaler.scale(loss).backward()
            if (step + 1) % CFG['grad_accum'] == 0:
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['max_grad_norm'])
                if torch.isfinite(grad_norm):
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    global_step += 1
                else:
                    scaler.update()
                    nan_count += 1
                opt.zero_grad()
            total_loss  += loss.item() * CFG['grad_accum']
            num_batches += 1
            if step % 10 == 0:
                del ids, lbl, out, loss
                torch.cuda.empty_cache()
            pbar.set_postfix({
                'loss': f'{total_loss/num_batches:.4f}',
                'lr':   f'{sched.get_last_lr()[0]:.2e}',
                'nan':  nan_count,
            })
            if time.time() - last_save > CFG['save_every_hours'] * 3600:
                save_path = f"{CFG['ckpt_dir']}/autosave_e{epoch+1}_s{step}"
                os.makedirs(save_path, exist_ok=True)
                model.save_pretrained(save_path)
                tok.save_pretrained(save_path)
                print(f"\nautosaved → {save_path}\n")
                last_save = time.time()
        avg_train_loss = total_loss / max(num_batches, 1)
        print(f"\nepoch {epoch+1} | loss={avg_train_loss:.4f} | nan_batches={nan_count} | {fmt_time(time.time()-epoch_start)}")
        print("validating...")
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for i, batch in enumerate(val_dl):
                if i >= 30:
                    break
                ids = batch['input_ids'].to(device)
                lbl = batch['labels'].to(device)
                with torch.amp.autocast('cuda'):
                    out = model(input_ids=ids, labels=lbl)
                if not torch.isnan(out.loss):
                    val_loss    += out.loss.item()
                    val_batches += 1
                del ids, lbl, out
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        ppl = math.exp(min(avg_val_loss, 20))
        print(f"val_loss={avg_val_loss:.4f} | ppl={ppl:.2f}")
        print_gpu()
        print("\ngeneration test:")
        print("-" * 40)
        for question in ["What is machine learning?", "Explain Python simply"]:
            prompt = f"<|user|>\n{question}\n<|assistant|>\n"
            inp    = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model.generate(
                    inp['input_ids'],
                    max_new_tokens=80,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            decoded  = tok.decode(out[0], skip_special_tokens=True)
            response = decoded.split('<|assistant|>')[-1].strip() if '<|assistant|>' in decoded else decoded
            print(f"Q: {question}")
            print(f"A: {response[:150]}")
            print("-" * 40)
            del inp, out
        torch.cuda.empty_cache()
        if ppl < best_ppl:
            best_ppl  = ppl
            best_path = f"{CFG['ckpt_dir']}/best_model"
            os.makedirs(best_path, exist_ok=True)
            model.save_pretrained(best_path)
            tok.save_pretrained(best_path)
            with open(f"{best_path}/info.json", 'w') as f:
                json.dump({
                    'epoch':    epoch + 1,
                    'loss':     avg_train_loss,
                    'val_loss': avg_val_loss,
                    'ppl':      ppl,
                    'params':   total_params,
                    'ts':       datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
            print(f"\nnew best! ppl={ppl:.2f} saved to {best_path}")
        epoch_ckpt = f"{CFG['ckpt_dir']}/epoch{epoch+1}"
        os.makedirs(epoch_ckpt, exist_ok=True)
        model.save_pretrained(epoch_ckpt)
        tok.save_pretrained(epoch_ckpt)
        elapsed   = time.time() - start_time
        remaining = (elapsed / (epoch + 1)) * (CFG['epochs'] - epoch - 1)
        print(f"\nelapsed: {fmt_time(elapsed)} | remaining: {fmt_time(remaining)}")
        if elapsed + remaining > 11.5 * 3600:
            print("warning: approaching kaggle 12hr limit!")
        gc.collect()
        torch.cuda.empty_cache()
    print(f"best ppl  : {best_ppl:.2f}")
    print(f"total time: {fmt_time(time.time() - start_time)}")
    print(f"model at  : {CFG['ckpt_dir']}/best_model/")
if __name__ == "__main__":
    train()
