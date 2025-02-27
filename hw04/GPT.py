# ==================================================================================== #
# This code is a simplified version of https://github.com/karpathy/nanoGPT             #
# ==================================================================================== #

import csv
import os
import time

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from transformers import GPT2Tokenizer

@dataclass
class GPTConfig:
    context_window: int = 64
    vocab_size: int = 50304
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 64

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        batch_size, context_window, emb_dim = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(batch_size, context_window, self.n_head, emb_dim // self.n_head).transpose(1, 2)
        q = q.view(batch_size, context_window, self.n_head, emb_dim // self.n_head).transpose(1, 2)
        v = v.view(batch_size, context_window, self.n_head, emb_dim // self.n_head).transpose(1, 2)
        
        y = self.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(batch_size, context_window, emb_dim)
        y = self.c_proj(y)
        return y

    # adopted from:
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    def scaled_dot_product_attention(self, query, key, value, scale=None) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / np.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device="cuda")
        temp_mask = torch.ones(L, S, dtype=torch.bool, device="cuda").tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        self.attn_weights = attn_weight
        return attn_weight @ value


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.context_window, config.n_embd),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.transformer.wte.weight = self.lm_head.weight  # No need to apply weight tight scheme for the small models
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        _, context_window = idx.size()
        pos = torch.arange(0, context_window, dtype=torch.long, device=idx.device) # shape (context_window)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoader:
    def __init__(self, batch_size: int, context_window: int):
        self.batch_size = batch_size
        self.context_window = context_window

        data_root = "../../xlstm/edu_fineweb10B_shuffled"
        shards = os.listdir(data_root)
        np.random.shuffle(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        batch_size, context_window = self.batch_size, self.context_window
        buf = self.tokens[self.current_position : self.current_position + batch_size * context_window + 1]
        x = (buf[:-1]).view(batch_size, context_window)
        y = (buf[1:]).view(batch_size, context_window)
        self.current_position += batch_size * context_window
        if self.current_position + (batch_size * context_window + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x, y

class Trainer:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

    def train(self, model: GPT, max_steps: int = 100_000, verbose: int = 250):
        
        MINI_BATCH_SIZE: int = 128
        BATCH_SIZE: int = 512
        gradient_accumulation_steps: int = BATCH_SIZE // MINI_BATCH_SIZE
        train_loader = DataLoader(batch_size=MINI_BATCH_SIZE, context_window=model.config.context_window)
        model.train()
        model.to("cuda")
        logs_csv_file = "training_logs.csv"
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, fused=True)
        scheduler = CosineAnnealingLR(optimizer=optimizer, eta_min=1e-4, T_max=1000)

        with open(logs_csv_file, mode="w", newline="", encoding="utf8") as file:
            writer = csv.writer(file)
            writer.writerow(["Step", "Loss", "Norm", "LR", "dt (ms)", "Tokens/sec"])
            start_step = 0

        model = torch.compile(model)
        for step in range(start_step, max_steps):
            model.train()
            t0 = time.time()
            last_step = (step == max_steps - 1)

            loss_accum = 0
            optimizer.zero_grad()
            for _ in range(gradient_accumulation_steps):

                inputs, targets = train_loader.next_batch()
                inputs, targets = inputs.to("cuda"), targets.to("cuda")

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(inputs)

                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                loss = loss / gradient_accumulation_steps
                loss.backward()
                loss_accum += loss.detach()

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if ((step > 0 and step % verbose == 0) or last_step):
                print(f"======= STEP {step}: =======")
                self._generate_text(model=model)

            torch.cuda.synchronize()
            t1 = time.time()
            dt = t1 - t0
            tokens_processed = gradient_accumulation_steps * train_loader.batch_size * train_loader.context_window
            tokens_per_sec = tokens_processed / dt
            with open(logs_csv_file, mode="a", newline="", encoding="utf8") as file:
                writer = csv.writer(file)
                writer.writerow([step, f"{loss_accum:.6f}", f"{norm:.4f}", f"{scheduler.get_lr()[0]:.4e}", f"{dt * 1000:.2f}", f"{tokens_per_sec:.2f}"])
        
        torch.save(model.state_dict(), f"gpt_cp_{step}.pth")

    def _generate_text(self, model, max_length: int = 32, prompt: str = "Once upon a time"):
        model.eval()
        xgen = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long, device="cuda").unsqueeze(0)

        while xgen.size(1) < max_length:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(xgen)[:, -1, :]
            xgen = torch.cat((xgen, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1)

        print(self.tokenizer.decode(xgen[0, :max_length].tolist()))
