import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0, bias=False):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.resid_drop(self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C)))

class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.0, bias=False):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0, bias=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, bias)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))

class GPT2(nn.Module):
    def __init__(self, vocab_size=50304, block_size=1024, n_layer=12, n_head=12, n_embd=768,
                 dropout=0.0, bias=False):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(vocab_size, n_embd),
            'wpe': nn.Embedding(block_size, n_embd),
            'drop': nn.Dropout(dropout),
            'h': nn.ModuleList([Block(n_embd, n_head, dropout, bias) for _ in range(n_layer)]),
            'ln_f': nn.LayerNorm(n_embd),
        })
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        T = idx.size(1)
        assert T <= self.block_size
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_parameters(self):

        return sum(p.numel() for p in self.parameters()) - self.transformer.wpe.weight.numel()

def gpt2_param_groups(model, weight_decay):

    params = [p for p in model.parameters() if p.requires_grad]
    return [{'params': [p for p in params if p.dim() >= 2], 'weight_decay': weight_decay},
            {'params': [p for p in params if p.dim() < 2], 'weight_decay': 0.0}]
