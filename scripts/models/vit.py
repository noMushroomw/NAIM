import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)
        self.drop_path = drop_path

    def _drop_path(self, x):
        if self.drop_path == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_path
        mask = (torch.rand(x.shape[0], 1, 1, device=x.device) < keep).float() / keep
        return x * mask

    def forward(self, x):
        x = x + self._drop_path(self.attn(self.norm1(x)))
        return x + self._drop_path(self.mlp(self.norm2(x)))

class ViT(nn.Module):
    def __init__(self, num_classes=1000, embed_dim=768, depth=12, num_heads=12, dropout=0.0,
                 drop_path=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(224, 16, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        rates = [drop_path * i / (depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, 4.0, dropout, r) for r in rates])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, 0]))

def vit_b16(num_classes=1000):

    return ViT(num_classes, embed_dim=768, depth=12, num_heads=12, drop_path=0.1)

def vit_s16(num_classes=1000):

    return ViT(num_classes, embed_dim=384, depth=12, num_heads=6, drop_path=0.1)
