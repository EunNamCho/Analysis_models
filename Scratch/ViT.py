from typing import Tuple, Optional

import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        


class ViT(nn.Module):
    """
        1. Input image를 patch 단위로 나눠서 쭉 핀다 R^3 => R^2
        2. 각 patch들 embedding layer 통과
        3. cls token 맨 앞에 부착후, PE 부여

    """
    def __init__(self, patch_size, input_size, emb_dim):
        super().__init__()

        self.patch_size: Tuple[int, int] = patch_size
        self.input_size: Tuple[int, int] = input_size

        if self.input_size[0] % self.patch_size[0] or self.input_size[1] % self.patch_size[1]:
            raise Exception("이미지를 패치만큼 나눌 수 없습니다.")
        
        self.emb_dim = emb_dim
        self.patch_embedding = nn.Embedding(self.patch_size[0], self.emb_dim)
        
        self.pos_emb = None
        self.cls_token = nn.Parameter(torch.zeros(1, self.patch_size[0]**2 * 3, self.emb_dim))
        nn.init.kaiming_uniform_(self.cls_token, nonlinearity='relu')
        self.layernorm1 = nn.LayerNorm(self.emb_dim)

        self.transformer = Transformer()
        self.layernorm2 = nn.LayerNorm()

        self.mlp = MLP()
        self.layernorm2 = nn.LayerNorm()
        
        
    def forward(self, x):
        B, C, H, W = x.shape

        x = x.reshape(B, -1, self.patch_size[0]**2 * C)    # B * N * (P^2 * C)
        x = self.patch_embedding(x)                     # B * N * (P^2 * C) * Embedding_dim

        self.cls_token = self.cls_token.expand(B, -1, -1, -1)
        x = torch.concat((self.cls_token, x), dim=1)     # B * (N+1) * (P^2 * C) * Embedding_dim
        x += self.pos_emb
        x = self.layernorm1(x)

        x = self.transformer(x)
        x = self.layernorm2(x) + x

        x = self.mlp(x) + x

        y = self.layernorm3(x)
        return y