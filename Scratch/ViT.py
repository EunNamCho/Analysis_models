from typing import Tuple

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

class Attention(nn.Module):
    def __init__(self):
        super().__init__()

class Transformer(nn.Module):
    def __init__(self, num_head):
        super().__init__()
        self.layernorm()                       # MSA 최종 출력을 알아야 함
        self.num_head = num_head
        self.msa = nn.Sequential()
        for i in range(self.num_head):
            self.msa.append(nn.Sequential(
                    Attention(),
                    FeedForward()
                ))
    
    def forward(self, x):
        for attn, ff in self.msa:
            x = attn(x) + x
            x = ff(x) + x
        return self.layernorm(x)

class ViT(nn.Module):
    """
        1. Input image를 patch 단위로 나눠서 쭉 핀다 R^3 => R^2
        2. 각 patch들 embedding layer 통과
        3. cls token 맨 앞에 부착후, PE 부여

    """
    def __init__(self, patch_size, input_size, emb_dim):
        super().__init__()

        self.patch_size: Tuple[int, int] = patch_size        # H, W
        self.input_size: Tuple[int, int, int] = input_size   # C, H, W

        if self.input_size[0] % self.patch_size[1] or self.input_size[2] % self.patch_size[1]:
            raise Exception("이미지를 패치만큼 나눌 수 없습니다.")
        else:
            self.patch_num = (self.input_size[1] * self.input_size[2]) // (self.patch_size[0] * self.patch_size[1]) # the number of patches in a channel
        
        self.patch_dim = self.patch_size[0] * self.patch_size[1] * self.input_size[0]
        self.emb_dim = emb_dim
        ##wrong## self.patch_embedding = nn.Embedding(self.patch_size[0], self.emb_dim)
        self.patch_embedding = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim)
        )

        ##wrong## self.pos_emb = None
        self.cls_token = nn.Parameter(torch.zeros(self.emb_dim))
        nn.init.kaiming_uniform_(self.cls_token, nonlinearity='relu')

        self.pos_emb = nn.Parameter(torch.zeros(self.patch_num+1, self.emb_dim))
        nn.init.kaiming_uniform_(self.cls_token, nonlinearity='relu')

        self.transformer = Transformer()
        ##self.layernorm1 = nn.LayerNorm(self.emb_dim)

        self.mlp = MLP()
        ##self.layernorm2 = nn.LayerNorm(self.emb_dim)
        
    def forward(self, x):
        B, C, H, W = x.shape

        x = x.reshape(B, self.patch_num, self.patch_dim)    # B * patch_num * (P^2 * C)
        x = self.patch_embedding(x)                         # B * patch_num * (P^2 * C) ==> B * patch_num * emb_dim

        self.cls_token = self.cls_token.expand(B, 1, -1)
        x = torch.concat((self.cls_token, x), dim=1)        # + class token    B * (patch_num + 1) * emb_dim
        x += self.pos_emb                                   # + pose embedding

        ##wrong## x = self.transformer(self.layernorm1(x)) + x        # Transformer encoder + residual connection: B * (patch_num + 1) * emb_dim
        ##wrong## x = self.mlp(self.layernorm2(x)) + x        각 layer내부에서 res connection을 해줌. 그리고, 입력에 LN을 하는게 아니라, 출력에 LN을 해줌
        """
        논문에 나온 연산만 넣은 코드인데. 실제로는 중간 LN을 넣음.

        self.cls_token = self.cls_token.expand(B, 1, -1)
        
        x = torch.concat((self.cls_token, x), dim=1)     # B * (N+1) * (P^2 * C) * Embedding_dim
        x += self.pos_emb
        x = self.layernorm1(x)

        x = self.transformer(x)
        x = self.layernorm2(x) + x

        x = self.mlp(x) + x

        y = self.layernorm3(x)
        return y
        """