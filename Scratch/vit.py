from typing import Tuple

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.LayerNorm(self.emb_dim),
            nn.Linear(self.emb_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: B * (patch_num + 1) * emb_dim
        return self.net(x)

class Attention(nn.Module):
    """
        실질적인 attention 연산이 일어나는 layer
        1. 입력을 q, k, v로 만듦
        2. softmax(q @ k / dim**0.5) * v
    """
    def __init__(self, emb_dim, num_head, qkv_dim, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_head = num_head
        self.qkv_dim = qkv_dim
        self.project_out = self.emb_dim != self.qkv_dim * self.num_head

        self.scale = self.qkv_dim ** -0.5

        self.to_qkv = nn.Linear(self.emb_dim, self.num_head * self.qkv_dim * 3, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # 최초 입력과 크기가 다르면, 똑같게 만들어 준다
        self.to_out = nn.Sequential(
            nn.Linear(qkv_dim * num_head, emb_dim),
            nn.Dropout(dropout)
        ) if self.project_out else nn.Identity()

    def forward(self, x):
        # x: B * (patch_num + 1) * emb_dim
        qkv = self.to_qkv(x).chunk(3, dim=-1)       # B * (patch_num + 1) * (qkv_dim * num_head)
        B, N, D = qkv[0].shape
        q, k, v = map(lambda t: t.reshape(B, self.num_head, N, -1))
        """
            동료랑 attention 연산 얘기할 때, 
            q @ k 하면, 하나의 query에 모든 key의 연산이 아니라 
            첫번째 q는 첫번째 k랑, 두번째 q는 두번째 k랑 연산을 해서 잘못된 거 아닌가?
            라는 엄청난 토론을 한 결과
            행렬곱의 특성으로 인해, k를 transpose하면, 모든 key와의 연산이 되지만,
            q를 transpose하면, 아까 말한 논쟁의 연산법이 된다.
            결론은, k.T를 해야한다.

            
        """
        dots = q @ k.transpose(-1,-2) * self.scale # B * num_head * (patch_num + 1) * (patch_num + 1) 각 query와 모든 key의 상관관계
        attention_score = self.softmax(dots)       # B * num_head * (patch_num + 1) * (patch_num + 1) 점수로 변환
        attention_score = self.dropout(attention_score)

        output = attention_score @ v               # B * num_head * (patch_num + 1) * qkv_dim                                # B * (patch_num + 1) * (qkv_dim * num_head) 
        output = output.reshape(B, N, -1)
        return output

class Transformer(nn.Module):
    """
        Multi head Self Attention만 있음
        입력과 출력의 크기 변화가 없음...? (아직 추측)
        일단 dimension의 변화는 없는 듯. 왜냐면, MSA지난 출력에도 emb_dim으로 LN을 해주기 때문.
    """
    def __init__(self, emb_dim, depth, num_head, qkv_dim, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.depth = depth
        self.num_head = num_head
        self.qkv_dim = qkv_dim
        
        self.layernorm(self.emb_dim)                       # MSA 최종 출력을 알아야 함
        
        self.msa = nn.Sequential()
        for i in range(self.depth):
            self.msa.append(nn.Sequential(
                    Attention(self.emb_dim, self.num_head, self.qkv_dim, dropout),
                    FeedForward(dropout)
                ))
    
    def forward(self, x):
        # x: B * (patch_num + 1) * emb_dim
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
    def __init__(self, patch_size, input_size, emb_dim, depth, num_head, qkv_dim, dropout, emb_dropout, num_class):
        super().__init__()

        self.patch_size: Tuple[int, int] = patch_size        # H, W
        self.input_size: Tuple[int, int, int] = input_size   # C, H, W
        self.emb_dim = emb_dim
        self.depth = depth
        self.num_head = num_head
        self.qkv_dim = qkv_dim
        self.num_class = num_class

        if self.input_size[0] % self.patch_size[1] or self.input_size[2] % self.patch_size[1]:
            raise Exception("이미지를 패치만큼 나눌 수 없습니다.")
        else:
            self.patch_num = (self.input_size[1] * self.input_size[2]) // (self.patch_size[0] * self.patch_size[1]) # the number of patches in a channel
        
        self.patch_dim = self.patch_size[0] * self.patch_size[1] * self.input_size[0]
        ##wrong## self.patch_embedding = nn.Embedding(self.patch_size[0], self.emb_dim)
        self.patch_embedding = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim)
        )

        ##wrong## self.pos_emb = None
        self.cls_token = nn.Parameter(torch.zeros(self.emb_dim))
        nn.init.kaiming_uniform_(self.cls_token, nonlinearity='relu')
        self.dropout = nn.Dropout(emb_dropout)

        self.pos_emb = nn.Parameter(torch.zeros(self.patch_num+1, self.emb_dim))
        nn.init.kaiming_uniform_(self.cls_token, nonlinearity='relu')

        self.transformer = Transformer(self.emb_dim, self.depth, self.num_head, self.qkv_dim, dropout)
        ##self.layernorm1 = nn.LayerNorm(self.emb_dim)

        self.mlp = nn.Linear(self.emb_dim, self.num_class)
        ##self.layernorm2 = nn.LayerNorm(self.emb_dim)
        
    def forward(self, x):
        B, C, H, W = x.shape

        x = x.reshape(B, self.patch_num, self.patch_dim)    # B * patch_num * (P^2 * C)
        x = self.patch_embedding(x)                         # B * patch_num * (P^2 * C) ==> B * patch_num * emb_dim

        self.cls_token = self.cls_token.expand(B, 1, -1)
        x = torch.concat((self.cls_token, x), dim=1)        # + class token    B * (patch_num + 1) * emb_dim
        x += self.pos_emb                                   # + pose embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  #### 아직 여기까지 이해못함

        x = self.to_latent(x)
        return self.mlp(x)

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