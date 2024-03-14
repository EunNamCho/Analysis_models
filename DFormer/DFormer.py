import warnings
from collections import OrderedDict
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.utils.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmcv.runner import (BaseModule, CheckpointLoader, ModuleList,
                         load_state_dict)
from mmcv.utils import to_2tuple
import math

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pos(x) + x
        x = x.permute(0, 2, 3, 1)
        x = self.act(x)
        x = self.fc2(x)

        return x


class attention(nn.Module):
    def __init__(self, dim, num_head=8, window=7, norm_cfg=dict(type='SyncBN', requires_grad=True),drop_depth=False):
        super().__init__()
        self.num_head = num_head
        self.window = window

        self.q = nn.Linear(dim, dim)
        self.q_cut = nn.Linear(dim, dim//2)
        self.a = nn.Linear(dim, dim)
        self.l = nn.Linear(dim, dim)
        self.conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.e_conv = nn.Conv2d(dim//2, dim//2, 7, padding=3, groups=dim//2)
        self.e_fore = nn.Linear(dim//2, dim//2)
        self.e_back = nn.Linear(dim//2, dim//2)
              
        self.proj = nn.Linear(dim//2*3, dim)
        if not drop_depth:
            self.proj_e = nn.Linear(dim//2*3, dim//2)
        if window != 0:
            self.short_cut_linear = nn.Linear(dim//2*3, dim//2)
            self.kv = nn.Linear(dim, dim)
            self.pool = nn.AdaptiveAvgPool2d(output_size=(7,7))
            self.proj = nn.Linear(dim * 2, dim)
            if not drop_depth:
                self.proj_e = nn.Linear(dim*2, dim//2)

        self.act = nn.GELU()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.norm_e = LayerNorm(dim//2, eps=1e-6, data_format="channels_last")
        self.drop_depth = drop_depth 

    def forward(self, x,x_e):
        B, H, W, C = x.size()
        e_B, e_H, e_W, e_C = x_e.size()
        #print(f"RGB: B, H, W, C: {B} X {H} X {W} X {C}============================")
        #print(f"Depth: B, H, W, C: {e_B} X {e_H} X {e_W} X {e_C}============================")
        """
            x, x_e shape

            **************Stage 1***************************
            RGB: B, H, W, C: 4 X 120 X 160 X 32============================
            Depth: B, H, W, C: 4 X 120 X 160 X 16============================

            ******************Stage2 **************************
            RGB: B, H, W, C: 4 X 60 X 80 X 64============================
            Depth: B, H, W, C: 4 X 60 X 80 X 32============================

            *******************Stage 3************************
            RGB: B, H, W, C: 4 X 30 X 40 X 128============================
            Depth: B, H, W, C: 4 X 30 X 40 X 64============================

            ********************Stage 4***********************
            RGB: B, H, W, C: 4 X 15 X 20 X 256============================
            Depth: B, H, W, C: 4 X 15 X 20 X 128============================
        """
        x = self.norm(x)            # RGB
        x_e = self.norm_e(x_e)      # Depth
        if self.window != 0:        # window 뭐지? 1 Stage에서만 0이다.
            short_cut = torch.cat([x,x_e],dim=3)##########      RGB + Depth concat by Channel
            short_cut = short_cut.permute(0,3,1,2)############# 다시 B*C*H*W 로 바꿈

        q = self.q(x)               
        cutted_x = self.q_cut(x)    
        x = self.l(x).permute(0, 3, 1, 2) 
        x = self.act(x)
            
        a = self.conv(x)            # Base: Depth-Wise Conv
        a = a.permute(0, 2, 3, 1)
        a = self.a(a)

        if self.window != 0:
            #prompt = self.prompt_proj(self.prompt_embeddings)

            b = x.permute(0, 2, 3, 1)   # B*W*C*H
            kv = self.kv(b)             # Linear
            kv = kv.reshape(B, H*W, 2, self.num_head, C // self.num_head // 2).permute(2, 0, 3, 1, 4)   # AGG: [Key, Value] B*HW*2*num_head*(C//num_head//2) ==> 2, B, num_head, HW, C//num_head//2
            k, v = kv.unbind(0)         # AGG: Key, Value -> B, num_head, HW, C//num_head//2
            short_cut = self.pool(short_cut).permute(0,2,3,1) # AGG pool: B*H*W*C+1
            short_cut = self.short_cut_linear(short_cut)      # AGG: Query with RGB, Depth 
            short_cut = short_cut.reshape(B, -1, self.num_head, C // self.num_head // 2).permute(0, 2, 1, 3)  # AGG: Query -> B, 2*HW, num_head, C//num_head//2 ==> B, num_head, 2*HW, C//num_head//2
            m = short_cut #+ prompt
            attn = (m * (C // self.num_head // 2) ** -0.5) @ k.transpose(-2, -1) # GAA: Attention -> B, num_head, 2*HW, HW
            attn = attn.softmax(dim=-1)
            attn = (attn @ v).reshape(B, self.num_head, self.window, self.window, C // self.num_head // 2).permute(0, 1, 4, 2, 3).reshape(B, C // 2, self.window, self.window)
            attn = F.interpolate(attn, (H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)  ##################### GAA output
            #print(f"Query shape: {short_cut.size()}")
            #print(f"Key, Value shape: {k.size()}")
            #print(f"GAA output shape: {attn.size()}")
            """
                Query, Key, Value shape

                **************Stage 1***************************
                None

                ******************Stage2 **************************
                Query shape: torch.Size([4, 2, 49, 16])
                Key, Value shape: torch.Size([4, 2, 4800, 16])
                GAA output shape: torch.Size([4, 60, 80, 32])

                *******************Stage 3************************
                Query shape: torch.Size([4, 4, 49, 16])
                Key, Value shape: torch.Size([4, 4, 1200, 16])
                GAA output shape: torch.Size([4, 30, 40, 64])

                ********************Stage 4***********************
                Query shape: torch.Size([4, 8, 49, 16])
                Key, Value shape: torch.Size([4, 8, 300, 16])
                GAA output shape: torch.Size([4, 15, 20, 128])
            """
        
        x_e = self.e_back(self.e_conv(self.e_fore(x_e).permute(0, 3, 1, 2)).permute(0, 2, 3, 1))    # LEA: Depth-Wise Conv 
        cutted_x = cutted_x * x_e   ######################## LEA output
        x = q * a                   ######################## Base output

        if self.window != 0:
            x = torch.cat([x, attn,cutted_x], dim=3)
        else:
            x = torch.cat([x,cutted_x], dim=3)
        if not self.drop_depth:
            x_e = self.proj_e(x)
        x = self.proj(x)

        return x,x_e

class Block(nn.Module):
    def __init__(self, index, dim, num_head, norm_cfg=dict(type='SyncBN', requires_grad=True), mlp_ratio=4.,block_index=0, last_block_index=50, window=7, dropout_layer=None,drop_depth=False):
        super().__init__()
        
        self.index = index
        layer_scale_init_value = 1e-6  
        if block_index>last_block_index:
            # 무슨 일이 있어도 window는 여기서 0이 안된다
            window=0 
        self.attn = attention(dim, num_head, window=window, norm_cfg=norm_cfg,drop_depth=drop_depth)
        self.mlp = MLP(dim, mlp_ratio, norm_cfg=norm_cfg)
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()
                 
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        
        if not drop_depth:
            self.layer_scale_1_e = nn.Parameter(layer_scale_init_value * torch.ones((dim//2)), requires_grad=True)
            self.layer_scale_2_e = nn.Parameter(layer_scale_init_value * torch.ones((dim//2)), requires_grad=True)
            self.mlp_e2 = MLP(dim//2, mlp_ratio)
        self.drop_depth=drop_depth

    def forward(self, x, x_e):
        #print(f"{self.index}th Block=================================")
        res_x,res_e=x,x_e
        x,x_e=self.attn(x,x_e)

        
        x = res_x + self.dropout_layer(self.layer_scale_1.unsqueeze(0).unsqueeze(0) * x )

        
        x = x + self.dropout_layer(self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(x))
        if not self.drop_depth:
            x_e = res_e + self.dropout_layer(self.layer_scale_1_e.unsqueeze(0).unsqueeze(0) * x_e)
            x_e = x_e + self.dropout_layer(self.layer_scale_2_e.unsqueeze(0).unsqueeze(0) * self.mlp_e2(x_e))

        return x,x_e


class DFormer(BaseModule):
    def __init__(self, in_channels=4, depths=(2, 2, 8, 2), dims=(32, 64, 128, 256), out_indices=(0, 1, 2, 3), windows=[7, 7, 7, 7], norm_cfg=dict(type='SyncBN', requires_grad=True),
                 mlp_ratios=[8, 8, 4, 4], num_heads=(2, 4, 10, 16),last_block=[50,50,50,50], drop_path_rate=0.1, init_cfg=None):
        """
        PARAM:
            in_channels: B*C*H*W
            depths: The depth of each Stage
            dims: Hierarchical encoder dimensions
            out_indices: ?
            windows: ?
            num_heads: the number of Attention each building block
        """
        super().__init__()
        print(drop_path_rate)
        self.depths = depths
        self.init_cfg = init_cfg
        self.out_indices = out_indices

        # RGB stem layer
        # "Building block"에 들어가기전 RGB feature를 뽑아내는 layer
        self.downsample_layers = nn.ModuleList() 
        stem = nn.Sequential(
                nn.Conv2d(3, dims[0] // 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dims[0] // 2),
                nn.GELU(),
                nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dims[0]),
        )

        # Depth stem layer
        # "Building block"에 들어가기전 Depth feature를 뽑아내는 layer
        self.downsample_layers_e = nn.ModuleList() 
        stem_e = nn.Sequential(
                nn.Conv2d(1, dims[0] // 4, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dims[0] // 4),
                nn.GELU(),
                nn.Conv2d(dims[0] // 4, dims[0]//2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dims[0]//2),
        )

        self.downsample_layers.append(stem)
        self.downsample_layers_e.append(stem_e)

        # GAA에서 Query만들 때 필요한 Pool 7x7
        for i in range(len(dims)-1):
            stride = 2
            downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, dims[i])[1],
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=stride, padding=1),
            )
            self.downsample_layers.append(downsample_layer)

            downsample_layer_e = nn.Sequential(
                    build_norm_layer(norm_cfg, dims[i]//2)[1],
                    nn.Conv2d(dims[i]//2, dims[i+1]//2, kernel_size=3, stride=stride, padding=1),
            )
            self.downsample_layers_e.append(downsample_layer_e)

        # RGB-D Backbone with 4 Stages
        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[Block(index=cur+j, # 모든 layer 통 틀어서 몇 번째 building block인지 
                        dim=dims[i], # 이번 Stage에서 Hieracrhical encoder 차원
                        window=windows[i], 
                        dropout_layer=dict(type='DropPath', drop_prob=dp_rates[cur + j]), # dropout에 관한 거라서 신경 쓰지 말자
                        num_head=num_heads[i], # 이번 Stage에서 Attention 몇 개 만들지
                        norm_cfg=norm_cfg,     
                        block_index=depths[i]-j,   # 이번 Stage에서 몇번째 block인지
                        last_block_index=last_block[i],
                        mlp_ratio=mlp_ratios[i],drop_depth=((i==3)&(j==depths[i]-1))) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

       
        for i in out_indices:
            layer = LayerNorm(dims[i], eps=1e-6, data_format="channels_first")
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)


    def init_weights(self,pretrained):
       
        _state_dict=torch.load(pretrained, map_location=torch.device('cpu'))
        print(f"{_state_dict.keys()}================================================================================================")
        if 'state_dict_ema' in _state_dict.keys():
            _state_dict=_state_dict['state_dict_ema']
        else:
            _state_dict=_state_dict['model']

        state_dict = OrderedDict()
        for k, v in _state_dict.items():
            if k.startswith('backbone.'):
                state_dict[k[9:]] = v
            else:
                state_dict[k] = v

        
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        
        load_state_dict(self, state_dict, strict=False)

    def forward(self, x,x_e):
        #print(f"Original x, x_e shape: {x.size()}, {x_e.size()}=================")
        # Stem에서 feature 추출
        if x_e is None:
            x_e=x
        if len(x.shape)==3:
            x=x.unsqueeze(0)
        if len(x_e.shape)==3:
            x_e=x_e.unsqueeze(0)

        x_e=x_e[:,0,:,:].unsqueeze(1)
        
        outs = []

        # RGB-D building block with 4 Stages 
        for i in range(4):
            #print(f"{i}th Stage========================")
            x = self.downsample_layers[i](x)
            x_e = self.downsample_layers_e[i](x_e)
           
            #print(f"Downsampled x, x_e: {x.size()}, {x_e.size()}=====================")
            # 입력 크기(B, H, W not C) 는 계속 변한다. 
            # 그에 따라, Query, Key, Value의 0, 2 축의 크기만 바뀜
            """
                Original x, x_e shape: torch.Size([4, 3, 480, 640]), torch.Size([4, 3, 480, 640])=================

                0th Stage========================
                Downsampled x, x_e: torch.Size([4, 32, 120, 160]), torch.Size([4, 16, 120, 160])=====================

                1th Stage========================
                Downsampled x, x_e: torch.Size([4, 64, 60, 80]), torch.Size([4, 32, 60, 80])=====================
                Query shape: torch.Size([4, 2, 49, 16])
                Key, Value shape: torch.Size([4, 2, 4800, 16])
                GAA output shape: torch.Size([4, 60, 80, 32])

                2th Stage========================
                Downsampled x, x_e: torch.Size([4, 128, 30, 40]), torch.Size([4, 64, 30, 40])=====================
                Query shape: torch.Size([4, 4, 49, 16])
                Key, Value shape: torch.Size([4, 4, 1200, 16])
                GAA output shape: torch.Size([4, 30, 40, 64])

                3th Stage========================
                Downsampled x, x_e: torch.Size([4, 256, 15, 20]), torch.Size([4, 128, 15, 20])=====================
                Query shape: torch.Size([4, 8, 49, 16])
                Key, Value shape: torch.Size([4, 8, 300, 16])
                GAA output shape: torch.Size([4, 15, 20, 128])
            """

            x = x.permute(0, 2, 3, 1)
            x_e = x_e.permute(0, 2, 3, 1)
            for blk in self.stages[i]:
                x,x_e = blk(x,x_e)
            x = x.permute(0, 3, 1, 2)
            x_e = x_e.permute(0, 3, 1, 2)
            outs.append(x)
        return outs

def DFormer_Tiny(pretrained=False, **kwargs):   # 81.5
    model = DFormer(dims=[32, 64, 128, 256], mlp_ratios=[8, 8, 4, 4], depths=[3, 3, 5, 2], num_heads=[1, 2, 4, 8], windows=[0, 7, 7, 7], **kwargs)
   
    if pretrained:
        model = load_model_weights(model, 'scnet', kwargs)
    return model

def DFormer_Small(pretrained=False, **kwargs):   # 81.0
    model = DFormer(dims=[64, 128, 256, 512], mlp_ratios=[8, 8, 4, 4], depths=[2, 2, 4, 2], num_heads=[1, 2, 4, 8], windows=[0, 7, 7, 7], **kwargs)
    if pretrained:
        model = load_model_weights(model, 'scnet', kwargs)
    return model

def DFormer_Base(pretrained=False,drop_path_rate=0.1, **kwargs):   # 82.1
    model = DFormer(dims=[64, 128, 256, 512], mlp_ratios=[8, 8, 4, 4], depths=[3, 3, 12, 2], num_heads=[1, 2, 4, 8], windows=[0, 7, 7, 7],drop_path_rate=drop_path_rate, **kwargs)
   
    if pretrained:
        model = load_model_weights(model, 'scnet', kwargs)
    return model

def DFormer_Large(pretrained=False,drop_path_rate=0.1, **kwargs):   # 82.1
    model = DFormer(dims=[96, 192, 288, 576], mlp_ratios=[8, 8, 4, 4], depths=[3, 3, 12, 2], num_heads=[1, 2, 4, 8], windows=[0, 7, 7, 7],drop_path_rate=drop_path_rate, **kwargs)
    if pretrained:
        model = load_model_weights(model, 'scnet', kwargs)
    return model

