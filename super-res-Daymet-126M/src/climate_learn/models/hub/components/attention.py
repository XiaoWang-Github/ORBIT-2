import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type

class VariableMapping_Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            fused_attn: bool = False,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn


        self.q = nn.Linear(dim, dim, bias=qkv_bias)
   
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, var_query: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        N_a = var_query.size(dim=1) #number of aggregated variables
        B, N_i, C = x.shape #B batch times sequence length, #N_i number of input variables, C embedding size 

        q = self.q(var_query).reshape(B, N_a, self.num_heads, self.head_dim ).permute(0, 2, 1, 3)

        #print("var_query.shape",var_query.shape,"self.q",self.q,"q.shape",q.shape,flush=True)
       
        kv = self.kv(x).reshape(B, N_i, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N_a, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
