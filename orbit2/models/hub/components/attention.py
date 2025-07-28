import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type
from orbit2.utils.dist_functions import F_Identity_B_AllReduce, F_Identity_B_AllReduce_VariableMapping, Grad_Inspect
from orbit2.utils.fused_attn import FusedAttn
import torch.distributed as dist

import xformers
from xformers.components.attention.core import scaled_dot_product_attention as xformers_sdpa

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            fused_attn: FusedAttn = FusedAttn.NONE,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            tensor_par_size = 1,
            tensor_par_group = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        self.tensor_par_size = tensor_par_size
        self.tensor_par_group = tensor_par_group

        self.qkv = nn.Linear(dim, dim * 3 //self.tensor_par_size, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim//self.tensor_par_size, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        if self.tensor_par_size>1:

            x= F_Identity_B_AllReduce(x, group=self.tensor_par_group)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads // self.tensor_par_size, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn == FusedAttn.CK:
            x = xformers.ops.memory_efficient_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                p=self.attn_drop.p,
                op=xformers.ops.MemoryEfficientAttentionCkOp
                # MemoryEfficientAttentionCkOp seems to work fine for now
                #op=xformers.ops.MemoryEfficientAttentionSplitKCkOp
                #op=xformers.ops.MemoryEfficientAttentionFlashAttentionOp
                #op=xformers.ops.fmha.MemoryEfficientAttentionCkOp
                #op=xformers.ops.fmha.MemoryEfficientAttentionCkDecoderOp
                #op=xformers.ops.MemoryEfficientAttentionOp
            )
        elif self.fused_attn == FusedAttn.DEFAULT:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            x = x.transpose(1, 2)
        else: # FusedAttn.NONE
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
            x = x.transpose(1, 2)

        x = x.reshape(B, N, C//self.tensor_par_size)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.tensor_par_size >1:
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.tensor_par_group)

        return x










class VariableMapping_Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            fused_attn: FusedAttn = FusedAttn.NONE,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            tensor_par_size: int = 1,
            tensor_par_group = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        self.tensor_par_size = tensor_par_size
        self.tensor_par_group = tensor_par_group

        self.q = nn.Linear(dim, dim//tensor_par_size, bias=qkv_bias)

        self.kv = nn.Linear(dim, dim * 2 //tensor_par_size, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim // tensor_par_size, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, var_query: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        if self.tensor_par_size >1:

            var_query= F_Identity_B_AllReduce_VariableMapping(var_query, group=self.tensor_par_group)
            x= F_Identity_B_AllReduce_VariableMapping(x, group=self.tensor_par_group)

        N_a = var_query.size(dim=1) #number of aggregated variables
        B, N_i, C = x.shape #B batch times sequence length, #N_i number of input variables, C embedding size

        q = self.q(var_query).reshape(B, N_a, self.num_heads // self.tensor_par_size, self.head_dim ).permute(0, 2, 1, 3)

        #print("var_query.shape",var_query.shape,"self.q",self.q,"q.shape",q.shape,flush=True)

        kv = self.kv(x).reshape(B, N_i, 2, self.num_heads // self.tensor_par_size, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn == FusedAttn.CK:
            x = xformers.ops.memory_efficient_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                p=self.attn_drop.p,
                op=xformers.ops.MemoryEfficientAttentionCkOp
                #op=xformers.ops.MemoryEfficientAttentionSplitKCkOp
                #op=xformers.ops.MemoryEfficientAttentionFlashAttentionOp
                #op=xformers.ops.fmha.MemoryEfficientAttentionCkOp
                #op=xformers.ops.fmha.MemoryEfficientAttentionCkDecoderOp
                #op=xformers.ops.MemoryEfficientAttentionOp
            )
        elif self.fused_attn == FusedAttn.DEFAULT:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            x = x.transpose(1, 2)
        else: # FusedAttn.NONE
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
            x = x.transpose(1, 2)

        x = x.reshape(B, N_a, C//self.tensor_par_size)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.tensor_par_size >1:
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.tensor_par_group)

        return x
