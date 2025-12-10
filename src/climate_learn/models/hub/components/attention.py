import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type
from climate_learn.utils.dist_functions import F_Identity_B_AllReduce, F_Identity_B_AllReduce_VariableMapping, Grad_Inspect
from climate_learn.utils.fused_attn import FusedAttn
import torch.distributed as dist
import os

import xformers
from xformers.components.attention.core import scaled_dot_product_attention as xformers_sdpa

# Import PureInt8Linear
from climate_learn.models.hub.components.pure_int8_linear import PureInt8Linear

def debug_print(*args, **kwargs):
    # Check rank using dist or env vars (fallback)
    rank = 0
    if dist.is_initialized():
        rank = dist.get_rank()
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
    
    # Print only for rank 0, or if all_ranks is True
    if rank == 0 or kwargs.pop("all_ranks", False):
        print(f"[ATTENTION_DEBUG_RANK_{rank}]", *args, **kwargs, flush=True)


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

        self.qkv = PureInt8Linear(dim, dim * 3 //self.tensor_par_size, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = PureInt8Linear(dim//self.tensor_par_size, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        debug_print(f"Attention.forward: Input x shape: {x.shape}, dtype: {x.dtype}")
        debug_print(f"Attention.forward: Input x NaN/Inf: NaN={torch.isnan(x).any()}, Inf={torch.isinf(x).any()}")
        
        B, N, C = x.shape

        if self.tensor_par_size>1:
            x= F_Identity_B_AllReduce(x, group=self.tensor_par_group)
            debug_print(f"Attention.forward: After F_Identity_B_AllReduce x NaN/Inf: NaN={torch.isnan(x).any()}, Inf={torch.isinf(x).any()}")

        try:
            qkv_output = self.qkv(x)
            debug_print(f"Attention.forward: After qkv (PureInt8Linear) qkv_output shape: {qkv_output.shape}, dtype: {qkv_output.dtype}")
            debug_print(f"Attention.forward: qkv_output NaN/Inf: NaN={torch.isnan(qkv_output).any()}, Inf={torch.isinf(qkv_output).any()}")
            qkv = qkv_output.reshape(B, N, 3, self.num_heads // self.tensor_par_size, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            debug_print(f"Attention.forward: After unbind q/k/v shapes: q={q.shape}, k={k.shape}, v={v.shape}")
            debug_print(f"Attention.forward: q NaN/Inf: NaN={torch.isnan(q).any()}, Inf={torch.isinf(q).any()}")
            debug_print(f"Attention.forward: k NaN/Inf: NaN={torch.isnan(k).any()}, Inf={torch.isinf(k).any()}")
            debug_print(f"Attention.forward: v NaN/Inf: NaN={torch.isnan(v).any()}, Inf={torch.isinf(v).any()}")

            q, k = self.q_norm(q), self.k_norm(k)
            debug_print(f"Attention.forward: After q_norm/k_norm q/k NaN/Inf: q_NaN={torch.isnan(q).any()}, q_Inf={torch.isinf(q).any()}, k_NaN={torch.isnan(k).any()}, k_Inf={torch.isinf(k).any()}")
        except Exception as e:
            debug_print(f"CRITICAL ERROR in Attention qkv/norm: {e}", all_ranks=True)
            raise e

        try:
            if self.fused_attn == FusedAttn.CK:
                attn_output = xformers.ops.memory_efficient_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                    p=self.attn_drop.p,
                    op=xformers.ops.MemoryEfficientAttentionCkOp
                )
            elif self.fused_attn == FusedAttn.DEFAULT:
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
                attn_output = attn_output.transpose(1, 2)
            else: # FusedAttn.NONE
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn_output = self.attn_drop(attn) @ v
                attn_output = attn_output.transpose(1, 2)
            
            x = attn_output.reshape(B, N, C//self.tensor_par_size)
            debug_print(f"Attention.forward: After attention mechanism x shape: {x.shape}, dtype: {x.dtype}")
            debug_print(f"Attention.forward: After attention mechanism x NaN/Inf: NaN={torch.isnan(x).any()}, Inf={torch.isinf(x).any()}")
        except Exception as e:
            debug_print(f"CRITICAL ERROR in Attention mechanism: {e}", all_ranks=True)
            raise e

        try:
            x = self.proj(x)
            x = self.proj_drop(x)
            debug_print(f"Attention.forward: After proj (PureInt8Linear) x NaN/Inf: NaN={torch.isnan(x).any()}, Inf={torch.isinf(x).any()}")
        except Exception as e:
            debug_print(f"CRITICAL ERROR in Attention proj: {e}", all_ranks=True)
            raise e

        if self.tensor_par_size >1:
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.tensor_par_group)
            debug_print(f"Attention.forward: After all_reduce x NaN/Inf: NaN={torch.isnan(x).any()}, Inf={torch.isinf(x).any()}")

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

        self.q = PureInt8Linear(dim, dim//tensor_par_size, bias=qkv_bias)

        self.kv = PureInt8Linear(dim, dim * 2 //tensor_par_size, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = PureInt8Linear(dim // tensor_par_size, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, var_query: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        debug_print(f"VariableMapping_Attention.forward: var_query shape: {var_query.shape}, x shape: {x.shape}")
        
        if self.tensor_par_size >1:
            var_query= F_Identity_B_AllReduce_VariableMapping(var_query, group=self.tensor_par_group)
            x= F_Identity_B_AllReduce_VariableMapping(x, group=self.tensor_par_group)

        N_a = var_query.size(dim=1) #number of aggregated variables
        B, N_i, C = x.shape #B batch times sequence length, #N_i number of input variables, C embedding size

        try:
            q_output = self.q(var_query)
            debug_print(f"VariableMapping_Attention.forward: q_output NaN/Inf: NaN={torch.isnan(q_output).any()}, Inf={torch.isinf(q_output).any()}")
            q = q_output.reshape(B, N_a, self.num_heads // self.tensor_par_size, self.head_dim ).permute(0, 2, 1, 3)

            kv_output = self.kv(x)
            debug_print(f"VariableMapping_Attention.forward: kv_output NaN/Inf: NaN={torch.isnan(kv_output).any()}, Inf={torch.isinf(kv_output).any()}")
            kv = kv_output.reshape(B, N_i, 2, self.num_heads // self.tensor_par_size, self.head_dim).permute(2, 0, 3, 1, 4)

            k, v = kv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
        except Exception as e:
            debug_print(f"CRITICAL ERROR in VariableMapping_Attention q/kv: {e}", all_ranks=True)
            raise e

        try:
            if self.fused_attn == FusedAttn.CK:
                x = xformers.ops.memory_efficient_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                    p=self.attn_drop.p,
                    op=xformers.ops.MemoryEfficientAttentionCkOp
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
        except Exception as e:
            debug_print(f"CRITICAL ERROR in VariableMapping_Attention attention mechanism: {e}", all_ranks=True)
            raise e

        x = x.reshape(B, N_a, C//self.tensor_par_size)
        
        try:
            x = self.proj(x)
            debug_print(f"VariableMapping_Attention.forward: After proj NaN/Inf: NaN={torch.isnan(x).any()}, Inf={torch.isinf(x).any()}")
            x = self.proj_drop(x)
        except Exception as e:
            debug_print(f"CRITICAL ERROR in VariableMapping_Attention proj: {e}", all_ranks=True)
            raise e

        if self.tensor_par_size >1:
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.tensor_par_group)

        return x
