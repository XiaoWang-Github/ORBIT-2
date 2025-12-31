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
from climate_learn.models.hub.components.pure_int8_linear import PureInt8Linear, QuantizedTensor

_ATTENTION_DEBUG_ENABLED = os.environ.get("ATTENTION_DEBUG", "0") == "1"
_INT8_SOFTMAX_ENABLED = os.environ.get("INT8_SOFTMAX", "0") == "1"
_INT8_SOFTMAX_LUT_RANGE = float(os.environ.get("INT8_SOFTMAX_LUT_RANGE", "8.0"))
_INT8_SOFTMAX_LUT_SIZE = int(os.environ.get("INT8_SOFTMAX_LUT_SIZE", "256"))
_INT8_SOFTMAX_LUT_SCALE = int(os.environ.get("INT8_SOFTMAX_LUT_SCALE", "32768"))
_INT8_SOFTMAX_LOG = os.environ.get("INT8_SOFTMAX_LOG", "1") == "1"
_INT8_SOFTMAX_TRITON = os.environ.get("INT8_SOFTMAX_TRITON", "0") == "1"
_INT8_ATTENTION_E2E = os.environ.get("INT8_ATTENTION_E2E", "0") == "1"
_INT8_ATTENTION_E2E_TRITON = os.environ.get("INT8_ATTENTION_E2E_TRITON", "1") == "1"
_INT8_ATTENTION_E2E_LOG = os.environ.get("INT8_ATTENTION_E2E_LOG", "1") == "1"
_INT8_ATTENTION_LOGITS_DTYPE = os.environ.get("INT8_ATTENTION_LOGITS_DTYPE", "fp32").lower()
_INT8_ATTENTION_LOGITS_TORCH_DTYPE = (
    torch.bfloat16 if _INT8_ATTENTION_LOGITS_DTYPE == "bf16" else torch.float32
)
_EXP_LUT_CACHE = {}
_INT8_SOFTMAX_LOGGED = False
_INT8_SOFTMAX_TRITON_LOGGED = False
_INT8_ATTENTION_E2E_LOGGED = False
_INT8_ATTENTION_E2E_TRITON_LOGGED = False

def debug_print(*args, **kwargs):
    if not _ATTENTION_DEBUG_ENABLED:
        return

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

def debug_tensor(label: str, tensor: torch.Tensor):
    """Print shape/dtype and NaN/Inf status only when attention debug is enabled."""
    if not _ATTENTION_DEBUG_ENABLED:
        return
    debug_print(f"{label} shape: {tensor.shape}, dtype: {tensor.dtype}")
    debug_print(f"{label} NaN/Inf: NaN={torch.isnan(tensor).any()}, Inf={torch.isinf(tensor).any()}")


def _get_exp_lut(device: torch.device):
    key = (device, _INT8_SOFTMAX_LUT_RANGE, _INT8_SOFTMAX_LUT_SIZE, _INT8_SOFTMAX_LUT_SCALE)
    if key in _EXP_LUT_CACHE:
        return _EXP_LUT_CACHE[key]
    step = _INT8_SOFTMAX_LUT_RANGE / max(_INT8_SOFTMAX_LUT_SIZE - 1, 1)
    idx = torch.arange(_INT8_SOFTMAX_LUT_SIZE, device=device, dtype=torch.float32)
    exp_vals = torch.exp(-idx * step)
    lut = torch.round(exp_vals * _INT8_SOFTMAX_LUT_SCALE).to(torch.int32)
    _EXP_LUT_CACHE[key] = lut
    return lut


def int8_softmax_lut(logits: torch.Tensor) -> torch.Tensor:
    """Approximate softmax with LUT on clipped logits; returns FP probs."""
    global _INT8_SOFTMAX_LOGGED
    if _INT8_SOFTMAX_LOG and not _INT8_SOFTMAX_LOGGED:
        rank = 0
        if dist.is_initialized():
            rank = dist.get_rank()
        if rank == 0:
            print(
                f"INT8_SOFTMAX LUT active: range={_INT8_SOFTMAX_LUT_RANGE}, "
                f"size={_INT8_SOFTMAX_LUT_SIZE}, scale={_INT8_SOFTMAX_LUT_SCALE}",
                flush=True,
            )
        _INT8_SOFTMAX_LOGGED = True
    if _INT8_SOFTMAX_TRITON and logits.is_cuda:
        global _INT8_SOFTMAX_TRITON_LOGGED
        try:
            from climate_learn.models.hub.components import triton_ops
            lut = _get_exp_lut(logits.device)
            if _INT8_SOFTMAX_LOG and not _INT8_SOFTMAX_TRITON_LOGGED:
                rank = 0
                if dist.is_initialized():
                    rank = dist.get_rank()
                if rank == 0:
                    print("INT8_SOFTMAX using Triton LUT kernel", flush=True)
                _INT8_SOFTMAX_TRITON_LOGGED = True
            return triton_ops.triton_lut_softmax(
                logits, lut, _INT8_SOFTMAX_LUT_RANGE, _INT8_SOFTMAX_LUT_SIZE
            )
        except Exception as exc:
            if _INT8_SOFTMAX_LOG and not _INT8_SOFTMAX_TRITON_LOGGED:
                rank = 0
                if dist.is_initialized():
                    rank = dist.get_rank()
                if rank == 0:
                    print(f"INT8_SOFTMAX Triton fallback: {exc}", flush=True)
                _INT8_SOFTMAX_TRITON_LOGGED = True
    if _INT8_SOFTMAX_LUT_SIZE <= 1:
        return logits.softmax(dim=-1)
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted = logits - max_logits
    shifted = torch.clamp(shifted, -_INT8_SOFTMAX_LUT_RANGE, 0.0)
    step = _INT8_SOFTMAX_LUT_RANGE / max(_INT8_SOFTMAX_LUT_SIZE - 1, 1)
    idx = torch.round(-shifted / step).to(torch.int64)
    idx = torch.clamp(idx, 0, _INT8_SOFTMAX_LUT_SIZE - 1)
    lut = _get_exp_lut(logits.device)
    exp_int = lut[idx]
    sum_int = exp_int.sum(dim=-1, keepdim=True).clamp_min(1)
    probs = exp_int.float() / sum_int.float()
    return probs.to(logits.dtype)


def _int8_bmm_qk(q_int8: torch.Tensor, k_int8: torch.Tensor) -> torch.Tensor:
    """Compute QK^T in int32 using per-(batch, head) int8 matmul."""
    global _INT8_ATTENTION_E2E_TRITON_LOGGED
    if _INT8_ATTENTION_E2E_TRITON and q_int8.is_cuda and k_int8.is_cuda:
        try:
            from climate_learn.models.hub.components import triton_ops
            if _INT8_ATTENTION_E2E_LOG and not _INT8_ATTENTION_E2E_TRITON_LOGGED:
                rank = 0
                if dist.is_initialized():
                    rank = dist.get_rank()
                if rank == 0:
                    print("INT8_ATTENTION_E2E using Triton QK kernel", flush=True)
                _INT8_ATTENTION_E2E_TRITON_LOGGED = True
            return triton_ops.triton_int8_bmm_qk(q_int8, k_int8)
        except Exception as exc:
            if _INT8_ATTENTION_E2E_LOG and not _INT8_ATTENTION_E2E_TRITON_LOGGED:
                rank = 0
                if dist.is_initialized():
                    rank = dist.get_rank()
                if rank == 0:
                    print(f"INT8_ATTENTION_E2E Triton fallback: {exc}", flush=True)
                _INT8_ATTENTION_E2E_TRITON_LOGGED = True
    # q_int8: (B, H, Nq, D), k_int8: (B, H, Nk, D)
    bsz, nheads, n_q, dim = q_int8.shape
    n_k = k_int8.shape[2]
    q_flat = q_int8.reshape(bsz * nheads, n_q, dim)
    k_flat = k_int8.reshape(bsz * nheads, n_k, dim)
    outputs = []
    for idx in range(q_flat.shape[0]):
        q_i = q_flat[idx].contiguous()
        k_i = k_flat[idx].contiguous()
        orig_rows = q_i.shape[0]
        orig_dim = q_i.shape[1]
        orig_cols = k_i.shape[0]
        if orig_rows < 17:
            # aten._int_mm requires M > 16; pad to 17 rows and trim back.
            pad_rows = 17 - orig_rows
            q_i = torch.nn.functional.pad(q_i, (0, 0, 0, pad_rows)).contiguous()
        pad_dim = (-orig_dim) % 8
        if pad_dim:
            # aten._int_mm requires K multiple of 8.
            q_i = torch.nn.functional.pad(q_i, (0, pad_dim, 0, 0)).contiguous()
            k_i = torch.nn.functional.pad(k_i, (0, pad_dim, 0, 0)).contiguous()
        pad_cols = (-orig_cols) % 8
        if pad_cols:
            # aten._int_mm requires N multiple of 8 (mat2.size(1)).
            k_i = torch.nn.functional.pad(k_i, (0, 0, 0, pad_cols)).contiguous()
        logits_i = torch.ops.aten._int_mm(q_i, k_i.t().contiguous())
        if orig_rows < 17:
            logits_i = logits_i[:orig_rows, :]
        if pad_cols:
            logits_i = logits_i[:, :orig_cols]
        if pad_dim:
            logits_i = logits_i[:, :orig_dim]
        outputs.append(logits_i)
    out = torch.stack(outputs, dim=0)
    return out.reshape(bsz, nheads, n_q, n_k)


def _maybe_log_e2e_int8():
    global _INT8_ATTENTION_E2E_LOGGED
    if _INT8_ATTENTION_E2E_LOG and not _INT8_ATTENTION_E2E_LOGGED:
        rank = 0
        if dist.is_initialized():
            rank = dist.get_rank()
        if rank == 0:
            print("INT8_ATTENTION_E2E path active (int8 QK^T with scale)", flush=True)
        _INT8_ATTENTION_E2E_LOGGED = True


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
        debug_tensor("Attention.forward: Input x", x)
        
        B, N, C = x.shape
        out_dtype = x.dtype
        use_int8_e2e = (
            _INT8_ATTENTION_E2E
            and self.fused_attn == FusedAttn.NONE
            and self.qkv.int8_enabled
        )

        if self.tensor_par_size>1:
            x= F_Identity_B_AllReduce(x, group=self.tensor_par_group)
            debug_tensor("Attention.forward: After F_Identity_B_AllReduce x", x)

        try:
            if not use_int8_e2e:
                qkv_output = self.qkv(x)
                debug_tensor("Attention.forward: After qkv (PureInt8Linear) qkv_output", qkv_output)
                qkv = qkv_output.reshape(B, N, 3, self.num_heads // self.tensor_par_size, self.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                if _ATTENTION_DEBUG_ENABLED:
                    debug_print(f"Attention.forward: After unbind q/k/v shapes: q={q.shape}, k={k.shape}, v={v.shape}")
                debug_tensor("Attention.forward: q", q)
                debug_tensor("Attention.forward: k", k)
                debug_tensor("Attention.forward: v", v)

                q, k = self.q_norm(q), self.k_norm(k)
                if _ATTENTION_DEBUG_ENABLED:
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
                if _INT8_ATTENTION_E2E and self.qkv.int8_enabled:
                    _maybe_log_e2e_int8()
                    qkv_qt = self.qkv.forward_int8(x)
                    if not isinstance(qkv_qt, QuantizedTensor):
                        raise RuntimeError("Expected QuantizedTensor from forward_int8")
                    qkv_int8 = qkv_qt.int8.reshape(
                        B, N, 3, self.num_heads // self.tensor_par_size, self.head_dim
                    ).permute(2, 0, 3, 1, 4)
                    q_int8, k_int8, v_int8 = qkv_int8.unbind(0)
                    logits_int32 = _int8_bmm_qk(q_int8, k_int8)
                    scale_qk = qkv_qt.scale * qkv_qt.scale
                    logits_scaled = logits_int32.to(_INT8_ATTENTION_LOGITS_TORCH_DTYPE) / scale_qk.to(_INT8_ATTENTION_LOGITS_TORCH_DTYPE)
                    attn = int8_softmax_lut(logits_scaled) if _INT8_SOFTMAX_ENABLED else logits_scaled.softmax(dim=-1)
                    attn = self.attn_drop(attn).to(out_dtype)
                    v_fp = v_int8.to(out_dtype) / qkv_qt.scale.to(out_dtype)
                    attn_output = attn @ v_fp
                    attn_output = attn_output.to(out_dtype).transpose(1, 2)
                else:
                    q = q * self.scale
                    attn = q @ k.transpose(-2, -1)
                    if _INT8_SOFTMAX_ENABLED:
                        attn = int8_softmax_lut(attn)
                    else:
                        attn = attn.softmax(dim=-1)
                    attn_output = self.attn_drop(attn) @ v
                    attn_output = attn_output.transpose(1, 2)
            
            x = attn_output.reshape(B, N, C//self.tensor_par_size)
            debug_tensor("Attention.forward: After attention mechanism x", x)
        except Exception as e:
            debug_print(f"CRITICAL ERROR in Attention mechanism: {e}", all_ranks=True)
            raise e

        try:
            x = self.proj(x)
            x = self.proj_drop(x)
            debug_tensor("Attention.forward: After proj (PureInt8Linear) x", x)
        except Exception as e:
            debug_print(f"CRITICAL ERROR in Attention proj: {e}", all_ranks=True)
            raise e

        if self.tensor_par_size >1:
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.tensor_par_group)
            debug_tensor("Attention.forward: After all_reduce x", x)

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
        if _ATTENTION_DEBUG_ENABLED:
            debug_print(f"VariableMapping_Attention.forward: var_query shape: {var_query.shape}, x shape: {x.shape}")
        
        out_dtype = x.dtype
        use_int8_e2e = (
            _INT8_ATTENTION_E2E
            and self.fused_attn == FusedAttn.NONE
            and self.q.int8_enabled
            and self.kv.int8_enabled
        )
        if self.tensor_par_size >1:
            var_query= F_Identity_B_AllReduce_VariableMapping(var_query, group=self.tensor_par_group)
            x= F_Identity_B_AllReduce_VariableMapping(x, group=self.tensor_par_group)

        N_a = var_query.size(dim=1) #number of aggregated variables
        B, N_i, C = x.shape #B batch times sequence length, #N_i number of input variables, C embedding size

        try:
            if not use_int8_e2e:
                q_output = self.q(var_query)
                debug_tensor("VariableMapping_Attention.forward: q_output", q_output)
                q = q_output.reshape(B, N_a, self.num_heads // self.tensor_par_size, self.head_dim ).permute(0, 2, 1, 3)

                kv_output = self.kv(x)
                debug_tensor("VariableMapping_Attention.forward: kv_output", kv_output)
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
                if _INT8_ATTENTION_E2E and self.q.int8_enabled and self.kv.int8_enabled:
                    _maybe_log_e2e_int8()
                    q_qt = self.q.forward_int8(var_query)
                    kv_qt = self.kv.forward_int8(x)
                    if not isinstance(q_qt, QuantizedTensor) or not isinstance(kv_qt, QuantizedTensor):
                        raise RuntimeError("Expected QuantizedTensor from forward_int8")
                    q_int8 = q_qt.int8.reshape(
                        B, N_a, self.num_heads // self.tensor_par_size, self.head_dim
                    ).permute(0, 2, 1, 3)
                    kv_int8 = kv_qt.int8.reshape(
                        B, N_i, 2, self.num_heads // self.tensor_par_size, self.head_dim
                    ).permute(2, 0, 3, 1, 4)
                    k_int8, v_int8 = kv_int8.unbind(0)
                    logits_int32 = _int8_bmm_qk(q_int8, k_int8)
                    scale_qk = q_qt.scale * kv_qt.scale
                    logits_scaled = logits_int32.to(_INT8_ATTENTION_LOGITS_TORCH_DTYPE) / scale_qk.to(_INT8_ATTENTION_LOGITS_TORCH_DTYPE)
                    attn = int8_softmax_lut(logits_scaled) if _INT8_SOFTMAX_ENABLED else logits_scaled.softmax(dim=-1)
                    attn = self.attn_drop(attn).to(out_dtype)
                    v_fp = v_int8.to(out_dtype) / kv_qt.scale.to(out_dtype)
                    x = attn @ v_fp
                    x = x.to(out_dtype).transpose(1, 2)
                else:
                    q = q * self.scale
                    attn = q @ k.transpose(-2, -1)
                    if _INT8_SOFTMAX_ENABLED:
                        attn = int8_softmax_lut(attn)
                    else:
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
            debug_tensor("VariableMapping_Attention.forward: After proj", x)
            x = self.proj_drop(x)
        except Exception as e:
            debug_print(f"CRITICAL ERROR in VariableMapping_Attention proj: {e}", all_ranks=True)
            raise e

        if self.tensor_par_size >1:
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.tensor_par_group)

        return x
