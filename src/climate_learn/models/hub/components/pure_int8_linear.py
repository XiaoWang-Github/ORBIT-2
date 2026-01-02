import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.autograd import Function
import math
import torch.distributed as dist
import os
# Use ROCm/ATen int8 GEMM (hits rocBLASLt on MI250x) instead of custom Triton
_int_mm = torch.ops.aten._int_mm
_DISABLE_STOCHASTIC_ROUNDING = os.environ.get("INT8_DISABLE_STOCHASTIC_ROUND", "0") == "1"
_WEIGHT_CACHE_STRATEGY = os.environ.get("INT8_WEIGHT_CACHE_STRATEGY", "step")  # step|epoch|off
_INPUT_SCALE_EMA_ALPHA = float(os.environ.get("INT8_INPUT_SCALE_EMA", "0"))
_INPUT_SCALE_UPDATE_EVERY = int(os.environ.get("INT8_INPUT_SCALE_UPDATE_EVERY", "1"))
_INPUT_SCALE_FREEZE_AFTER = int(os.environ.get("INT8_INPUT_SCALE_FREEZE_AFTER", "0"))
_INPUT_SCALE_SAMPLE_STRIDE = int(os.environ.get("INT8_INPUT_SCALE_SAMPLE_STRIDE", "1"))
_OUTPUT_SCALE_SAMPLE_STRIDE = int(os.environ.get("INT8_OUTPUT_SCALE_SAMPLE_STRIDE", "1"))
_OUTPUT_REQUANT_METHOD = os.environ.get("INT8_OUTPUT_REQUANT_METHOD", "auto").lower()
_INT8_OUTPUT_DTYPE = torch.bfloat16


def set_int8_output_dtype(dtype: torch.dtype) -> None:
    """Set the output dtype for INT8 path (e.g., torch.float32 or torch.bfloat16)."""
    global _INT8_OUTPUT_DTYPE
    _INT8_OUTPUT_DTYPE = dtype


@dataclass
class QuantizedTensor:
    """Container for int8 tensor + scale factor (float tensor)."""
    int8: torch.Tensor
    scale: torch.Tensor


def quantize_activation_to_int8(tensor_fp32):
    """Quantize activation to int8 and return (int8, scale_factor)."""
    sample_stride = max(_OUTPUT_SCALE_SAMPLE_STRIDE, 1)
    if sample_stride > 1:
        flat = tensor_fp32.reshape(-1)
        sampled = flat[::sample_stride]
        if sampled.numel() == 0:
            sampled = flat
        abs_max = sampled.abs().max()
    else:
        abs_max = tensor_fp32.abs().max()
    shift, scale = get_scale_shift(abs_max)
    int8 = quantize_to_int8_shifted(tensor_fp32, scale, stochastic=False)
    return QuantizedTensor(int8=int8, scale=scale)

# --- Quantization and Dequantization Helper Functions ---

def get_scale_shift(tensor_abs_max):
    """
    Calculates a bit-shift amount to scale tensor values into the INT8 range.
    Returns (shift_amount, actual_scale_factor) as Tensors to avoid CPU-GPU sync.
    """
    # Use a small epsilon to avoid division by zero
    # We keep operations on the device.
    
    # Formula: shift = log2(127 / abs_max)
    # Using 1e-9 for numerical stability if abs_max is extremely small (but not 0)
    shift_amount_float = torch.log2(127.0 / (tensor_abs_max + 1e-9))
    
    # Handle potential NaNs or Infs (e.g. if tensor_abs_max was NaN/Inf)
    # Also if tensor_abs_max is 0, the log2 term becomes huge. We want shift=0 for input=0.
    shift_amount_float = torch.nan_to_num(shift_amount_float, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Explicitly force 0 shift if input max is 0
    shift_amount_float = torch.where(tensor_abs_max == 0, torch.tensor(0.0, device=tensor_abs_max.device, dtype=shift_amount_float.dtype), shift_amount_float)

    shift_amount = torch.round(shift_amount_float)
    shift_amount = torch.clamp(shift_amount, -15, 15)
    
    actual_scale_factor = torch.pow(2.0, shift_amount)
    
    # Ensure types are correct (shift_amount usually int, but keep as float tensor for pow then cast if needed? 
    # quantize function expects shift_amount to be broadcastable.
    # Let's keep them as float tensors (or whatever type log2 returns, usually float)
    
    return shift_amount, actual_scale_factor

def quantize_to_int8_shifted(tensor_fp32, scale_factor, stochastic=False):
    """
    Quantizes a float32 tensor to int8 using a bit-shift-like scaling.
    Accepts scale_factor (2.0 ** shift_amount) directly to avoid re-computation.
    """
    # Scale the tensor
    scaled_tensor_fp32 = tensor_fp32 * scale_factor

    # Apply stochastic rounding
    if stochastic:
        # Add uniform noise [-0.5, 0.5] before rounding
        scaled_tensor_fp32 = scaled_tensor_fp32 + (torch.rand_like(scaled_tensor_fp32) - 0.5)

    # Round to nearest integer
    quantized_tensor_int = torch.round(scaled_tensor_fp32)

    # Clip to INT8 range
    quantized_tensor_int = torch.clamp(quantized_tensor_int, -128, 127)
    return quantized_tensor_int.to(torch.int8)

def dequantize_from_int8_shifted(tensor_int8, scale_factor):
    """
    Dequantizes an int8 tensor back to float32 using the inverse bit-shift scaling.
    Accepts scale_factor (2.0 ** shift_amount) directly.
    """
    return tensor_int8.to(torch.float32) / scale_factor


# --- Custom Autograd Function for Pure INT8 Matmul ---

class PureInt8Matmul(Function):
    """
    Custom Autograd Function for performing matrix multiplication with INT8
    inputs and producing INT8 outputs (after scaling).
    Backward pass also uses INT8 for gradient computations.
    """
    @staticmethod
    def forward(ctx, input_fp32, weight_fp32, bias_fp32, weight_int8_cached=None, weight_shift_cached=None, weight_scale_cached=None, weight_int8_t_cached=None, input_abs_override=None):
        # 1. Determine shift amounts for input and weight
        input_abs_max = input_fp32.abs().max() if input_abs_override is None else input_abs_override
        input_shift, input_scale_factor = get_scale_shift(input_abs_max)

        # If cached quantized weight is provided, reuse it; otherwise compute fresh
        if weight_int8_cached is None or weight_shift_cached is None or weight_scale_cached is None or weight_int8_t_cached is None:
            weight_abs_max = weight_fp32.abs().max()
            weight_shift, weight_scale_factor = get_scale_shift(weight_abs_max)
            weight_int8 = quantize_to_int8_shifted(weight_fp32, weight_scale_factor, stochastic=False)
            weight_int8_t = weight_int8.t().contiguous()
        else:
            weight_int8 = weight_int8_cached
            weight_shift = weight_shift_cached
            weight_scale_factor = weight_scale_cached
            weight_int8_t = weight_int8_t_cached
        
        # 2. Quantize input and weight to INT8
        input_int8 = quantize_to_int8_shifted(input_fp32, input_scale_factor, stochastic=False)

        # 3. Perform INT8 matrix multiplication using rocBLASLt via ATen _int_mm
        # Reshape input to 2D
        input_int8_flattened = input_int8.reshape(-1, input_int8.shape[-1])
        if not input_int8_flattened.is_contiguous():
            input_int8_flattened = input_int8_flattened.contiguous()
        # weight_int8_t already contiguous if cached; otherwise computed above  # (In, Out)

        # Output is INT32
        output_int32_accum_flattened = _int_mm(input_int8_flattened, weight_int8_t)

        # Reshape back to original dimensions
        output_shape = list(input_fp32.shape)
        output_shape[-1] = weight_int8.shape[0] # out_features
        output_int32_accum = output_int32_accum_flattened.reshape(output_shape)
        
        # Determine output scale and dequantize int32 accumulators
        output_scale_factor = input_scale_factor * weight_scale_factor  # per-tensor scale
        output_dequant = dequantize_from_int8_shifted(
            output_int32_accum.to(torch.float32), output_scale_factor
        )

        # 4. Apply bias
        if bias_fp32 is not None:
            output_dequant += bias_fp32 

        # Store quantized inputs, weights, and shifts for backward
        ctx.save_for_backward(input_int8, weight_int8, weight_int8_t)
        ctx.input_shift = input_shift
        ctx.weight_shift = weight_shift
        ctx.input_fp32_shape = input_fp32.shape
        ctx.has_bias = bias_fp32 is not None

        # Cast output to configured dtype for INT8 path.
        return output_dequant.to(_INT8_OUTPUT_DTYPE)

    @staticmethod
    def backward(ctx, grad_output_fp32):
        input_int8, weight_int8, weight_int8_t = ctx.saved_tensors
        input_shift = ctx.input_shift
        weight_shift = ctx.weight_shift

        grad_input = grad_weight = grad_bias = None

        # 1. Quantize grad_output to INT8 using a new dynamic shift (with stochastic rounding)
        grad_output_abs_max = grad_output_fp32.abs().max()
        grad_output_shift, grad_output_scale_factor = get_scale_shift(grad_output_abs_max)
        grad_output_int8 = quantize_to_int8_shifted(
            grad_output_fp32,
            grad_output_scale_factor,
            stochastic=not _DISABLE_STOCHASTIC_ROUNDING,
        )

        # 2. Calculate gradients using INT8 matmul
        # Flatten tensors once for reuse in both grad paths
        grad_output_int8_flattened = grad_output_int8.reshape(-1, grad_output_int8.shape[-1])
        if not grad_output_int8_flattened.is_contiguous():
            grad_output_int8_flattened = grad_output_int8_flattened.contiguous()
        input_int8_flattened = input_int8.reshape(-1, input_int8.shape[-1])
        if not input_int8_flattened.is_contiguous():
            input_int8_flattened = input_int8_flattened.contiguous()

        # dW = input.T @ grad_output
        if ctx.needs_input_grad[1]:
            # grad_output is (Batch, Out). input is (Batch, In).
            grad_output_t_contiguous = grad_output_int8_flattened.t().contiguous()

            grad_weight_int32_accum = _int_mm(grad_output_t_contiguous, input_int8_flattened)
            
            # Result is (Out, In) which matches weight shape.
            
            grad_weight_dequant_shift = input_shift + grad_output_shift
            grad_weight_scale_factor = torch.pow(2.0, grad_weight_dequant_shift)
            grad_weight = dequantize_from_int8_shifted(grad_weight_int32_accum.to(torch.float32), grad_weight_scale_factor)

        # dX = grad_output @ weight.T
        if ctx.needs_input_grad[0]:
            # weight_int8 is (Out, In).
            weight_int8_contiguous = weight_int8 if weight_int8.is_contiguous() else weight_int8.contiguous()
            grad_input_int32_accum_flattened = _int_mm(grad_output_int8_flattened, weight_int8_contiguous)
            
            # Reshape back to original input shape
            grad_input_int32_accum = grad_input_int32_accum_flattened.reshape(ctx.input_fp32_shape)

            grad_input_dequant_shift = grad_output_shift + weight_shift
            grad_input_scale_factor = torch.pow(2.0, grad_input_dequant_shift)
            grad_input = dequantize_from_int8_shifted(grad_input_int32_accum.to(torch.float32), grad_input_scale_factor) 
            
            grad_input = grad_input.reshape(ctx.input_fp32_shape)

        # dBias = grad_output.sum(0)
        if ctx.needs_input_grad[2] and ctx.has_bias:
            grad_bias = grad_output_fp32.sum(0) # Bias gradient remains FP32 for now

        # No gradients for cached tensors passed through forward
        # Extra None for cached transposed weight tensor and input_abs_override
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


# --- Pure INT8 Linear Layer Module ---

class PureInt8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weights are stored as float and quantized on-the-fly for now,
        # but the goal is to store them as INT8 and handle updates in INT8.
        # This is a complex part of "Pure INT8 Training".
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.int8_enabled = False # Flag to control precision mode
        # Cache quantized weight to avoid re-quantizing every forward
        self._cached_weight_int8 = None
        self._cached_weight_int8_t = None
        self._cached_weight_shift = None
        self._cached_weight_scale = None
        self._cached_weight_version = None
        # Optional EMA for input abs max
        self._input_abs_ema = None
        self._input_abs_step = 0

        self.reset_parameters()

    def reset_parameters(self):
        # Standard kaiming uniform initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.int8_enabled:
            input_abs_override = None
            if 0.0 < _INPUT_SCALE_EMA_ALPHA < 1.0:
                input_abs_override = self._maybe_update_input_abs(input)
            # PureInt8Matmul will handle the quantization internally
            weight_int8, weight_int8_t, weight_shift, weight_scale = self._get_cached_weight_quant()
            return PureInt8Matmul.apply(
                input,
                self.weight,
                self.bias,
                weight_int8,
                weight_shift,
                weight_scale,
                weight_int8_t,
                input_abs_override,
            )
        else:
            # Standard PyTorch linear (bfloat16/float32)
            return F.linear(input, self.weight, self.bias)

    def forward_int8(self, input):
        """Return INT8 output with scale for end-to-end int8 path."""
        if self.int8_enabled:
            input_abs_override = None
            if 0.0 < _INPUT_SCALE_EMA_ALPHA < 1.0:
                input_abs_override = self._maybe_update_input_abs(input)
            input_abs_max = input.abs().max() if input_abs_override is None else input_abs_override
            _, input_scale = get_scale_shift(input_abs_max)
            input_int8 = quantize_to_int8_shifted(input, input_scale, stochastic=False)

            weight_int8, weight_int8_t, weight_shift, weight_scale = self._get_cached_weight_quant()
            input_int8_flattened = input_int8.reshape(-1, input_int8.shape[-1])
            if not input_int8_flattened.is_contiguous():
                input_int8_flattened = input_int8_flattened.contiguous()
            output_int32_flat = _int_mm(input_int8_flattened, weight_int8_t)

            output_shape = list(input.shape)
            output_shape[-1] = weight_int8.shape[0]
            output_int32 = output_int32_flat.reshape(output_shape)

            output_scale = input_scale * weight_scale
            if self.bias is None and _OUTPUT_REQUANT_METHOD == "direct":
                output_int8 = torch.clamp(
                    torch.round(output_int32.to(torch.float32) / output_scale), -127, 127
                ).to(torch.int8)
                return QuantizedTensor(int8=output_int8, scale=output_scale)
            output_fp = output_int32.to(torch.float32) / output_scale
            if self.bias is not None:
                output_fp += self.bias

            qt = quantize_activation_to_int8(output_fp)
            return qt

        # FP path: quantize output for consistency
        output_fp = F.linear(input, self.weight, self.bias)
        return quantize_activation_to_int8(output_fp)

    def _maybe_update_input_abs(self, input: torch.Tensor):
        """Update input abs max EMA at a reduced frequency to cut overhead."""
        with torch.no_grad():
            sample_stride = max(_INPUT_SCALE_SAMPLE_STRIDE, 1)
            if sample_stride > 1:
                flat = input.detach().reshape(-1)
                sampled = flat[::sample_stride]
                if sampled.numel() == 0:
                    sampled = flat
                current_abs = sampled.abs().max()
            else:
                current_abs = input.detach().abs().max()
            if self._input_abs_ema is None:
                # Initialize once so input_abs_override is always available.
                self._input_abs_ema = current_abs
                return self._input_abs_ema
        self._input_abs_step += 1
        if _INPUT_SCALE_FREEZE_AFTER > 0 and self._input_abs_step >= _INPUT_SCALE_FREEZE_AFTER:
            return self._input_abs_ema
        if _INPUT_SCALE_UPDATE_EVERY <= 1 or self._input_abs_step % _INPUT_SCALE_UPDATE_EVERY == 0:
            with torch.no_grad():
                sample_stride = max(_INPUT_SCALE_SAMPLE_STRIDE, 1)
                if sample_stride > 1:
                    flat = input.detach().reshape(-1)
                    sampled = flat[::sample_stride]
                    if sampled.numel() == 0:
                        sampled = flat
                    current_abs = sampled.abs().max()
                else:
                    current_abs = input.detach().abs().max()
                self._input_abs_ema = (
                    _INPUT_SCALE_EMA_ALPHA * self._input_abs_ema
                    + (1.0 - _INPUT_SCALE_EMA_ALPHA) * current_abs
                )
        return self._input_abs_ema

    def _get_cached_weight_quant(self):
        """Quantize weight once per weight version to reduce overhead."""
        current_version = self.weight._version
        cache_disabled = _WEIGHT_CACHE_STRATEGY == "off"
        need_refresh = (
            cache_disabled
            or self._cached_weight_version != current_version
            or self._cached_weight_int8 is None
            or self._cached_weight_int8_t is None
            or self._cached_weight_shift is None
            or self._cached_weight_scale is None
        )
        if need_refresh:
            with torch.no_grad():
                weight_abs_max = self.weight.abs().max()
                weight_shift, weight_scale = get_scale_shift(weight_abs_max)
                self._cached_weight_int8 = quantize_to_int8_shifted(
                    self.weight, weight_scale, stochastic=False
                )
                self._cached_weight_int8_t = self._cached_weight_int8.t().contiguous()
                self._cached_weight_shift = weight_shift
                self._cached_weight_scale = weight_scale
                # For "off", force recompute every forward by not latching version
                self._cached_weight_version = None if cache_disabled else current_version
        return self._cached_weight_int8, self._cached_weight_int8_t, self._cached_weight_shift, self._cached_weight_scale

    def refresh_int8_cache(self, force: bool = False):
        """Refresh cached weight quantization based on strategy."""
        if _WEIGHT_CACHE_STRATEGY == "off":
            # Disable cache: clear so each forward recomputes
            self._cached_weight_version = None
            self._cached_weight_int8 = None
            self._cached_weight_int8_t = None
            self._cached_weight_shift = None
            self._cached_weight_scale = None
            return
        if force or _WEIGHT_CACHE_STRATEGY in ("step", "epoch"):
            # Recompute cache regardless of version (step boundary)
            self._cached_weight_version = None
            self._get_cached_weight_quant()

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, int8_enabled={self.int8_enabled}'
