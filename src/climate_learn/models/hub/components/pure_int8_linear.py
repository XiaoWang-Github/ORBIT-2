import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
import torch.distributed as dist
import os
from .triton_ops import triton_int8_matmul

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
    def forward(ctx, input_fp32, weight_fp32, bias_fp32):
        # 1. Determine shift amounts for input and weight
        input_abs_max = input_fp32.abs().max()
        weight_abs_max = weight_fp32.abs().max()

        input_shift, input_scale_factor = get_scale_shift(input_abs_max)
        weight_shift, weight_scale_factor = get_scale_shift(weight_abs_max)
        
        # 2. Quantize input and weight to INT8
        input_int8 = quantize_to_int8_shifted(input_fp32, input_scale_factor, stochastic=False)
        weight_int8 = quantize_to_int8_shifted(weight_fp32, weight_scale_factor, stochastic=False)

        # 3. Perform INT8 matrix multiplication using explicit torch._int_mm for hardware acceleration
        # Reshape input to 2D
        input_int8_flattened = input_int8.reshape(-1, input_int8.shape[-1]).contiguous()
        
        # We need (Batch, Out) = (Batch, In) @ (In, Out). 
        # weight_int8 is (Out, In). So we use weight_int8.t() which is (In, Out).
        # We pass the transposed view directly without .contiguous() to allow Triton to use 
        # stride-1 access along the K dimension (which is stride(0) of the transposed view).
        weight_int8_t = weight_int8.t()
        
        # Output is INT32
        output_int32_accum_flattened = triton_int8_matmul(input_int8_flattened, weight_int8_t)
        
        # Reshape back to original dimensions
        output_shape = list(input_fp32.shape)
        output_shape[-1] = weight_int8.shape[0] # out_features
        output_int32_accum = output_int32_accum_flattened.reshape(output_shape)
        
        # Determine the output shift based on the product of input and weight scales.
        output_abs_max_int32 = output_int32_accum.abs().max()
        output_shift, output_scale_factor = get_scale_shift(output_abs_max_int32.to(torch.float32)) 
        
        output_int8 = quantize_to_int8_shifted(output_int32_accum.to(torch.float32), output_scale_factor)
        
        output_dequant = dequantize_from_int8_shifted(output_int8, output_scale_factor)

        # 4. Apply bias
        if bias_fp32 is not None:
            output_dequant += bias_fp32 

        # Store quantized inputs, weights, and shifts for backward
        ctx.save_for_backward(input_int8, weight_int8)
        ctx.input_shift = input_shift
        ctx.weight_shift = weight_shift
        ctx.output_shift = output_shift 
        ctx.input_fp32_shape = input_fp32.shape
        ctx.has_bias = bias_fp32 is not None

        # CK Attention requires bfloat16 or float16 input. Cast output to bfloat16.
        return output_dequant.to(torch.bfloat16)

    @staticmethod
    def backward(ctx, grad_output_fp32):
        input_int8, weight_int8 = ctx.saved_tensors
        input_shift = ctx.input_shift
        weight_shift = ctx.weight_shift

        grad_input = grad_weight = grad_bias = None

        # 1. Quantize grad_output to INT8 using a new dynamic shift (with stochastic rounding)
        grad_output_abs_max = grad_output_fp32.abs().max()
        grad_output_shift, grad_output_scale_factor = get_scale_shift(grad_output_abs_max)
        grad_output_int8 = quantize_to_int8_shifted(grad_output_fp32, grad_output_scale_factor, stochastic=True)

        # 2. Calculate gradients using INT8 matmul
        # dW = input.T @ grad_output
        if ctx.needs_input_grad[1]:
            # Reshape for matmul
            input_int8_flattened = input_int8.reshape(-1, input_int8.shape[-1]).contiguous()
            grad_output_int8_flattened = grad_output_int8.reshape(-1, grad_output_int8.shape[-1]).contiguous()
            
            # We need (Out, In) = (Out, Batch) @ (Batch, In)
            # grad_output is (Batch, Out). input is (Batch, In).
            # So we compute grad_output.T @ input.
            grad_output_t_contiguous = grad_output_int8_flattened.t().contiguous()
            
            grad_weight_int32_accum = triton_int8_matmul(grad_output_t_contiguous, input_int8_flattened)
            
            # Result is (Out, In) which matches weight shape.
            
            grad_weight_dequant_shift = input_shift + grad_output_shift
            grad_weight_scale_factor = torch.pow(2.0, grad_weight_dequant_shift)
            grad_weight = dequantize_from_int8_shifted(grad_weight_int32_accum.to(torch.float32), grad_weight_scale_factor) 

        # dX = grad_output @ weight.T
        if ctx.needs_input_grad[0]:
            # Reshape for matmul
            # grad_output_int8_flattened already exists
            # weight_int8 is (Out, In).
            # We need (Batch, In) = (Batch, Out) @ (Out, In).
            # _int_mm(grad_output, weight)
            
            # Ensure weight is contiguous
            weight_int8_contiguous = weight_int8.contiguous()
            
            grad_input_int32_accum_flattened = triton_int8_matmul(grad_output_int8_flattened, weight_int8_contiguous)
            
            # Reshape back to original input shape (batch dims + in_features)
            grad_input_int32_accum = grad_input_int32_accum_flattened.reshape(ctx.input_fp32_shape[0], ctx.input_fp32_shape[1], ctx.input_fp32_shape[2])

            grad_input_dequant_shift = grad_output_shift + weight_shift
            grad_input_scale_factor = torch.pow(2.0, grad_input_dequant_shift)
            grad_input = dequantize_from_int8_shifted(grad_input_int32_accum.to(torch.float32), grad_input_scale_factor) 
            
            grad_input = grad_input.reshape(ctx.input_fp32_shape)

        # dBias = grad_output.sum(0)
        if ctx.needs_input_grad[2] and ctx.has_bias:
            grad_bias = grad_output_fp32.sum(0) # Bias gradient remains FP32 for now

        return grad_input, grad_weight, grad_bias


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
            # PureInt8Matmul will handle the quantization internally
            return PureInt8Matmul.apply(input, self.weight, self.bias)
        else:
            # Standard PyTorch linear (bfloat16/float32)
            return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, int8_enabled={self.int8_enabled}'
