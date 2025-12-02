"""Quantization utilities for ORBIT-2 models.

This module provides ROCm-compatible quantization functions for applying
hybrid quantization strategies to Res-slim-VIT models. It supports:
- Dynamic quantization (weights only)
- Static quantization (weights + activations)  
- Selective quantization of attention layers while preserving CNN layers

Designed for AMD MI250X GPU with ROCm 6.4 environment.
"""

import os
# Force Triton cache to Lustre to avoid home directory quota limits
os.environ["TRITON_CACHE_DIR"] = "/lustre/orion/proj-shared/lrn036/yoonh/cache/triton"

import torch
import torch.nn as nn
from typing import Set, List, Optional
import sys


def check_rocm_quantization_support() -> dict:
    """Check ROCm environment and quantization support.
    
    Returns:
        dict: Environment information including PyTorch version, ROCm support,
              and quantization availability.
    """
    env_info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "rocm_available": hasattr(torch.version, 'hip') and torch.version.hip is not None,
        "quantization_available": hasattr(torch, 'quantization'),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }
    
    if env_info["rocm_available"]:
        env_info["rocm_version"] = torch.version.hip
    
    return env_info


def get_quantization_backend(device: torch.device) -> str:
    """Get appropriate quantization backend for the device.
    
    Args:
        device: Target device (CPU or GPU)
        
    Returns:
        str: Backend name ('fbgemm' for CPU, 'native' for ROCm GPU)
    """
    if device.type == 'cpu':
        return 'fbgemm'  # x86 optimized
    else:
        # For ROCm GPUs, use native PyTorch ops
        return 'native'


def get_attention_modules(model: nn.Module) -> List[nn.Module]:
    """Identify all attention-related modules in the model.
    
    This function traverses the model and identifies:
    - Attention layers (self.attn in Block)
    - VariableMapping_Attention (self.var_agg)
    - Linear layers within attention (qkv, proj, q, kv)
    
    Args:
        model: PyTorch model (Res_Slim_ViT)
        
    Returns:
        List of attention module references
    """
    from climate_learn.models.hub.components.attention import Attention, VariableMapping_Attention
    from climate_learn.models.hub.components.vit_blocks import Block
    
    attention_modules = []
    
    for name, module in model.named_modules():
        # Check if this is an Attention or VariableMapping_Attention module
        if isinstance(module, (Attention, VariableMapping_Attention)):
            attention_modules.append((name, module))
            print(f"Found attention module: {name} ({type(module).__name__})", flush=True)
    
    return attention_modules


def get_attention_linear_layers(model: nn.Module) -> List[nn.Linear]:
    """Get all Linear layers within attention modules.
    
    Args:
        model: PyTorch model
        
    Returns:
        List of Linear layers to quantize
    """
    from climate_learn.models.hub.components.attention import Attention, VariableMapping_Attention
    
    linear_layers = []
    
    for name, module in model.named_modules():
        # Check if parent is an attention module
        if isinstance(module, (Attention, VariableMapping_Attention)):
            # Get Linear layers within this attention module
            for subname, submodule in module.named_children():
                if isinstance(submodule, nn.Linear):
                    full_name = f"{name}.{subname}"
                    linear_layers.append((full_name, submodule))
                    print(f"  Found Linear layer: {full_name}", flush=True)
    
    return linear_layers


def apply_dynamic_quantization(
    model: nn.Module,
    attention_only: bool = True,
    dtype: torch.dtype = torch.qint8,
    device = None
) -> nn.Module:
    """Apply dynamic quantization to model.
    
    Dynamic quantization quantizes weights offline and activations dynamically
    during runtime. This is simpler than static quantization and doesn't require
    calibration data.
    
    NOTE: PyTorch quantization only works on CPU. This function will:
    1. Move model to CPU
    2. Apply quantization
    3. Move model back to original device (if provided)
    
    Args:
        model: PyTorch model to quantize
        attention_only: If True, only quantize attention layers (hybrid strategy)
        dtype: Quantization data type (torch.qint8)
        device: Original device to move model back to after quantization (can be int or torch.device)
        
    Returns:
        Quantized model (on CPU or moved back to device if specified)
    """
    print("\n" + "="*80, flush=True)
    print("APPLYING DYNAMIC QUANTIZATION", flush=True)
    print("="*80, flush=True)
    
    # Handle device conversion
    original_device = None
    if device is not None:
        if isinstance(device, int):
            original_device = torch.device(f'cuda:{device}')
        elif isinstance(device, torch.device):
            original_device = device
        else:
            original_device = device
    
    # Check for bitsandbytes availability
    try:
        import bitsandbytes as bnb
        use_bitsandbytes = True
        print("✓ bitsandbytes found! Using GPU-native 8-bit quantization.", flush=True)
    except ImportError as e:
        use_bitsandbytes = False
        import sys
        print(f"❌ CRITICAL ERROR: bitsandbytes not found or failed to load: {e}", flush=True)
        print(f"  Current sys.path: {sys.path}", flush=True)
        print("  Please install bitsandbytes to use GPU quantization.", flush=True)
        # print("  Falling back to PyTorch CPU quantization.", flush=True) # Disabled fallback
        raise e # Stop execution to prevent misleading success

    if use_bitsandbytes and (original_device is None or original_device.type != 'cpu'):
        # GPU Quantization using bitsandbytes
        print(f"Applying bitsandbytes 8-bit quantization on {original_device}...", flush=True)
        
        if attention_only:
            print("Strategy: Hybrid (Attention + MLP INT8, CNN FP16/32) using bitsandbytes", flush=True)
            from climate_learn.models.hub.components.attention import Attention, VariableMapping_Attention
            from climate_learn.models.hub.components.vit_blocks import Mlp
            
            quantized_count = 0
            for name, module in model.named_modules():
                if isinstance(module, (Attention, VariableMapping_Attention)):
                    for attr_name in ['qkv', 'q', 'kv', 'proj']:
                        if hasattr(module, attr_name):
                            layer = getattr(module, attr_name)
                            if isinstance(layer, nn.Linear):
                                # Replace with bitsandbytes Linear8bitLt
                                # Note: has_fp16_weights=False for inference to save memory
                                bnb_layer = bnb.nn.Linear8bitLt(
                                    layer.in_features,
                                    layer.out_features,
                                    bias=layer.bias is not None,
                                    has_fp16_weights=False,
                                    threshold=6.0,
                                )
                                # Copy weights and bias
                                bnb_layer.weight.data = layer.weight.data.clone()
                                if layer.bias is not None:
                                    bnb_layer.bias.data = layer.bias.data.clone()
                                
                                # Move to device immediately
                                bnb_layer = bnb_layer.to(original_device)
                                
                                setattr(module, attr_name, bnb_layer)
                                quantized_count += 1
                                print(f"  ✓ Quantized {name}.{attr_name} (bnb 8-bit)", flush=True)
                
                elif isinstance(module, Mlp):
                    for attr_name in ['fc1', 'fc2']:
                        if hasattr(module, attr_name):
                            layer = getattr(module, attr_name)
                            if isinstance(layer, nn.Linear):
                                # Replace with bitsandbytes Linear8bitLt
                                bnb_layer = bnb.nn.Linear8bitLt(
                                    layer.in_features,
                                    layer.out_features,
                                    bias=layer.bias is not None,
                                    has_fp16_weights=False,
                                    threshold=6.0,
                                )
                                # Copy weights and bias
                                bnb_layer.weight.data = layer.weight.data.clone()
                                if layer.bias is not None:
                                    bnb_layer.bias.data = layer.bias.data.clone()
                                
                                # Move to device immediately
                                bnb_layer = bnb_layer.to(original_device)
                                
                                setattr(module, attr_name, bnb_layer)
                                quantized_count += 1
                                print(f"  ✓ Quantized {name}.{attr_name} (bnb 8-bit)", flush=True)
            
            print(f"\n✓ Selectively quantized {quantized_count} layers with bitsandbytes", flush=True)
            
        else:
            # Full model quantization with bitsandbytes
            print("Strategy: Full model quantization using bitsandbytes", flush=True)
            # Implementation for full model would be similar recursive replacement
            # For now, we focus on hybrid as requested
            print("WARNING: Full model bitsandbytes quantization not fully implemented yet, skipping.", flush=True)

        return model

    else:
        # Fallback to PyTorch CPU Quantization (Original Logic)
        print("Using PyTorch Native CPU Quantization...", flush=True)
        
        if original_device is not None and original_device.type != 'cpu':
            print(f"NOTE: PyTorch quantization requires CPU. Moving model from {original_device} to CPU...", flush=True)
            model = model.cpu()
            print("✓ Model moved to CPU", flush=True)
        
        if attention_only:
            print("Strategy: Hybrid (Attention INT8, CNN FP16/32)", flush=True)
            
            # Get all attention modules
            from climate_learn.models.hub.components.attention import Attention, VariableMapping_Attention
            
            quantized_count = 0
            
            # Traverse the model and quantize attention modules manually
            for name, module in model.named_modules():
                if isinstance(module, (Attention, VariableMapping_Attention)):
                    # Quantize Linear layers within this attention module
                    for attr_name in ['qkv', 'q', 'kv', 'proj']:
                        if hasattr(module, attr_name):
                            layer = getattr(module, attr_name)
                            if isinstance(layer, nn.Linear):
                                quantized_layer = torch.quantization.quantize_dynamic(
                                    layer,
                                    {nn.Linear},
                                    dtype=dtype,
                                    inplace=False
                                )
                                setattr(module, attr_name, quantized_layer)
                                quantized_count += 1
                                print(f"  ✓ Quantized {name}.{attr_name}", flush=True)
            
            if quantized_count == 0:
                print("WARNING: No attention Linear layers found to quantize!", flush=True)
            else:
                print(f"\n✓ Selectively quantized {quantized_count} Linear layers in attention modules", flush=True)
            
        else:
            print("Strategy: Full model quantization", flush=True)
            torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=dtype,
                inplace=True
            )
            print("✓ Full model quantization applied", flush=True)
        
        # Move model back to original device
        if original_device is not None and original_device.type != 'cpu':
            print(f"\nNOTE: Attempting to move quantized model back to {original_device}...", flush=True)
            try:
                model = model.to(original_device)
                print(f"✓ Model moved back to {original_device}", flush=True)
            except Exception as e:
                print(f"⚠ WARNING: Could not move quantized model to {original_device}: {e}", flush=True)
                print("  Quantized model will remain on CPU for inference.", flush=True)
        
        return model


def apply_selective_dynamic_quantization(
    model: nn.Module, 
    attention_only: bool = True
) -> nn.Module:
    """Apply dynamic quantization selectively to attention layers only.
    
    This is a more surgical approach that replaces only attention Linear layers
    with their quantized versions, leaving CNN layers untouched.
    
    Args:
        model: PyTorch model
        attention_only: If True, only quantize attention layers
        
    Returns:
        Modified model with selective quantization
    """
    from climate_learn.models.hub.components.attention import Attention, VariableMapping_Attention
    
    print("\n" + "="*80, flush=True)
    print("APPLYING SELECTIVE DYNAMIC QUANTIZATION", flush=True)
    print("="*80, flush=True)
    print("Strategy: Surgical replacement of attention Linear layers only", flush=True)
    
    quantized_count = 0
    
    # Traverse the model and quantize attention modules
    for name, module in model.named_modules():
        if isinstance(module, (Attention, VariableMapping_Attention)):
            print(f"\nProcessing attention module: {name}", flush=True)
            
            # Quantize Linear layers within this attention module
            for attr_name in ['qkv', 'q', 'kv', 'proj']:
                if hasattr(module, attr_name):
                    layer = getattr(module, attr_name)
                    if isinstance(layer, nn.Linear):
                        # Apply dynamic quantization to this specific layer
                        quantized_layer = nn.quantized.dynamic.Linear.from_float(layer)
                        setattr(module, attr_name, quantized_layer)
                        quantized_count += 1
                        print(f"  ✓ Quantized {attr_name}: {layer.in_features} -> {layer.out_features}", flush=True)
    
    print(f"\n✓ Selectively quantized {quantized_count} Linear layers in attention modules", flush=True)
    print("="*80 + "\n", flush=True)
    
    return model


def print_model_quantization_summary(model: nn.Module):
    """Print summary of which layers are quantized.
    
    Args:
        model: PyTorch model
    """
    print("\n" + "="*80, flush=True)
    print("MODEL QUANTIZATION SUMMARY", flush=True)
    print("="*80, flush=True)
    
    total_params = 0
    quantized_params = 0
    
    layer_info = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            num_params = sum(p.numel() for p in module.parameters())
            total_params += num_params
            layer_info.append((name, "FP32", num_params))
        elif isinstance(module, (nn.quantized.Linear, nn.quantized.dynamic.Linear)):
            # Quantized layers - estimate params (weight is quantized INT8)
            num_params = sum(p.numel() for p in module.parameters() if hasattr(p, 'numel'))
            quantized_params += num_params
            total_params += num_params
            layer_info.append((name, "INT8", num_params))
    
    # Print table
    print(f"\n{'Layer Name':<50} {'Precision':<10} {'Parameters':>15}", flush=True)
    print("-" * 80, flush=True)
    
    for name, precision, params in layer_info[:20]:  # Show first 20
        print(f"{name:<50} {precision:<10} {params:>15,}", flush=True)
    
    if len(layer_info) > 20:
        print(f"... and {len(layer_info) - 20} more layers", flush=True)
    
    print("-" * 80, flush=True)
    print(f"Total parameters: {total_params:,}", flush=True)
    print(f"Quantized parameters (INT8): {quantized_params:,}", flush=True)
    
    if total_params > 0:
        quant_ratio = (quantized_params / total_params) * 100
        print(f"Quantization ratio: {quant_ratio:.2f}%", flush=True)
    
    print("="*80 + "\n", flush=True)
