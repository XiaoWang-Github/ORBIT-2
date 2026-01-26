"""QAT (Quantization-Aware Training) utilities for ORBIT-2 models.

This module provides functions for training models with quantization awareness,
allowing the model to adapt to INT8 representation during training for better
accuracy compared to Post-Training Quantization (PTQ).

Key Concepts:
- Fake Quantization: Simulates INT8 quantization during forward pass while
  keeping gradients in FP32 for backward pass
- Selective QAT: Only quantize Attention and MLP layers, keep CNN in FP16/32
- Two-Phase Training: Normal training → QAT fine-tuning
- Uses torch.ao.quantization (torch.quantization): PyTorch's native quantization API
- NOT using bitsandbytes: bitsandbytes is for PTQ only (see quantization_utils.py)

Workflow:
1. Training: prepare_model_for_qat() → train with FakeQuantize → save checkpoint with metadata
2. Inference: load checkpoint → detect metadata → convert_qat_to_quantized() → INT8 model

Designed for AMD MI250X GPU with ROCm 6.4 environment.
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Optional


def prepare_model_for_qat(
    model: nn.Module,
    qconfig_spec: Optional[quant.QConfig] = None,
    attention_only: bool = True,
    tensor_par_size: int = 1,
) -> nn.Module:
    """Prepare model for Quantization-Aware Training.
    
    This function adds FakeQuantize modules to specified layers, enabling
    the model to learn INT8-friendly weights during training.
    
    Args:
        model: Res_Slim_ViT model to prepare for QAT
        qconfig_spec: Custom quantization config (default: fbgemm for x86)
        attention_only: If True, only quantize Attention + MLP (hybrid strategy)
        tensor_par_size: Size of tensor parallelism (default: 1). When > 1, Linear layers
                        are split across tensor parallel ranks, which is handled automatically.
        
    Returns:
        Model prepared for QAT with FakeQuantize modules inserted
        
    Example:
        >>> model = load_model(...)
        >>> model_qat = prepare_model_for_qat(model, attention_only=True)
        >>> train(model_qat)  # Train with fake quantization
        >>> model_int8 = convert_qat_to_quantized(model_qat)  # Convert to INT8
    """
    print("\n" + "="*80)
    print("PREPARING MODEL FOR QAT (Quantization-Aware Training)")
    print("="*80)
    
    if tensor_par_size > 1:
        print(f"NOTE: Tensor parallelism enabled (size={tensor_par_size}). "
              f"QAT will be applied to split Linear layers.", flush=True)
    
    try:
        from climate_learn.models.hub.components.attention import (
            Attention, 
            VariableMapping_Attention
        )
        from climate_learn.models.hub.components.vit_blocks import Mlp
    except ImportError as e:
        print(f"ERROR: Could not import model components: {e}")
        raise
    
    # Use default fbgemm qconfig if not specified
    if qconfig_spec is None:
        qconfig_spec = quant.get_default_qat_qconfig('fbgemm')
        print("Using default fbgemm QAT config")
    
    # Disable quantization for entire model by default
    model.qconfig = None
    
    if attention_only:
        print("Strategy: Hybrid QAT (Attention + MLP INT8, CNN FP16/32)")
        quantized_count = 0
        
        # Enable qconfig only for Attention and MLP modules
        for name, module in model.named_modules():
            if isinstance(module, (Attention, VariableMapping_Attention)):
                module.qconfig = qconfig_spec
                quantized_count += 1
                print(f"  ✓ Enabled QAT for Attention: {name}")
            elif isinstance(module, Mlp):
                module.qconfig = qconfig_spec
                quantized_count += 1
                print(f"  ✓ Enabled QAT for MLP: {name}")
        
        print(f"\n✓ Enabled QAT for {quantized_count} modules")
    else:
        print("Strategy: Full model QAT")
        model.qconfig = qconfig_spec
        print("  ✓ Enabled QAT for all modules")
    
    # Prepare model - this inserts FakeQuantize modules
    print("\nInserting FakeQuantize modules...")
    # Use inplace=True to avoid deepcopy issues with FSDP-wrapped models
    quant.prepare_qat(model, inplace=True)
    
    print("✓ Model prepared for QAT")
    print("=" * 80 + "\n")
    
    return model


def convert_qat_to_quantized(model: nn.Module) -> nn.Module:
    """Convert QAT-trained model to fully quantized INT8 model.
    
    This removes FakeQuantize modules and replaces layers with true
    INT8 quantized versions for efficient inference.
    
    Args:
        model: QAT-trained model with FakeQuantize modules
        
    Returns:
        Fully quantized INT8 model ready for inference
        
    Note:
        Model must be in eval mode before calling this function.
    """
    print("\n" + "="*80)
    print("CONVERTING QAT MODEL TO INT8")
    print("="*80)
    
    if model.training:
        print("WARNING: Model is in training mode. Setting to eval mode...")
        model.eval()
    
    print("Removing FakeQuantize modules and applying true quantization...")
    # Use inplace=True to avoid deepcopying FSDP ProcessGroup objects which are not picklable
    model_quantized = quant.convert(model, inplace=True)
    
    print("✓ Model converted to INT8")
    print("=" * 80 + "\n")
    
    return model_quantized


def enable_qat_mode(model: nn.Module, enable: bool = True) -> None:
    """Enable or disable QAT mode for a prepared model.
    
    This is useful for switching between normal training and QAT training
    in a two-phase training setup.
    
    Args:
        model: QAT-prepared model
        enable: If True, enable fake quantization. If False, disable it.
        
    Example:
        >>> # Phase 1: Normal training (epochs 0-50)
        >>> enable_qat_mode(model, enable=False)
        >>> train(model, epochs=50)
        >>> 
        >>> # Phase 2: QAT fine-tuning (epochs 50-60)
        >>> enable_qat_mode(model, enable=True)
        >>> train(model, epochs=10, lr=lr*0.1)
    """
    if enable:
        print("Enabling QAT mode (fake quantization active)...")
        model.apply(quant.enable_fake_quant)
        model.apply(quant.enable_observer)
    else:
        print("Disabling QAT mode (fake quantization inactive)...")
        model.apply(quant.disable_fake_quant)
        model.apply(quant.disable_observer)


def check_qat_status(model: nn.Module) -> dict:
    """Check QAT status of a model.
    
    Args:
        model: Model to check
        
    Returns:
        Dictionary with QAT status information
    """
    status = {
        "has_fake_quant": False,
        "num_fake_quant_modules": 0,
        "qconfig_set": False,
    }
    
    # Check for FakeQuantize modules
    for module in model.modules():
        if isinstance(module, quant.FakeQuantize):
            status["has_fake_quant"] = True
            status["num_fake_quant_modules"] += 1
    
    # Check if qconfig is set
    if hasattr(model, 'qconfig') and model.qconfig is not None:
        status["qconfig_set"] = True
    
    return status
