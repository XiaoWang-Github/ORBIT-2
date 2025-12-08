#!/usr/bin/env python
"""Convert QAT checkpoints to INT8 quantized models - Simple Version

This script marks QAT checkpoints as INT8-ready for deployment.
The actual INT8 conversion happens during model loading in inference.

NOTE: This script is now OPTIONAL for newly trained checkpoints.
- New checkpoints (trained with updated code) already include quantization metadata
- visualize.py will automatically detect and convert QAT checkpoints
- This script is only needed for legacy checkpoints that lack metadata

Usage:
    python convert_to_int8_simple.py
    
    Optional:
    python convert_to_int8_simple.py --checkpoint_dir checkpoints/climate --epoch 5
"""

import os
import sys
import argparse
import glob
import torch
from pathlib import Path


def get_checkpoint_filename(cp_save_path, epoch, rank=0, tensor_par_size=1):
    """Get checkpoint filename matching training format."""
    # Actual format: interm_epoch_48.ckpt
    return os.path.join(cp_save_path, f"interm_epoch_{epoch}.ckpt")


def find_checkpoint_epochs(checkpoint_dir, tensor_par_size=1):
    """Find all available checkpoint epochs."""
    pattern = os.path.join(checkpoint_dir, "interm_epoch_*.ckpt")
    
    checkpoint_files = glob.glob(pattern)
    epochs = []
    
    for file_path in checkpoint_files:
        basename = os.path.basename(file_path)
        # Extract: interm_epoch_48.ckpt -> 48
        epoch_str = basename.replace('interm_epoch_', '').replace('.ckpt', '')
        
        try:
            epoch = int(epoch_str)
            epochs.append(epoch)
        except ValueError:
            continue
    
    return sorted(epochs)


def mark_checkpoint_as_int8(checkpoint_path, output_path):
    """Mark checkpoint as INT8-ready.
    
    This creates a copy with INT8 metadata.
    Actual conversion happens at inference time using convert_qat_to_quantized().
    
    Args:
        checkpoint_path: Path to QAT checkpoint
        output_path: Path to save INT8-marked checkpoint
        
    Returns:
        True if successful
    """
    try:
        print(f"Processing: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        epoch = checkpoint['epoch']
        
        # Add INT8 metadata
        checkpoint['quantization'] = {
            'enabled': True,
            'precision': 'int8',
            'method': 'qat',
            'note': 'Use qat_utils.convert_qat_to_quantized() before inference',
        }
        
        # Save marked checkpoint
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(checkpoint, output_path)
        
        print(f"  ✓ Marked epoch {epoch} as INT8-ready")
        print(f"  → {output_path}\n")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(description='Mark QAT checkpoints as INT8-ready')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/climate',
                        help='Directory containing QAT checkpoints')
    parser.add_argument('--output_dir', type=str, default='checkpoints/climate_int8',
                        help='Directory to save INT8-marked checkpoints')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Mark specific epoch only (default: all)')
    parser.add_argument('--tensor_par_size', type=int, default=1,
                        help='Tensor parallelism size')
    
    args = parser.parse_args()
    
    print("="*80)
    print("QAT Checkpoint → INT8 Marker")
    print("="*80)
    print(f"Source: {args.checkpoint_dir}")
    print(f"Target: {args.output_dir}")
    print("="*80 + "\n")
    
    # Find epochs to process
    if args.epoch is not None:
        epochs = [args.epoch]
    else:
        epochs = find_checkpoint_epochs(args.checkpoint_dir, args.tensor_par_size)
        print(f"Found {len(epochs)} checkpoints: {epochs}\n")
    
    if not epochs:
        print("No checkpoints found!")
        return
    
    # Process each epoch
    success = 0
    for epoch in epochs:
        checkpoint_path = get_checkpoint_filename(
            args.checkpoint_dir, epoch, 0, args.tensor_par_size
        )
        
        if not os.path.exists(checkpoint_path):
            print(f"Not found: {checkpoint_path}")
            continue
        
        output_path = os.path.join(args.output_dir, f"interm_epoch_{epoch}_int8.ckpt")
        
        if os.path.exists(output_path):
            print(f"Epoch {epoch}: Already marked, skipping\n")
            continue
        
        if mark_checkpoint_as_int8(checkpoint_path, output_path):
            success += 1
    
    # Summary
    print("="*80)
    print(f"Processed: {success}/{len(epochs)} checkpoints")
    print(f"Output: {args.output_dir}/")
    print("="*80)
    
    if success > 0:
        print("\n✓ Checkpoints marked as INT8-ready!")
        print("\nNext steps:")
        print("1. Use these checkpoints in visualize.py")
        print("2. Load with: model = torch.load(...)")
        print("3. Convert with: model_int8 = qat_utils.convert_qat_to_quantized(model)")
        print("4. Run inference!")


if __name__ == "__main__":
    main()
