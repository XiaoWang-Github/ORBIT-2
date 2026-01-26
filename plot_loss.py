import matplotlib.pyplot as plt
import numpy as np

# Loss data extracted from log
epochs = list(range(49))
losses = [
    8463.1328125, 415.24554443359375, 218.28842163085938, 169.69732666015625,
    157.38116455078125, 149.36325073242188, 147.26235961914062, 145.03553771972656,
    143.53717041015625, 140.93492126464844, 140.68014526367188, 139.1283416748047,
    139.70755004882812, 138.08872985839844, 135.9098358154297, 136.95352172851562,
    137.9393310546875, 136.09185791015625, 133.9751739501953, 133.4627227783203,
    134.03823852539062, 136.19900512695312, 133.33226013183594, 133.437255859375,
    133.60153198242188, 132.98565673828125, 133.35354614257812, 132.67649841308594,
    131.62477111816406, 131.53074645996094, 132.43348693847656, 131.99090576171875,
    132.0423126220703, 133.62713623046875, 132.24771118164062, 133.2369842529297,
    130.29208374023438, 132.71336364746094, 131.92689514160156, 132.43502807617188,
    132.28248596191406, 132.2678985595703, 134.0597686767578, 132.0416259765625,
    130.91258239746094, 132.1168670654297, 132.91207885742188, 130.97398376464844,
    131.45223999023438
]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Full loss curve
ax1.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
ax1.axvline(x=5, color='r', linestyle='--', linewidth=2, label='QAT Activation (Epoch 5)')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('QAT Training Loss - Full Range', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_xlim(0, 48)

# Plot 2: Zoomed in (epoch 1-48, excluding the huge initial loss)
ax2.plot(epochs[1:], losses[1:], 'b-', linewidth=2, label='Training Loss')
ax2.axvline(x=5, color='r', linestyle='--', linewidth=2, label='QAT Activation (Epoch 5)')
ax2.fill_betweenx([min(losses[1:]), max(losses[1:])], 0, 5, alpha=0.2, color='green', label='Normal Training')
ax2.fill_betweenx([min(losses[1:]), max(losses[1:])], 5, 48, alpha=0.2, color='orange', label='QAT Fine-tuning')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('QAT Training Loss - Zoomed (Epochs 1-48)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_xlim(1, 48)

# Add statistics
pre_qat_loss = np.mean(losses[1:5])
post_qat_loss = np.mean(losses[5:])
final_loss = losses[-1]

stats_text = f'Pre-QAT Avg (Epoch 1-4): {pre_qat_loss:.2f}\n'
stats_text += f'Post-QAT Avg (Epoch 5-48): {post_qat_loss:.2f}\n'
stats_text += f'Final Loss (Epoch 48): {final_loss:.2f}'

ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         fontsize=10)

plt.tight_layout()
plt.savefig('qat_training_loss.png', dpi=150, bbox_inches='tight')
print("Plot saved to: qat_training_loss.png")
plt.close()

# Print summary
print("\n" + "="*60)
print("QAT Training Loss Summary")
print("="*60)
print(f"Total Epochs: {len(epochs)}")
print(f"Initial Loss (Epoch 0): {losses[0]:.2f}")
print(f"Pre-QAT Average (Epoch 1-4): {pre_qat_loss:.2f}")
print(f"Post-QAT Average (Epoch 5-48): {post_qat_loss:.2f}")
print(f"Final Loss (Epoch 48): {final_loss:.2f}")
print(f"Loss Reduction: {losses[0]:.2f} → {final_loss:.2f} ({(1-final_loss/losses[0])*100:.1f}% reduction)")
print("="*60)
