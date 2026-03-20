"""Test counting inference."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from inference_count import load_model, run

model, device = load_model(use_box=False)
img_path = "example_imgs/cnt/047cell.png"
result = run(model, img_path, device=device)

print(f"Count: {result['count']:.1f}")
print(f"Density map shape: {result['density_map'].shape}")

# Visualization — same style as app.py
img = plt.imread(img_path)
if img.dtype == np.uint8:
    img = img.astype(np.float32) / 255.0
if img.ndim == 3 and img.shape[2] == 4:
    img = img[:, :, :3]
if img.ndim == 2:
    img = np.stack([img] * 3, axis=-1)
img = (img - img.min()) / (img.max() - img.min() + 1e-8)

density = result['density_map'].squeeze()
density_norm = (density - density.min()) / (density.max() - density.min() + 1e-8)
density_colored = cm.get_cmap("jet")(density_norm)[:, :, :3]

alpha = 0.5
overlay = img.copy()
threshold = 0.01
significant = density_norm > threshold
overlay[significant] = (1 - alpha) * overlay[significant] + alpha * density_colored[significant]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(img)
axes[0].set_title("Input")
axes[1].imshow(np.clip(overlay, 0, 1))
axes[1].set_title(f"Density map (count: {result['count']:.0f})")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/counting_result.png", dpi=300)
plt.show()
print("Results visualized and saved to outputs/counting_result.png")