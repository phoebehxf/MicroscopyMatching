"""Test segmentation inference."""
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from inference_seg import load_model, run

model, device = load_model(use_box=False)
img_path = "example_imgs/seg/1977_Well_F-5_Field_1.png"
mask = run(model, img_path, device=device)

print(f"Mask shape: {mask.shape}")
print(f"Mask dtype: {mask.dtype}")


# Visualization 
def get_well_spaced_color(track_id):
    golden_ratio = 0.618033988749895
    hue = (track_id * golden_ratio) % 1.0
    return np.array(colorsys.hsv_to_rgb(hue, 0.9, 0.95))


img = plt.imread(img_path)
if img.dtype == np.uint8:
    img = img.astype(np.float32) / 255.0
if img.ndim == 3 and img.shape[2] == 4:
    img = img[:, :, :3]

overlay = img.copy()
alpha = 0.4
for inst_id in np.unique(mask):
    if inst_id == 0:
        continue
    binary_mask = (mask == inst_id).astype(np.uint8)
    color = get_well_spaced_color(inst_id)
    overlay[binary_mask == 1] = (1 - alpha) * overlay[binary_mask == 1] + alpha * color

    contours = measure.find_contours(binary_mask, 0.5)
    for contour in contours:
        contour = contour.astype(np.int32)
        valid_y = np.clip(contour[:, 0], 0, overlay.shape[0] - 1)
        valid_x = np.clip(contour[:, 1], 0, overlay.shape[1] - 1)
        overlay[valid_y, valid_x] = [1.0, 1.0, 0.0]  # Yellow contour

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(img)
axes[0].set_title("Input")
axes[1].imshow(np.clip(overlay, 0, 1))
axes[1].set_title("Segmentation mask")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/segmentation_result.png", dpi=300)
plt.show()
print("Results visualized and saved to outputs/segmentation_result.png")