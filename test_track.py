"""Test tracking inference."""
import colorsys
import zipfile
import tempfile
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure
from natsort import natsorted
from glob import glob
import tifffile
from inference_track import load_model, run

# Extract test sequence from zip
tmp_dir = tempfile.mkdtemp()
with zipfile.ZipFile("example_imgs/tra/tracking_test_sequence2.zip", "r") as z:
    z.extractall(tmp_dir)

# Find the frame directory (may be nested)
frame_dir = tmp_dir
for root, dirs, files in os.walk(tmp_dir):
    tifs = [f for f in files if f.endswith(".tif")]
    if tifs:
        frame_dir = root
        break

print(f"Frame dir: {frame_dir}")
print(f"Num frames: {len([f for f in os.listdir(frame_dir) if f.endswith('.tif')])}")

model, device = load_model(use_box=False)
result = run(model, frame_dir, device=device, output_dir="tracked_results")

print(f"Masks shape: {result['masks'].shape}")
print(f"Output dir: {result['output_dir']}")


# Visualization — same style as app.py
def get_well_spaced_color(track_id):
    golden_ratio = 0.618033988749895
    hue = (track_id * golden_ratio) % 1.0
    return np.array(colorsys.hsv_to_rgb(hue, 0.9, 0.95))


valid_tif_files = natsorted(glob(os.path.join(frame_dir, "*.tif")))
mask_files = natsorted(glob(os.path.join(result['output_dir'], "mask*.tif")) +
                       glob(os.path.join(result['output_dir'], "res_track*.tif")))
alpha = 0.3

frames = []
num_frames = min(len(valid_tif_files), len(mask_files))
for i in range(num_frames):
    img_np = tifffile.imread(valid_tif_files[i])
    if img_np.dtype == np.uint8:
        img_np = img_np.astype(np.float32) / 255.0
    elif img_np.dtype == np.uint16:
        img_min, img_max = img_np.min(), img_np.max()
        if img_max > img_min:
            img_np = (img_np.astype(np.float32) - img_min) / (img_max - img_min)
        else:
            img_np = img_np.astype(np.float32) / 65535.0
    else:
        img_np = img_np.astype(np.float32)
        img_min, img_max = img_np.min(), img_np.max()
        if img_max > img_min:
            img_np = (img_np - img_min) / (img_max - img_min)
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)

    mask = tifffile.imread(mask_files[i])
    overlay = img_np.copy()

    for track_id in np.unique(mask):
        if track_id == 0:
            continue
        binary_mask = (mask == track_id).astype(np.uint8)
        color = get_well_spaced_color(int(track_id))
        overlay[binary_mask == 1] = (1 - alpha) * overlay[binary_mask == 1] + alpha * color

        contours = measure.find_contours(binary_mask, 0.5)
        for contour in contours:
            contour = contour.astype(np.int32)
            valid_y = np.clip(contour[:, 0], 0, overlay.shape[0] - 1)
            valid_x = np.clip(contour[:, 1], 0, overlay.shape[1] - 1)
            overlay[valid_y, valid_x] = [1.0, 1.0, 0.0]  # Yellow contour

    frames.append(np.clip(overlay * 255, 0, 255).astype(np.uint8))

# Save as GIF
pil_frames = [Image.fromarray(f) for f in frames]
gif_path = "outputs/tracking_result.gif"
pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=200, loop=0)

# Show a few sample frames
n_show = min(6, num_frames)
indices = np.linspace(0, num_frames - 1, n_show, dtype=int)
fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
if n_show == 1:
    axes = [axes]
for ax, idx in zip(axes, indices):
    ax.imshow(frames[idx])
    ax.set_title(f"Frame {idx}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/tracking_result.png", dpi=300)
plt.show()
print(f"Results visualized and saved to {gif_path}")
