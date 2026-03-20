import gradio as gr
from gradio_bbox_annotator import BBoxAnnotator
from PIL import Image
import numpy as np
import torch
import os
import shutil
import time
import json
import uuid
from pathlib import Path
import tempfile
import zipfile
from skimage import measure
from matplotlib import cm
from glob import glob
from natsort import natsorted
from huggingface_hub import HfApi, upload_file
# import spaces

from inference_seg import load_model as load_seg_model, run as run_seg
from inference_count import load_model as load_count_model, run as run_count
from inference_track import load_model as load_track_model, run as run_track

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "phoebe777777/celltool_feedback"


print("===== clearing cache =====")
# cache_path = os.path.expanduser("~/.cache/")
cache_path = os.path.expanduser("~/.cache/huggingface/gradio")
if os.path.exists(cache_path):
    try:
        shutil.rmtree(cache_path)
        # print("✅ Deleted ~/.cache/")
        print("✅ Deleted ~/.cache/huggingface/gradio")
    except:
        pass

SEG_MODEL = None
SEG_DEVICE = torch.device("cpu")

COUNT_MODEL = None
COUNT_DEVICE = torch.device("cpu")

TRACK_MODEL = None
TRACK_DEVICE = torch.device("cpu")

def load_all_models():
    global SEG_MODEL, SEG_DEVICE
    global COUNT_MODEL, COUNT_DEVICE
    global TRACK_MODEL, TRACK_DEVICE
    
    print("\n" + "="*60)
    print("📦 Loading Segmentation Model")
    print("="*60)
    SEG_MODEL, SEG_DEVICE = load_seg_model(use_box=False)
    
    print("\n" + "="*60)
    print("📦 Loading Counting Model")
    print("="*60)
    COUNT_MODEL, COUNT_DEVICE = load_count_model(use_box=False)
    
    print("\n" + "="*60)
    print("📦 Loading Tracking Model")
    print("="*60)
    TRACK_MODEL, TRACK_DEVICE = load_track_model(use_box=False)
    
    print("\n" + "="*60)
    print("✅ All Models Loaded Successfully")
    print("="*60)

load_all_models()

DATASET_DIR = Path("solver_cache")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

def save_feedback_to_hf(query_id, feedback_type, feedback_text=None, img_path=None, bboxes=None):
    """Save feedback to Hugging Face Dataset"""
    
    if not HF_TOKEN:
        print("⚠️ No HF_TOKEN found, using local storage")
        save_feedback(query_id, feedback_type, feedback_text, img_path, bboxes)
        return
    
    feedback_data = {
        "query_id": query_id,
        "feedback_type": feedback_type,
        "feedback_text": feedback_text,
        "image_path": img_path,
        "bboxes": str(bboxes),  # 转为字符串
        "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": time.time()
    }
    
    try:
        api = HfApi()
        
        filename = f"feedback_{query_id}_{int(time.time())}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, indent=2, ensure_ascii=False)
        
        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=f"data/{filename}",
            repo_id=DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN
        )
        
        os.remove(filename)
        
        print(f"✅ Feedback saved to HF Dataset: {DATASET_REPO}")
        
    except Exception as e:
        print(f"⚠️ Failed to save to HF Dataset: {e}")
        save_feedback(query_id, feedback_type, feedback_text, img_path, bboxes)


def save_feedback(query_id, feedback_type, feedback_text=None, img_path=None, bboxes=None):
    """Save feedback to local JSON file"""
    feedback_data = {
        "query_id": query_id,
        "feedback_type": feedback_type,
        "feedback_text": feedback_text,
        "image": img_path,
        "bboxes": bboxes,
        "datetime": time.strftime("%Y%m%d_%H%M%S")
    }
    feedback_file = DATASET_DIR / query_id / "feedback.json"
    feedback_file.parent.mkdir(parents=True, exist_ok=True)
    
    if feedback_file.exists():
        with feedback_file.open("r") as f:
            existing = json.load(f)
            if not isinstance(existing, list):
                existing = [existing]
            existing.append(feedback_data)
            feedback_data = existing
    else:
        feedback_data = [feedback_data]
    
    with feedback_file.open("w") as f:
        json.dump(feedback_data, f, indent=4, ensure_ascii=False)

def parse_first_bbox(bboxes):
    """Parse the first bounding box from the annotation input, supports dict or list format"""
    if not bboxes:
        return None
    b = bboxes[0]
    if isinstance(b, dict):
        x, y = float(b.get("x", 0)), float(b.get("y", 0))
        w, h = float(b.get("width", 0)), float(b.get("height", 0))
        return x, y, x + w, y + h
    if isinstance(b, (list, tuple)) and len(b) >= 4:
        return float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return None

def parse_bboxes(bboxes):
    """Parse all bounding boxes from the annotation input"""
    if not bboxes:
        return None
    
    result = []
    for b in bboxes:
        if isinstance(b, dict):
            x, y = float(b.get("x", 0)), float(b.get("y", 0))
            w, h = float(b.get("width", 0)), float(b.get("height", 0))
            result.append([x, y, x + w, y + h])
        elif isinstance(b, (list, tuple)) and len(b) >= 4:
            result.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
    
    return result

def colorize_mask(mask: np.ndarray, num_colors: int = 512) -> np.ndarray:
    """Convert a 2D mask of instance IDs to a color image for visualization."""
    def hsv_to_rgb(h, s, v):
        i = int(h * 6.0)
        f = h * 6.0 - i
        i = i % 6
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        if i == 0: r, g, b = v, t, p
        elif i == 1: r, g, b = q, v, p
        elif i == 2: r, g, b = p, v, t
        elif i == 3: r, g, b = p, q, v
        elif i == 4: r, g, b = t, p, v
        else: r, g, b = v, p, q
        return int(r * 255), int(g * 255), int(b * 255)

    palette = [(0, 0, 0)]
    for i in range(1, num_colors):
        h = (i % num_colors) / float(num_colors)
        palette.append(hsv_to_rgb(h, 1.0, 0.95))

    palette_arr = np.array(palette, dtype=np.uint8)
    color_idx = mask % num_colors
    return palette_arr[color_idx]


def render_seg_overlay(img_np, inst_mask, overlay_alpha):
    """Render segmentation overlay from cached image/mask."""
    if img_np is None or inst_mask is None:
        return None

    overlay = img_np.copy()
    alpha = float(np.clip(overlay_alpha, 0.0, 1.0))

    for inst_id in np.unique(inst_mask):
        if inst_id == 0:
            continue
        binary_mask = (inst_mask == inst_id).astype(np.uint8)
        color = get_well_spaced_color(inst_id)
        overlay[binary_mask == 1] = (1 - alpha) * overlay[binary_mask == 1] + alpha * color

        contours = measure.find_contours(binary_mask, 0.5)
        for contour in contours:
            contour = contour.astype(np.int32)
            valid_y = np.clip(contour[:, 0], 0, overlay.shape[0] - 1)
            valid_x = np.clip(contour[:, 1], 0, overlay.shape[1] - 1)
            overlay[valid_y, valid_x] = [1.0, 1.0, 0.0]

    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def render_count_overlay(img_np, density_normalized, overlay_alpha):
    """Render counting heatmap overlay from cached image/density."""
    if img_np is None or density_normalized is None:
        return None

    alpha = float(np.clip(overlay_alpha, 0.0, 1.0))
    cmap = cm.get_cmap("jet")
    density_colored = cmap(density_normalized)[:, :, :3]

    overlay = img_np.copy()
    threshold = 0.01
    significant_mask = density_normalized > threshold
    overlay[significant_mask] = (1 - alpha) * overlay[significant_mask] + alpha * density_colored[significant_mask]
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def update_seg_overlay_alpha(overlay_alpha, seg_vis_cache):
    """Live update segmentation visualization without rerunning inference."""
    if not seg_vis_cache:
        return None
    return render_seg_overlay(seg_vis_cache.get("img_np"), seg_vis_cache.get("inst_mask"), overlay_alpha)


def update_count_overlay_alpha(overlay_alpha, count_vis_cache):
    """Live update counting visualization without rerunning inference."""
    if not count_vis_cache:
        return None
    return render_count_overlay(count_vis_cache.get("img_np"), count_vis_cache.get("density_normalized"), overlay_alpha)


def update_tracking_overlay_alpha(overlay_alpha, track_vis_cache):
    """Regenerate tracking visualization at new opacity using cached outputs."""
    if not track_vis_cache:
        return None

    tif_dir = track_vis_cache.get("tif_dir")
    output_dir = track_vis_cache.get("output_dir")
    valid_tif_files = track_vis_cache.get("valid_tif_files")
    if not tif_dir or not output_dir or not valid_tif_files:
        return None

    try:
        return create_tracking_visualization(
            tif_dir=tif_dir,
            output_dir=output_dir,
            valid_tif_files=valid_tif_files,
            overlay_alpha=overlay_alpha
        )
    except Exception as e:
        print(f"⚠️ Failed to update tracking opacity: {e}")
        return None


def cleanup_tracking_cache(track_vis_cache):
    """Delete cached tracking temp directories from the previous run."""
    if not track_vis_cache:
        return
    for key in ["input_temp_dir", "output_dir"]:
        path = track_vis_cache.get(key)
        if path and os.path.isdir(path):
            try:
                shutil.rmtree(path)
            except Exception:
                pass


# @spaces.GPU
def segment_with_choice(use_box_choice, annot_value, overlay_alpha):
    """Segmentation handler - supports bounding box, returns colorized overlay and original mask path"""
    if annot_value is None or len(annot_value) < 1:
        print("❌ No annotation input")
        return None, None, {}

    img_path = annot_value[0]
    bboxes = annot_value[1] if len(annot_value) > 1 else []

    print(f"🖼️ Image path: {img_path}")
    box_array = None
    if use_box_choice == "Yes" and bboxes:
        box = parse_bboxes(bboxes)
        if box:
            box_array = box
            print(f"📦 Using bounding boxes: {box_array}")


    try:
        mask = run_seg(SEG_MODEL, img_path, box=box_array, device=SEG_DEVICE)
        print("📏 mask shape:", mask.shape, "dtype:", mask.dtype)
    except Exception as e:
        print(f"❌ Inference failed: {str(e)}")
        return None, None, {}

    temp_mask_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    mask_img = Image.fromarray(mask.astype(np.uint16))
    mask_img.save(temp_mask_file.name)
    print(f"💾 Original mask saved to: {temp_mask_file.name}")

    try:
        img = Image.open(img_path)
        print("📷 Image mode:", img.mode, "size:", img.size)
    except Exception as e:
        print(f"❌ Failed to open image: {e}")
        return None, None, {}

    try:
        img_rgb = img.convert("RGB").resize(mask.shape[::-1], resample=Image.BILINEAR)
        img_np = np.array(img_rgb, dtype=np.float32)
        if img_np.max() > 1.5:
            img_np = img_np / 255.0
    except Exception as e:
        print(f"❌ Error in image conversion/resizing: {e}")
        return None, None, {}

    mask_np = np.array(mask)
    inst_mask = mask_np.astype(np.int32)
    unique_ids = np.unique(inst_mask)
    num_instances = len(unique_ids[unique_ids != 0])
    if num_instances == 0:
        print("⚠️ No instance found, returning dummy red image")
        return Image.new("RGB", mask.shape[::-1], (255, 0, 0)), None, {}

    overlay_img = render_seg_overlay(img_np, inst_mask, overlay_alpha)
    seg_vis_cache = {"img_np": img_np, "inst_mask": inst_mask}
    return overlay_img, temp_mask_file.name, seg_vis_cache


# @spaces.GPU
def count_cells_handler(use_box_choice, annot_value, overlay_alpha):
    """Counting handler - supports bounding box, returns only density map"""
    if annot_value is None or len(annot_value) < 1:
        return None, None, "⚠️ Please provide an image.", {}
    
    image_path = annot_value[0]
    bboxes = annot_value[1] if len(annot_value) > 1 else []

    print(f"🖼️ Image path: {image_path}")
    box_array = None
    if use_box_choice == "Yes" and bboxes:
        box = parse_bboxes(bboxes)
        if box:
            box_array = box
            print(f"📦 Using bounding boxes: {box_array}")
    
    try:
        print(f"🔢 Counting - Image: {image_path}")
        
        result = run_count(
            COUNT_MODEL,
            image_path,
            box=box_array,
            device=COUNT_DEVICE,
            visualize=True
        )
        
        if 'error' in result:
            return None, None, f"❌ Counting failed: {result['error']}", {}
        
        count = result['count']
        density_map = result['density_map']
        temp_density_file = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
        np.save(temp_density_file.name, density_map)
        print(f"💾 Density map saved to {temp_density_file.name}")
        

        try:
            img = Image.open(image_path)
            print("📷 Image mode:", img.mode, "size:", img.size)
        except Exception as e:
            print(f"❌ Failed to open image: {e}")
            return None, None, f"❌ Failed to open image: {str(e)}", {}

        try:
            img_rgb = img.convert("RGB").resize(density_map.shape[::-1], resample=Image.BILINEAR)
            img_np = np.array(img_rgb, dtype=np.float32)
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            if img_np.max() > 1.5:
                img_np = img_np / 255.0
        except Exception as e:
            print(f"❌ Error in image conversion/resizing: {e}")
            return None, None, f"❌ Error in image conversion/resizing: {str(e)}", {}

        
        density_normalized = density_map.copy()
        if density_normalized.max() > 0:
            density_normalized = (density_normalized - density_normalized.min()) / (density_normalized.max() - density_normalized.min())
        
        overlay_img = render_count_overlay(img_np, density_normalized, overlay_alpha)
        result_text = f"✅ Detected {round(count)} objects"
        if use_box_choice == "Yes" and box_array:
            result_text += f"\n📦 Using bounding box: {box_array}"
        

        print(f"✅ Counting done - Count: {count:.1f}")

        count_vis_cache = {"img_np": img_np, "density_normalized": density_normalized}
        return overlay_img, temp_density_file.name, result_text, count_vis_cache
        
        
    except Exception as e:
        print(f"❌ Counting error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, f"❌ Counting failed: {str(e)}", {}


def find_tif_dir(root_dir):
    """Recursively find the first directory containing .tif files"""
    for dirpath, _, filenames in os.walk(root_dir):
        if '__MACOSX' in dirpath:
            continue
        if any(f.lower().endswith('.tif') for f in filenames):
            return dirpath
    return None

def is_valid_tiff(filepath):
    """Check if a file is a valid TIFF image"""
    try:
        with Image.open(filepath) as img:
            img.verify()
            return True
    except Exception as e:
        return False

def find_valid_tif_dir(root_dir):
    """Recursively find the first directory containing valid .tif files"""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '__MACOSX' in dirpath:
            continue
        
        potential_tifs = [
            os.path.join(dirpath, f) 
            for f in filenames 
            if f.lower().endswith(('.tif', '.tiff')) and not f.startswith('._')
        ]
        
        if not potential_tifs:
            continue
        
        valid_tifs = [f for f in potential_tifs if is_valid_tiff(f)]
        
        if valid_tifs:
            print(f"✅ Found {len(valid_tifs)} valid TIFF files in: {dirpath}")
            return dirpath
    
    return None

def create_ctc_results_zip(output_dir):
    """
    Create a ZIP file with CTC format results
    
    Parameters:
    -----------
    output_dir : str
        Directory containing tracking results (res_track.txt, etc.)
    
    Returns:
    --------
    zip_path : str
        Path to created ZIP file
    """
    # Create temp directory for ZIP
    temp_zip_dir = tempfile.mkdtemp()
    zip_filename = f"tracking_results_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = os.path.join(temp_zip_dir, zip_filename)
    
    print(f"📦 Creating results ZIP: {zip_path}")
    
    # Create ZIP with all tracking results
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all files from output directory
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)
                print(f"  📄 Added: {arcname}")
        
        # Add a README with summary
        readme_content = f"""Tracking Results Summary
            ========================

            Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

            Files:
            ------
            - res_track.txt: CTC format tracking data
            Format: track_id start_frame end_frame parent_id
            
            - Segmentation masks

            For more information on CTC format:
            http://celltrackingchallenge.net/
            """
        zipf.writestr("README.txt", readme_content)
    
    print(f"✅ ZIP created: {zip_path} ({os.path.getsize(zip_path) / 1024:.1f} KB)")
    return zip_path


def get_well_spaced_color(track_id, num_colors=256):
    """Generate well-spaced colors, using contrasting colors for adjacent IDs"""

    golden_ratio = 0.618033988749895
    hue = (track_id * golden_ratio) % 1.0

    import colorsys
    rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
    return np.array(rgb)


def extract_first_frame(tif_dir):
    """
    Extract the first frame from a directory of TIF files
    
    Returns:
    --------
    first_frame_path : str
        Path to the first TIF frame
    """
    tif_files = natsorted(glob(os.path.join(tif_dir, "*.tif")) + 
                      glob(os.path.join(tif_dir, "*.tiff")))
    valid_tif_files = [f for f in tif_files 
                      if not os.path.basename(f).startswith('._') and is_valid_tiff(f)]
    
    if valid_tif_files:
        return valid_tif_files[0]
    return None

def create_tracking_visualization(tif_dir, output_dir, valid_tif_files, overlay_alpha=0.3):
    """
    Create an animated GIF/video showing tracked objects with consistent colors
    
    Parameters:
    -----------
    tif_dir : str
        Directory containing input TIF frames
    output_dir : str
        Directory containing tracking results (masks)
    valid_tif_files : list
        List of valid TIF file paths
    
    Returns:
    --------
    video_path : str
        Path to generated visualization (GIF or first frame)
    """
    import numpy as np
    from matplotlib import colormaps
    from skimage import measure
    import tifffile
    
    # Look for tracking mask files in output directory
    # Common CTC formats: man_track*.tif, mask*.tif, or numbered masks
    mask_files = natsorted(glob(os.path.join(output_dir, "mask*.tif")) + 
                       glob(os.path.join(output_dir, "man_track*.tif")) +
                       glob(os.path.join(output_dir, "*.tif")))
    
    if not mask_files:
        print("⚠️  No mask files found in output directory")
        # Return first frame as fallback
        return valid_tif_files[0]
    
    print(f"📊 Found {len(mask_files)} mask files")

    
    frames = []
    alpha = float(np.clip(overlay_alpha, 0.0, 1.0))  # Transparency for overlay
    
    # Process each frame
    num_frames = min(len(valid_tif_files), len(mask_files))
    for i in range(num_frames):
        try:
            # Load original image using tifffile (handles ZSTD compression)
            try:
                img_np = tifffile.imread(valid_tif_files[i])

                # Normalize to [0, 1] range based on actual data type and values
                if img_np.dtype == np.uint8:
                    img_np = img_np.astype(np.float32) / 255.0
                elif img_np.dtype == np.uint16:
                    # Normalize uint16 to [0, 1] using actual min/max
                    img_min, img_max = img_np.min(), img_np.max()
                    if img_max > img_min:
                        img_np = (img_np.astype(np.float32) - img_min) / (img_max - img_min)
                    else:
                        img_np = img_np.astype(np.float32) / 65535.0
                else:
                    # For float or other types, normalize based on actual range
                    img_np = img_np.astype(np.float32)
                    img_min, img_max = img_np.min(), img_np.max()
                    if img_max > img_min:
                        img_np = (img_np - img_min) / (img_max - img_min)
                    else:
                        img_np = np.clip(img_np, 0, 1)

                # Convert to RGB if grayscale
                if img_np.ndim == 2:
                    img_np = np.stack([img_np]*3, axis=-1)
                img_np = img_np.astype(np.float32)
                if img_np.max() > 1.5:
                    img_np = img_np / 255.0
            except Exception as e:
                print(f"⚠️  Error loading image frame {i}: {e}")
                # Fallback to PIL
                img = Image.open(valid_tif_files[i]).convert("RGB")
                img_np = np.array(img, dtype=np.float32) / 255.0
            
            # Load tracking mask using tifffile (handles ZSTD compression)
            try:
                mask = tifffile.imread(mask_files[i])
            except Exception as e:
                print(f"⚠️  Error loading mask frame {i}: {e}")
                # Fallback to PIL
                mask = np.array(Image.open(mask_files[i]))
            
            # Resize mask to match image if needed
            if mask.shape[:2] != img_np.shape[:2]:
                from scipy.ndimage import zoom
                zoom_factors = [img_np.shape[0] / mask.shape[0], img_np.shape[1] / mask.shape[1]]
                mask = zoom(mask, zoom_factors, order=0).astype(mask.dtype)
            
            # Create overlay
            overlay = img_np.copy()
            
            # Get unique track IDs (excluding background 0)
            track_ids = np.unique(mask)
            track_ids = track_ids[track_ids != 0]
            
            # Color each tracked object
            for track_id in track_ids:
                # Create binary mask for this track
                binary_mask = (mask == track_id)
                
                # Get consistent color for this track ID
                # color = np.array(cmap(int(track_id) % 256)[:3])
                color = get_well_spaced_color(int(track_id))
                
                # Blend color onto image
                overlay[binary_mask] = (1 - alpha) * overlay[binary_mask] + alpha * color
                
                # Draw contours (optional, adds yellow boundaries)
                try:
                    contours = measure.find_contours(binary_mask.astype(np.uint8), 0.5)
                    for contour in contours:
                        contour = contour.astype(np.int32)
                        valid_y = np.clip(contour[:, 0], 0, overlay.shape[0] - 1)
                        valid_x = np.clip(contour[:, 1], 0, overlay.shape[1] - 1)
                        overlay[valid_y, valid_x] = [1.0, 1.0, 0.0]  # Yellow contour
                except:
                    pass  # Skip contours if they fail
            
            # Convert to uint8
            overlay_uint8 = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
            frames.append(Image.fromarray(overlay_uint8))
            
            if i % 10 == 0 or i == num_frames - 1:
                print(f"  📸 Processed frame {i+1}/{num_frames}")
        
        except Exception as e:
            print(f"⚠️  Error processing frame {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not frames:
        print("⚠️  No frames were processed successfully")
        return valid_tif_files[0]
    
    # Save as animated GIF
    try:
        temp_gif = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
        frames[0].save(
            temp_gif.name,
            save_all=True,
            append_images=frames[1:],
            duration=200,  # 200ms per frame = 5fps
            loop=0
        )
        temp_gif.close()  # Close the file handle
        print(f"✅ Created tracking visualization GIF: {temp_gif.name}")
        print(f"   Size: {os.path.getsize(temp_gif.name)} bytes, Frames: {len(frames)}")
        return temp_gif.name
    except Exception as e:
        print(f"⚠️  Failed to create GIF: {e}")
        import traceback
        traceback.print_exc()
        # Return first frame as static image fallback
        try:
            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            frames[0].save(temp_img.name)
            temp_img.close()
            return temp_img.name
        except:
            return valid_tif_files[0]

# @spaces.GPU
def track_video_handler(use_box_choice, first_frame_annot, zip_file_obj, overlay_alpha, prev_track_vis_cache):
    """
    Tracking handler - processes a ZIP of TIF frames, supports bounding box, returns visualization and results ZIP
    
    Parameters:
    -----------
    use_box_choice : str
        "Yes" or "No" - whether to use bounding box annotation for tracking
    first_frame_annot : tuple or None
        (image_path, bboxes) from BBoxAnnotator, only used if user annotated first frame
    zip_file_obj : File
        Uploaded ZIP file containing TIF sequence
    """
    if zip_file_obj is None:
        return None, "⚠️  Please upload a ZIP file containing video frames (.zip)", None, None, {}
    
    cleanup_tracking_cache(prev_track_vis_cache)
    temp_dir = None
    output_temp_dir = None
    
    try:
        # Parse bounding box if provided
        box_array = None
        if use_box_choice == "Yes" and first_frame_annot is not None:
            if isinstance(first_frame_annot, (list, tuple)) and len(first_frame_annot) > 1:
                bboxes = first_frame_annot[1]
                if bboxes:
                    box = parse_bboxes(bboxes)
                    if box:
                        box_array = box
                        print(f"📦 Using bounding boxes: {box_array}")
        
        # Extract input ZIP
        temp_dir = tempfile.mkdtemp()
        print(f"\n📦 Extracting to temporary directory: {temp_dir}")

        with zipfile.ZipFile(zip_file_obj.name, 'r') as zip_ref:
            extracted_count = 0
            skipped_count = 0
            
            for member in zip_ref.namelist():
                basename = os.path.basename(member)
                
                if ('__MACOSX' in member or 
                    basename.startswith('._') or 
                    basename.startswith('.DS_Store') or
                    member.endswith('/')):
                    skipped_count += 1
                    continue
                
                try:
                    zip_ref.extract(member, temp_dir)
                    extracted_count += 1
                    if basename.lower().endswith(('.tif', '.tiff')):
                        print(f"📄 Extracted TIFF: {basename}")
                except Exception as e:
                    print(f"⚠️  Failed to extract {member}: {e}")

            print(f"\n📊 Extracted: {extracted_count} files, Skipped: {skipped_count} files")

        # Find valid TIFF directory
        tif_dir = find_valid_tif_dir(temp_dir)
        
        if tif_dir is None:
            return None, "❌ Did not find valid TIF directory", None, None, {}
        
        # Validate TIFF files
        tif_files = natsorted(glob(os.path.join(tif_dir, "*.tif")) + 
                          glob(os.path.join(tif_dir, "*.tiff")))
        valid_tif_files = [f for f in tif_files 
                          if not os.path.basename(f).startswith('._') and is_valid_tiff(f)]
        
        if len(valid_tif_files) == 0:
            return None, "❌ Did not find valid TIF files", None, None, {}

        print(f"📈 Using {len(valid_tif_files)} TIF files")

        # Store paths for later visualization
        first_frame_path = valid_tif_files[0]

        # Create temporary output directory for CTC results
        output_temp_dir = tempfile.mkdtemp()
        print(f"💾 CTC-format results will be saved to: {output_temp_dir}")

        # Run tracking with optional bounding box
        result = run_track(
            TRACK_MODEL,
            video_dir=tif_dir,
            box=box_array,  # Pass bounding box if specified
            device=TRACK_DEVICE,
            output_dir=output_temp_dir
        )
        
        if 'error' in result:
            return None, f"❌ Tracking failed: {result['error']}", None, None, {}
        
        # Create visualization video of tracked objects
        print("\n🎬 Creating tracking visualization...")
        try:
            tracking_video = create_tracking_visualization(
                tif_dir, 
                output_temp_dir, 
                valid_tif_files,
                overlay_alpha=overlay_alpha
            )
        except Exception as e:
            print(f"⚠️  Failed to create visualization: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to first frame if visualization fails
            try:
                tracking_video = Image.open(first_frame_path)
            except:
                tracking_video = None
        
        # Create downloadable ZIP with results
        try:
            results_zip = create_ctc_results_zip(output_temp_dir)
        except Exception as e:
            print(f"⚠️  Failed to create ZIP: {e}")
            results_zip = None
        
        bbox_info = ""
        if box_array:
            bbox_info = f"\n🔲 Using bounding box: [{box_array[0][0]}, {box_array[0][1]}, {box_array[0][2]}, {box_array[0][3]}]"

        result_text = f"""✅ Tracking completed!

            🖼️  Processed frames: {len(valid_tif_files)}{bbox_info}

            📥 Click the button below to download CTC-format results
            The results include:
            - res_track.txt (CTC-format tracking data)
            - Other tracking-related files
            - README.txt (Results description)
            """

        if use_box_choice == "Yes" and box_array:
            result_text += f"\n📦 Using bounding box: {box_array}"

        print(f"\n✅ Tracking completed")

        track_vis_cache = {
            "tif_dir": tif_dir,
            "valid_tif_files": valid_tif_files,
            "output_dir": output_temp_dir,
            "input_temp_dir": temp_dir,
        }

        return results_zip, result_text, gr.update(visible=True), tracking_video, track_vis_cache

    except zipfile.BadZipFile:
        return None, "❌ Not a valid ZIP file", None, None, {}
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # Clean up on error
        for d in [temp_dir, output_temp_dir]:
            if d:
                try:
                    shutil.rmtree(d)
                except:
                    pass
        return None, f"❌ Tracking failed: {str(e)}", None, None, {}



# ===== Example Images =====
example_images_seg = [f for f in glob("example_imgs/seg/*")]
example_images_cnt = [f for f in glob("example_imgs/cnt/*")]
example_tracking_zips = [f for f in glob("example_imgs/tra/*.zip")]

# ===== Gradio UI =====
CSS = """
/* ── Layout ──────────────────────────────────────────── */
.gradio-container {
    max-width: 1380px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

/* ── Header markdown polish ───────────────────────────── */
.gradio-container .prose h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #1e293b !important;
    letter-spacing: -0.5px !important;
    margin-bottom: 10px !important;
}
.gradio-container .prose h3 {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #0284c7 !important;
    margin-top: 14px !important;
    margin-bottom: 4px !important;
}
.gradio-container .prose p {
    margin-top: 4px !important;
    margin-bottom: 6px !important;
    color: #475569 !important;
    line-height: 1.7 !important;
}
.gradio-container .prose ul,
.gradio-container .prose ol {
    margin-top: 4px !important;
    margin-bottom: 6px !important;
}
.gradio-container .prose li {
    color: #475569 !important;
    line-height: 1.7 !important;
}

/* ── Top-level header section ─────────────────────────── */
.gradio-container > .gap > .prose:first-child {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #f0fdf4 100%) !important;
    border: 1px solid #bae6fd !important;
    border-radius: 16px !important;
    padding: 28px 36px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 4px 20px rgba(14,165,233,0.08) !important;
}

/* ── Tabs ────────────────────────────────────────────── */
.tabs > .tab-nav {
    border-bottom: 2px solid #e2e8f0 !important;
    margin-bottom: 20px !important;
    gap: 4px !important;
}
.tabs button {
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 11px 24px !important;
    border-radius: 8px 8px 0 0 !important;
    color: #64748b !important;
    transition: color 0.15s, background 0.15s !important;
}
.tabs button:hover {
    color: #0ea5e9 !important;
    background: #f0f9ff !important;
}
.tabs button.selected {
    color: #0284c7 !important;
    border-bottom: 3px solid #0284c7 !important;
    background: transparent !important;
}

/* ── Buttons ─────────────────────────────────────────── */
button.primary {
    background: linear-gradient(135deg, #0284c7 0%, #0ea5e9 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    box-shadow: 0 3px 12px rgba(14,165,233,0.35) !important;
    transition: transform 0.12s ease, box-shadow 0.15s ease !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(14,165,233,0.45) !important;
}
button.secondary {
    border-radius: 10px !important;
    font-weight: 500 !important;
    border: 1.5px solid #cbd5e1 !important;
    color: #475569 !important;
    transition: border-color 0.12s, color 0.12s, background 0.12s !important;
}
button.secondary:hover {
    border-color: #94a3b8 !important;
    color: #1e293b !important;
    background: #f8fafc !important;
}

/* ── Blocks and panels ───────────────────────────────── */
.gradio-container .block { border-radius: 14px !important; }
.gradio-container .gr-form,
.gradio-container .gr-box,
.gradio-container .gr-panel {
    border-radius: 14px !important;
    border-color: #e2e8f0 !important;
}

/* ── Labels ──────────────────────────────────────────── */
label { font-weight: 500 !important; color: #374151 !important; }

/* ── Image output ────────────────────────────────────── */
.uniform-height {
    height: 480px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 12px !important;
    background: #f8fafc !important;
}
.uniform-height img, .uniform-height canvas {
    max-height: 480px !important;
    object-fit: contain !important;
}

/* ── Density map output ──────────────────────────────── */
#density_map_output { height: 480px !important; }
#density_map_output .image-container { height: 480px !important; }
#density_map_output img {
    height: 460px !important;
    width: auto !important;
    max-width: 95% !important;
    object-fit: contain !important;
}

/* ── Tab content description markdown ───────────────── */
.tabitem .prose h2 {
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    color: #1e293b !important;
    margin-top: 0 !important;
    margin-bottom: 10px !important;
    padding-bottom: 8px !important;
    border-bottom: 2px solid #e0f2fe !important;
}
.tabitem .prose:nth-child(2) {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 12px 18px !important;
    margin-bottom: 16px !important;
}
.tabitem .prose:nth-child(2) p,
.tabitem .prose:nth-child(2) li {
    font-size: 0.91rem !important;
    color: #64748b !important;
}
.tabitem .prose:nth-child(2) strong {
    color: #0f172a !important;
}

/* ════════════════════════════════════════════════════════
   DARK MODE  (.dark is added to <html> by Gradio)
   ════════════════════════════════════════════════════════ */

/* ── Header text ─────────────────────────────────────── */
.dark .gradio-container .prose h1 {
    color: #e2e8f0 !important;
}
.dark .gradio-container .prose h3 {
    color: #38bdf8 !important;
}
.dark .gradio-container .prose p,
.dark .gradio-container .prose li {
    color: #94a3b8 !important;
}

/* ── Top-level header card ───────────────────────────── */
.dark .gradio-container > .gap > .prose:first-child {
    background: linear-gradient(135deg, #0c1a2e 0%, #0f2942 50%, #0d1f12 100%) !important;
    border-color: #1e3a5f !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
}

/* ── Tabs ────────────────────────────────────────────── */
.dark .tabs > .tab-nav {
    border-bottom-color: #334155 !important;
}
.dark .tabs button {
    color: #94a3b8 !important;
}
.dark .tabs button:hover {
    color: #38bdf8 !important;
    background: rgba(56,189,248,0.08) !important;
}
.dark .tabs button.selected {
    color: #38bdf8 !important;
    border-bottom-color: #38bdf8 !important;
}

/* ── Buttons ─────────────────────────────────────────── */
.dark button.secondary {
    border-color: #475569 !important;
    color: #94a3b8 !important;
    background: transparent !important;
}
.dark button.secondary:hover {
    border-color: #64748b !important;
    color: #e2e8f0 !important;
    background: rgba(255,255,255,0.05) !important;
}

/* ── Blocks / panels ─────────────────────────────────── */
.dark .gradio-container .gr-form,
.dark .gradio-container .gr-box,
.dark .gradio-container .gr-panel {
    border-color: #334155 !important;
}

/* ── Labels ──────────────────────────────────────────── */
.dark label {
    color: #cbd5e1 !important;
}

/* ── Image output area ───────────────────────────────── */
.dark .uniform-height {
    background: #1e293b !important;
}

/* ── Tab content markdown ────────────────────────────── */
.dark .tabitem .prose h2 {
    color: #e2e8f0 !important;
    border-bottom-color: #1e3a5f !important;
}
.dark .tabitem .prose:nth-child(2) {
    background: #1e293b !important;
    border-color: #334155 !important;
}
.dark .tabitem .prose:nth-child(2) p,
.dark .tabitem .prose:nth-child(2) li {
    color: #94a3b8 !important;
}
.dark .tabitem .prose:nth-child(2) strong {
    color: #e2e8f0 !important;
}
"""

with gr.Blocks(
    title="Microscopy Analysis Suite",
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.sky,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CSS,
) as demo:
    gr.Markdown(
        """
        # 🔬 MicroscopyMatching: Microscopy Image Analysis Suite
        
        ### Supporting three key tasks:
        - 🎨 **Segmentation**: Instance segmentation of microscopic objects
        - 🔢 **Counting**: Counting microscopic objects based on density maps
        - 🎬 **Tracking**: Tracking microscopic objects in video sequences

        ### 💡 Technical Details:

        **MicroscopyMatching** - A general-purpose microscopy image analysis toolkit based on pre-trained Latent Diffusion Model
        
        ### 📒 Note:

        This project is currently available with usage limits for research trial use and feedback collection. We plan to release a free public version in the future. We are actively improving the toolkit and greatly appreciate your feedback!

        """
    )
    
    # 全局状态
    current_query_id = gr.State(str(uuid.uuid4()))
    user_uploaded_examples = gr.State(example_images_seg.copy())
    seg_vis_state = gr.State({})
    count_vis_state = gr.State({})
    track_vis_state = gr.State({})
    
    with gr.Tabs():
        # ===== Tab 1: Segmentation =====
        with gr.Tab("🎨 Segmentation"):
            gr.Markdown("## Instance Segmentation of Microscopic Objects")
            gr.Markdown(
                        """
                        **Instructions:**
                        1. Upload an image or select an example image (supports various formats: .png, .jpg, .tif)
                        2. (Optional) Specify a target object with a bounding box and select "Yes", or click "Run Segmentation" directly
                        3. Click "Run Segmentation"
                        4. View the segmentation results (you can adjust the overlay opacity by sliding the opacity bar below the visualization), download the original predicted mask (.tif format); if needed, click "Clear Selection" to choose a new image
                        
                        🤘 Rate and submit feedback to help us improve the model!
                        """
                    )
            
            with gr.Row():
                with gr.Column(scale=1):
                    annotator = BBoxAnnotator(
                        label="🖼️ Upload Image (Optional: Provide a Bounding Box)",
                        categories=["cell"],
                    )
                    
                    # Example Images Gallery
                    example_gallery = gr.Gallery(
                        label="📁 Example Image Gallery",
                        columns=len(example_images_seg),
                        rows=1,
                        height=120,
                        object_fit="cover",
                        show_download_button=False
                    )
                    
                    
                    with gr.Row():
                        use_box_radio = gr.Radio(
                            choices=["Yes", "No"],
                            value="No",
                            label="🔲 Specify Bounding Box?"
                        )
                    with gr.Row():
                        run_seg_btn = gr.Button("▶️ Run Segmentation", variant="primary", size="lg")
                        clear_btn = gr.Button("🔄 Clear Selection", variant="secondary")

                    # Upload Example Image
                    image_uploader = gr.Image(
                        label="➕ Upload New Example Image to Gallery",
                        type="filepath"
                    )


                with gr.Column(scale=2):
                    seg_output = gr.Image(
                        type="pil",
                        label="📸 Segmentation Result",
                        elem_classes="uniform-height"
                    )
                    seg_alpha_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.5,
                        label="🪄 Overlay Opacity"
                    )
                    
                    # Download Original Prediction
                    download_mask_btn = gr.File(
                        label="📥 Download Original Prediction (.tif format)",
                        visible=True,
                        height=40,
                    )

                    # Satisfaction Rating
                    score_slider = gr.Slider(
                        minimum=1,
                        maximum=5,
                        step=1,
                        value=5,
                        label="🌟 Satisfaction Rating (1-5)"
                    )

                    # Feedback Textbox
                    feedback_box = gr.Textbox(
                        placeholder="Please enter your feedback...",
                        lines=2,
                        label="💬 Feedback"
                    )

                    # Submit Button
                    submit_feedback_btn = gr.Button("💾 Submit Feedback", variant="secondary")

                    feedback_status = gr.Textbox(
                        label="✅ Submission Status",
                        lines=1,
                        visible=False
                    )
            
            # click event for segmentation
            run_seg_btn.click(
                fn=segment_with_choice,
                inputs=[use_box_radio, annotator, seg_alpha_slider],
                outputs=[seg_output, download_mask_btn, seg_vis_state]
            )
            seg_alpha_slider.input(
                fn=update_seg_overlay_alpha,
                inputs=[seg_alpha_slider, seg_vis_state],
                outputs=seg_output
            )

            # click event for clear button
            clear_btn.click(
                fn=lambda: (None, {}),
                inputs=None,
                outputs=[annotator, seg_vis_state]
            )
            
            # init Gallery with example images
            demo.load(
                fn=lambda: example_images_seg.copy(),
                outputs=example_gallery
            )
            
            # click event for image uploader
            def add_to_gallery(img_path, current_imgs):
                if not img_path:
                    return current_imgs
                try:
                    if img_path not in current_imgs:
                        current_imgs.append(img_path)
                    return current_imgs
                except:
                    return current_imgs
            
            image_uploader.change(
                fn=add_to_gallery,
                inputs=[image_uploader, user_uploaded_examples],
                outputs=user_uploaded_examples
            ).then(
                fn=lambda imgs: imgs,
                inputs=user_uploaded_examples,
                outputs=example_gallery
            )
            
            # click event for Gallery selection
            def load_from_gallery(evt: gr.SelectData, all_imgs):
                if evt.index is not None and evt.index < len(all_imgs):
                    return all_imgs[evt.index]
                return None
            
            example_gallery.select(
                fn=load_from_gallery,
                inputs=user_uploaded_examples,
                outputs=annotator
            )
            
            # click event for submitting feedback
            def submit_user_feedback(query_id, score, comment, annot_val):
                try:
                    img_path = annot_val[0] if annot_val and len(annot_val) > 0 else None
                    bboxes = annot_val[1] if annot_val and len(annot_val) > 1 else []
                    
                    # save_feedback(
                    #     query_id=query_id,
                    #     feedback_type=f"score_{int(score)}",
                    #     feedback_text=comment,
                    #     img_path=img_path,
                    #     bboxes=bboxes
                    # )

                    save_feedback_to_hf(
                        query_id=query_id,
                        feedback_type=f"score_{int(score)}",
                        feedback_text=comment,
                        img_path=img_path,
                        bboxes=bboxes
                    )
                    return "✅ Feedback submitted, thank you!", gr.update(visible=True)
                except Exception as e:
                    return f"❌ Submission failed: {str(e)}", gr.update(visible=True)

            submit_feedback_btn.click(
                fn=submit_user_feedback,
                inputs=[current_query_id, score_slider, feedback_box, annotator],
                outputs=[feedback_status, feedback_status]
            )
        
        # ===== Tab 2: Counting =====
        with gr.Tab("🔢 Counting"):
            gr.Markdown("## Microscopy Object Counting Analysis")
            gr.Markdown(
                """
                **Usage Instructions:**
                1. Upload an image or select an example image (supports multiple formats: .png, .jpg, .tif)
                2. (Optional) Specify a target object with a bounding box and select "Yes", or click "Run Counting" directly
                3. Click "Run Counting"
                4. View the density map (you can adjust the density opacity by sliding the opacity bar below the visualization), download the original prediction (.npy format); if needed, click "Clear Selection" to choose a new image to run
                
                🤘 Rate and submit feedback to help us improve the model!
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    count_annotator = BBoxAnnotator(
                        label="🖼️ Upload Image (Optional: Provide a Bounding Box)",
                        categories=["cell"],
                    )
                    
                    # Example gallery with "add" functionality
                    with gr.Row():
                        count_example_gallery = gr.Gallery(
                            label="📁 Example Image Gallery",
                            columns=len(example_images_cnt),
                            rows=1,
                            object_fit="cover",
                            height=120,
                            value=example_images_cnt.copy(),  # Initialize with examples
                            show_download_button=False
                        )
                    
                    
                    with gr.Row():
                        count_use_box_radio = gr.Radio(
                            choices=["Yes", "No"],
                            value="No",
                            label="🔲 Specify Bounding Box?"
                        )

                    with gr.Row():
                        count_btn = gr.Button("▶️ Run Counting", variant="primary", size="lg")
                        clear_btn = gr.Button("🔄 Clear Selection", variant="secondary")
                    
                    # Add button to upload new examples
                    with gr.Row():
                        count_image_uploader = gr.File(
                            label="➕ Add Example Image to Gallery",
                            file_types=["image"],
                            type="filepath"
                        )

                
                with gr.Column(scale=2):
                    count_output = gr.Image(
                        label="📸 Density Map",
                        type="filepath",
                        elem_id="density_map_output"
                        
                    )
                    count_alpha_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.3,
                        label="🪄 Density Opacity"
                    )
                    count_status = gr.Textbox(
                        label="📊 Statistics",
                        lines=2
                    )
                    download_density_btn = gr.File(
                        label="📥 Download Original Prediction (.npy format)",
                        visible=True
                    )

                    # Satisfaction rating
                    score_slider = gr.Slider(
                        minimum=1,
                        maximum=5,
                        step=1,
                        value=5,
                        label="🌟 Satisfaction Rating (1-5)"
                    )

                    # Feedback textbox
                    feedback_box = gr.Textbox(
                        placeholder="Please enter your feedback...",
                        lines=2,
                        label="💬 Feedback"
                    )

                    # Submit button
                    submit_feedback_btn = gr.Button("💾 Submit Feedback", variant="secondary")

                    feedback_status = gr.Textbox(
                        label="✅ Submission Status",
                        lines=1,
                        visible=False
                    )
            
            # State for managing gallery images
            count_user_examples = gr.State(example_images_cnt.copy())
            
            # Function to add image to gallery
            def add_to_count_gallery(new_img_file, current_imgs):
                """Add uploaded image to gallery"""
                if new_img_file is None:
                    return current_imgs, current_imgs
                
                try:
                    # Add new image path to list
                    if new_img_file not in current_imgs:
                        current_imgs.append(new_img_file)
                        print(f"✅ Added image to gallery: {new_img_file}")
                except Exception as e:
                    print(f"⚠️  Failed to add image: {e}")
                
                return current_imgs, current_imgs
            
            # When user uploads a new image file
            count_image_uploader.upload(
                fn=add_to_count_gallery,
                inputs=[count_image_uploader, count_user_examples],
                outputs=[count_user_examples, count_example_gallery]
            )
            
            # When user selects from gallery, load into annotator
            def load_from_count_gallery(evt: gr.SelectData, all_imgs):
                """Load selected image from gallery into annotator"""
                if evt.index is not None and evt.index < len(all_imgs):
                    selected_img = all_imgs[evt.index]
                    print(f"📸 Loading image from gallery: {selected_img}")
                    return selected_img
                return None
            
            count_example_gallery.select(
                fn=load_from_count_gallery,
                inputs=count_user_examples,
                outputs=count_annotator
            )
            
            # Run counting
            count_btn.click(
                fn=count_cells_handler,
                inputs=[count_use_box_radio, count_annotator, count_alpha_slider],
                outputs=[count_output, download_density_btn, count_status, count_vis_state]
            )
            count_alpha_slider.input(
                fn=update_count_overlay_alpha,
                inputs=[count_alpha_slider, count_vis_state],
                outputs=count_output
            )

            # Clear selection
            clear_btn.click(
                fn=lambda: (None, {}),
                inputs=None,
                outputs=[count_annotator, count_vis_state]
            )

            # Submit feedback
            def submit_user_feedback(query_id, score, comment, annot_val):
                try:
                    img_path = annot_val[0] if annot_val and len(annot_val) > 0 else None
                    bboxes = annot_val[1] if annot_val and len(annot_val) > 1 else []
                    
                    # save_feedback(
                    #     query_id=query_id,
                    #     feedback_type=f"score_{int(score)}",
                    #     feedback_text=comment,
                    #     img_path=img_path,
                    #     bboxes=bboxes
                    # )

                    save_feedback_to_hf(
                        query_id=query_id,
                        feedback_type=f"score_{int(score)}",
                        feedback_text=comment,
                        img_path=img_path,
                        bboxes=bboxes
                    )
                    return "✅ Feedback submitted successfully, thank you!", gr.update(visible=True)
                except Exception as e:
                    return f"❌ Submission failed: {str(e)}", gr.update(visible=True)

            submit_feedback_btn.click(
                fn=submit_user_feedback,
                inputs=[current_query_id, score_slider, feedback_box, annotator],
                outputs=[feedback_status, feedback_status]
            )
        
        # ===== Tab 3: Tracking =====
        with gr.Tab("🎬 Tracking"):
            gr.Markdown("## Microscopy Object Video Tracking - Supports ZIP Upload")
            gr.Markdown(
                        """
                        **Instructions:**
                        1. Upload a ZIP file or select from the example library. The ZIP should contain a sequence of TIF images named in chronological order (e.g., t000.tif, t001.tif...)
                        2. (Optional) Specify a target object with a bounding box on the first frame and select "Yes", or click "Run Tracking" directly
                        3. Click "Run Tracking"
                        4. View the tracking results (you can adjust the overlay opacity by sliding the opacity bar below the visualization), download the CTC format results; if needed, click "Clear Selection" to choose a new ZIP file to run
                        
                        🤘 Rate and submit feedback to help us improve the model!

                        """
                    )
            
            with gr.Row():
                with gr.Column(scale=1):
                    track_zip_upload = gr.File(
                        label="📦 Upload Image Sequence in ZIP File",
                        file_types=[".zip"]
                    )

                    # First frame annotation for bounding box
                    track_first_frame_annotator = BBoxAnnotator(
                        label="🖼️ (Optional) First Frame Bounding Box Annotation",
                        categories=["cell"],
                        visible=False,  # Hidden initially
                    )

                    # Example ZIP gallery
                    track_example_gallery = gr.Gallery(
                        label="📁 Example Video Gallery (Click to Select)",
                        columns=10,
                        rows=1,
                        height=120,
                        object_fit="contain",
                        show_download_button=False
                    )
                    
                    with gr.Row():
                        track_use_box_radio = gr.Radio(
                            choices=["Yes", "No"],
                            value="No",
                            label="🔲 Specify Bounding Box?"
                        )

                    with gr.Row():
                        track_btn = gr.Button("▶️ Run Tracking", variant="primary", size="lg")
                        clear_btn = gr.Button("🔄 Clear Selection", variant="secondary")
                    
                    # Add to gallery button
                    track_gallery_upload = gr.File(
                        label="➕ Add ZIP to Example Gallery",
                        file_types=[".zip"],
                        type="filepath"
                    )
                
                with gr.Column(scale=2):
                    track_first_frame_preview = gr.Image(
                        label="📸 Tracking Visualization",
                        type="filepath",
                        # height=400,
                        elem_classes="uniform-height",
                        interactive=False
                    )
                    track_alpha_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.3,
                        label="🪄 Overlay Opacity"
                    )
                    
                    track_output = gr.Textbox(
                        label="📊 Tracking Information",
                        lines=8,
                        interactive=False
                    )
                    
                    track_download = gr.File(
                        label="📥 Download Tracking Results (CTC Format)",
                        visible=False
                    )

                    # Satisfaction rating
                    score_slider = gr.Slider(
                        minimum=1,
                        maximum=5,
                        step=1,
                        value=5,
                        label="🌟 Satisfaction Rating (1-5)"
                    )

                    # Feedback textbox
                    feedback_box = gr.Textbox(
                        placeholder="Please enter your feedback...",
                        lines=2,
                        label="💬 Feedback"
                    )

                    # Submit button
                    submit_feedback_btn = gr.Button("💾 Submit Feedback", variant="secondary")

                    feedback_status = gr.Textbox(
                        label="✅ Submission Status",
                        lines=1,
                        visible=False
                    )
            
            # State for tracking examples
            track_user_examples = gr.State(example_tracking_zips.copy())
            
            # Function to get preview image from ZIP
            def get_zip_preview(zip_path):
                """Extract first frame from ZIP for gallery preview"""
                try:
                    temp_dir = tempfile.mkdtemp()
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        for member in zip_ref.namelist():
                            basename = os.path.basename(member)
                            if ('__MACOSX' not in member and 
                                not basename.startswith('._') and
                                basename.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))):
                                zip_ref.extract(member, temp_dir)
                                extracted_path = os.path.join(temp_dir, member)
                                
                                # Load and normalize for preview
                                import tifffile
                                import numpy as np
                                
                                img_np = tifffile.imread(extracted_path)
                                if img_np.dtype == np.uint16:
                                    img_min, img_max = img_np.min(), img_np.max()
                                    if img_max > img_min:
                                        img_np = ((img_np.astype(np.float32) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                                
                                if img_np.ndim == 2:
                                    img_np = np.stack([img_np]*3, axis=-1)
                                
                                # Save preview
                                preview_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                                Image.fromarray(img_np).save(preview_path.name)
                                return preview_path.name
                except:
                    pass
                return None
            
            # Initialize gallery with previews
            def init_tracking_gallery():
                """Create preview images for ZIP examples"""
                previews = []
                for zip_path in example_tracking_zips:
                    if os.path.exists(zip_path):
                        preview = get_zip_preview(zip_path)
                        if preview:
                            previews.append(preview)
                return previews
            
            # Load gallery on startup
            demo.load(
                fn=init_tracking_gallery,
                outputs=track_example_gallery
            )
            
            # Add ZIP to gallery
            def add_zip_to_gallery(zip_path, current_zips):
                if not zip_path:
                    return current_zips, track_example_gallery
                try:
                    if zip_path not in current_zips:
                        current_zips.append(zip_path)
                        print(f"✅ Added ZIP to gallery: {zip_path}")
                    # Regenerate previews
                    previews = []
                    for zp in current_zips:
                        preview = get_zip_preview(zp)
                        if preview:
                            previews.append(preview)
                    return current_zips, previews
                except Exception as e:
                    print(f"⚠️ Error: {e}")
                    return current_zips, []
            
            track_gallery_upload.upload(
                fn=add_zip_to_gallery,
                inputs=[track_gallery_upload, track_user_examples],
                outputs=[track_user_examples, track_example_gallery]
            )
            
            # Select ZIP from gallery
            def load_zip_from_gallery(evt: gr.SelectData, all_zips):
                if evt.index is not None and evt.index < len(all_zips):
                    selected_zip = all_zips[evt.index]
                    print(f"📁 Selected ZIP from gallery: {selected_zip}")
                    return selected_zip
                return None
            
            track_example_gallery.select(
                fn=load_zip_from_gallery,
                inputs=track_user_examples,
                outputs=track_zip_upload
            )

            # Load first frame when ZIP is uploaded
            def load_first_frame_for_annotation(zip_file_obj):
                '''Load and normalize first frame from ZIP for annotation'''
                if zip_file_obj is None:
                    return None, gr.update(visible=False)
                
                import tifffile
                import numpy as np
                
                try:
                    temp_dir = tempfile.mkdtemp()
                    with zipfile.ZipFile(zip_file_obj.name, 'r') as zip_ref:
                        for member in zip_ref.namelist():
                            basename = os.path.basename(member)
                            if ('__MACOSX' not in member and 
                                not basename.startswith('._') and
                                basename.lower().endswith(('.tif', '.tiff'))):
                                zip_ref.extract(member, temp_dir)
                    
                    tif_dir = find_valid_tif_dir(temp_dir)
                    if tif_dir:
                        first_frame = extract_first_frame(tif_dir)
                        if first_frame:
                            # Load and normalize the first frame
                            try:
                                img_np = tifffile.imread(first_frame)
                                
                                # Normalize to [0, 255] uint8 range for display
                                if img_np.dtype == np.uint8:
                                    pass  # Already uint8
                                elif img_np.dtype == np.uint16:
                                    # Normalize uint16 using actual min/max
                                    img_min, img_max = img_np.min(), img_np.max()
                                    if img_max > img_min:
                                        img_np = ((img_np.astype(np.float32) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                                    else:
                                        img_np = (img_np.astype(np.float32) / 65535.0 * 255).astype(np.uint8)
                                else:
                                    # Float or other types
                                    img_np = img_np.astype(np.float32)
                                    img_min, img_max = img_np.min(), img_np.max()
                                    if img_max > img_min:
                                        img_np = ((img_np - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                                    else:
                                        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                                
                                # Convert to RGB if grayscale
                                if img_np.ndim == 2:
                                    img_np = np.stack([img_np]*3, axis=-1)
                                elif img_np.ndim == 3 and img_np.shape[2] > 3:
                                    img_np = img_np[:, :, :3]
                                
                                # Save normalized image to temp file
                                temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                                Image.fromarray(img_np).save(temp_img.name)
                                
                                print(f"✅ Loaded and normalized first frame: {first_frame}")
                                print(f"   Original dtype: {tifffile.imread(first_frame).dtype}")
                                print(f"   Normalized to uint8 RGB for annotation")
                                
                                return temp_img.name, gr.update(visible=True)
                            except Exception as e:
                                print(f"⚠️  Error normalizing first frame: {e}")
                                import traceback
                                traceback.print_exc()
                                # Fallback to original file
                                return first_frame, gr.update(visible=True)
                except Exception as e:
                    print(f"⚠️  Error loading first frame: {e}")
                    import traceback
                    traceback.print_exc()
                return None, gr.update(visible=False)
            
            # Load first frame when ZIP is uploaded
            track_zip_upload.change(
                fn=load_first_frame_for_annotation,
                inputs=track_zip_upload,
                outputs=[track_first_frame_annotator, track_first_frame_annotator]
            )
            
            # Run tracking
            track_btn.click(
                fn=track_video_handler,
                inputs=[track_use_box_radio, track_first_frame_annotator, track_zip_upload, track_alpha_slider, track_vis_state],
                outputs=[track_download, track_output, track_download, track_first_frame_preview, track_vis_state]
            )
            track_alpha_slider.change(
                fn=update_tracking_overlay_alpha,
                inputs=[track_alpha_slider, track_vis_state],
                outputs=track_first_frame_preview
            )

            # Clear selection
            clear_btn.click(
                fn=lambda: (None, {}),
                inputs=None,
                outputs=[track_first_frame_annotator, track_vis_state]
            )

            # Submit feedback
            def submit_user_feedback(query_id, score, comment, annot_val):
                try:
                    img_path = annot_val[0] if annot_val and len(annot_val) > 0 else None
                    bboxes = annot_val[1] if annot_val and len(annot_val) > 1 else []
                    
                    # save_feedback(
                    #     query_id=query_id,
                    #     feedback_type=f"score_{int(score)}",
                    #     feedback_text=comment,
                    #     img_path=img_path,
                    #     bboxes=bboxes
                    # )

                    save_feedback_to_hf(
                        query_id=query_id,
                        feedback_type=f"score_{int(score)}",
                        feedback_text=comment,
                        img_path=img_path,
                        bboxes=bboxes
                    )
                    return "✅ Feedback submitted successfully, thank you!", gr.update(visible=True)
                except Exception as e:
                    return f"❌ Submission failed: {str(e)}", gr.update(visible=True)

            submit_feedback_btn.click(
                fn=submit_user_feedback,
                inputs=[current_query_id, score_slider, feedback_box, annotator],
                outputs=[feedback_status, feedback_status]
            )
    
    

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        ssr_mode=False,
        show_error=True,
    )
