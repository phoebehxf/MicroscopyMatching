# inference_count.py

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import os
from huggingface_hub import hf_hub_download
from counting import CountingModule

MODEL = None
DEVICE = torch.device("cpu")

def load_model(use_box=False):
    """
    load counting model from Hugging Face Hub
    
    Args:
        use_box: use bounding box as input (default: False)
    
    Returns:
        model: loaded counting model
        device: device
    """
    global MODEL, DEVICE
    
    try:
        print("🔄 Loading counting model...")

        MODEL = CountingModule(use_box=use_box)
        
        ckpt_path = hf_hub_download(
            repo_id="phoebe777777/111",
            filename="microscopy_matching_cnt.pth",
            token=None,
            force_download=False
        )
        
        print(f"✅ Checkpoint downloaded: {ckpt_path}")
        
        MODEL.load_state_dict(
            torch.load(ckpt_path, map_location="cpu"), 
            strict=True
        )
        MODEL.eval()
        
        if torch.cuda.is_available():
            DEVICE = torch.device("cuda")
            MODEL.move_to_device(DEVICE)
            print("✅ Model moved to CUDA")
        else:
            DEVICE = torch.device("cpu")
            MODEL.move_to_device(DEVICE)
            print("✅ Model on CPU")
        
        print("✅ Counting model loaded successfully")
        return MODEL, DEVICE
        
    except Exception as e:
        print(f"❌ Error loading counting model: {e}")
        import traceback
        traceback.print_exc()
        return None, torch.device("cpu")


@torch.no_grad()
def run(model, img_path, box=None, device="cpu", visualize=True):
    """
    Run counting inference on a single image
    
    Args:
        model: loaded counting model
        img_path: image path
        box: bounding box [[x1, y1, x2, y2], ...] or None
        device: device
        visualize: whether to generate visualization
    
    Returns:
        result_dict: {
            'density_map': numpy array,
            'count': float,
            'visualized_path': str (if visualize=True)
        }
    """
    print("DEVICE:", device)
    model.move_to_device(device)
    model.eval()
    if box is not None:
        use_box = True
    else:
        use_box = False
    model.use_box = use_box

    if model is None:
        return {
            'density_map': None,
            'count': 0,
            'visualized_path': None,
            'error': 'Model not loaded'
        }
    
    try:
        print(f"🔄 Running counting inference on {img_path}")
        
        with torch.no_grad():
            density_map, count = model(img_path, box)
        
        print(f"✅ Counting result: {count:.1f} objects")
        
        result = {
            'density_map': density_map,
            'count': count,
            'visualized_path': None
        }

        
        return result
        
    except Exception as e:
        print(f"❌ Counting inference error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'density_map': None,
            'count': 0,
            'visualized_path': None,
            'error': str(e)
        }


def visualize_result(image_path, density_map, count):
    """
    Visualize counting results (consistent with your original visualization code)
    
    Args:
        image_path: original image path
        density_map: numpy array of predicted density map
        count
    
    Returns:
        output_path: temporary file path of the visualization result
    """
    try:
        import skimage.io as io
        
        img = io.imread(image_path)
        
        if len(img.shape) == 3 and img.shape[2] > 3:
            img = img[:, :, :3]
        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)
        
        img_show = img.squeeze()
        density_map_show = density_map.squeeze()
        
        img_show = (img_show - np.min(img_show)) / (np.max(img_show) - np.min(img_show) + 1e-8)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.imshow(img_show)
        ax.imshow(density_map_show, cmap='jet', alpha=0.5)
        ax.axis('off')
        
        plt.tight_layout()
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_file.name, dpi=300)
        plt.close()
        
        print(f"✅ Visualization saved to {temp_file.name}")
        return temp_file.name
        
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        import traceback
        traceback.print_exc()
        return image_path


if __name__ == "__main__":
    print("="*60)
    print("Testing Counting Model")
    print("="*60)

    model, device = load_model(use_box=False)
    
    if model is not None:
        print("\n" + "="*60)
        print("Model loaded successfully, testing inference...")
        print("="*60)
        
        test_image = "example_imgs/1977_Well_F-5_Field_1.png"
        
        if os.path.exists(test_image):
            result = run(
                model,
                test_image,
                box=None,
                device=device,
                visualize=True
            )
            
            if 'error' not in result:
                print("\n" + "="*60)
                print("Inference Results:")
                print("="*60)
                print(f"Count: {result['count']:.1f}")
                print(f"Density map shape: {result['density_map'].shape}")
                if result['visualized_path']:
                    print(f"Visualization saved to: {result['visualized_path']}")
            else:
                print(f"\n❌ Inference failed: {result['error']}")
        else:
            print(f"\n⚠️ Test image not found: {test_image}")
    else:
        print("\n❌ Model loading failed")
