# inference_track.py

import torch
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from tracking_one import TrackingModule
from models.tra_post_model.tracking import graph_to_ctc

MODEL = None
DEVICE = torch.device("cpu")

def load_model(use_box=False):
    """
    load tracking model from Hugging Face Hub
    
    Args:
        use_box: use bounding box as input (default: False)
    
    Returns:
        model: loaded tracking model
        device
    """
    global MODEL, DEVICE
    
    try:
        print("🔄 Loading tracking model...")
        
        # 初始化模型
        MODEL = TrackingModule(use_box=use_box)
        
        # Load checkpoint from Hugging Face Hub
        ckpt_path = hf_hub_download(
            repo_id="phoebe777777/111",
            filename="microscopy_matching_tra.pth",
            token=None,
            force_download=False
        )
        
        print(f"✅ Checkpoint downloaded: {ckpt_path}")
        
        # Load weights
        MODEL.load_state_dict(
            torch.load(ckpt_path, map_location="cpu"), 
            strict=True
        )
        MODEL.eval()
        
        # Move model to device
        if torch.cuda.is_available():
            DEVICE = torch.device("cuda")
            MODEL.move_to_device(DEVICE)
            print("✅ Model moved to CUDA")
        else:
            DEVICE = torch.device("cpu")
            MODEL.move_to_device(DEVICE)
            print("✅ Model on CPU")
        
        print("✅ Tracking model loaded successfully")
        return MODEL, DEVICE
        
    except Exception as e:
        print(f"❌ Error loading tracking model: {e}")
        import traceback
        traceback.print_exc()
        return None, torch.device("cpu")


@torch.no_grad()
def run(model, video_dir, box=None, device="cpu", output_dir="tracked_results"):
    """
    run tracking inference on video frames
    
    Args:
        model: loaded tracking model
        video_dir: directory of video frame sequence (contains consecutive image files)
        box: bounding box (optional)
        device: device
        output_dir: output directory
    
    Returns:
        result_dict: {
            'track_graph': TrackGraph object containing tracking results,
            'masks': tracked masks (T, H, W),
            'output_dir': output directory path,
            'num_tracks': number of tracked trajectories
        }
    """
    if model is None:
        return {
            'track_graph': None,
            'masks': None,
            'output_dir': None,
            'num_tracks': 0,
            'error': 'Model not loaded'
        }
    
    try:
        print(f"🔄 Running tracking inference on {video_dir}")
        
        # Run tracking
        track_graph, masks = model.track(
            file_dir=video_dir,
            boxes=box,
            mode="greedy",  # Optional: "greedy", "greedy_nodiv", "ilp"
            dataname="tracking_result"
        )
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Convert tracking results to CTC format and save
        print("🔄 Converting to CTC format...")
        ctc_tracks, masks_tracked = graph_to_ctc(
            track_graph,
            masks,
            outdir=output_dir,
        )
        print(f"✅ CTC results saved to {output_dir}")
        
        
        print(f"✅ Tracking completed")
        
        result = {
            'track_graph': track_graph,
            'masks': masks,
            'masks_tracked': masks_tracked,
            'output_dir': output_dir,
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Tracking inference error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'track_graph': None,
            'masks': None,
            'output_dir': None,
            'num_tracks': 0,
            'error': str(e)
        }


def visualize_tracking_result(masks_tracked, output_path):
    """
    visualize tracking results
    
    Args:
        masks_tracked: masks with tracking results (T, H, W)
        output_path: output video file path
    
    Returns:
        output_path: output video file path
    """
    try:
        import cv2
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        T, H, W = masks_tracked.shape
        
        # create a color map for unique track IDs
        unique_ids = np.unique(masks_tracked)
        num_colors = len(unique_ids)
        cmap = cm.get_cmap('tab20', num_colors)
        
        # create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 5.0, (W, H))
        
        for t in range(T):
            frame = masks_tracked[t]
            
            # create colored image
            colored_frame = np.zeros((H, W, 3), dtype=np.uint8)
            for i, obj_id in enumerate(unique_ids):
                if obj_id == 0:
                    continue
                mask = (frame == obj_id)
                color = np.array(cmap(i % num_colors)[:3]) * 255
                colored_frame[mask] = color
            
            # convert to BGR (OpenCV format)
            colored_frame_bgr = cv2.cvtColor(colored_frame, cv2.COLOR_RGB2BGR)
            out.write(colored_frame_bgr)
        
        out.release()
        print(f"✅ Visualization saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return None
