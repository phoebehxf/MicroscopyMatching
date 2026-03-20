import torch
import numpy as np
from huggingface_hub import hf_hub_download
from segmentation import SegmentationModule  

MODEL = None
DEVICE = torch.device("cpu")

def load_model(use_box=False):
    global MODEL, DEVICE
    MODEL = SegmentationModule(use_box=use_box)

    ckpt_path = hf_hub_download(
        repo_id="phoebe777777/111",
        filename="microscopy_matching_seg.pth",
        token=None,
        force_download=False
    )
    MODEL.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    MODEL.eval()
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        MODEL.move_to_device(DEVICE)
        print("✅ Model moved to CUDA")
    else:
        DEVICE = torch.device("cpu")
        MODEL.move_to_device(DEVICE)
        print("✅ Model on CPU")
    return MODEL, DEVICE


@torch.no_grad()
def run(model, img_path, box=None, device="cpu"):
    print("DEVICE:", device)
    model.move_to_device(device)
    model.eval()
    with torch.no_grad():
        if box is not None:
            use_box = True
        else:
            use_box = False
        model.use_box = use_box
        output = model(img_path, box=box)
    mask = output
    return mask
