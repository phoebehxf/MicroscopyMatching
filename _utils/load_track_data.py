import os
from glob import glob
from pathlib import Path
from natsort import natsorted
from PIL import Image
import numpy as np
import tifffile
import skimage.io as io
import torchvision.transforms as T
import cv2
from tqdm import tqdm
from models.tra_post_model.utils import normalize_01, normalize
IMG_SIZE = 512

def _load_tiffs(folder: Path, dtype=None):
    """Load a sequence of tiff files from a folder into a 3D numpy array."""
    images = glob(str(folder / "*.tif"))
    test_data = tifffile.imread(images[0])
    if len(test_data.shape) == 3:
        turn_gray = True
    else:
        turn_gray = False
    end_frame = len(images)
    if not turn_gray:
        x = np.stack([
            tifffile.imread(f).astype(dtype)
            for f in tqdm(
                sorted(folder.glob("*.tif"))[0 : end_frame : 1],
                leave=False,
                desc=f"Loading [0:{end_frame}]",
            )
        ])
    else:
        x = []
        for f in tqdm(
            sorted(folder.glob("*.tif"))[0 : end_frame : 1],
            leave=False,
            desc=f"Loading [0:{end_frame}]",
        ):
            img = tifffile.imread(f).astype(dtype)
            if img.ndim == 3:
                if img.shape[-1] > 3:
                    img = img[..., :3]
                img = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])
            x.append(img)
        x = np.stack(x)
    return x


def load_track_images(file_dir):
    
    def find_tif_dir(root_dir):
        tif_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            if '__MACOSX' in dirpath:
                continue
            for f in filenames:
                if f.lower().endswith('.tif'):
                    tif_files.append(os.path.join(dirpath, f))
        return tif_files

    tif_dir = find_tif_dir(file_dir)
    print(f"Found {len(tif_dir)} tif images in {file_dir}")
    print(f"First 5 tif images: {tif_dir[:5]}")
    assert len(tif_dir) > 0, f"No tif images found in {file_dir}"
    images = natsorted(tif_dir)
    imgs = []
    imgs_raw = []
    images_stable = []
    # load images for seg and track
    for img_path in tqdm(images, desc="Loading images"):
        img = tifffile.imread(img_path)
        img_raw = io.imread(img_path)
    
        if img.dtype == 'uint16':
            img = ((img - img.min()) / (img.max() - img.min() + 1e-6) * 255).astype(np.uint8)
            img = np.stack([img] * 3, axis=-1)
            w, h = img.shape[1], img.shape[0]
        else:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size

        img = T.Compose([
            T.ToTensor(),
            T.Resize((IMG_SIZE, IMG_SIZE)),
        ])(img)

        image_stable = img - 0.5
        img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)


        imgs.append(img)
        imgs_raw.append(img_raw)
        images_stable.append(image_stable)

    height = h
    width = w
    imgs = np.stack(imgs, axis=0)
    imgs_raw = np.stack(imgs_raw, axis=0)
    images_stable = np.stack(images_stable, axis=0)

    # track data
    imgs_ = _load_tiffs(Path(file_dir), dtype=np.float32)
    imgs_01 = np.stack([
                normalize_01(_x) for _x in tqdm(imgs_, desc="Normalizing", leave=False)
            ])
    imgs_ = np.stack([
                normalize(_x) for _x in tqdm(imgs_, desc="Normalizing", leave=False)
            ])

    return imgs, imgs_raw, images_stable, imgs_, imgs_01, height, width

