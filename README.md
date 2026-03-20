# 🔬 MicroscopyMatching

A unified framework for microscopy image analysis, built on a pre-trained Latent Diffusion Model (Stable Diffusion). MicroscopyMatching supports three tasks through a single backbone:

- **Instance Segmentation** — segment individual cells/objects in microscopy images
- **Object Counting** — estimate object counts via density map prediction
- **Cell Tracking** — track objects across time-lapse image sequences

Model weights are automatically downloaded from [HuggingFace Hub](https://huggingface.co/phoebe777777/111) on first use.

## Installation

```bash
conda create -n microscopy python=3.11 -y
conda activate microscopy
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Quick Start

### Segmentation

```python
from inference_seg import load_model, run

# Load model (downloads weights on first run)
model, device = load_model()

# Run inference
mask = run(model, "example_imgs/seg/003_img.png", device=device)
# mask: uint16 numpy array, each unique value = one instance
```

With bounding box (ROI):

```python
model, device = load_model(use_box=True)
box = [[100, 100, 400, 400]]  # [x1, y1, x2, y2]
mask = run(model, "image.png", box=box, device=device)
```

For a complete example with visualization, see `test_seg.py`:

```bash
python test_seg.py
```

### Counting

```python
from inference_count import load_model, run, visualize_result

model, device = load_model()

result = run(model, "example_imgs/cnt/047cell.png", device=device)
print(f"Estimated count: {result['count']:.1f}")

# Visualize density map overlay
vis_path = visualize_result("example_imgs/cnt/047cell.png", result['density_map'], result['count'])
```

For a complete example with visualization, see `test_count.py`:

```bash
python test_count.py
```

### Tracking

Input: a directory of sequential TIF frames (e.g., `t000.tif`, `t001.tif`, ...).

```python
from inference_track import load_model, run, visualize_tracking_result

model, device = load_model()

result = run(model, "path/to/frame_dir/", device=device, output_dir="tracked_results")
# Outputs CTC-format results: res_track.txt + mask TIFs in output_dir

# Visualize as video
visualize_tracking_result(result['masks_tracked'], "tracking_vis.mp4")
```

For a complete example with visualization, see `test_track.py`:

```bash
python test_track.py
```

## Interactive Demo

Visit our [online demo](https://huggingface.co/spaces/VisionLanguageGroup/MicroscopyMatching) for an interactive experience. Note that this online interactive demo currently has usage limit and is for research trial use and feedback collection. You can also run the interactive demo locally without usage limit.
To run locally, first install the dependencies as described above, then run the following command (requires a GPU for best performance):


```bash
python app.py
```

Then open `http://localhost:7860` in your browser.

## Project Structure

```
├── inference_seg.py        # Segmentation inference API
├── inference_count.py      # Counting inference API
├── inference_track.py      # Tracking inference API
├── segmentation.py         # SegmentationModule (PyTorch Lightning)
├── counting.py             # CountingModule (PyTorch Lightning)
├── tracking_one.py         # TrackingModule (PyTorch Lightning)
├── config.py               # Stable Diffusion denoising config
├── app.py                  # Gradio web demo
├── models/
│   ├── model.py            # Task-specific adapter heads
│   ├── enc_model/          # Encoder architectures (LOCA, ViT, etc.)
│   ├── seg_post_model/     # Segmentation post-processing
│   └── tra_post_model/     # Tracking transformer & graph algorithms
├── _utils/                 # Shared utilities (SD loader, attention, etc.)
├── example_imgs/           # Example inputs for each task
│   ├── seg/                # Segmentation examples (.png)
│   ├── cnt/                # Counting examples (.png)
│   └── tra/                # Tracking examples (.zip of TIF sequences)
└── requirements.txt
```

## Example Images

Example inputs are provided in `example_imgs/` for each task. These cover various microscopy modalities including phase contrast, fluorescence, and tissue imaging.

<!-- ## Citation

If you use MicroscopyMatching in your research, please cite:

```bibtex
@article{microscopymatching2025,
  title={MicroscopyMatching},
  author={},
  year={2025}
}
```

## License

TBD -->
