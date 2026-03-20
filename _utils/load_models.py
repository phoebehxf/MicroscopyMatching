from config import RunConfig
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
import torch.nn as nn

def load_stable_diffusion_model(config: RunConfig):
    device = torch.device('cpu')

    stable_diffusion_version = "CompVis/stable-diffusion-v1-4"
    stable = StableDiffusionPipeline.from_pretrained(stable_diffusion_version).to(device)
    return stable

