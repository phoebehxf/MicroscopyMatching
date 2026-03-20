# stable diffusion x loca
import os
import pprint
from typing import Any, List, Optional
import argparse
from huggingface_hub import hf_hub_download
import pyrallis
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import os
from PIL import Image
import numpy as np
from config import RunConfig
from _utils import attn_utils_new as attn_utils
from _utils.attn_utils_new import AttentionStore
from _utils.misc_helper import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pytorch_lightning as pl
from _utils.load_models import load_stable_diffusion_model
from models.model import Counting_with_SD_features_loca as Counting
from models.enc_model.loca import build_model as build_loca_model
import time
import torchvision.transforms as T
import skimage.io as io

SCALE = 1


class CountingModule(pl.LightningModule):
    def __init__(self, use_box=True):
        super().__init__()
        self.use_box = use_box
        self.config = RunConfig()   # config for stable diffusion
        self.initialize_model()
        

    def initialize_model(self):
        
        # load loca model
        self.loca_model = build_loca_model()

        self.counting_adapter = Counting(scale_factor=SCALE)
        # if os.path.isfile(self.args.adapter_weight):
        #     adapter_weight = torch.load(self.args.adapter_weight,map_location=torch.device('cpu'))
        #     self.counting_adapter.load_state_dict(adapter_weight, strict=False)
            
        ### load stable diffusion and its controller
        self.stable = load_stable_diffusion_model(config=self.config)
        self.noise_scheduler = self.stable.scheduler
        self.controller = AttentionStore(max_size=64)
        attn_utils.register_attention_control(self.stable, self.controller)
        attn_utils.register_hier_output(self.stable)

        ##### initialize token_emb #####
        placeholder_token = "<task-prompt>"
        self.task_token = "repetitive objects"
        # Add the placeholder token in tokenizer
        num_added_tokens = self.stable.tokenizer.add_tokens(placeholder_token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )
        try:
            task_embed_from_pretrain = hf_hub_download(
                repo_id="phoebe777777/111",
                filename="task_embed.pth",
                token=None,
                force_download=False
            )
            placeholder_token_id = self.stable.tokenizer.convert_tokens_to_ids(placeholder_token)
            self.stable.text_encoder.resize_token_embeddings(len(self.stable.tokenizer))

            token_embeds = self.stable.text_encoder.get_input_embeddings().weight.data
            token_embeds[placeholder_token_id] = task_embed_from_pretrain
        except:
            initializer_token = "count"
            token_ids = self.stable.tokenizer.encode(initializer_token, add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id = token_ids[0]
            placeholder_token_id = self.stable.tokenizer.convert_tokens_to_ids(placeholder_token)

            self.stable.text_encoder.resize_token_embeddings(len(self.stable.tokenizer))

            token_embeds = self.stable.text_encoder.get_input_embeddings().weight.data
            token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

        # others
        self.placeholder_token = placeholder_token
        self.placeholder_token_id = placeholder_token_id
    

    def move_to_device(self, device):
        self.stable.to(device)
        if self.loca_model is not None and self.counting_adapter is not None:
            self.loca_model.to(device)
            self.counting_adapter.to(device)
        self.to(device)

    def forward(self, data_path, box=None):
        filename = data_path.split("/")[-1]
        img = Image.open(data_path).convert("RGB")
        width, height = img.size
        input_image = T.Compose([T.ToTensor(), T.Resize((512, 512))])(img)
        input_image_stable = input_image - 0.5
        input_image = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(input_image)
        if box is not None:
            boxes = torch.tensor(box) / torch.tensor([width, height, width, height]) * 512  # xyxy, normalized
            assert self.use_box == True
        else:
            boxes = torch.tensor([[100,100,130,130], [200,200,250,250]], dtype=torch.float32)  # dummy box
            assert self.use_box == False

        # move to device
        input_image = input_image.unsqueeze(0).to(self.device)
        boxes = boxes.unsqueeze(0).to(self.device)
        input_image_stable = input_image_stable.unsqueeze(0).to(self.device)
        


        latents = self.stable.vae.encode(input_image_stable).latent_dist.sample().detach()
        latents = latents * 0.18215
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        timesteps = torch.tensor([20], device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        input_ids_ = self.stable.tokenizer(
            self.placeholder_token + " repetitive objects",
            # "object",
            padding="max_length",
            truncation=True,
            max_length=self.stable.tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = input_ids_["input_ids"].to(self.device)
        attention_mask = input_ids_["attention_mask"].to(self.device)
        encoder_hidden_states = self.stable.text_encoder(input_ids, attention_mask)[0]

        input_image = input_image.to(self.device)
        boxes = boxes.to(self.device)
        

        task_loc_idx = torch.nonzero(input_ids == self.placeholder_token_id)
        if self.use_box:
            loca_out = self.loca_model.forward_before_reg(input_image, boxes)
            loca_feature_bf_regression =  loca_out["feature_bf_regression"]
            adapted_emb = self.counting_adapter.adapter(loca_feature_bf_regression, boxes)      # shape [1, 768]
            if task_loc_idx.shape[0] == 0:
                encoder_hidden_states[0,2,:] = adapted_emb.squeeze()  
            else:
                encoder_hidden_states[0,task_loc_idx[0, 1]+1,:] = adapted_emb.squeeze()  

        # Predict the noise residual
        noise_pred, feature_list = self.stable.unet(noisy_latents, timesteps, encoder_hidden_states)
        noise_pred = noise_pred.sample
        attention_store = self.controller.attention_store


        attention_maps = []
        exemplar_attention_maps = []
        exemplar_attention_maps1 = []
        exemplar_attention_maps2 = []
        exemplar_attention_maps3 = []

        cross_self_task_attn_maps = []
        cross_self_exe_attn_maps = []

        # only use 64x64 self-attention
        self_attn_aggregate = attn_utils.aggregate_attention( # [res, res, 4096]
                prompts=[self.config.prompt],        
                attention_store=self.controller,     
                res=64,
                from_where=("up", "down"),
                is_cross=False,
                select=0
            )
        self_attn_aggregate32 = attn_utils.aggregate_attention( # [res, res, 4096]
                prompts=[self.config.prompt],        
                attention_store=self.controller,     
                res=32,
                from_where=("up", "down"),
                is_cross=False,
                select=0
            )
        self_attn_aggregate16 = attn_utils.aggregate_attention( # [res, res, 4096]
                prompts=[self.config.prompt],        
                attention_store=self.controller,     
                res=16,
                from_where=("up", "down"),
                is_cross=False,
                select=0
            )

        # cross attention
        for res in [32, 16]:
            attn_aggregate = attn_utils.aggregate_attention( # [res, res, 77]
                prompts=[self.config.prompt],        
                attention_store=self.controller,     
                res=res,
                from_where=("up", "down"),
                is_cross=True,
                select=0
            )

            task_attn_ = attn_aggregate[:, :, 1].unsqueeze(0).unsqueeze(0) # [1, 1, res, res]
            attention_maps.append(task_attn_)
            if self.use_box:
                exemplar_attns = attn_aggregate[:, :, 2].unsqueeze(0).unsqueeze(0) 
                exemplar_attention_maps.append(exemplar_attns)
            else:
                exemplar_attns1 = attn_aggregate[:, :, 2].unsqueeze(0).unsqueeze(0)
                exemplar_attns2 = attn_aggregate[:, :, 3].unsqueeze(0).unsqueeze(0)
                exemplar_attns3 = attn_aggregate[:, :, 4].unsqueeze(0).unsqueeze(0)
                exemplar_attention_maps1.append(exemplar_attns1)
                exemplar_attention_maps2.append(exemplar_attns2)
                exemplar_attention_maps3.append(exemplar_attns3)


        scale_factors = [(64 // attention_maps[i].shape[-1]) for i in range(len(attention_maps))]
        attns = torch.cat([F.interpolate(attention_maps[i_], scale_factor=scale_factors[i_], mode="bilinear") for i_ in range(len(attention_maps))])
        task_attn_64 = torch.mean(attns, dim=0, keepdim=True)
        cross_self_task_attn = attn_utils.self_cross_attn(self_attn_aggregate, task_attn_64)
        cross_self_task_attn_maps.append(cross_self_task_attn)

        if self.use_box:
            scale_factors = [(64 // exemplar_attention_maps[i].shape[-1]) for i in range(len(exemplar_attention_maps))]
            attns = torch.cat([F.interpolate(exemplar_attention_maps[i_], scale_factor=scale_factors[i_], mode="bilinear") for i_ in range(len(exemplar_attention_maps))])
            exemplar_attn_64 = torch.mean(attns, dim=0, keepdim=True)

            cross_self_exe_attn = attn_utils.self_cross_attn(self_attn_aggregate, exemplar_attn_64)
            cross_self_exe_attn_maps.append(cross_self_exe_attn)
        else:
            scale_factors = [(64 // exemplar_attention_maps1[i].shape[-1]) for i in range(len(exemplar_attention_maps1))]
            attns = torch.cat([F.interpolate(exemplar_attention_maps1[i_], scale_factor=scale_factors[i_], mode="bilinear") for i_ in range(len(exemplar_attention_maps1))])
            exemplar_attn_64_1 = torch.mean(attns, dim=0, keepdim=True)

            scale_factors = [(64 // exemplar_attention_maps2[i].shape[-1]) for i in range(len(exemplar_attention_maps2))]
            attns = torch.cat([F.interpolate(exemplar_attention_maps2[i_], scale_factor=scale_factors[i_], mode="bilinear") for i_ in range(len(exemplar_attention_maps2))])
            exemplar_attn_64_2 = torch.mean(attns, dim=0, keepdim=True)

            scale_factors = [(64 // exemplar_attention_maps3[i].shape[-1]) for i in range(len(exemplar_attention_maps3))]
            attns = torch.cat([F.interpolate(exemplar_attention_maps3[i_], scale_factor=scale_factors[i_], mode="bilinear") for i_ in range(len(exemplar_attention_maps3))])
            exemplar_attn_64_3 = torch.mean(attns, dim=0, keepdim=True)


            cross_self_task_attn = attn_utils.self_cross_attn(self_attn_aggregate, task_attn_64)
            cross_self_task_attn_maps.append(cross_self_task_attn)

            # if self.args.merge_exemplar == "average":
            cross_self_exe_attn1 = attn_utils.self_cross_attn(self_attn_aggregate, exemplar_attn_64_1)
            cross_self_exe_attn2 = attn_utils.self_cross_attn(self_attn_aggregate, exemplar_attn_64_2)
            cross_self_exe_attn3 = attn_utils.self_cross_attn(self_attn_aggregate, exemplar_attn_64_3)
            exemplar_attn_64 = (exemplar_attn_64_1 + exemplar_attn_64_2 + exemplar_attn_64_3) / 3
            cross_self_exe_attn = (cross_self_exe_attn1 + cross_self_exe_attn2 + cross_self_exe_attn3) / 3

            exemplar_attn_64 = (exemplar_attn_64 - exemplar_attn_64.min()) / (exemplar_attn_64.max() - exemplar_attn_64.min() + 1e-6)

        attn_stack = [exemplar_attn_64 / 2, cross_self_exe_attn / 2, exemplar_attn_64, cross_self_exe_attn]
        attn_stack = torch.cat(attn_stack, dim=1)
        
        if not self.use_box:

            loca_out = self.loca_model.forward_before_reg(input_image, boxes)
            loca_feature_bf_regression =  loca_out["feature_bf_regression"]
        attn_out = self.loca_model.forward_reg(loca_out, attn_stack, feature_list[-1])
        pred_density = attn_out["pred"].squeeze().cpu().numpy()
        pred_cnt = pred_density.sum().item()

        # resize pred_density to original image size
        pred_density_rsz = cv2.resize(pred_density, (width, height), interpolation=cv2.INTER_CUBIC)
        pred_density_rsz = pred_density_rsz / pred_density_rsz.sum() * pred_cnt

        return pred_density_rsz, pred_cnt

    
def inference(data_path, box=None, save_path="./example_imgs", visualize=False):
    if box is not None:
        use_box = True
    else:
        use_box = False
    model = CountingModule(use_box=use_box)
    load_msg = model.load_state_dict(torch.load("pretrained/microscopy_matching_cnt.pth"), strict=True)
    model.eval()
    with torch.no_grad():
        density_map, cnt = model(data_path, box)
    
    if visualize:
        img = io.imread(data_path)
        if len(img.shape) == 3 and img.shape[2] > 3:
            img = img[:,:,:3]
        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)
        img_show = img.squeeze()
        density_map_show = density_map.squeeze()
        os.makedirs(save_path, exist_ok=True)
        filename = data_path.split("/")[-1]
        img_show = (img_show - np.min(img_show)) / (np.max(img_show) - np.min(img_show))
        fig, ax = plt.subplots(1,2, figsize=(12,6))
        ax[0].imshow(img_show)
        ax[0].axis('off')
        ax[0].set_title(f"Input image")
        ax[1].imshow(img_show)
        ax[1].imshow(density_map_show, cmap='jet', alpha=0.5)  # Overlay density map with some transparency
        ax[1].axis('off')
        ax[1].set_title(f"Predicted density map, count: {cnt:.1f}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, filename.split(".")[0]+"_cnt.png"), dpi=300)
        plt.close()
    return density_map

def main():
    
    inference(
        data_path = "example_imgs/1977_Well_F-5_Field_1.png",
        # box=[[150, 60, 183, 87]],
        save_path = "./example_imgs",
        visualize = True
    )

if __name__ == "__main__":
    main()