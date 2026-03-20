import os
from typing import Any, List, Optional
from huggingface_hub import hf_hub_download
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from PIL import Image
import numpy as np
import tifffile
from config import RunConfig
from _utils import attn_utils_new as attn_utils
from _utils.attn_utils_new import AttentionStore
from _utils.misc_helper import *
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import cv2
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pytorch_lightning as pl
from _utils.load_models import load_stable_diffusion_model
from models.model import Counting_with_SD_features_track as Counting
from models.enc_model.loca import build_model as build_loca_model
import time
from models.tra_post_model.model import TrackingTransformer
from models.tra_post_model.utils import (
    normalize,
)
from models.tra_post_model.data import build_windows_sd, get_features
from models.tra_post_model.tracking import TrackGraph, build_graph, track_greedy
import torchvision.transforms as T
from pathlib import Path
import dask.array as da
from typing import Dict, List, Optional, Union, Literal
from scipy.sparse import SparseEfficiencyWarning, csr_array
import tracemalloc
import gc
from _utils.load_track_data import load_track_images

SCALE = 1

def get_instance_boxes(mask):
    # Convert to int64 if needed
    if mask.dtype != torch.long:
        mask = mask.to(torch.long)

    boxes = []
    instance_ids = torch.unique(mask)
    instance_ids = instance_ids[instance_ids != 0]  # skip background

    for inst_id in instance_ids:
        inst_mask = mask == inst_id
        y_indices, x_indices = torch.where(inst_mask)

        if len(x_indices) == 0 or len(y_indices) == 0:
            continue

        x_min = torch.min(x_indices).item()
        x_max = torch.max(x_indices).item()
        y_min = torch.min(y_indices).item()
        y_max = torch.max(y_indices).item()

        boxes.append([x_min, y_min, x_max, y_max])
    boxes = torch.tensor(boxes, dtype=torch.float32)
    return boxes

class TrackingModule(pl.LightningModule):
    def __init__(self, use_box=False):
        super().__init__()
        self.use_box = use_box
        self.config = RunConfig()   # config for stable diffusion
        self.initialize_model()

    def initialize_model(self):
        
        # load loca model
        self.loca_model = build_loca_model()

        self.counting_adapter = Counting(scale_factor=SCALE)
            
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
            #@title Get token ids for our placeholder and initializer token. This code block will complain if initializer string is not a single token
            # Convert the initializer_token, placeholder_token to ids
            initializer_token = "track"
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
        
        fpath = Path("_utils/config.yaml")

        model = TrackingTransformer.from_cfg(
            cfg_path=fpath,
        )

        self.track_model = model
    

    def move_to_device(self, device):
        self.stable.to(device)
        self.counting_adapter.to(device)
        self.loca_model.to(device)
        self.track_model.to(device)

        self.to(device)

    def on_train_start(self) -> None:
        device = self.device
        dtype = self.dtype
        self.stable.to(device,dtype)
    
    def on_validation_start(self) -> None:
        device = self.device
        dtype = self.dtype
        self.stable.to(device,dtype)

    def forward(self, data):

        input_image_stable = data["image_stable"]
        boxes = data["boxes"]
        input_image = data["img_enc"]
        mask = data["mask"]
        latents = self.stable.vae.encode(input_image_stable).latent_dist.sample().detach()
        latents = latents * 0.18215
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.tensor([20], device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        input_ids_ = self.stable.tokenizer(
            self.placeholder_token,
            # "object",
            padding="max_length",
            truncation=True,
            max_length=self.stable.tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = input_ids_["input_ids"].to(self.device)
        attention_mask = input_ids_["attention_mask"].to(self.device)
        encoder_hidden_states = self.stable.text_encoder(input_ids, attention_mask)[0]
        encoder_hidden_states = encoder_hidden_states.repeat(bsz, 1, 1)

        time1 = time.time()
        input_image = input_image.to(self.device)
        boxes = boxes.to(self.device)

        loca_out = self.loca_model.forward_before_reg(input_image, boxes)
        loca_feature_bf_regression =  loca_out["feature_bf_regression"]
        # time2 = time.time()

        task_loc_idx = torch.nonzero(input_ids == self.placeholder_token_id)
        adapted_emb = self.counting_adapter.adapter(loca_feature_bf_regression, boxes)      # shape [1, 768]

        if task_loc_idx.shape[0] == 0:
            encoder_hidden_states[0,2,:] = adapted_emb.squeeze()
        else:
            encoder_hidden_states[:,task_loc_idx[0, 1]+1,:] = adapted_emb.squeeze()  

        # Predict the noise residual
        noise_pred, feature_list = self.stable.unet(noisy_latents, timesteps, encoder_hidden_states)
        time3 = time.time()
        noise_pred = noise_pred.sample

        attention_store = self.controller.attention_store

        # print(time2-time1, time3-time2)

        attention_maps = []
        exemplar_attention_maps = []

        cross_self_task_attn_maps = []
        cross_self_exe_attn_maps = []

        # only use 64x64 self-attention
        self_attn_aggregate = attn_utils.aggregate_attention( # [res, res, 4096]
                prompts=[self.config.prompt for i in range(bsz)], 
                attention_store=self.controller,     
                res=64,
                from_where=("up", "down"),
                is_cross=False,
                select=0
            )
        self_attn_aggregate32 = attn_utils.aggregate_attention( # [res, res, 4096]
                prompts=[self.config.prompt for i in range(bsz)], 
                attention_store=self.controller,     
                res=32,
                from_where=("up", "down"),
                is_cross=False,
                select=0
            )
        self_attn_aggregate16 = attn_utils.aggregate_attention( # [res, res, 4096]
                prompts=[self.config.prompt for i in range(bsz)], 
                attention_store=self.controller,     
                res=16,
                from_where=("up", "down"),
                is_cross=False,
                select=0
            )

        # cross attention
        for res in [32, 16]:
            attn_aggregate = attn_utils.aggregate_attention( # [res, res, 77]
                prompts=[self.config.prompt for i in range(bsz)], 
                attention_store=self.controller,     
                res=res,
                from_where=("up", "down"),
                is_cross=True,
                select=0
            )

            task_attn_ = attn_aggregate[:, :, 1].unsqueeze(0).unsqueeze(0) # [1, 1, res, res]
            attention_maps.append(task_attn_)
            exemplar_attns = attn_aggregate[:, :, 2].unsqueeze(0).unsqueeze(0) 
            exemplar_attention_maps.append(exemplar_attns)


        scale_factors = [(64 // attention_maps[i].shape[-1]) for i in range(len(attention_maps))]
        attns = torch.cat([F.interpolate(attention_maps[i_], scale_factor=scale_factors[i_], mode="bilinear") for i_ in range(len(attention_maps))])
        task_attn_64 = torch.mean(attns, dim=0, keepdim=True)


        scale_factors = [(64 // exemplar_attention_maps[i].shape[-1]) for i in range(len(exemplar_attention_maps))]
        attns = torch.cat([F.interpolate(exemplar_attention_maps[i_], scale_factor=scale_factors[i_], mode="bilinear") for i_ in range(len(exemplar_attention_maps))])
        exemplar_attn_64 = torch.mean(attns, dim=0, keepdim=True)

        cross_self_task_attn = attn_utils.self_cross_attn(self_attn_aggregate, task_attn_64)
        cross_self_exe_attn = attn_utils.self_cross_attn(self_attn_aggregate, exemplar_attn_64)
        cross_self_task_attn_maps.append(cross_self_task_attn)
        cross_self_exe_attn_maps.append(cross_self_exe_attn)

        task_attn_64 = (task_attn_64 - task_attn_64.min()) / (task_attn_64.max() - task_attn_64.min() + 1e-6)
        cross_self_task_attn = (cross_self_task_attn - cross_self_task_attn.min()) / (cross_self_task_attn.max() - cross_self_task_attn.min() + 1e-6)
        exemplar_attn_64 = (exemplar_attn_64 - exemplar_attn_64.min()) / (exemplar_attn_64.max() - exemplar_attn_64.min() + 1e-6)
        cross_self_exe_attn = (cross_self_exe_attn - cross_self_exe_attn.min()) / (cross_self_exe_attn.max() - cross_self_exe_attn.min() + 1e-6)

        attn_stack = [task_attn_64 / 2, cross_self_task_attn / 2, exemplar_attn_64, cross_self_exe_attn]
        attn_stack = torch.cat(attn_stack, dim=1)

            
        attn_after_new_regressor, loss = self.counting_adapter.regressor(input_image, attn_stack, feature_list, mask.cpu().numpy(), training=False) 

        return {
                "attn_after_new_regressor":attn_after_new_regressor, 
                "task_attn_64":task_attn_64, 
                "cross_self_task_attn":cross_self_task_attn, 
                "exemplar_attn_64": exemplar_attn_64,
                "cross_self_exe_attn": cross_self_exe_attn,
                "noise_pred":noise_pred,
                "noise":noise,
                "self_attn_aggregate":self_attn_aggregate,
                "self_attn_aggregate32":self_attn_aggregate32,
                "self_attn_aggregate16":self_attn_aggregate16,
                "loss": loss
                }

    def forward_sd(self, input_image_stable, input_image, boxes, height, width, mask=None):

        input_image_stable = input_image_stable.to(self.device)
        # density = data["density"]
        if boxes is not None:
            boxes = boxes.to(self.device)
        input_image = input_image.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        else:
            mask = torch.zeros((input_image.shape[0], 1, input_image.shape[2], input_image.shape[3])).to(self.device)

        latents = self.stable.vae.encode(input_image_stable).latent_dist.sample().detach()
        latents = latents * 0.18215
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.tensor([20], device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        input_ids_ = self.stable.tokenizer(
                self.placeholder_token + " " + self.task_token,
                # "object",
                padding="max_length",
                truncation=True,
                max_length=self.stable.tokenizer.model_max_length,
                return_tensors="pt",
            )
        input_ids = input_ids_["input_ids"].to(self.device)
        attention_mask = input_ids_["attention_mask"].to(self.device)
        encoder_hidden_states = self.stable.text_encoder(input_ids, attention_mask)[0]
        encoder_hidden_states = encoder_hidden_states.repeat(bsz, 1, 1)


        if boxes is not None and not self.training:
            if self.adapt_emb is None:
                loca_out_ = self.loca_model.forward_before_reg(input_image, boxes)
                loca_feature_bf_regression_ =  loca_out_["feature_bf_regression"]
                adapted_emb = self.counting_adapter.adapter(loca_feature_bf_regression_, boxes)      # shape [1, 768]
            else:
                adapted_emb = self.adapt_emb.to(self.device)
            task_loc_idx = torch.nonzero(input_ids == self.placeholder_token_id)
            if task_loc_idx.shape[0] == 0:
                encoder_hidden_states[0,5,:] = adapted_emb.squeeze() 
            else:
                encoder_hidden_states[:,task_loc_idx[0, 1]+4,:] = adapted_emb.squeeze()  

        # Predict the noise residual
        noise_pred, feature_list = self.stable.unet(noisy_latents, timesteps, encoder_hidden_states)
        noise_pred = noise_pred.sample
        attention_store = self.controller.attention_store


        attention_maps = []
        exemplar_attention_maps = []
        exemplar_attention_maps1 = []
        exemplar_attention_maps2 = []
        exemplar_attention_maps3 = []
        exemplar_attention_maps4 = []

        cross_self_task_attn_maps = []
        cross_self_exe_attn_maps = []

        # only use 64x64 self-attention
        self_attn_aggregate = attn_utils.aggregate_attention( # [res, res, 4096]
                prompts=[self.config.prompt for i in range(bsz)],       
                attention_store=self.controller,     
                res=64,
                from_where=("up", "down"),
                is_cross=False,
                select=0
            )

        # cross attention
        for res in [32, 16]:
            attn_aggregate = attn_utils.aggregate_attention( # [res, res, 77]
                prompts=[self.config.prompt for i in range(bsz)],        
                attention_store=self.controller,     
                res=res,
                from_where=("up", "down"),
                is_cross=True,
                select=0
            )

            task_attn_ = attn_aggregate[:, :, 1].unsqueeze(0).unsqueeze(0) # [1, 1, res, res]
            attention_maps.append(task_attn_)
            # if self.boxes is not None and not self.training:
            exemplar_attns1 = attn_aggregate[:, :, 2].unsqueeze(0).unsqueeze(0) 
            exemplar_attention_maps1.append(exemplar_attns1)
            exemplar_attns2 = attn_aggregate[:, :, 3].unsqueeze(0).unsqueeze(0) 
            exemplar_attention_maps2.append(exemplar_attns2)
            exemplar_attns3 = attn_aggregate[:, :, 4].unsqueeze(0).unsqueeze(0) 
            exemplar_attention_maps3.append(exemplar_attns3)
            exemplar_attns4 = attn_aggregate[:, :, 5].unsqueeze(0).unsqueeze(0) 
            exemplar_attention_maps4.append(exemplar_attns4)
            


        scale_factors = [(64 // attention_maps[i].shape[-1]) for i in range(len(attention_maps))]
        attns = torch.cat([F.interpolate(attention_maps[i_], scale_factor=scale_factors[i_], mode="bilinear") for i_ in range(len(attention_maps))])
        task_attn_64 = torch.mean(attns, dim=0, keepdim=True)
        cross_self_task_attn = attn_utils.self_cross_attn(self_attn_aggregate, task_attn_64)
        cross_self_task_attn_maps.append(cross_self_task_attn)

        # if not self.training:
        scale_factors = [(64 // exemplar_attention_maps1[i].shape[-1]) for i in range(len(exemplar_attention_maps1))]
        attns = torch.cat([F.interpolate(exemplar_attention_maps1[i_], scale_factor=scale_factors[i_], mode="bilinear") for i_ in range(len(exemplar_attention_maps1))])
        exemplar_attn_64_1 = torch.mean(attns, dim=0, keepdim=True)

        scale_factors = [(64 // exemplar_attention_maps2[i].shape[-1]) for i in range(len(exemplar_attention_maps2))]
        attns = torch.cat([F.interpolate(exemplar_attention_maps2[i_], scale_factor=scale_factors[i_], mode="bilinear") for i_ in range(len(exemplar_attention_maps2))])
        exemplar_attn_64_2 = torch.mean(attns, dim=0, keepdim=True)

        scale_factors = [(64 // exemplar_attention_maps3[i].shape[-1]) for i in range(len(exemplar_attention_maps3))]
        attns = torch.cat([F.interpolate(exemplar_attention_maps3[i_], scale_factor=scale_factors[i_], mode="bilinear") for i_ in range(len(exemplar_attention_maps3))])
        exemplar_attn_64_3 = torch.mean(attns, dim=0, keepdim=True)
        
        if boxes is not None:
            scale_factors = [(64 // exemplar_attention_maps4[i].shape[-1]) for i in range(len(exemplar_attention_maps4))]
            attns = torch.cat([F.interpolate(exemplar_attention_maps4[i_], scale_factor=scale_factors[i_], mode="bilinear") for i_ in range(len(exemplar_attention_maps4))])
            exemplar_attn_64_4 = torch.mean(attns, dim=0, keepdim=True)

        exes = []
        cross_exes = []
        cross_self_exe_attn1 = attn_utils.self_cross_attn(self_attn_aggregate, exemplar_attn_64_1)
        cross_self_exe_attn2 = attn_utils.self_cross_attn(self_attn_aggregate, exemplar_attn_64_2)
        cross_self_exe_attn3 = attn_utils.self_cross_attn(self_attn_aggregate, exemplar_attn_64_3)
        
        # # average
        exemplar_attn_64_1 = (exemplar_attn_64_1 - exemplar_attn_64_1.min()) / (exemplar_attn_64_1.max() - exemplar_attn_64_1.min() + 1e-6)
        exemplar_attn_64_2 = (exemplar_attn_64_2 - exemplar_attn_64_2.min()) / (exemplar_attn_64_2.max() - exemplar_attn_64_2.min() + 1e-6)
        exemplar_attn_64_3 = (exemplar_attn_64_3 - exemplar_attn_64_3.min()) / (exemplar_attn_64_3.max() - exemplar_attn_64_3.min() + 1e-6)
        cross_self_exe_attn1 = (cross_self_exe_attn1 - cross_self_exe_attn1.min()) / (cross_self_exe_attn1.max() - cross_self_exe_attn1.min() + 1e-6)
        cross_self_exe_attn2 = (cross_self_exe_attn2 - cross_self_exe_attn2.min()) / (cross_self_exe_attn2.max() - cross_self_exe_attn2.min() + 1e-6)
        cross_self_exe_attn3 = (cross_self_exe_attn3 - cross_self_exe_attn3.min()) / (cross_self_exe_attn3.max() - cross_self_exe_attn3.min() + 1e-6)
        exes = [exemplar_attn_64_1, exemplar_attn_64_2, exemplar_attn_64_3]
        cross_exes = [cross_self_exe_attn1, cross_self_exe_attn2, cross_self_exe_attn3]
        if boxes is not None:
            cross_self_exe_attn4 = attn_utils.self_cross_attn(self_attn_aggregate, exemplar_attn_64_4)
            exemplar_attn_64_4 = (exemplar_attn_64_4 - exemplar_attn_64_4.min()) / (exemplar_attn_64_4.max() - exemplar_attn_64_4.min() + 1e-6)
            cross_self_exe_attn4 = (cross_self_exe_attn4 - cross_self_exe_attn4.min()) / (cross_self_exe_attn4.max() - cross_self_exe_attn4.min() + 1e-6)
            exes.append(exemplar_attn_64_4)
            cross_exes.append(cross_self_exe_attn4)
        exemplar_attn_64 = sum(exes) / len(exes)
        cross_self_exe_attn = sum(cross_exes) / len(cross_exes)

        

        if self.use_box:
            attn_stack = [task_attn_64 / 2, cross_self_task_attn / 2, exemplar_attn_64, cross_self_exe_attn]
        else:
            attn_stack = [exemplar_attn_64 / 2, cross_self_exe_attn / 2, exemplar_attn_64, cross_self_exe_attn]
        attn_stack = torch.cat(attn_stack, dim=1)

            
        attn_after_new_regressor, loss, _ = self.counting_adapter.regressor.forward_seg(input_image, attn_stack, feature_list, mask.cpu().numpy(), self.training) 

        if not self.training:
            pred_mask = attn_after_new_regressor.detach().cpu()
            pred_boxes = get_instance_boxes(pred_mask.squeeze())
            
            self.boxes = pred_boxes.unsqueeze(0)

            if pred_boxes.shape[0] == 0:
                print("No instances detected in the predicted mask.")
                self.adapt_emb = adapted_emb.detach().cpu() # reuse emb
            else:
                pred_boxes = pred_boxes.unsqueeze(0).to(self.device)
                loca_out_ = self.loca_model.forward_before_reg(input_image, pred_boxes)
                loca_feature_bf_regression_ =  loca_out_["feature_bf_regression"]
                adapted_emb_ = self.counting_adapter.adapter(loca_feature_bf_regression_, pred_boxes)      # shape [1, 768]
                self.adapt_emb = adapted_emb_.detach().cpu()
        
        # resize to original image size
        mask_np = attn_after_new_regressor.squeeze().detach().cpu().numpy()
        mask_resized = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_NEAREST)

        return mask_resized

    def forward_boxes(self, input_image_stable, boxes, input_image):

        latents = self.stable.vae.encode(input_image_stable).latent_dist.sample().detach()
        latents = latents * 0.18215
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.tensor([20], device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        input_ids_ = self.stable.tokenizer(
                self.placeholder_token,
                # "object",
                padding="max_length",
                truncation=True,
                max_length=self.stable.tokenizer.model_max_length,
                return_tensors="pt",
            )
        input_ids = input_ids_["input_ids"].to(self.device)
        attention_mask = input_ids_["attention_mask"].to(self.device)
        encoder_hidden_states = self.stable.text_encoder(input_ids, attention_mask)[0]
        encoder_hidden_states = encoder_hidden_states.repeat(bsz, 1, 1)

        time1 = time.time()
        input_image = input_image.to(self.device)
        boxes = boxes.to(self.device)

        loca_out = self.loca_model.forward_before_reg(input_image, boxes)
        loca_feature_bf_regression =  loca_out["feature_bf_regression"]
        # time2 = time.time()

        task_loc_idx = torch.nonzero(input_ids == self.placeholder_token_id)
        adapted_emb = self.counting_adapter.adapter.forward_boxes(loca_feature_bf_regression, boxes)      # shape [n_instance, 768]
        n_instance = adapted_emb.shape[0]
        n_forward = int(np.ceil(n_instance / 74)) # in total 75 prompts including 1 task prompt and 74 object prompts?
        
        task_cross_attention = []
        instances_cross_attention = []

        for n in range(n_forward):
            len_ = min(74, n_instance - n * 74)
            encoder_hidden_states[:,(task_loc_idx[0, 1]+1):(task_loc_idx[0, 1]+1+len_),:] = adapted_emb[n*74:n*74+len_].squeeze()  


            # Predict the noise residual
            noise_pred, feature_list = self.stable.unet(noisy_latents, timesteps, encoder_hidden_states)
            noise_pred = noise_pred.sample



            attention_maps = []
            exemplar_attention_maps = []

            # cross attention
            for res in [32, 16]:
                attn_aggregate = attn_utils.aggregate_attention( # [res, res, 77]
                    prompts=[self.config.prompt for i in range(bsz)],        
                    attention_store=self.controller,     
                    res=res,
                    from_where=("up", "down"),
                    is_cross=True,
                    select=0
                )

                task_attn_ = attn_aggregate[:, :, 1].unsqueeze(0).unsqueeze(0) # [1, 1, res, res]
                attention_maps.append(task_attn_)
                try:
                    exemplar_attns = attn_aggregate[:, :, (task_loc_idx[0, 1]+1):(task_loc_idx[0, 1]+1+len_)].unsqueeze(0) 
                except:
                    print(n_instance, len_)
                exemplar_attns = torch.permute(exemplar_attns, (0, 3, 1, 2)) # [1, len_, res, res]
                exemplar_attention_maps.append(exemplar_attns)



            scale_factors = [(64 // attention_maps[i].shape[-1]) for i in range(len(attention_maps))]
            attns = torch.cat([F.interpolate(attention_maps[i_], scale_factor=scale_factors[i_], mode="bilinear") for i_ in range(len(attention_maps))])
            task_attn_64 = torch.mean(attns, dim=0, keepdim=True)

            try:
                scale_factors = [(64 // exemplar_attention_maps[i].shape[-1]) for i in range(len(exemplar_attention_maps))]
                attns = torch.cat([F.interpolate(exemplar_attention_maps[i_], scale_factor=scale_factors[i_], mode="bilinear") for i_ in range(len(exemplar_attention_maps))])
            except:
                print("exemplar_attention_maps shape mismatch, n_instance: {}, len_: {}".format(n_instance, len_))
                print(exemplar_attention_maps[0].shape)
                print(exemplar_attention_maps[1].shape)
                print(exemplar_attention_maps[2].shape)
            exemplar_attn_64 = torch.mean(attns, dim=0, keepdim=True)

            
            task_attn_64 = (task_attn_64 - task_attn_64.min()) / (task_attn_64.max() - task_attn_64.min() + 1e-6)
            exemplar_attn_64 = (exemplar_attn_64 - exemplar_attn_64.min()) / (exemplar_attn_64.max() - exemplar_attn_64.min() + 1e-6)

            task_cross_attention.append(task_attn_64)
            instances_cross_attention.append(exemplar_attn_64)

        task_cross_attention = torch.cat(task_cross_attention, dim=0) # [n_forward, 1, 64, 64]
        task_cross_attention = torch.mean(task_cross_attention, dim=0, keepdim=True) # [1, 1, 64, 64]
        instances_cross_attention = torch.cat(instances_cross_attention, dim=1) # [1, n_instance, 64, 64]
        assert instances_cross_attention.shape[1] == n_instance, "instances_cross_attention shape mismatch"
        attn_stack = [task_cross_attention / 2, instances_cross_attention]
        attn_stack = torch.cat(attn_stack, dim=1)

        del exemplar_attention_maps, attention_maps, attns, task_attn_64, exemplar_attn_64, latents
        del input_ids_, input_ids, attention_mask, encoder_hidden_states, timesteps, noisy_latents
        del loca_out, loca_feature_bf_regression, adapted_emb
        torch.cuda.empty_cache()

        return {
                "task_attn_64":task_cross_attention, 
                "exemplar_attn_64": instances_cross_attention,
                "noise_pred":noise_pred,
                "noise":noise,
                "attn_stack": attn_stack,
                "feature_list": feature_list,
                }

    
    def common_step(self, batch):
        mask = batch["mask_t"].to(torch.float32).to(self.device)
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        
        image_stable = batch["image_stable"]
        boxes = batch["boxes"]
        input_image = batch["img_enc"]
        input_image = input_image.to(self.device)
        image_stable = image_stable.to(self.device)
        keep_boxes = None
        if image_stable.dim() == 4:
            image_stable = image_stable.unsqueeze(0)
        if input_image.dim() == 4:
            input_image = input_image.unsqueeze(0)


        # segmentation part
        n_frames = mask.shape[1]
        masks_pred = []
        
        for i in range(n_frames):
            mask_ = mask[:, i, :, :].unsqueeze(0)  # [1, 1, H, W]
            mask_ = F.interpolate(mask_.float(), size=(512, 512), mode='nearest')  # [1, 1, 512, 512]
            mask_ = mask_.to(torch.int64).squeeze(0).detach().to(self.device)  # [1, 512, 512]
            masks_pred.append(mask_)
            del mask_



        # if True:
        attns_emb = []
        for i in range(n_frames):
            image_stable_prev = image_stable[:, max(0, i-1), :, :, :]
            image_stable_after = image_stable[:, min(n_frames-1, i+1), :, :, :]
            input_image_curr = input_image[:, i, :, :, :]

            mask_ = masks_pred[i].detach()
            unique_labels = torch.unique(mask_)  # tensor([0, 1, 2, ...])
            boxes_all = []
            for label in unique_labels:
                if label.item() == 0:
                    continue
                binary_mask = (mask_[0] == label).to(torch.uint8)  # [H, W]

                # 找非零点坐标
                y_coords, x_coords = torch.nonzero(binary_mask, as_tuple=True)
                if len(x_coords) == 0 or len(y_coords) == 0:
                    continue
                x_min = torch.min(x_coords)
                y_min = torch.min(y_coords)
                x_max = torch.max(x_coords)
                y_max = torch.max(y_coords)
                boxes_all.append([x_min.item(), y_min.item(), x_max.item(), y_max.item()])
            boxes_all_t = torch.tensor(boxes_all, dtype=torch.float32).unsqueeze(0)
            
            
            output_prev = self.forward_boxes(image_stable_prev, boxes_all_t, input_image_curr)
            attn_prev = output_prev["exemplar_attn_64"]
            feature_list_prev = output_prev["feature_list"]

            output_after = self.forward_boxes(image_stable_after, boxes_all_t, input_image_curr)
            # attn_stack = output["attn_stack"]
            attn_after = output_after["exemplar_attn_64"] # [1, n_instance, 64, 64]
            feature_list_after = output_after["feature_list"] # [1, n_channels, res, res]

            attn_prev = torch.permute(attn_prev, (1, 0, 2, 3)) # [n_instance, 1, 64, 64]
            attn_after = torch.permute(attn_after, (1, 0, 2, 3))
            attn_emb = self.counting_adapter.regressor(attn_prev, feature_list_prev, attn_after, feature_list_after)
            attns_emb.append(attn_emb.detach())
            
        attns_emb = torch.cat(attns_emb, dim=1)  # [1, n_instance, 4]
        # tracking part

        feats = batch["features_t"]
        coords = batch["coords_t"]

        with torch.no_grad():
        
            A_pred = self.track_model(coords, feats, attn_feat=attns_emb).detach()

        del masks_pred, feats, coords, batch
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        return A_pred
    
    # @profile
    def _predict_batch(self, batch):
        feats = batch["features_t"].to(self.device)
        coords = batch["coords_t"].to(self.device)
        timepoints = batch["timepoints_t"].to(self.device)
        # Hack that assumes that all parameters of a model are on the same device
        device = next(self.track_model.parameters()).device
        feats = feats.unsqueeze(0).to(device)
        timepoints = timepoints.unsqueeze(0).to(device)
        coords = coords.unsqueeze(0).to(device)

        # Concat timepoints to coordinates
        coords = torch.cat((timepoints.unsqueeze(2).float(), coords), dim=2)
        batch["coords_t"] = coords
        batch["features_t"] = feats
        with torch.no_grad():
            A = self.common_step(batch)
        torch.cuda.empty_cache()
        gc.collect()

        A = self.track_model.normalize_output(A, timepoints, coords)

        A = A.squeeze(0).detach().cpu().numpy()

        del feats, coords, timepoints, batch
        
        return A

    # @profile
    def predict_windows(self,
        windows: List[dict],
        features: list,
        model,
        imgs_enc: Optional[np.ndarray] = None,
        imgs_stable: Optional[np.ndarray] = None,
        intra_window_weight: float = 0,
        delta_t: int = 1,
        edge_threshold: float = 0.05,
        spatial_dim: int = 3,
        progbar_class=tqdm,
    ) -> dict:
        
        # first get all objects/coords
        time_labels_to_id = dict()
        node_properties = list()
        max_id = np.sum([len(f.labels) for f in features])

        all_timepoints = np.concatenate([f.timepoints for f in features])
        all_labels = np.concatenate([f.labels for f in features])
        all_coords = np.concatenate([f.coords for f in features])
        all_coords = all_coords[:, -spatial_dim:]

        for i, (t, la, c) in enumerate(zip(all_timepoints, all_labels, all_coords)):
            time_labels_to_id[(t, la)] = i
            node_properties.append(
                dict(
                    id=i,
                    coords=tuple(c),
                    time=t,
                    # index=ix,
                    label=la,
                )
            )

        # create assoc matrix between ids
        sp_weights, sp_accum = (
            csr_array((max_id, max_id), dtype=np.float32),
            csr_array((max_id, max_id), dtype=np.float32),
        )

        tracemalloc.start()

        for t in progbar_class(
            range(len(windows)),
            desc="Computing associations",
        ):
            # This assumes that the samples in the dataset are ordered by time and start at 0.
            batch = windows[t]
            timepoints = batch["timepoints"]
            labels = batch["labels"]

            A = self._predict_batch(batch)

            dt = timepoints[None, :] - timepoints[:, None]
            time_mask = np.logical_and(dt <= delta_t, dt > 0)
            A[~time_mask] = 0
            ii, jj = np.where(A >= edge_threshold)

            if len(ii) == 0:
                continue

            labels_ii = labels[ii]
            labels_jj = labels[jj]
            ts_ii = timepoints[ii]
            ts_jj = timepoints[jj]
            nodes_ii = np.array(
                tuple(time_labels_to_id[(t, lab)] for t, lab in zip(ts_ii, labels_ii))
            )
            nodes_jj = np.array(
                tuple(time_labels_to_id[(t, lab)] for t, lab in zip(ts_jj, labels_jj))
            )

            # weight middle parts higher
            t_middle = t + (model.config["window"] - 1) / 2
            ddt = timepoints[:, None] - t_middle * np.ones_like(dt)
            window_weight = np.exp(-intra_window_weight * ddt**2)  # default is 1
            # window_weight = np.exp(4*A) # smooth max
            sp_weights[nodes_ii, nodes_jj] += window_weight[ii, jj] * A[ii, jj]
            sp_accum[nodes_ii, nodes_jj] += window_weight[ii, jj]


            del batch, A, ii, jj, labels_ii, labels_jj, ts_ii, ts_jj, nodes_ii, nodes_jj, dt, time_mask
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        sp_weights_coo = sp_weights.tocoo()
        sp_accum_coo = sp_accum.tocoo()
        assert np.allclose(sp_weights_coo.col, sp_accum_coo.col) and np.allclose(
            sp_weights_coo.row, sp_accum_coo.row
        )

        # Normalize weights by the number of times they were written from different sliding window positions
        weights = tuple(
            ((i, j), v / a)
            for i, j, v, a in zip(
                sp_weights_coo.row,
                sp_weights_coo.col,
                sp_weights_coo.data,
                sp_accum_coo.data,
            )
        )

        results = dict()
        results["nodes"] = node_properties
        results["weights"] = weights

        return results


    def _predict(
        self,
        imgs: Union[np.ndarray, da.Array],
        masks: Union[np.ndarray, da.Array],
        imgs_enc: Optional[np.ndarray] = None,
        imgs_stable: Optional[np.ndarray] = None,
        boxes: Optional[np.ndarray] = None,
        edge_threshold: float = 0.05,
        n_workers: int = 0,
        normalize_imgs: bool = True,
        progbar_class=tqdm,
    ):
        print("Predicting weights for candidate graph")
        if normalize_imgs:
            if isinstance(imgs, da.Array):
                imgs = imgs.map_blocks(normalize)
            else:
                imgs = normalize(imgs)

        self.eval()

        features = get_features(
            detections=masks,
            imgs=imgs,
            ndim=self.track_model.config["coord_dim"],
            n_workers=n_workers,
            progbar_class=progbar_class,
        )
        print("Building windows")
        windows = build_windows_sd(
            features,
            imgs_enc=imgs_enc,
            imgs_stable=imgs_stable,
            boxes=boxes,
            imgs=imgs,
            masks=masks,
            window_size=self.track_model.config["window"],
            progbar_class=progbar_class,
        )

        print("Predicting windows")
        with torch.no_grad():
            predictions = self.predict_windows(
                windows=windows,
                features=features,
                imgs_enc=imgs_enc,
                imgs_stable=imgs_stable,
                model=self.track_model,
                edge_threshold=edge_threshold,
                spatial_dim=masks.ndim - 1,
                progbar_class=progbar_class,
            )

        return predictions

    def _track_from_predictions(
        self,
        predictions,
        mode: Literal["greedy_nodiv", "greedy", "ilp"] = "greedy",
        use_distance: bool = False,
        max_distance: int = 256,
        max_neighbors: int = 10,
        delta_t: int = 1,
        **kwargs,
    ):
        print("Running greedy tracker")
        nodes = predictions["nodes"]
        weights = predictions["weights"]

        candidate_graph = build_graph(
            nodes=nodes,
            weights=weights,
            use_distance=use_distance,
            max_distance=max_distance,
            max_neighbors=max_neighbors,
            delta_t=delta_t,
        )
        if mode == "greedy":
            return track_greedy(candidate_graph)
        elif mode == "greedy_nodiv":
            return track_greedy(candidate_graph, allow_divisions=False)
        elif mode == "ilp":
            from models.tra_post_model.tracking.ilp import track_ilp

            return track_ilp(candidate_graph, ilp_config="gt", **kwargs)
        else:
            raise ValueError(f"Tracking mode {mode} does not exist.")

    def track(
        self,
        file_dir: str,
        boxes: Optional[torch.Tensor] = None,
        mode: Literal["greedy_nodiv", "greedy", "ilp"] = "greedy",
        normalize_imgs: bool = True,
        progbar_class=tqdm,
        n_workers: int = 0,
        dataname: Optional[str] = None,
        **kwargs,
    ) -> TrackGraph:
        """Track objects across time frames.

        This method links segmented objects across time frames using the specified
        tracking mode. No hyperparameters need to be chosen beyond the tracking mode.

        Args:
            imgs: Input images of shape (T,(Z),Y,X) (numpy or dask array)
            masks: Instance segmentation masks of shape (T,(Z),Y,X).
            mode: Tracking mode:
                - "greedy_nodiv": Fast greedy linking without division
                - "greedy": Fast greedy linking with division
                - "ilp": Integer Linear Programming based linking (more accurate but slower)
            progbar_class: Progress bar class to use.
            n_workers: Number of worker processes for feature extraction.
            normalize_imgs: Whether to normalize the images.
            **kwargs: Additional arguments passed to tracking algorithm.

        Returns:
            TrackGraph containing the tracking results.
        """

        self.eval()
        imgs, imgs_raw, images_stable, tra_imgs, imgs_01, height, width = load_track_images(file_dir)
        imgs_stable = torch.from_numpy(images_stable).float().to(self.device)
        imgs_enc = torch.from_numpy(imgs).float().to(self.device)


        """get segmentation masks first"""
        self.boxes = None
        self.adapt_emb = None
        masks = []
        for i, (input_image, input_image_stable) in tqdm(enumerate(zip(imgs_enc, imgs_stable))):
            input_image = input_image.unsqueeze(0)
            input_image_stable = input_image_stable.unsqueeze(0)
            if i == 0:
                if self.use_box and boxes is not None:
                    self.boxes = boxes.to(self.device)
                else:
                    self.boxes = None
            
            with torch.no_grad():
                mask = self.forward_sd(input_image_stable, input_image, self.boxes, height=height, width=width)
            masks.append(mask)

        masks = np.stack(masks, axis=0)  # (T, H, W)

# -------------------------
        if not masks.shape == tra_imgs.shape:
            raise RuntimeError(
                f"Img shape {tra_imgs.shape} and mask shape {masks.shape} do not match."
            )

        if not tra_imgs.ndim == self.track_model.config["coord_dim"] + 1:
            raise RuntimeError(
                f"images should be a sequence of {self.track_model.config['coord_dim']}D images"
            )

        predictions = self._predict(
            tra_imgs,
            masks,
            imgs_enc=imgs_enc,
            imgs_stable=imgs_stable,
            boxes=boxes,
            normalize_imgs=normalize_imgs,
            progbar_class=progbar_class,
            n_workers=n_workers,
        )
        track_graph = self._track_from_predictions(predictions, mode=mode, **kwargs)
        
        return track_graph, masks
