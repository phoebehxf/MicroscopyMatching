import os
from typing import Any, List, Optional
from huggingface_hub import hf_hub_download
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
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pytorch_lightning as pl
from _utils.load_models import load_stable_diffusion_model
from models.model import Counting_with_SD_features_dino_vit_c3 as Counting
from models.enc_model.loca import build_model as build_loca_model
import time
from models.seg_post_model import metrics
from datetime import datetime
import json
import logging
from PIL import Image
import torchvision.transforms as T
import cv2
from skimage import io, measure
logging.getLogger('models.seg_post_model.models').setLevel(logging.ERROR)

SCALE = 1



class SegmentationModule(pl.LightningModule):
    def __init__(self, use_box=True):
        super().__init__()
        self.use_box = use_box
        self.config = RunConfig()   # config for stable diffusion
        self.initialize_model()
        

    def initialize_model(self):
        
        # load loca model
        self.loca_model = build_loca_model()
        self.loca_model.eval()

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
            initializer_token = "segment"
            token_ids = self.stable.tokenizer.encode(initializer_token, add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                # raise ValueError("The initializer token must be a single token.")
                token_ids = token_ids[:1]

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
        self.counting_adapter.to(device)
        self.loca_model.to(device)

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
            boxes = torch.tensor([[0,0,512,512]])
            assert self.use_box == False
        img_raw = io.imread(data_path)
        if len(img_raw.shape) == 3 and img_raw.shape[2] > 3:
            img_raw = img_raw[:,:,:3]
        img_raw = cv2.resize(img_raw, (512, 512))

        # move to device
        input_image = input_image.unsqueeze(0).to(self.device)
        img_raw = torch.from_numpy(img_raw).unsqueeze(0).float().to(self.device)
        boxes = boxes.unsqueeze(0).to(self.device)
        input_image_stable = input_image_stable.unsqueeze(0).to(self.device)
        
        latents = self.stable.vae.encode(input_image_stable).latent_dist.sample().detach()
        latents = latents * 0.18215
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.tensor([20], device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        input_ids_ = self.stable.tokenizer(
            self.placeholder_token + " " + self.task_token,
            padding="max_length",
            truncation=True,
            max_length=self.stable.tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = input_ids_["input_ids"].to(self.device)
        attention_mask = input_ids_["attention_mask"].to(self.device)
        encoder_hidden_states = self.stable.text_encoder(input_ids, attention_mask)[0]
        encoder_hidden_states = encoder_hidden_states.repeat(bsz, 1, 1)



        task_loc_idx = torch.nonzero(input_ids == self.placeholder_token_id)

        if self.use_box:
            loca_out = self.loca_model.forward_before_reg(input_image, boxes)
            loca_feature_bf_regression =  loca_out["feature_bf_regression"]
            adapted_emb = self.counting_adapter.adapter(loca_feature_bf_regression, boxes)      # shape [1, 768]
            # adapted_emb = self.counting_adapter.adapter(data['crops_dino'], self.dino)      # shape [1, 768]
            if task_loc_idx.shape[0] == 0:
                encoder_hidden_states[0,2,:] = adapted_emb.squeeze()  # 放在task prompt下一位
            else:
                encoder_hidden_states[:,task_loc_idx[0, 1]+1,:] = adapted_emb.squeeze()  # 放在task prompt下一位

        # Predict the noise residual
        noise_pred, feature_list = self.stable.unet(noisy_latents, timesteps, encoder_hidden_states)
        time3 = time.time()
        noise_pred = noise_pred.sample

        attention_store = self.controller.attention_store


        attention_maps = []
        exemplar_attention_maps1 = []
        exemplar_attention_maps2 = []
        exemplar_attention_maps3 = []

        # only use 64x64 self-attention
        self_attn_aggregate = attn_utils.aggregate_attention( # [res, res, 4096]
                prompts=[self.config.prompt for i in range(bsz)],        # 这里要改么
                attention_store=self.controller,     
                res=64,
                from_where=("up", "down"),
                is_cross=False,
                select=0
            )

        # cross attention
        for res in [32, 16]:
            attn_aggregate = attn_utils.aggregate_attention( # [res, res, 77]
                prompts=[self.config.prompt for i in range(bsz)],        # 这里要改么
                attention_store=self.controller,     
                res=res,
                from_where=("up", "down"),
                is_cross=True,
                select=0
            )

            task_attn_ = attn_aggregate[:, :, 1].unsqueeze(0).unsqueeze(0) # [1, 1, res, res]
            attention_maps.append(task_attn_)
            exemplar_attns1 = attn_aggregate[:, :, 2].unsqueeze(0).unsqueeze(0) # 取exemplar的attn
            exemplar_attention_maps1.append(exemplar_attns1)
            exemplar_attns2 = attn_aggregate[:, :, 3].unsqueeze(0).unsqueeze(0) # 取exemplar的attn
            exemplar_attention_maps2.append(exemplar_attns2)
            exemplar_attns3 = attn_aggregate[:, :, 4].unsqueeze(0).unsqueeze(0) # 取exemplar的attn
            exemplar_attention_maps3.append(exemplar_attns3)


        scale_factors = [(64 // attention_maps[i].shape[-1]) for i in range(len(attention_maps))]
        attns = torch.cat([F.interpolate(attention_maps[i_], scale_factor=scale_factors[i_], mode="bilinear") for i_ in range(len(attention_maps))])
        task_attn_64 = torch.mean(attns, dim=0, keepdim=True)
        cross_self_task_attn = attn_utils.self_cross_attn(self_attn_aggregate, task_attn_64)
        task_attn_64 = (task_attn_64 - task_attn_64.min()) / (task_attn_64.max() - task_attn_64.min() + 1e-6)
        cross_self_task_attn = (cross_self_task_attn - cross_self_task_attn.min()) / (cross_self_task_attn.max() - cross_self_task_attn.min() + 1e-6)

        scale_factors = [(64 // exemplar_attention_maps1[i].shape[-1]) for i in range(len(exemplar_attention_maps1))]
        attns = torch.cat([F.interpolate(exemplar_attention_maps1[i_], scale_factor=scale_factors[i_], mode="bilinear") for i_ in range(len(exemplar_attention_maps1))])
        exemplar_attn_64_1 = torch.mean(attns, dim=0, keepdim=True)

        if self.use_box:
            exemplar_attn_64 = exemplar_attn_64_1
            cross_self_exe_attn = attn_utils.self_cross_attn(self_attn_aggregate, exemplar_attn_64)
            exemplar_attn_64 = (exemplar_attn_64 - exemplar_attn_64.min()) / (exemplar_attn_64.max() - exemplar_attn_64.min() + 1e-6)
            cross_self_exe_attn = (cross_self_exe_attn - cross_self_exe_attn.min()) / (cross_self_exe_attn.max() - cross_self_exe_attn.min() + 1e-6)
        else:

            scale_factors = [(64 // exemplar_attention_maps2[i].shape[-1]) for i in range(len(exemplar_attention_maps2))]
            attns = torch.cat([F.interpolate(exemplar_attention_maps2[i_], scale_factor=scale_factors[i_], mode="bilinear") for i_ in range(len(exemplar_attention_maps2))])
            exemplar_attn_64_2 = torch.mean(attns, dim=0, keepdim=True)

            scale_factors = [(64 // exemplar_attention_maps3[i].shape[-1]) for i in range(len(exemplar_attention_maps3))]
            attns = torch.cat([F.interpolate(exemplar_attention_maps3[i_], scale_factor=scale_factors[i_], mode="bilinear") for i_ in range(len(exemplar_attention_maps3))])
            exemplar_attn_64_3 = torch.mean(attns, dim=0, keepdim=True)

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

            exemplar_attn_64 = (exemplar_attn_64_1 + exemplar_attn_64_2 + exemplar_attn_64_3) / 3
            cross_self_exe_attn = (cross_self_exe_attn1 + cross_self_exe_attn2 + cross_self_exe_attn3) / 3

            
        
        
        
        if self.use_box:
            attn_stack = [task_attn_64 / 2, cross_self_task_attn / 2, exemplar_attn_64, cross_self_exe_attn]
        else:
            attn_stack = [exemplar_attn_64 / 2, cross_self_exe_attn / 2, exemplar_attn_64, cross_self_exe_attn]
        attn_stack = torch.cat(attn_stack, dim=1)
        
            
        attn_after_new_regressor = self.counting_adapter.regressor(img_raw, attn_stack, feature_list)      # 直接用自己的
        
        input_image = cv2.resize(input_image[0].permute(1,2,0).cpu().numpy(), (width, height))
        pred = cv2.resize(attn_after_new_regressor.squeeze().cpu().numpy(), (width, height), interpolation=cv2.INTER_NEAREST)
        return pred

    



def inference(data_path, box=None, save_path="./example_imgs", visualize=False):
    if box is not None:
        use_box = True
    else:
        use_box = False
    model = SegmentationModule(use_box=use_box)
    load_msg = model.load_state_dict(torch.load("pretrained/microscopy_matching_seg.pth"), strict=True)
    model.eval()
    with torch.no_grad():
        mask = model(data_path, box)

    
    # visualize
    if visualize:
        img = io.imread(data_path)
        if len(img.shape) == 3 and img.shape[2] > 3:
            img = img[:,:,:3]
        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)
        img_show = img.squeeze()
        mask_show = mask.squeeze()
        os.makedirs(save_path, exist_ok=True)
        filename = data_path.split("/")[-1]
        fig, ax = plt.subplots(1,2, figsize=(12,6))
        ax[0].imshow(img_show)
        if use_box:
            boxes = np.array(box)
            for box in boxes:
                rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='r', facecolor='none')
                ax[0].add_patch(rect)
            ax[0].set_title("Input Image with Box")
        else:
            ax[0].set_title("Input Image")
        ax[0].axis("off")
        ax[1].imshow(img_show)
        for inst_id in np.unique(mask_show):
            if inst_id == 0:  # 0 background
                continue
            # 生成二值 mask
            binary_mask = (mask_show == inst_id).astype(np.uint8)
            contours = measure.find_contours(binary_mask, 0.5)
            for contour in contours:
                ax[1].plot(contour[:, 1], contour[:, 0], linewidth=1.5, linestyle="--", color='yellow')
        ax[1].imshow(overlay_instances(img_show, mask_show, alpha=0.3))
        ax[1].set_title("Segmentation Result")
        ax[1].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, filename.split(".")[0]+"_seg.png"), dpi=300)
        plt.close()
    
    return mask


def main():
    inference(
        data_path="example_imgs/1977_Well_F-5_Field_1.png", 
    #   box=[[724, 864, 900, 966]], 
        save_path="./example_imgs",
        visualize=True
        )


from matplotlib import cm

def overlay_instances(img, mask, alpha=0.5, cmap_name="tab20"):
    img = img.astype(np.float32)
    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=-1)
    if img.max() > 1.5:
        img = img / 255.0


    overlay = img.copy()
    cmap = cm.get_cmap(cmap_name, np.max(mask)+1)

    for inst_id in np.unique(mask):
        if inst_id == 0:  
            continue
        color = np.array(cmap(inst_id)[:3])  # RGB
        overlay[mask == inst_id] = (1 - alpha) * overlay[mask == inst_id] + alpha * color

    return overlay

if __name__ == "__main__":
    main()