import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import clip
import sys
import numpy as np
from models.seg_post_model.models import SegModel

from torchvision.ops import roi_align


class Counting_with_SD_features_loca(nn.Module):
    def __init__(self, scale_factor):
        super(Counting_with_SD_features_loca, self).__init__()
        self.adapter = adapter_roi_loca()
        self.regressor = regressor_with_SD_features()


class Counting_with_SD_features_dino_vit_c3(nn.Module):
    def __init__(self, scale_factor, vit=None):
        super(Counting_with_SD_features_dino_vit_c3, self).__init__()
        self.adapter = adapter_roi_loca()
        self.regressor = regressor_with_SD_features_seg_vit_c3()

class Counting_with_SD_features_track(nn.Module):
    def __init__(self, scale_factor, vit=None):
        super(Counting_with_SD_features_track, self).__init__()
        self.adapter = adapter_roi_loca()
        self.regressor = regressor_with_SD_features_tra()


class adapter_roi_loca(nn.Module):
    def __init__(self, pool_size=[3, 3]):
        super(adapter_roi_loca, self).__init__()
        self.pool_size = pool_size
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(256 * 3 * 3, 768)
        self.initialize_weights()
    def forward(self, x, boxes):
            num_of_boxes = boxes.shape[1]
            rois = []
            bs, _, h, w = x.shape
            if h != 512 or w != 512:
                x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
            if bs == 1:
                boxes = torch.cat([
                    torch.arange(
                        bs, requires_grad=False
                    ).to(boxes.device).repeat_interleave(num_of_boxes).reshape(-1, 1),
                    boxes.flatten(0, 1),
                ], dim=1)
                rois = roi_align(
                    x,
                    boxes=boxes, output_size=3,
                    spatial_scale=1.0 / 8, aligned=True
                )
                rois = torch.mean(rois, dim=0, keepdim=True)
            else:
                boxes = torch.cat([
                    boxes.flatten(0, 1),
                ], dim=1).split(num_of_boxes, dim=0)
                rois = roi_align(
                    x,
                    boxes=boxes, output_size=3,
                    spatial_scale=1.0 / 8, aligned=True
                )
                rois = rois.split(num_of_boxes, dim=0)
                rois = torch.stack(rois, dim=0)
                rois = torch.mean(rois, dim=1, keepdim=False)
            x = self.conv1(rois)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    def forward_boxes(self, x, boxes):
            num_of_boxes = boxes.shape[1]
            rois = []
            bs, _, h, w = x.shape
            if h != 512 or w != 512:
                x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
            if bs == 1:
                boxes = torch.cat([
                    torch.arange(
                        bs, requires_grad=False
                    ).to(boxes.device).repeat_interleave(num_of_boxes).reshape(-1, 1),
                    boxes.flatten(0, 1),
                ], dim=1)
                rois = roi_align(
                    x,
                    boxes=boxes, output_size=3,
                    spatial_scale=1.0 / 8, aligned=True
                )
                # rois = torch.mean(rois, dim=0, keepdim=True)
            else:
                raise NotImplementedError
            x = self.conv1(rois)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



class regressor_with_SD_features(nn.Module):
    def __init__(self):
        super(regressor_with_SD_features, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(324, 256, kernel_size=1, stride=1),
            nn.LeakyReLU(),
            nn.LayerNorm((64, 64))
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU()
        )
        self.norm = nn.LayerNorm(normalized_shape=(64, 64))
        self.initialize_weights()

    def forward(self, attn_stack, feature_list):
        attn_stack = self.norm(attn_stack)
        unet_feature = feature_list[-1]
        attn_stack_mean = torch.mean(attn_stack, dim=1, keepdim=True)
        unet_feature = unet_feature * attn_stack_mean
        unet_feature = torch.cat([unet_feature, attn_stack], dim=1) # [1, 324, 64, 64]
        x = self.layer1(unet_feature)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.conv(x)
        return out / 100
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

from models.enc_model.unet_parts import *


class regressor_with_SD_features_seg_vit_c3(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=False):
        super(regressor_with_SD_features_seg_vit_c3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.norm = nn.LayerNorm(normalized_shape=(64, 64))
        self.inc_0 = nn.Conv2d(n_channels, 3, kernel_size=3, padding=1)
        self.vit_model = SegModel(gpu=True, nchan=3, pretrained_model="", use_bfloat16=False)
        self.vit = self.vit_model.net

    def forward(self, img, attn_stack, feature_list):
        attn_stack = attn_stack[:, [1,3], ...]
        attn_stack = self.norm(attn_stack)
        unet_feature = feature_list[-1]
        unet_feature_mean = torch.mean(unet_feature, dim=1, keepdim=True)

        x = torch.cat([unet_feature_mean, attn_stack], dim=1) # [1, 324, 64, 64]

        if x.shape[-1] != 512:
            x = F.interpolate(x, size=(512, 512), mode="bilinear")
        x = self.inc_0(x)


        
        out = self.vit_model.eval(img.squeeze().cpu().numpy(), feat=x.squeeze().cpu().numpy())
        if out.dtype == np.uint16:
            out = out.astype(np.int16)
        out = torch.from_numpy(out).unsqueeze(0).to(x.device)
        return out
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class regressor_with_SD_features_tra(nn.Module):
    def __init__(self, n_channels=2, n_classes=2, bilinear=False):
        super(regressor_with_SD_features_tra, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.norm = nn.LayerNorm(normalized_shape=(64, 64))

        # segmentation
        self.inc_0 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.vit_model = SegModel(gpu=True, nchan=3, pretrained_model="", use_bfloat16=False)
        self.vit = self.vit_model.net

        self.inc_1 = nn.Conv2d(n_channels, 1, kernel_size=3, padding=1)
        self.mlp = nn.Linear(64 * 64, 320)

    def forward_seg(self, img, attn_stack, feature_list, mask, training=False):
        attn_stack = attn_stack[:, [1,3], ...]
        attn_stack = self.norm(attn_stack)
        unet_feature = feature_list[-1]
        unet_feature_mean = torch.mean(unet_feature, dim=1, keepdim=True)
        x = torch.cat([unet_feature_mean, attn_stack], dim=1) # [1, 324, 64, 64]

        if x.shape[-1] != 512:
            x = F.interpolate(x, size=(512, 512), mode="bilinear")
        x = self.inc_0(x)
        feat = x
        
        out = self.vit_model.eval(img.squeeze().cpu().numpy(), feat=x.squeeze().cpu().numpy())
        if out.dtype == np.uint16:
            out = out.astype(np.int16)
        out = torch.from_numpy(out).unsqueeze(0).to(x.device)
        return out, 0., feat

    def forward(self, attn_prev, feature_list_prev, attn_after, feature_list_after):
        assert attn_prev.shape == attn_after.shape, "attn_prev and attn_after must have the same shape"
        n_instances = attn_prev.shape[0] 
        attn_prev = self.norm(attn_prev) # [n_instances, 1, 64, 64]
        attn_after = self.norm(attn_after)
        
        x = torch.cat([attn_prev, attn_after], dim=1)  # n_instances, 2, 64, 64
        
        x = self.inc_1(x)
        x = x.view(1, n_instances, -1)  # Flatten the tensor to [n_instances, 64*64*4]
        x = self.mlp(x)  # Apply the MLP to get the output
        
        return x  # Output shape will be [n_instances, 4]

        
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
