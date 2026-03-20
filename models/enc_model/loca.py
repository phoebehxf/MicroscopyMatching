from .backbone import Backbone
from .transformer import TransformerEncoder
from .ope import OPEModule
from .positional_encoding import PositionalEncodingsFixed
from .regression_head import DensityMapRegressor

import torch
from torch import nn
from torch.nn import functional as F


class LOCA(nn.Module):

    def __init__(
        self,
        image_size: int,
        num_encoder_layers: int,
        num_ope_iterative_steps: int,
        num_objects: int,
        emb_dim: int,
        num_heads: int,
        kernel_dim: int,
        backbone_name: str,
        swav_backbone: bool,
        train_backbone: bool,
        reduction: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        norm: bool,
        zero_shot: bool,
    ):

        super(LOCA, self).__init__()

        self.emb_dim = emb_dim
        self.num_objects = num_objects
        self.reduction = reduction
        self.kernel_dim = kernel_dim
        self.image_size = image_size
        self.zero_shot = zero_shot
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers

        self.backbone = Backbone(
            backbone_name, pretrained=True, dilation=False, reduction=reduction,
            swav=swav_backbone, requires_grad=train_backbone
        )
        self.input_proj = nn.Conv2d(
            self.backbone.num_channels, emb_dim, kernel_size=1
        )

        if num_encoder_layers > 0:
            self.encoder = TransformerEncoder(
                num_encoder_layers, emb_dim, num_heads, dropout, layer_norm_eps,
                mlp_factor, norm_first, activation, norm
            )

        self.ope = OPEModule(
            num_ope_iterative_steps, emb_dim, kernel_dim, num_objects, num_heads,
            reduction, layer_norm_eps, mlp_factor, norm_first, activation, norm, zero_shot
        )

        self.regression_head = DensityMapRegressor(emb_dim, reduction)
        self.aux_heads = nn.ModuleList([
            DensityMapRegressor(emb_dim, reduction)
            for _ in range(num_ope_iterative_steps - 1)
        ])

        self.pos_emb = PositionalEncodingsFixed(emb_dim)

        self.attn_norm = nn.LayerNorm(normalized_shape=(64, 64))
        self.fuse = nn.Sequential(
            nn.Conv2d(324, 256, kernel_size=1, stride=1),
            nn.LeakyReLU(),
            nn.LayerNorm((64, 64))
        )

    def forward_before_reg(self, x, bboxes):
        num_objects = bboxes.size(1) if not self.zero_shot else self.num_objects
        # backbone
        backbone_features = self.backbone(x)
        # prepare the encoder input
        src = self.input_proj(backbone_features)
        bs, c, h, w = src.size()
        pos_emb = self.pos_emb(bs, h, w, src.device).flatten(2).permute(2, 0, 1)
        src = src.flatten(2).permute(2, 0, 1)

        # push through the encoder
        if self.num_encoder_layers > 0:
            image_features = self.encoder(src, pos_emb, src_key_padding_mask=None, src_mask=None)
        else:
            image_features = src

        # prepare OPE input
        f_e = image_features.permute(1, 2, 0).reshape(-1, self.emb_dim, h, w)

        all_prototypes = self.ope(f_e, pos_emb, bboxes) # [3, 27, 1, 256]

        response_maps_list = []
        for i in range(all_prototypes.size(0)):
            prototypes = all_prototypes[i, ...].permute(1, 0, 2).reshape(
                bs, num_objects, self.kernel_dim, self.kernel_dim, -1
            ).permute(0, 1, 4, 2, 3).flatten(0, 2)[:, None, ...] # [768, 1, 3, 3]

            response_maps = F.conv2d(
                torch.cat([f_e for _ in range(num_objects)], dim=1).flatten(0, 1).unsqueeze(0),
                prototypes,
                bias=None,
                padding=self.kernel_dim // 2,
                groups=prototypes.size(0)
            ).view(
                bs, num_objects, self.emb_dim, h, w
            ).max(dim=1)[0]

            response_maps_list.append(response_maps)

        out = {
            "feature_bf_regression": response_maps_list[-1],
            "aux_feature_bf_regression": response_maps_list[:-1]
        }

        return out

    def forward_reg(self, response_maps, attn_stack, unet_feature):
        attn_stack = self.attn_norm(attn_stack)
        attn_stack_mean = torch.mean(attn_stack, dim=1, keepdim=True)
        unet_feature = torch.cat([unet_feature, attn_stack], dim=1) # [1, 324, 64, 64]
        unet_feature = unet_feature * attn_stack_mean
        if unet_feature.shape[1] == 322:
            unet_feature = self.fuse1(unet_feature)
        else:
            unet_feature = self.fuse(unet_feature)

        response_maps = response_maps["aux_feature_bf_regression"] + [response_maps["feature_bf_regression"]]

        outputs = []
        for i in range(len(response_maps)):
            response_map = response_maps[i] + unet_feature
            if i == len(response_maps) - 1:
                predicted_dmaps = self.regression_head(response_map)
            else:
                predicted_dmaps = self.aux_heads[i](response_map)
            outputs.append(predicted_dmaps)
        
        return {"pred": outputs[-1], "aux_pred": outputs[:-1]}
    
    # def forward_reg1(self, response_maps, self_attn):        

    #     response_maps = response_maps["aux_feature_bf_regression"] + [response_maps["feature_bf_regression"]]

    #     outputs = []
    #     for i in range(len(response_maps)):
    #         response_map = response_maps[i] + self_attn
    #         if i == len(response_maps) - 1:
    #             predicted_dmaps = self.regression_head(response_map)
    #         else:
    #             predicted_dmaps = self.aux_heads[i](response_map)
    #         outputs.append(predicted_dmaps)
        
    #     return {"pred": outputs[-1], "aux_pred": outputs[:-1]}
    
    # def forward_reg_without_unet(self, response_maps, attn_stack):
    #     # attn_stack = self.attn_norm(attn_stack)
    #     attn_stack_mean = torch.mean(attn_stack, dim=1, keepdim=True)

    #     response_maps = response_maps["aux_feature_bf_regression"] + [response_maps["feature_bf_regression"]]

    #     outputs = []
    #     for i in range(len(response_maps)):
    #         response_map = response_maps[i] * attn_stack_mean * 0.5 + response_maps[i]
    #         if i == len(response_maps) - 1:
    #             predicted_dmaps = self.regression_head(response_map)
    #         else:
    #             predicted_dmaps = self.aux_heads[i](response_map)
    #         outputs.append(predicted_dmaps)
        
    #     return {"pred": outputs[-1], "aux_pred": outputs[:-1]}


def build_model():
    """
    Build LOCA with a fixed configuration based on defaults in `loca_args.py`.
    The `args` parameter is accepted for backward compatibility but ignored.
    """
    return LOCA(
        image_size=512,
        num_encoder_layers=3,
        num_ope_iterative_steps=3,
        num_objects=3,
        zero_shot=False,
        emb_dim=256,
        num_heads=8,
        kernel_dim=3,
        backbone_name="resnet50",
        swav_backbone=True,
        train_backbone=False,  # backbone_lr default is 0 in loca_args.py
        reduction=8,
        dropout=0.1,
        layer_norm_eps=1e-5,
        mlp_factor=8,
        norm_first=True,
        activation=nn.GELU,
        norm=True,
    )
