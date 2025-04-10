import torch.nn as nn
from utils import img2mse
import torch

from ZYW_PBR_functions.field_ZYW import AppShadingNetwork


class Criterion(nn.Module):
    default_cfg = {
        # shader network for NeRO method
        'shader_config': {},
    }

    def __init__(self):
        super().__init__()
        self.cfg = {**self.default_cfg}
        self.NeRO_PBR = AppShadingNetwork(self.cfg['shader_config'])


    def forward(self, outputs, ray_batch, scalars_to_log):
        """
        training criterion
        """
        # Zhenyi Wan [2025/4/10] Extract the RGB output and calculate the differences between the output and ground truth using MSE method
        pred_rgb = outputs["rgb"]
        if "mask" in outputs:
            pred_mask = outputs["mask"].float()
        else:
            pred_mask = None
        gt_rgb = ray_batch["rgb"]# Zhenyi Wan [2025/4/10] The ground truth RGB

        loss = img2mse(pred_rgb, gt_rgb, pred_mask)

        return loss, scalars_to_log

    def BRDF_loss(self, outputs, roughness_outputs, metallic_outputs, albedo_outputs, normals_outputs, ray_batch, scalars_to_log):
        """
        BRDF criterion
        """
        # Zhenyi Wan [2025/4/10] First extract the BRDF values and use PBR method to rerender the RGB and calculate the MSE with ground truth
        pred_roughness = roughness_outputs["roughness"]
        pred_metallic = metallic_outputs["metallic"]
        pred_albedo = albedo_outputs["albedo"]
        pred_normals = normals_outputs["normals"]
        points = outputs["points"]



    def sky_loss_depth(self,outputs, ray_batch,scalars_to_log):
        sky_depth = outputs["depth_sky"]
        loss_sky = torch.sum(torch.exp(-sky_depth))
        return loss_sky

    def sky_loss_rgb(self,outputs, ray_batch):
        pred_sky_rgb = outputs["rgb_sky"]

        gt_rgb = ray_batch["rgb"]

        sky_mask = ray_batch["sky_mask"]

        gt_sky_rgb = gt_rgb*(1-sky_mask)

        loss_sky = img2mse(pred_sky_rgb, gt_sky_rgb)

        return loss_sky

    'LinGaoyuan_operation_20240901: add depth loss'
    def depth_loss(self, outputs, ray_batch, train_depth_prior):
        ''
        'both pred_depth_value and gt_depth_value must have the shape (N_rand, 1)'
        pred_depth_value = outputs['depth'][..., None]

        if train_depth_prior is None:
            gt_depth_value = ray_batch["depth_value"]
        else:
            gt_depth_value = train_depth_prior[ray_batch['selected_inds']]
        # pred_depth_value = (pred_depth_value - pred_depth_value.min()) / (pred_depth_value.max() - pred_depth_value.min()) * 255.0
        # loss_depth = torch.sqrt(torch.sum((pred_depth_value - gt_depth_value) * (pred_depth_value - gt_depth_value)))

        'LinGaoyuan_operation_20240906: add zhengzhisheng depth loss'

        loss_depth = torch.mean((pred_depth_value - gt_depth_value) * (pred_depth_value - gt_depth_value))
        # N_rand = len(gt_depth_value)
        # loss_depth = (1/N_rand)*(torch.sum((pred_depth_value - gt_depth_value) * (pred_depth_value - gt_depth_value)))

        return loss_depth
        


