import torch.nn as nn
from utils import img2mse
import torch

from ZYW_PBR_functions.field_ZYW import AppShadingNetwork
from ZYW_PBR_functions.PBR_MI_ZYW import NeILFPBR


class Criterion(nn.Module):
    default_cfg = {
        # shader network for NeRO method
        'shader_config': {},
    }

    def __init__(self):
        super().__init__()
        self.cfg = {**self.default_cfg}
        self.NeRO_PBR = AppShadingNetwork(self.cfg['shader_config'])
        self.NeILF_PBR = NeILFPBR()


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

    def NeRO_loss(self, outputs, roughness_outputs, metallic_outputs, albedo_outputs, normals_outputs, ray_batch, scalars_to_log):
        """
        BRDF criterion
        """
        # Zhenyi Wan [2025/4/10] First extract the BRDF values and use PBR method to rerender the RGB and calculate the MSE with ground truth
        pred_roughness = roughness_outputs["roughness"]
        pred_metallic = metallic_outputs["metallic"]
        pred_albedo = albedo_outputs["albedo"]
        pred_normals = normals_outputs["normals"]
        points = outputs["points"]
        ray_d = ray_batch["ray_d"]

        # Zhenyi Wan [2025/4/15] NeRO Method: Which use LUT based slit-sum approximation to calculate the color. Different is, NeRO calculate
        #each sampled points along the rays. We already get the surface points, so only calculate each surface points.
        color_NeRO, occ_info = self.NeRO_PBR(points, pred_roughness, pred_metallic, pred_albedo, pred_normals, ray_d)

        if "mask" in outputs:
            pred_mask = outputs["mask"].float()
        else:
            pred_mask = None
        gt_rgb = ray_batch["rgb"]# Zhenyi Wan [2025/4/10] The ground truth RGB

        loss_NeRO = img2mse(color_NeRO, gt_rgb, pred_mask)

        return loss_NeRO, scalars_to_log



    def NeILF_loss(self, outputs, roughness_outputs, metallic_outputs, albedo_outputs, normals_outputs, ray_batch, scalars_to_log):
        pred_roughness = roughness_outputs["roughness"]
        pred_metallic = metallic_outputs["metallic"]
        pred_albedo = albedo_outputs["albedo"]
        pred_normals = normals_outputs["normals"]
        points = outputs["points"]
        ray_d = ray_batch["ray_d"]
        view_d = -ray_d # Zhenyi Wan [2025/4/15] View direction is the opposite to ray_d

        # Zhenyi Wan [2025/4/15] NeILF PBR, which uses traditional BRDF render method.
        color_NeILF = self.NeILF_PBR(points, pred_roughness, pred_metallic, pred_albedo, pred_normals, view_d)

        if "mask" in outputs:
            pred_mask = outputs["mask"].float()
        else:
            pred_mask = None
        gt_rgb = ray_batch["rgb"]# Zhenyi Wan [2025/4/10] The ground truth RGB

        loss_NeILF = img2mse(color_NeILF, gt_rgb, pred_mask)

        return loss_NeILF, scalars_to_log


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
        


