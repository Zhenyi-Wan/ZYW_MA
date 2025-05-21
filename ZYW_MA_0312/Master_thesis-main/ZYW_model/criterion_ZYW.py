import torch.nn as nn
from utils import img2mse
import torch


class Criterion(nn.Module):

    def __init__(self):
        super().__init__()

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

    def NeRO_loss(self, outputs, ray_batch, scalars_to_log):
        """
        BRDF criterion
        """
        color_NeRO = outputs["color_NeRO"]
        if "mask" in outputs:
            pred_mask = outputs["mask"].float()
        else:
            pred_mask = None
        gt_rgb = ray_batch["rgb"]# Zhenyi Wan [2025/4/10] The ground truth RGB

        loss_NeRO = img2mse(color_NeRO, gt_rgb, pred_mask)

        return loss_NeRO, scalars_to_log



    def NeILF_loss(self, outputs, roughness_output, metallic_output, ray_batch, gradients, scalars_to_log):
        color_NeILF = outputs["color_NeILF"]
        if "mask" in outputs:
            pred_mask = outputs["mask"].float()
        else:
            pred_mask = None
        gt_rgb = ray_batch["rgb"]# Zhenyi Wan [2025/4/10] The ground truth RGB
        # RGB loss
        loss_NeILF = img2mse(color_NeILF, gt_rgb, pred_mask)
        # set the weight of smoothness loss and lambertian loss
        lambertian_weight = 0.005  # We want to keep more details in the road
        smoothness_weight = 0.0005  
        # smoothness loss
        rgb_grad = ray_batch["rgb_grad"].cuda().reshape(-1) #[H*W]
        brdf_grads = gradients["NeILF_grad"] #[N_rand, 2, 3]
        smooth_loss = (brdf_grads.norm(dim=-1).sum(dim=-1) * (-rgb_grad).exp()).mean()
        # lambertian loss
        roughness = roughness_output["roughness"]
        metallic = metallic_output["metallic"]
        reg_loss = reg_loss = ((roughness - 1).abs()).mean() + ((metallic - 0).abs()).mean()
        # combine the loss
        loss_NeILF = loss_NeILF + lambertian_weight * reg_loss + smoothness_weight * smooth_loss

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
        


