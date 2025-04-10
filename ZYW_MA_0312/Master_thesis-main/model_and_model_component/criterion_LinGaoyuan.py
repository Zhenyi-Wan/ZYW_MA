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
        pred_rgb = outputs["rgb"]
        if "mask" in outputs:
            pred_mask = outputs["mask"].float()
        else:
            pred_mask = None
        gt_rgb = ray_batch["rgb"]

        loss = img2mse(pred_rgb, gt_rgb, pred_mask)

        return loss, scalars_to_log

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
        


