import os
import time
import numpy as np
import shutil
import torch
import torchvision

from model_and_model_component.render_image_LinGaoyuan import render_single_image
from model_and_model_component.sample_ray_LinGaoyuan import RaySamplerSingleImage


def get_ret(
        global_step,
        args,
        model,
        ray_sampler,
        projector,
        # gt_img,
        render_stride=1,
        # prefix="",
        # out_folder="",
        ret_alpha=False,
        single_net=True,
        sky_style_code=None,
        # sky_style_model=None,
        sky_model=None,
):
    model.switch_to_eval()
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()

        if model.feature_net is not None:
            featmaps = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
        else:
            featmaps = [None, None]
        ret = render_single_image(
            args,
            ray_sampler=ray_sampler,
            ray_batch=ray_batch,
            model=model,
            projector=projector,
            chunk_size=args.chunk_size,
            N_samples=args.N_samples,
            inv_uniform=args.inv_uniform,
            det=True,
            N_importance=args.N_importance,
            white_bkgd=args.white_bkgd,
            render_stride=render_stride,
            featmaps=featmaps,
            ret_alpha=ret_alpha,
            single_net=single_net,
            sky_style_code=sky_style_code,
            # sky_style_model=sky_style_model,
            sky_model=sky_model,
        )
        return ret

def update_prior_depth_value(
        global_step,
        train_dataloader,
        val_dataloader,
        args,
        model,
        # ray_sampler,
        projector,
        # gt_img,
        render_stride=1,
        prefix="",
        out_folder="",
        ret_alpha=False,
        single_net=True,
        sky_style_code=None,
        sky_style_model=None,
        sky_model=None,
):
    device = "cuda:{}".format(args.local_rank)
    time0 = time.time()


    train_updated_depth_values = torch.zeros((len(train_dataloader), 900,1600)).cuda()

    train_idx_list = []
    for train_data in train_dataloader:

        if train_data['idx'] in train_idx_list:
            print('already in idx_list')
        else:
            train_idx_list.append(train_data['idx'])

        print('begin of update prior depth value of image_{} in train dataset,'.format(train_data['idx'][0]), ' this is the {}st image'.format(len(train_idx_list)))

        original_gt_depth_value = train_data['depth_value']

        train_ray_sample = RaySamplerSingleImage(
                        train_data, device, render_stride=args.render_stride)
        train_ret = get_ret(
            global_step,
            args,
            model,
            train_ray_sample,
            projector,
            # gt_img,
            render_stride=args.render_stride,
            # prefix="val/",
            # out_folder=out_folder,
            # ret_alpha=args.N_importance > 0,
            ret_alpha=ret_alpha,
            single_net=args.single_net,
            sky_style_code=sky_style_code,
            # sky_style_model=sky_style_model,
            sky_model=sky_model,
        )
        update_depth_value = train_ret['outputs_coarse']['depth'][None, ...]
        # train_data['depth_value'] = update_depth_value
        train_updated_depth_values[train_data['idx'][0], ...] = update_depth_value


        print('end of update prior depth value of image_{} in train dataset'.format(train_data['idx'][0]))

    val_updated_depth_values = torch.zeros((len(val_dataloader), 900, 1600)).cuda()

    val_idx_list = []
    for val_data in val_dataloader:

        if val_data['idx'] in val_idx_list:
            print('already in val_idx_list')
        else:
            val_idx_list.append(val_data['idx'])

        print('begin of update prior depth value of image_{} in val dataset,'.format(val_data['idx'][0]), ' this is the {}st image'.format(len(val_idx_list)))

        original_gt_depth_value = val_data['depth_value']

        val_ray_sample = RaySamplerSingleImage(
                        val_data, device, render_stride=args.render_stride)
        val_ret = get_ret(
            global_step,
            args,
            model,
            val_ray_sample,
            projector,
            # gt_img,
            render_stride=args.render_stride,
            # prefix="val/",
            # out_folder=out_folder,
            # ret_alpha=args.N_importance > 0,
            ret_alpha=ret_alpha,
            single_net=args.single_net,
            sky_style_code=sky_style_code,
            # sky_style_model=sky_style_model,
            sky_model=sky_model,
        )
        update_depth_value = val_ret['outputs_coarse']['depth'][None, ...]
        # val_data['depth_value'] = update_depth_value

        val_updated_depth_values[val_data['idx'][0], ...] = update_depth_value

        print('end of update prior depth value of image_{} in val dataset'.format(val_data['idx'][0]))

    dt = time.time() - time0
    print('total update time is {:.05f} seconds'.format(dt))
    return train_updated_depth_values, val_updated_depth_values