import os
import time
import numpy as np
import shutil
import torch
import torch.utils.data.distributed

from torch.utils.data import DataLoader

from model_and_model_component.data_loaders import dataset_dict
from ZYW_model.render_ray_ZYW import render_rays
from ZYW_model.render_image_ZYW import render_single_image
from ZYW_model.model_ZYW import Model
from model_and_model_component.sample_ray_LinGaoyuan import RaySamplerSingleImage
from ZYW_model.criterion_ZYW import Criterion
from utils import img2mse, mse2psnr, img_HWC2CHW, colorize, cycle, img2psnr
import config
import torch.distributed as dist
from model_and_model_component.projection import Projector
from model_and_model_component.data_loaders.create_training_dataset import create_training_dataset
import imageio
from PIL import Image

import torchvision

import matplotlib.pyplot as plt
import cv2

from LinGaoyuan_function.sky_network import SKYMLP, StyleMLP, SkyModel
from LinGaoyuan_function.sky_transformer_network import SkyTransformer, SkyTransformerModel
from LinGaoyuan_function.update_prior_depth_value import update_prior_depth_value
from LinGaoyuan_function.image_resize import resize_img
import json

from ZYW_model.ReTR_model_ZYW import ZYW_ReTR_model
from LinGaoyuan_function.ReTR_function.ReTR_feature_extractor import FPN_FeatureExtractor
from LinGaoyuan_function.ReTR_function.ReTR_feature_volume import FeatureVolume

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def train(args):


    device = "cuda:{}".format(args.local_rank)
    out_folder = os.path.join(args.rootdir, "out", args.expname)
    print("outputs will be saved to {}".format(out_folder))
    os.makedirs(out_folder, exist_ok=True)

    # save the args and config files
    f = os.path.join(out_folder, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))

    if args.config is not None:
        f = os.path.join(out_folder, "config.txt")
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

    # create training dataset
    train_dataset, train_sampler = create_training_dataset(args)
    # currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
    # please use distributed parallel on multiple GPUs to train multiple target views per batch
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        worker_init_fn=lambda _: np.random.seed(),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=True if train_sampler is None else False,
    )

    'Lingaoyuan_operation_20240907: add a image resizer if args.resize_image is True'
    'LinGaoyuan_operation_20240905: create a tensor array that save the prior depth value of all training image'
    if args.resize_image is True:
        image_resizer = torchvision.transforms.Resize((args.image_resize_H, args.image_resize_W))# Zhenyi Wan [2025/4/10] Resize a fig into[args.image_resize_H, args.image_resize_W]
        train_prior_depth_values = torch.zeros((len(train_loader), args.image_resize_H, args.image_resize_W)).cuda()# Zhenyi Wan [2025/4/10] create a tensor to store prior values
        #[len(train_loader), args.image_resize_H, args.image_resize_W]
    else:
        train_prior_depth_values = torch.zeros((len(train_loader), args.image_H, args.image_W)).cuda()# Zhenyi Wan [2025/4/10] [len(train_loader), args.image_H, args.image_W]

    '''
    LinGaoyuan_operation_20240920: if exit a saved train_prior_depth_values.pt file in actual out_folder, load it as train_prior_depth_values
    args.save_prior_depth is set to false by default
    '''
    if os.path.exists(os.path.join(out_folder, "train_prior_depth_values.pt")) is True and args.save_prior_depth is True:
        print('laad exist train_prior_depth_values from {}'.format(os.path.join(out_folder, "train_prior_depth_values.pt")))
        train_prior_depth_values = torch.load(os.path.join(out_folder, "train_prior_depth_values.pt"))
    else:# Zhenyi Wan [2025/4/10] if donot have a prior depth value
        print('create initial train_prior_depth_values')
        for idx in range(len(train_dataset.depth_value_files)):
            with open(train_dataset.depth_value_files[idx], 'r') as f:
                depth_value_dictionary = json.load(f)# Zhenyi Wan [2025/4/10] save the depth value in a dictionary
            train_depth_value = torch.tensor(depth_value_dictionary['depth_value_pred'])[None, ...]# Zhenyi Wan [2025/4/10] turn the depth value into pytorch tensor.[1,N]

            if args.resize_image is True:
                train_depth_value = resize_img(train_depth_value, image_resizer)

            train_prior_depth_values[idx, ...] = train_depth_value



    # create validation dataset
    val_dataset = dataset_dict[args.eval_dataset](args, "validation", scenes=args.eval_scenes)

    val_loader = DataLoader(val_dataset, batch_size=1)
    val_loader_iterator = iter(cycle(val_loader))

    'LinGaoyuan_operation_20240905: create a tensor array that save the prior depth value of all val image'
    if args.resize_image is True:
        val_prior_depth_values = torch.zeros((len(val_loader), args.image_resize_H, args.image_resize_W)).cuda()
    else:
        val_prior_depth_values = torch.zeros((len(val_loader), args.image_H, args.image_W)).cuda()

    for idx in range(len(val_dataset.depth_value_files)):
        with open(val_dataset.depth_value_files[idx], 'r') as f:
            depth_value_dictionary = json.load(f)
        val_depth_value = torch.tensor(depth_value_dictionary['depth_value_pred'])[None, ...]

        if args.resize_image is True:
            val_depth_value = resize_img(val_depth_value, image_resizer)

        val_prior_depth_values[idx, ...] = val_depth_value


    # Create GNT model
    '''
    LinGaoyuan_operation_20240918: now the model can contain retr model according to the indicator of args
    '''
    model = Model(
        args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler
    )

    # # # create ReTR model
    # retr_model = LinGaoyuan_ReTR_model(args, in_feat_ch=32, posenc_dim=3, viewenc_dim=3, ret_alpha=False, use_volume_feature=args.use_volume_feature).to(device)
    #
    # # # Create ReTR feature extractor
    # retr_feature_extractor = FPN_FeatureExtractor(out_ch=32).to(device)
    #
    # retr_feature_volume = FeatureVolume(volume_reso=100).to(device)


    # create projector
    projector = Projector(device=device)

    # Create criterion
    criterion = Criterion()
    scalars_to_log = {}

    global_step = model.start_step + 1
    epoch = 0

    'model for sky color'

    if args.sky_model_type == 'mlp':
        sky_model = SkyModel(args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler)
    else:
        sky_model = SkyTransformerModel(args, ray_d_input=3, style_dim=128, dim_embed=256, num_head=2, input_embedding=False,
                                   dropout=0.05, num_layers=5).to(device)

    # 'two model for sky color'
    # style_dims = 128
    # # batch_size = args.num_source_views
    # batch_size = 1  # only one image will be used in each iteration of train_loader
    # sky_style_model = StyleMLP(style_dim = style_dims, out_dim = style_dims).to(device)
    # 'each ray_d has three coordinate, each ray_d through the SKYMLP will map to a rgb value'
    # print("sky_model_type: {}".format(args.sky_model_type))
    # if args.sky_model_type == 'mlp':
    #     sky_model = SKYMLP(in_channels = 3, style_dim=style_dims).to(device)
    # else:
    #     sky_model = SkyTransformer(ray_d_input=3, style_dim=128, dim_embed=256, num_head=2, input_embedding = False,
    #                                dropout=0.05, num_layers=5).to(device)
    #
    # sky_optimizer = torch.optim.Adam(sky_model.parameters(), lr=args.lrate_sky_model)
    # sky_style_optimizer = torch.optim.Adam(sky_style_model.parameters(), lr=args.lrate_sky_style_model)
    #
    # sky_scheduler = torch.optim.lr_scheduler.StepLR(
    #     sky_optimizer, step_size=args.lrate_decay_steps_sky_model, gamma=args.lrate_decay_factor_sky_model
    # )
    #
    # sky_style_scheduler = torch.optim.lr_scheduler.StepLR(
    #     sky_style_optimizer, step_size=args.lrate_decay_steps_sky_style_model, gamma=args.lrate_decay_factor_sky_style_model
    # )

    style_dims = 128
    # batch_size = args.num_source_views
    batch_size = 1  # only one image will be used in each iteration of train_loader

    if sky_model.start_step > 0:
        print('load pretrained sky style code')
        z = sky_model.sky_style_code
    else:
        print('create initial sky style code')
        z = torch.randn(batch_size, style_dims, dtype=torch.float32, device=device)

    # while global_step < model.start_step + args.n_iters + 1:
    for epoch in range(args.max_epochs):
        np.random.seed()

        'LinGaoyuan_operation_20240920: calculate right epoch when load ckpt'
        load_epoch = int(global_step/len(train_loader))

        print('----------------------------------The {}st epoch begin----------------------------------'.format(epoch))

        epoch_step = 0

        total_time_NeRO = 0.0
        total_time_NeILF = 0.0

        for train_data in train_loader:

            time0 = time.time()

            if args.distributed:
                train_sampler.set_epoch(epoch)

            # Start of core optimization loop

            'render_stride is not used during training'
            # load training rays
            ray_sampler = RaySamplerSingleImage(train_data, device)
            N_rand = int(
                1.0 * args.N_rand * args.num_source_views / train_data["src_rgbs"][0].shape[0]
            )# Zhenyi Wan [2025/4/10] Dynamically adjusts the number of rays sampled according to the number of source images and the number of pixels per image
            ray_batch = ray_sampler.random_sample(
                N_rand,
                sample_mode=args.sample_mode,
                center_ratio=args.center_ratio,
            )# Zhenyi Wan [2025/4/10] Randomly sample rays

            # Zhenyi Wan [2025/4/10] Extract featmaps from scr_rgbs. Use either ReTR method or GNT method
            if args.use_retr_feature_extractor is False:# Zhenyi Wan [2025/4/10] from GNT: ResUNET
                featmaps = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
            else:# Zhenyi Wan [2025/4/10] from ReTR: FPN_FeatureExtractor
                featmaps, fpn = model.retr_feature_extractor(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
            if args.use_volume_feature is True and args.use_retr_feature_extractor is True:# Zhenyi Wan [2025/4/10] from ReTR: feature volume
                feature_volume = model.retr_feature_volume(fpn, ray_batch)
            else:
                feature_volume = None

            'LinGaoyuan_operation_20240830: set self.ret_alpha = True in order to always return depth prediction'

            # ret_alpha = args.N_importance > 0
            ret_alpha = True

            'LinGaoyuan_operation_20240905: set a indicator use_updated_prior_depth = True if epoch reach preset value'
            update_prior_depth_indicator = True
            '''v
            LinGaoyuan_operation_20240920: use load_epoch in if condition because the epoch from for_loop can not represent the 
            right epoch of model when this model is loaded from exist checkpoint. the right epoch of model = global_step/length of train dataset
            '''
            # if epoch >= args.update_prior_depth_epochs:
            if load_epoch >= args.update_prior_depth_epochs:
                use_updated_prior_depth = True
                train_depth_prior = (train_prior_depth_values[ray_batch["idx"], ...]).reshape(-1, 1)
            else:
                use_updated_prior_depth = False
                train_depth_prior = None

            if epoch == args.update_prior_depth_epochs and epoch_step == 0:
                print('The updating process of prior depth process will begin at epoch: {}'.format(epoch), 'step: {}'.format(global_step) )

            # Zhenyi Wan [2025/4/10] render part!
            ret, z = render_rays(
                args = args,
                ray_batch=ray_batch,
                model=model,
                projector=projector,
                featmaps=featmaps,
                N_samples=args.N_samples,
                inv_uniform=args.inv_uniform,
                N_importance=args.N_importance,
                det=args.det,
                white_bkgd=args.white_bkgd,
                # ret_alpha=args.N_importance > 0,
                ret_alpha=ret_alpha,
                single_net=args.single_net,
                sky_style_code=z,
                # sky_style_model = sky_style_model,
                sky_model=sky_model,
                use_updated_prior_depth=use_updated_prior_depth,
                train_depth_prior=train_depth_prior,
                feature_volume=feature_volume,
            )


            'LinGaoyuan_operation_20240906: (optional) use cov of depth from Uncle SLAM formular 5 to determine whether update depth prior or not'
            if args.cov_criteria is True:
                pred_depth_cov = ret["outputs_coarse"]["depth_cov"]
                if pred_depth_cov < args.preset_depth_cov:
                    use_updated_prior_depth = True

            # compute loss
            model.optimizer.zero_grad()

            'set sky model and style model to zero_grad'
            if args.sky_model_type == 'mlp':
                sky_model.sky_optimizer.zero_grad()
                sky_model.sky_style_optimizer.zero_grad()
            else:
                sky_model.optimizer.zero_grad()
            # sky_optimizer.zero_grad()
            # sky_style_optimizer.zero_grad()

            loss, scalars_to_log = criterion(ret["outputs_coarse"], ray_batch, scalars_to_log)

            loss_NeRO = None
            loss_NeILF = None
            if args.use_NeROPBR is True:
                # NeRO loss
                loss_NeRO, scalars_to_log = criterion.NeRO_loss(ret["color_NeRO"], ray_batch, scalars_to_log)
            if args.use_NeILFPBR is True:
                # NeILF loss
                loss_NeILF, scalars_to_log = criterion.NeILF_loss(ret["color_NeILF"], ray_batch, scalars_to_log)

            'loss of sky area, add by LinGaoyuan'
            loss_sky_rgb = criterion.sky_loss_rgb(ret["outputs_coarse"], ray_batch)

            'LinGaoyuan_operation_20240907: the depth loss will not be added to total loss after epoch reach a preset value'
            if ret_alpha is True:
                loss_depth_value = criterion.depth_loss(ret["outputs_coarse"], ray_batch, train_depth_prior)
                if epoch < args.update_prior_depth_epochs:
                    loss = args.lambda_rgb * loss + args.lambda_depth * loss_depth_value

            'LinGaoyuan_operation_20240920: (optional) use depth loss to determine whether update depth prior or not'
            if args.depth_loss_criteria is True:
                if loss_depth_value < args.preset_depth_loss:
                    use_updated_prior_depth = True

            if ret["outputs_fine"] is not None:
                fine_loss, scalars_to_log = criterion(
                    ret["outputs_fine"], ray_batch, scalars_to_log
                )
                loss += fine_loss

            'LinGaoyuan_operation_20240905: update prior depth with depth prediction if epoch reach preset value'
            'LinGaoyuan_operation_20240920: add a indicator(args.update_prior_depth) to determine whether update depth prior or not'
            if use_updated_prior_depth and args.update_prior_depth is True:
                train_depth_pred = ret["outputs_coarse"]["depth"].detach() # Zhenyi Wan [2025/4/16] (N_rand,)
                # Zhenyi Wan [2025/4/16] write back the depth map for prior depth
                train_depth_prior[ray_batch["selected_inds"]] = train_depth_pred[...,None]
                if args.resize_image is True:
                    train_depth_prior = train_depth_prior.reshape(1, args.image_resize_H, args.image_resize_W)
                else:
                    train_depth_prior = train_depth_prior.reshape(1,args.image_H, args.image_W)
                train_prior_depth_values[ray_batch["idx"], ...] = train_depth_prior
                # print('finish update train prior depth value in epoch: {}'.format(epoch), 'step: {}'.format(global_step))

            #loss.backward()
            total_loss = loss
            if loss_NeRO is not None:# Zhenyi Wan [2025/4/16] Add the PBR loss to total loss for backward
                total_loss += loss_NeRO
            if loss_NeILF is not None:
                total_loss += loss_NeILF
            total_loss.backward()

            scalars_to_log["loss"] = loss.item()
            if loss_NeRO is not None:# Zhenyi Wan [2025/4/16] Add PBr loss to log
                scalars_to_log["loss_NeRO"] = loss_NeRO.item()
            if loss_NeILF is not None:
                scalars_to_log["loss_NeILF"] = loss_NeILF.item()
            model.optimizer.step()
            model.scheduler.step()

            loss_sky_rgb.backward()
            if args.sky_model_type == 'mlp':
                sky_model.sky_optimizer.step()
                sky_model.sky_scheduler.step()
            else:
                sky_model.optimizer.step()
                sky_model.scheduler.step()
            # sky_optimizer.step()
            # sky_scheduler.step()

            if args.sky_model_type == 'mlp':
                sky_model.sky_style_optimizer.step()
                sky_model.sky_style_scheduler.step()
            # sky_style_optimizer.step()
            # sky_style_scheduler.step()

            scalars_to_log["lr"] = model.scheduler.get_last_lr()[0]
            # end of core optimization loop
            dt = time.time() - time0

            # Track PBR computation times
            if ret.get("color_NeRO") is not None:
                total_time_NeRO += ret["color_NeRO"]["time_NeRO"]
            if ret.get("color_NeILF") is not None:
                total_time_NeILF += ret["color_NeILF"]["time_NeILF"]


            if args.sky_model_type == 'mlp':
                sky_model_lr = sky_model.sky_scheduler.get_last_lr()[0]
                sky_style_lr = sky_model.sky_style_scheduler.get_last_lr()[0]
            else:
                sky_model_lr = sky_model.scheduler.get_last_lr()[0]
                sky_style_lr = sky_model.scheduler.get_last_lr()[1]
            # sky_model_lr = sky_scheduler.get_last_lr()[0]
            # sky_style_lr = sky_style_scheduler.get_last_lr()[0]

            # Rest is logging
            if args.local_rank == 0:
                if global_step % args.i_print == 0 or global_step < 10:
                    # write mse and psnr stats
                    mse_error = img2mse(ret["outputs_coarse"]["rgb"], ray_batch["rgb"]).item()
                    scalars_to_log["train/coarse-loss"] = mse_error
                    scalars_to_log["train/coarse-psnr-training-batch"] = mse2psnr(mse_error)

                    # Zhenyi Wan [2025/4/16] NeROPBR loss and PSNR
                    if loss_NeRO is not None:
                        scalars_to_log["train/NeROPBR-loss"] = loss_NeRO.item()
                        scalars_to_log["train/NeROPBR-psnr-training-batch"] = mse2psnr(loss_NeRO.item())

                    # Zhenyi Wan [2025/4/16] NeILFPBR loss and PSNR
                    if loss_NeILF is not None:
                        scalars_to_log["train/NeILFPBR-loss"] = loss_NeILF.item()
                        scalars_to_log["train/NeILFPBR-psnr-training-batch"] = mse2psnr(loss_NeILF.item())

                    if ret["outputs_fine"] is not None:
                        mse_error = img2mse(ret["outputs_fine"]["rgb"], ray_batch["rgb"]).item()
                        scalars_to_log["train/fine-loss"] = mse_error
                        scalars_to_log["train/fine-psnr-training-batch"] = mse2psnr(mse_error)

                    logstr = "{} Epoch: {}  step: {} ".format(args.expname, epoch, global_step)
                    for k in scalars_to_log.keys():
                        logstr += " {}: {:.6f}".format(k, scalars_to_log[k])
                    print(logstr)

                    print("depth loss: {}".format(loss_depth_value), "depth cov: {}".format(ret["outputs_coarse"]["depth_cov"]))

                    print("sky model lr: {}".format(sky_model_lr), "sky_style_lr: {}".format(sky_style_lr), "sky_loss: {}".format(loss_sky_rgb))

                    if loss_NeRO is not None:
                        print(">>> NeROPBR loss: {:.6f}".format(loss_NeRO))

                    if loss_NeILF is not None:
                        print(">>> NeILFPBR loss: {:.6f}".format(loss_NeILF))

                    print("each iter time {:.05f} seconds".format(dt))
                    print(f"Total NeRO_PBR time: {total_time_NeRO:.2f} seconds.")
                    print(f"Total NeILF_PBR time: {total_time_NeILF:.2f} seconds.")

                if global_step % args.i_weights == 0: # Zhenyi Wan [2025/4/16] every i_weights step save the model
                    print("Saving checkpoints at {} to {}...".format(global_step, out_folder))
                    fpath = os.path.join(out_folder, "model_{:06d}.pth".format(global_step))
                    model.save_model(fpath)

                    print("Saving checkpoints of sky model at {} to {}...".format(global_step, out_folder))
                    fpath = os.path.join(out_folder, "sky_model_{:06d}.pth".format(global_step))
                    sky_model.save_model(fpath)

                    'LinGaoyuan_operation_20240920: save train_prior_depth_values as .pt file when args.save_prior_depth is True'
                    if args.save_prior_depth is True:
                        print("Saving train_prior_depth_values at {} to {}...".format(global_step, out_folder))
                        fpath = os.path.join(out_folder, "train_prior_depth_values.pt")
                        torch.save(train_prior_depth_values, fpath)





                'LinGaoyuan_20240905_operation: add a function that can update prior depth value of each image'
                'LinGaoyuan_20240906_operation: use another approach for updating prior depth during each training iteration'
                # if global_step % args.i_prior_depth_update == 0 and args.update_prior_depth is True:
                #     print("updating prior depth value of all images")
                #
                #     train_updated_depth_values, val_updated_depth_values = update_prior_depth_value(
                #         global_step,
                #         train_loader,
                #         val_loader,
                #         args,
                #         model,
                #         # tmp_ray_sampler,
                #         projector,
                #         # gt_img,
                #         render_stride=args.render_stride,
                #         prefix="val/",
                #         out_folder=out_folder,
                #         # ret_alpha=args.N_importance > 0,
                #         ret_alpha=ret_alpha,
                #         single_net=args.single_net,
                #         sky_style_code=z,
                #         # sky_style_model=sky_style_model,
                #         sky_model=sky_model,
                #
                #     )
                #
                #     print("end updating prior depth value of all images")

                if global_step % args.i_img == 0:
                    print("Logging a random validation view...")



                    val_data = next(val_loader_iterator)

                    # plt.imshow((val_data['rgb'].squeeze()))
                    # plt.show()

                    # plt.imshow((val_data['depth_value']).squeeze()*255)
                    # plt.show()

                    print(val_data['rgb_path'], val_data['depth_value_path'])


                    'render_stride is used during validation'
                    tmp_ray_sampler = RaySamplerSingleImage(
                        val_data, device, render_stride=args.render_stride
                    )
                    H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
                    gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)

                    'LinGaoyuan_operation_20240830: set create depth image by default even N_importance is 0 in order to create depth image'
                    'LinGaoyuan_operation_20240920: set data_mode to val when use val dataset in val process'
                    log_view(
                        global_step,
                        args,
                        model,
                        tmp_ray_sampler,
                        projector,
                        gt_img,
                        render_stride=args.render_stride,
                        prefix="val/",
                        out_folder=out_folder,
                        # ret_alpha=args.N_importance > 0,
                        ret_alpha=ret_alpha,
                        single_net=args.single_net,
                        sky_style_code=z,
                        # sky_style_model=sky_style_model,
                        sky_model=sky_model,
                        use_updated_prior_depth=use_updated_prior_depth,
                        data_mode='val',
                    )
                    torch.cuda.empty_cache()

                    print("Logging current training view...")
                    tmp_ray_train_sampler = RaySamplerSingleImage(
                        train_data, device, render_stride=1
                    )
                    H, W = tmp_ray_train_sampler.H, tmp_ray_train_sampler.W
                    gt_img = tmp_ray_train_sampler.rgb.reshape(H, W, 3)

                    # filename_gt = os.path.join(out_folder, 'img_gt.png')
                    # torchvision.io.write_png(torch.tensor(gt_img * 255).to(torch.uint8).permute(2, 0, 1), filename_gt)

                    'LinGaoyuan_operation_20240830: set create depth image by default even N_importance is 0'
                    '''
                    LinGaoyuan_operation_20240918: use updated prior depth if use train sampler and use_updated_prior_depth is True. 
                    set data_mode=train in this situation
                    '''
                    log_view(
                        global_step,
                        args,
                        model,
                        tmp_ray_train_sampler,
                        projector,
                        gt_img,
                        render_stride=1,
                        prefix="train/",
                        out_folder=out_folder,
                        # ret_alpha=args.N_importance > 0,
                        ret_alpha=ret_alpha,
                        single_net=args.single_net,
                        sky_style_code=z,
                        # sky_style_model=sky_style_model,
                        sky_model=sky_model,
                        data_mode='train',
                        train_prior_depth_values=train_prior_depth_values,
                        use_updated_prior_depth=use_updated_prior_depth,
                    )
            global_step += 1

            epoch_step += 1

            if global_step > model.start_step + args.n_iters + 1:
                break
        # epoch += 1


@torch.no_grad()
def log_view(
    global_step,
    args,
    model,
    ray_sampler,
    projector,
    gt_img,
    render_stride=1,
    prefix="",
    out_folder="",
    ret_alpha=False,
    single_net=True,
    sky_style_code=None,
    sky_style_model=None,
    sky_model=None,
    data_mode=None,
    train_prior_depth_values=None,
    use_updated_prior_depth=False,
):
    model.switch_to_eval() # Zhenyi Wan [2025/4/17] switch to validation mode
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()

        # if model.feature_net is not None:
        #     featmaps = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
        # else:
        #     featmaps = [None, None]

        '''
        LinGaoyuan_operation_20240918: In val process, if the data is from training dataset(which mean data_mode is train) 
        and use_updated_prior_depth is True, the updated prior depth will be used.
        '''
        if data_mode == 'train' and use_updated_prior_depth is True:
            print('use updated prior depth for training dataset in val process')
            train_depth_prior = (train_prior_depth_values[ray_batch["idx"], ...]).reshape(-1, 1) # Zhenyi Wan [2025/4/17] [N_rays,1]
        else:
            train_depth_prior = None

        if args.use_retr_feature_extractor is False:
            featmaps = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
        else:
            featmaps, fpn = model.retr_feature_extractor(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
        if args.use_volume_feature is True and args.use_retr_feature_extractor is True:
            feature_volume = model.retr_feature_volume(fpn, ray_batch)
        else:
            feature_volume = None

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
            feature_volume=feature_volume,
            use_updated_prior_depth=use_updated_prior_depth,
            train_depth_prior=train_depth_prior,
            data_mode=data_mode,
        )

    color_NeRO = None
    color_NeILF = None
    roughness_map = None
    metallic_map = None
    albedo_map = None
    normals_map = None
    BRDF_im = None

    average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1)) # Zhenyi Wan [2025/4/17] Average to every batch and view [H,W,3]

    'get the mask of sky area'
    H = ray_sampler.H
    W = ray_sampler.W
    sky_mask = ray_sampler.sky_mask.reshape(H, W, 1) # Zhenyi Wan [2025/4/17] [H,W,1]
    # sky_mask = sky_mask.permute(1, 2, 0)

    depth_value = ray_sampler.depth_value.reshape(H, W, 1).squeeze().cpu()
    # plt.imshow(depth_value)
    # plt.show()

    # Zhenyi Wan [2025/4/17] down sampling
    if args.render_stride != 1:
        print(gt_img.shape)
        gt_img = gt_img[::render_stride, ::render_stride] # Zhenyi Wan [2025/4/17] [H//s, W//s, 3]
        average_im = average_im[::render_stride, ::render_stride]# Zhenyi Wan [2025/4/17] [H//s, W//s, 3]

        sky_mask = sky_mask[::render_stride, ::render_stride] # Zhenyi Wan [2025/4/17] [H//s, W//s, 3]

    rgb_gt = img_HWC2CHW(gt_img) # Zhenyi Wan [2025/4/17] [3,H//s,W//s]
    average_im = img_HWC2CHW(average_im)# Zhenyi Wan [2025/4/17] [3,H//s,W//s]

    rgb_pred = img_HWC2CHW(ret["outputs_coarse"]["rgb"].detach().cpu())# Zhenyi Wan [2025/4/17] [3,H//s,W//s]
    if ret.get("color_NeRO") is not None:
        color_NeRO = img_HWC2CHW(ret["color_NeRO"].detach().cpu())

    if ret.get("color_NeILF") is not None:
        color_NeILF = img_HWC2CHW(ret["color_NeILF"].detach().cpu())

    if ret.get("outputs_roughness") is not None and ret["outputs_roughness"].get("roughness") is not None:
        roughness_map = img_HWC2CHW(ret["outputs_roughness"]["roughness"].detach().cpu())  # [1,H//s,W//s]

    if ret.get("outputs_metallic") is not None and ret["outputs_metallic"].get("metallic") is not None:
        metallic_map = img_HWC2CHW(ret["outputs_metallic"]["metallic"].detach().cpu())  # [1,H//s,W//s]

    if ret.get("outputs_albedo") is not None and ret["outputs_albedo"].get("albedo") is not None:
        albedo_map = img_HWC2CHW(ret["outputs_albedo"]["albedo"].detach().cpu())  # [3,H//s,W//s]

    if ret.get("outputs_normals") is not None and ret["outputs_normals"].get("normals") is not None:
        normals_map = img_HWC2CHW(ret["outputs_normals"]["normals"].detach().cpu())  # [3,H//s,W//s]

    rgb_sky = img_HWC2CHW(ret["outputs_coarse"]["rgb_sky"].detach().cpu())

    print("sky rgb max value: {}".format(rgb_sky.max()), "sky rgb min value: {}".format(rgb_sky.min()))
    # plt.imshow((rgb_sky*255).permute(1, 2, 0))
    # plt.show()

    rgb_sky = torch.clamp(rgb_sky, 0, 1)

    # softmax_func = torch.nn.Softmax(dim=2)
    # rgb_sky = softmax_func(rgb_sky)

    sky_mask = sky_mask.permute(2,0,1)# Zhenyi Wan [2025/4/17] [3, H//s, W//s]
    'the sky color should be considered to different color combination'
    # sky_color = torch.ones((3,), dtype=rgb_pred.type)
    # sky_color = torch.ones((3,))
    # sky_color = torch.tensor([203/255, 206/255, 211/255])

    rgb_pred = rgb_pred * sky_mask + rgb_sky * (1 - sky_mask)
    if color_NeRO is not None:
        color_NeRO = color_NeRO * sky_mask + rgb_sky * (1 - sky_mask)
    if color_NeILF is not None:
        color_NeILF = color_NeILF * sky_mask + rgb_sky * (1 - sky_mask)
    if roughness_map is not None:
        roughness_map = roughness_map * sky_mask + 1 * (1 - sky_mask) # Zhenyi Wan [2025/4/17] In roughness map the sky is treated as maximum roughness
    if metallic_map is not None:
        metallic_map = metallic_map * sky_mask + 0 * (1 - sky_mask)# Zhenyi Wan [2025/4/17] [1,H//s,W//s]The sky is treated as no metallic
    if albedo_map is not None:
        albedo_map = albedo_map * sky_mask + rgb_sky * (1 - sky_mask)  # Zhenyi Wan [2025/4/17] [3,H//s,W//s]
    if normals_map is not None:
        default_normal = torch.tensor([0.0, 0.0, 1.0], device=normals_map.device).view(3, 1, 1) # Zhenyi Wan [2025/4/18] Set the normals of the sky as default in z-direction
        normals_map = normals_map * sky_mask + default_normal * (1 - sky_mask)  # Zhenyi Wan [2025/4/17] [3,H//s,W//s]


    h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], average_im.shape[-2])
    w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], average_im.shape[-1])

    # rgb_im = torch.zeros(3, h_max, 3 * w_max)
    # rgb_im[:, : average_im.shape[-2], : average_im.shape[-1]] = average_im
    # rgb_im[:, : rgb_gt.shape[-2], w_max : w_max + rgb_gt.shape[-1]] = rgb_gt
    # rgb_im[:, : rgb_pred.shape[-2], 2 * w_max : 2 * w_max + rgb_pred.shape[-1]] = rgb_pred

    rgb_im = torch.zeros(3, h_max, 2 * w_max) # Zhenyi Wan [2025/4/18] create a black background[3,H_max,2*max]
    rgb_im[:, : rgb_gt.shape[-2], : rgb_gt.shape[-1]] = rgb_gt # Zhenyi Wan [2025/4/18] place the rgb_gt in the upper left corner of the background
    rgb_im[:, : rgb_pred.shape[-2], w_max: w_max + rgb_pred.shape[-1]] = rgb_pred
    if color_NeRO is not None:
        color_NeRO_ = torch.zeros(3, h_max, w_max)
        color_NeRO_[:, : color_NeRO.shape[-2], : color_NeRO.shape[-1]] = color_NeRO
        rgb_im = torch.cat((rgb_im, color_NeRO_), dim=-1)
    if color_NeILF is not None:
        color_NeILF_ = torch.zeros(3, h_max, w_max)
        color_NeILF_[:, : color_NeILF.shape[-2], : color_NeILF.shape[-1]] = color_NeILF
        rgb_im = torch.cat((rgb_im, color_NeILF_), dim=-1)


    if roughness_map is not None and metallic_map is not None and albedo_map is not None and normals_map is not None:
        BRDF_im = torch.zeros(3, h_max, 4 * w_max)  # Zhenyi Wan [2025/4/18] create a black background[3,H_max,4*max]
        BRDF_im[:, : roughness_map.shape[-2], : roughness_map.shape[-1]] = roughness_map.repeat(3, 1, 1)  # Zhenyi Wan [2025/4/18] The third fig is roughness_map after converting to 3-channel
        BRDF_im[:, : metallic_map.shape[-2], w_max: w_max + metallic_map.shape[-1]] = metallic_map.repeat(3, 1, 1)  # Zhenyi Wan [2025/4/18] The fourth fig is metallic_map after converting to 3-channel
        BRDF_im[:, : albedo_map.shape[-2], 2 * w_max: 2 * w_max + albedo_map.shape[-1]] = albedo_map  # Zhenyi Wan [2025/4/18] The fifth fig is albedo map
        BRDF_im[:, : normals_map.shape[-2], 3 * w_max: 3 * w_max + normals_map.shape[-1]] = normals_map  # Zhenyi Wan [2025/4/18] The fifth fig is normals map

    depth_matrix = {}

    if "depth" in ret["outputs_coarse"].keys():
        depth_pred = ret["outputs_coarse"]["depth"].detach().cpu()

        # print(torch.max(depth_pred), torch.min(depth_pred))
        print('depth predict max value: {}'.format(torch.max(depth_pred)), 'depth predict min value: {}'.format(torch.min(depth_pred)))

        'this depth prediction should be save as .json file'
        depth_matrix.update({'depth_value_pred': np.array(depth_pred).tolist()})

        depth_sky = torch.full_like(depth_pred, 200)
        depth_pred_replace_sky = (depth_pred * sky_mask + depth_sky * (1 - sky_mask)).squeeze()

        depth_replace_sky = ((depth_pred_replace_sky - depth_pred_replace_sky.min()) / (depth_pred_replace_sky.max() - depth_pred_replace_sky.min())) * 255.0

        'LinGaoyuan_operation_20240927: define depth_replace_sky_float for following calculation of psnr of depth prediction'
        depth_replace_sky_float = depth_replace_sky/255.0

        depth_replace_sky = depth_replace_sky.cpu().numpy().astype(np.uint8)

        # plt.imshow(depth_replace_sky)
        # plt.title('Depth img replaced sky area')
        # plt.show()

        # plt.imshow(depth_pred)
        # plt.show()
        depth = ((depth_pred - depth_pred.min()) / (depth_pred.max() - depth_pred.min())) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)
        # plt.imshow(depth)
        # plt.title('normalized depth image')
        # plt.show()
        # print(torch.min(depth_pred),torch.max(depth_pred))
        # depth_range = [torch.min(depth_pred), torch.max(depth_pred)]

        depth_im = img_HWC2CHW(colorize(depth_pred, cmap_name="jet"))
        # print(torch.max(depth_im), torch.min(depth_im))
        # plt.title('RGB depth image')
        # plt.imshow(depth_im.permute(1,2,0))
        # plt.show()

    else:
        depth_im = None

    if ret["outputs_fine"] is not None:
        rgb_fine = img_HWC2CHW(ret["outputs_fine"]["rgb"].detach().cpu())
        rgb_fine_ = torch.zeros(3, h_max, w_max)
        rgb_fine_[:, : rgb_fine.shape[-2], : rgb_fine.shape[-1]] = rgb_fine

        'update sky area for rgb_fine'
        rgb_fine_ = rgb_fine_ * sky_mask + rgb_sky * (1 - sky_mask)

        rgb_im = torch.cat((rgb_im, rgb_fine_), dim=-1)
        depth_pred = torch.cat((depth_pred, ret["outputs_fine"]["depth"].detach().cpu()), dim=-1)
        depth_im = img_HWC2CHW(colorize(depth_pred, cmap_name="jet"))

    rgb_im = rgb_im.permute(1, 2, 0).detach().cpu().numpy() # Zhenyi Wan [2025/4/19] (n*H, W, 3)
    rgb_im = np.array(Image.fromarray((rgb_im * 255).astype(np.uint8)))
    filename = os.path.join(out_folder, prefix[:-1] + "_{:03d}.png".format(global_step))

    torchvision.io.write_png(torch.tensor(rgb_im).permute(2, 0, 1), filename)

    if BRDF_im is not None:
        BRDF_im = BRDF_im.permute(1, 2, 0).detach().cpu().numpy()
        BRDF_im = np.array(Image.fromarray((BRDF_im * 255).astype(np.uint8)))
        filename = os.path.join(out_folder, prefix[:-1] + "BRDF_{:03d}.png".format(global_step))
        torchvision.io.write_png(torch.tensor(BRDF_im).permute(2, 0, 1), filename)

    # imageio.imwrite(filename, rgb_im)
    if depth_im is not None:
        depth_im = depth_im.permute(1, 2, 0).detach().cpu().numpy()
        filename = os.path.join(out_folder, prefix[:-1] + "depth_{:03d}.png".format(global_step))
        depth_im = np.array(Image.fromarray((depth_im * 255).astype(np.uint8)))

        # R = depth_im[:,:,0]
        # G = depth_im[:, :, 1]
        # B = depth_im[:, :, 2]
        # normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
        #
        # print(np.max(normalized), np.min(normalized))
        #
        # plt.imshow(normalized)
        # plt.show()
        # plt.imsave(filename, normalized, cmap='gray')

        imageio.imwrite(filename, depth_replace_sky)
        # imageio.imwrite(filename, depth_im)

        'save depth prediction matrix as .json file'
        # depth_matrix_filename = os.path.join(out_folder, prefix[:-1] + "_depth_{:03d}.json".format(global_step))
        # with open(depth_matrix_filename,'w') as f:
        #     json.dump(depth_matrix, f)

    # write scalar
    pred_rgb = (
        ret["outputs_fine"]["rgb"]
        if ret["outputs_fine"] is not None
        else ret["outputs_coarse"]["rgb"]
    )
    psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    print(prefix + "psnr_image_Original: ", psnr_curr_img)
    if color_NeRO is not None:
        psnr_NeRO_img = img2psnr(ret["color_NeRO"].detach().cpu(), gt_img)
        print(prefix + "psnr_image_NeROPBR: ", psnr_NeRO_img)
    if color_NeILF is not None:
        psnr_NeILF_img = img2psnr(ret["color_NeILF"].detach().cpu(), gt_img)
        print(prefix + "psnr_image_NeILFPBR: ", psnr_NeILF_img)




    'LinGaoyuan_operation_20240927: calculate psnr of depth image'
    depth_value_gt = ray_batch['depth_value'].reshape(H,W,1).squeeze().cpu()
    depth_gt_replace_sky = (depth_value_gt * sky_mask + depth_sky * (1 - sky_mask)).squeeze()
    depth_img_gt_replace_sky = ((depth_gt_replace_sky - depth_gt_replace_sky.min()) / (depth_gt_replace_sky.max() - depth_gt_replace_sky.min()))

    psnr_curr_depth_img = img2psnr(torch.tensor(depth_replace_sky_float).detach().cpu(), depth_img_gt_replace_sky.cpu())
    print(prefix + "psnr_depth_image: ", psnr_curr_depth_img)

    model.switch_to_train()


if __name__ == "__main__":
    parser = config.config_parser()
    args = parser.parse_args()

    # print(args.aliasing_filter, args.contraction_type, args.aliasing_filter_type)

    if args.distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.local_rank = int(os.environ.get("LOCAL_RANK"))
        torch.cuda.set_device(args.local_rank)

    train(args)
