import os
import time
import numpy as np
import shutil
import torch
import torch.utils.data.distributed

from torch.utils.data import DataLoader

from model_and_model_component.data_loaders import dataset_dict
from model_and_model_component.render_ray_LinGaoyuan_clip import render_rays
from model_and_model_component.render_image_LinGaoyuan_clip import render_single_image
from model_and_model_component.model_LinGaoyuan_clip import GNTModel
from model_and_model_component.sample_ray_LinGaoyuan import RaySamplerSingleImage
from model_and_model_component.criterion_LinGaoyuan_clip import Criterion
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

from utils import img2mse
import json

from model_and_model_component.ReTR_model_LinGaoyuan import LinGaoyuan_ReTR_model
from LinGaoyuan_function.ReTR_function.ReTR_feature_extractor import FPN_FeatureExtractor
from LinGaoyuan_function.ReTR_function.ReTR_feature_volume import FeatureVolume

from LinGaoyuan_function.clip_function import mapper, deformation, mapper_attention, Embedder, Loss_clip, Loss_clip_version_2

import clip

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
        image_resizer = torchvision.transforms.Resize((args.image_resize_H, args.image_resize_W))
        train_prior_depth_values = torch.zeros((len(train_loader), args.image_resize_H, args.image_resize_W)).cuda()
    else:
        train_prior_depth_values = torch.zeros((len(train_loader), args.image_H, args.image_W)).cuda()

    '''
    LinGaoyuan_operation_20240920: if exit a saved train_prior_depth_values.pt file in actual out_folder, load it as train_prior_depth_values
    args.save_prior_depth is set to false by default
    '''
    if os.path.exists(os.path.join(out_folder, "train_prior_depth_values.pt")) is True and args.save_prior_depth is True:
        print('laad exist train_prior_depth_values from {}'.format(os.path.join(out_folder, "train_prior_depth_values.pt")))
        train_prior_depth_values = torch.load(os.path.join(out_folder, "train_prior_depth_values.pt"))
    else:
        print('create initial train_prior_depth_values')
        for idx in range(len(train_dataset.depth_value_files)):
            with open(train_dataset.depth_value_files[idx], 'r') as f:
                depth_value_dictionary = json.load(f)
            train_depth_value = torch.tensor(depth_value_dictionary['depth_value_pred'])[None, ...]

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
    LinGaoyuan_operation_20240925: create two model for building and street
    '''
    # model = GNTModel(
    #     args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler
    # )
    # Zhenyi Wan [2025/3/12] the name is GNT, but it contains both retr and gnt model
    model_building = GNTModel(
        args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler
    )
    model_street = GNTModel(
        args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler
    )
    'LinGaoyuan_20240929: only one model for car'
    model_car = GNTModel(
        args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler, dim_clip_shape_code=128,
    )

    'LinGaoyuan_20240930: use a individual model for sky'
    model_sky = GNTModel(
        args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler
    )

    ### create clip model and shape&appearance code
    '''
    LinGaoyuan_operation_20240924: create clip feature encoder and mapper for shape code & appearance code and deformation network
    
    LinGaoyuan_operation_20240927: add car shape mapper, appearance mapper and deformation network
    
    LinGaoyuan_operation_20240929: only one model for car
    '''

    clip_model_name = "ViT-L/14@336px"
    use_attention_network_in_mapper = False
    use_torch_attention_model = False
    if clip_model_name == "ViT-L/14@336px":

        clip_model, clip_preprocess = clip.load(clip_model_name, device=device)
        if use_attention_network_in_mapper is False:
            clip_mapper_building = mapper(input_dim=768)
            clip_mapper_street = mapper(input_dim=768)

            clip_shape_mapper_car = mapper(input_dim=768)
            clip_appearance_mapper_car = mapper(input_dim=768)
        else:
            if use_torch_attention_model is False:
                clip_mapper_building = mapper_attention(input_dim=768)
                clip_mapper_street = mapper_attention(input_dim=768)

                clip_shape_mapper_car = mapper_attention(input_dim=768)
                clip_appearance_mapper_car = mapper_attention(input_dim=768)
            else:
                clip_mapper_building = torch.nn.MultiheadAttention(768, 1)
                clip_mapper_street = torch.nn.MultiheadAttention(768, 1)

                clip_shape_mapper_car = mapper_attention(input_dim=768)
                clip_appearance_mapper_car = mapper_attention(input_dim=768)
    elif clip_model_name == "ViT-B/32":
        clip_model, clip_preprocess = clip.load(clip_model_name, device=device)
        if use_attention_network_in_mapper is False:
            clip_mapper_building  = mapper(input_dim=512)
            clip_mapper_street = mapper(input_dim=512)

            clip_shape_mapper_car = mapper(input_dim=512)
            clip_appearance_mapper_car = mapper(input_dim=512)
        else:
            if use_torch_attention_model is False:
                clip_mapper_building = mapper_attention(input_dim=512)
                clip_mapper_street = mapper_attention(input_dim=512)

                clip_shape_mapper_car = mapper_attention(input_dim=512)
                clip_appearance_mapper_car = mapper_attention(input_dim=512)
            else:
                clip_mapper_building = torch.nn.MultiheadAttention(512, 1)
                clip_mapper_street = torch.nn.MultiheadAttention(512, 1)

                clip_shape_mapper_car = torch.nn.MultiheadAttention(512, 1)
                clip_appearance_mapper_car = torch.nn.MultiheadAttention(512, 1)
    'LinGaoyuan_20240926: deformation network'
    clip_deformation_building = deformation(input_dim=63 + 128)  # dim_out_pos_emb = 63, dim_shape_code = 128
    clip_deformation_street = deformation(input_dim=63 + 128)  # dim_out_pos_emb = 63, dim_shape_code = 128

    clip_deformation_car = deformation(input_dim=63 + 128)  # dim_out_pos_emb = 63, dim_shape_code = 128
    clip_embedder = Embedder(
        input_dims=3,
        include_input=True,
        max_freq_log2=9,
        num_freqs=10,
        log_sampling=True,
        periodic_fns=[torch.sin, torch.cos],
    )
    'LinGaoyuan_20240924: create initial shape code and appearance code'
    # shape_code = torch.randn(1, 128, dtype=torch.float32, device=device)
    # appearance_code = torch.randn(1, 128, dtype=torch.float32, device=device)
    latent_code_building = torch.randn(1, 128, dtype=torch.float32, device=device)
    latent_code_street = torch.randn(1, 128, dtype=torch.float32, device=device)

    'LinGaoyuan_operation_20240929: create a several initial shape codes and appearance codes according to the number of vehicle'
    latent_shape_code_car_list = {}
    latent_appearance_code_car_list = {}
    for car_id in train_dataset['car_id']:
        latent_shape_code_car = torch.randn(1, 128, dtype=torch.float32, device=device)
        latent_appearance_code_car = torch.randn(1, 128, dtype=torch.float32, device=device)
        latent_shape_code_car_list.update({car_id:latent_shape_code_car})
        latent_appearance_code_car_list.update({car_id:latent_appearance_code_car})


    # create projector
    projector = Projector(device=device)

    # Create criterion
    criterion = Criterion()
    scalars_to_log = {}

    'LinGaoyuan_operation_20240925: use model_building to calculate global_step during clip test process'
    # global_step = model.start_step + 1
    global_step = model_building.start_step + 1


    epoch = 0

    'model for sky color'

    if args.sky_model_type == 'mlp':
        sky_model = SkyModel(args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler)
    else:
        sky_model = SkyTransformerModel(args, ray_d_input=3, style_dim=128, dim_embed=256, num_head=2, input_embedding=False,
                                   dropout=0.05, num_layers=5).to(device)

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

        for train_data in train_loader:

            time0 = time.time()

            if args.distributed:
                train_sampler.set_epoch(epoch)

            # Start of core optimization loop

            'LinGaoyuan_operation_20240925: use clip image encoder to extract feature from target image to update latent code'
            input_img = train_data['rgb'].squeeze()
            black_background = torch.zeros((3,)).cuda()
            'LinGaoyuan_20240925: TO DO: building_mask and street_mask should be added in the future'
            building_mask = None  # building area = 0, other area = 1
            input_img_building = input_img * (1.0 - building_mask) + black_background * (building_mask)
            street_mask = None  # street area = 0, other area = 1
            input_img_street = input_img * (1.0 - street_mask) + black_background * (street_mask)

            with torch.no_grad():
                clip_features_building = clip_model.encode_image(input_img_building)
                clip_features_street = clip_model.encode_image(input_img_street)

            'LinGaoyuan_20240925: (optional) normalize for the clip feature'
            clip_features_building /= clip_features_building.norm(dim=-1, keepdim=True).to(torch.float32)
            clip_features_street /= clip_features_street.norm(dim=-1, keepdim=True).to(torch.float32)


            'LinGaoyuan_operation_20240925: use MLP mapper or attention mapper to create delta latent code'
            delta_latent_code_building = clip_mapper_building(clip_features_building)
            delta_latent_code_street = clip_mapper_street(clip_features_street)


            # delta_latent_code_building = clip_attention_building_mapper(clip_features_building)
            # delta_latent_code_street = clip_attention_street_mapper(clip_features_street)

            latent_code_building = latent_code_building + delta_latent_code_building
            latent_code_street = latent_code_street + delta_latent_code_street

            'LinGaoyuan_20240927: create corresponding data for car'

            'LinGaoyuan_operation_20240929: create a list to save data from multiple cars'

            input_img_car_list = {}
            car_mask_list = {}
            for car_id in train_data['car_id']:
                car_mask = None  # car area = 0, other area = 1
                car_mask_list.update({car_id:car_mask})
                input_img_car = input_img * (1.0 - car_mask) + black_background * (car_mask)
                input_img_car_list.update({car_id:input_img_car})

                with torch.no_grad():
                    clip_features_car = clip_model.encode_image(input_img_car)

                'normalization'
                clip_features_car /= clip_features_car.norm(dim=-1, keepdim=True).to(torch.float32)

                delta_latent_shape_code_car = clip_shape_mapper_car(clip_features_car)
                delta_latent_appearance_code_car = clip_appearance_mapper_car(clip_features_car)

                latent_shape_code_car = latent_shape_code_car_list[car_id]
                latent_appearance_code_car = latent_appearance_code_car_list[car_id]

                latent_shape_code_car = latent_shape_code_car + delta_latent_shape_code_car
                latent_appearance_code_car = latent_appearance_code_car + delta_latent_appearance_code_car

                latent_shape_code_car_list[car_id] = latent_shape_code_car
                latent_appearance_code_car_list[car_id] = latent_appearance_code_car


            'LinGaoyuan_operation_20240925: the ray sample should be reconsiderated when original model divide to building and steet model'
            'render_stride is not used during training'
            'LinGaoyuan_20240930: ray_batch only used for calculate the sky loss'
            # load training rays
            ray_sampler = RaySamplerSingleImage(train_data, device)
            N_rand = int(
                1.0 * args.N_rand * args.num_source_views / train_data["src_rgbs"][0].shape[0]
            )
            ray_batch = ray_sampler.random_sample(
                N_rand,
                sample_mode=args.sample_mode,
                center_ratio=args.center_ratio,
            )

            '''
            LinGaoyuan_operation_20240925: in order to calculate the clip loss, we must get the complete rendering image, which 
            mean that the returned rendering image should has the same size as the input image. So based on this idea, I plan to generate
            two rendering image based on two seperated model, which mean that I will generated rendered_img_building and rendered_img_steet.
            these two rendered image will have the same size as input image. In other words, we need to call the render_single_image() twice 
            to seperately input the model_building and model_street
            '''
            train_data_building = train_data.copy()
            train_data_building['rgb']=input_img_building[None,...]

            'LinGaoyuan_operation_20241001: extract building area of src image according to building mask'
            'LinGaoyuan_20241001: TO DO: the building masks of src image should be add in the future, here I assume building masks of src image is exist'
            train_data_building_src_rgbs = train_data_building['src_rgbs']  # shape: (batch, num_src, H, W, 3)
            train_data_building_src_building_masks = train_data_building['src_building_masks']  # shape: (batch, num_src, H, W)
            num_src = train_data_building_src_rgbs.shape[1]
            for i in range(num_src):
                src_building_mask = train_data_building_src_building_masks[:,i,:,:]
                train_data_building_src_rgbs[:,i,:,:,:] = train_data_building_src_rgbs[:,i,:,:,:] * (1.0 - src_building_mask) + black_background * (src_building_mask)
                train_data_building['src_rgbs'][:,i,:,:,:] = train_data_building_src_rgbs[:,i,:,:,:]

            ray_sampler_building = RaySamplerSingleImage(train_data_building, device)
            ray_batch_building = ray_sampler_building.get_all()


            train_data_street = train_data.copy()
            train_data_street['rgb']=input_img_street[None,...]

            'LinGaoyuan_operation_20241001: extract street area of src image according to street mask'
            'LinGaoyuan_20241001: TO DO: the street masks of src image should be add in the future, here I assume street masks of src image is exist'
            train_data_street_src_rgbs = train_data_street['src_rgbs']  # shape: (batch, num_src, H, W, 3)
            train_data_street_src_street_masks = train_data_street['src_street_masks']  # shape: (batch, num_src, H, W)
            num_src = train_data_street_src_rgbs.shape[1]
            for i in range(num_src):
                src_street_mask = train_data_street_src_street_masks[:,i,:,:]
                train_data_street_src_rgbs[:,i,:,:,:] = train_data_street_src_rgbs[:,i,:,:,:] * (1.0 - src_street_mask) + black_background * (src_street_mask)
                train_data_street['src_rgbs'][:,i,:,:,:] = train_data_street_src_rgbs[:,i,:,:,:]

            ray_sampler_street = RaySamplerSingleImage(train_data_street, device)
            ray_batch_street = ray_sampler_street.get_all()

            # if args.use_retr_feature_extractor is False:
            #     featmaps = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
            # else:
            #     featmaps, fpn = model.retr_feature_extractor(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
            # if args.use_volume_feature is True and args.use_retr_feature_extractor is True:
            #     feature_volume = model.retr_feature_volume(fpn, ray_batch)
            # else:
            #     feature_volume = None

            '''
            LinGaoyuan_operation_20240925: extract 2d feature and volume feature for building image and street image.
            TO DO: the extracted feature of building and street is the same right now, which mean that both based on a complete image.
            But this approach may not suitable for seperated model, I not sure whether should extract the feature_building only based on building image or not. 
            The street has also this problem.
            '''

            if args.use_retr_feature_extractor is False:
                featmaps_building = model_building.feature_net(ray_batch_building["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
            else:
                featmaps_building, fpn_building = model_building.retr_feature_extractor(ray_batch_building["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
            if args.use_volume_feature is True and args.use_retr_feature_extractor is True:
                feature_volume_building = model_building.retr_feature_volume(fpn_building, ray_batch_building)
            else:
                feature_volume_building = None

            if args.use_retr_feature_extractor is False:
                featmaps_street = model_street.feature_net(ray_batch_street["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
            else:
                featmaps_street, fpn_street = model_street.retr_feature_extractor(ray_batch_street["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
            if args.use_volume_feature is True and args.use_retr_feature_extractor is True:
                feature_volume_street = model_street.retr_feature_volume(fpn_street, ray_batch_street)
            else:
                feature_volume_street = None

            'LinGaoyuan_operation_20240830: set self.ret_alpha = True in order to always return depth prediction'

            # ret_alpha = args.N_importance > 0
            ret_alpha = True

            'LinGaoyuan_operation_20240905: set a indicator use_updated_prior_depth = True if epoch reach preset value'
            update_prior_depth_indicator = True
            '''
            LinGaoyuan_operation_20240920: use load_epoch in if condition because the epoch from for_loop can not represent the 
            right epoch of model when this model is loaded from exist checkpoint. the right epoch of model = global_step/length of train dataset
            
            LinGaoyuan_20240925: we will not update depth during clip test process, so uncomment following code 
            '''

            # if load_epoch >= args.update_prior_depth_epochs:
            #     use_updated_prior_depth = True
            #     train_depth_prior = (train_prior_depth_values[ray_batch["idx"], ...]).reshape(-1, 1)
            # else:
            #     use_updated_prior_depth = False
            #     train_depth_prior = None
            #
            # if epoch == args.update_prior_depth_epochs and epoch_step == 0:
            #     print('The updating process of prior depth process will begin at epoch: {}'.format(epoch), 'step: {}'.format(global_step) )
            use_updated_prior_depth = False
            train_depth_prior = None




            'LinGaoyuan_20240925: we should generated two complete rendering image during clip test process, so we use render_single_image() to replace render_rays()'
            '''
            LinGaoyuan_20240930: the render_rays() will be used only for generating the rgb of sky area, so actually we can use randomly featmaps and feature volume as input.
            because the rendering process of sky area can not be affected by featmaps and feature volume. in actually, model_sky has no effect for the sky area, the rgb of sky 
            area is generated from sky_model, the input of sky_model are only the sky style code(z) and ray_d. So the model_sky will not be optimized during the training process,
            only the sky_model will be optimized.
            '''
            ret_sky, z = render_rays(
                args = args,
                ray_batch=ray_batch,
                # model=model,
                projector=projector,
                featmaps=featmaps_building,
                N_samples=args.N_samples,
                model=model_sky,
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
                feature_volume=feature_volume_building,
            )

            'LinGaoyuan_operation_20240925: a complete image must be generated in order to calculate the clip loss'
            '------------------------------------ generate the completed rendering image of building--------------------------------------'
            ret_building = render_single_image(
                args,
                ray_sampler=ray_sampler_building,
                ray_batch=ray_batch_building,
                projector=projector,
                chunk_size=args.chunk_size,
                N_samples=args.N_samples,
                model=model_building,
                inv_uniform=args.inv_uniform,
                det=True,
                N_importance=args.N_importance,
                white_bkgd=args.white_bkgd,
                render_stride=args.render_stride,
                featmaps=featmaps_building,
                ret_alpha=ret_alpha,
                single_net=args.single_net,
                sky_style_code=z,
                # sky_style_model=sky_style_model,
                sky_model=sky_model,
                feature_volume=feature_volume_building,
                mode='train',
                use_updated_prior_depth=use_updated_prior_depth,
                train_depth_prior=train_depth_prior,
                data_mode='train',
                latent_code=latent_code_building,
                deformation_network=clip_deformation_building,
                pos_embedder=clip_embedder,
                # model_building=model_building,
                # model_street=model_street,
                # featmaps_building=featmaps_building,
                # featmaps_street=featmaps_street,
                # feature_volume_building=feature_volume_building,
                # feature_volume_street=feature_volume_street,
                # latent_code_building=latent_code_building,
                # latent_code_street=latent_code_street,
            )

            '------------------------------------ generate the completed rendering image of street--------------------------------------'
            ret_street = render_single_image(
                args,
                ray_sampler=ray_sampler_street,
                ray_batch=ray_batch_street,
                projector=projector,
                chunk_size=args.chunk_size,
                N_samples=args.N_samples,
                model=model_street,
                inv_uniform=args.inv_uniform,
                det=True,
                N_importance=args.N_importance,
                white_bkgd=args.white_bkgd,
                render_stride=args.render_stride,
                featmaps=featmaps_street,
                ret_alpha=ret_alpha,
                single_net=args.single_net,
                sky_style_code=z,
                # sky_style_model=sky_style_model,
                sky_model=sky_model,
                feature_volume=feature_volume_street,
                mode='train',
                use_updated_prior_depth=use_updated_prior_depth,
                train_depth_prior=train_depth_prior,
                data_mode='train',
                latent_code=latent_code_street,
                deformation_network=clip_deformation_street,
                pos_embedder=clip_embedder,
                # model_building=model_building,
                # model_street=model_street,
                # featmaps_building=featmaps_building,
                # featmaps_street=featmaps_street,
                # feature_volume_building=feature_volume_building,
                # feature_volume_street=feature_volume_street,
                # latent_code_building=latent_code_building,
                # latent_code_street=latent_code_street,
            )

            'LinGaoyuan_operation_20240929: add the code related to the generation of rendering image of car'
            '------------------------------------ generate the completed rendering image of cars--------------------------------------'
            ret_cars_list = {}
            for car_id in train_data['car_id']:
                'LinGaoyuan_operation_20240929: create ray_batch for car object'
                train_data_car = train_data.copy()
                input_img_car = input_img_car_list[car_id]
                train_data_car['rgb'] = input_img_car[None, ...]

                'LinGaoyuan_operation_20241001: extract car area of src image according to car mask'
                'LinGaoyuan_20241001: TO DO: the car masks of src image should be add in the future, here I assume car masks of src image is exist'
                train_data_car_src_rgbs = train_data_car['src_rgbs'][car_id]   # shape: (batch, num_src, H, W, 3)
                train_data_car_src_car_masks = train_data_car[
                    'src_car_masks'][car_id]  # shape: (batch, num_src, H, W)
                num_src = train_data_car_src_rgbs.shape[1]
                for i in range(num_src):
                    src_car_mask = train_data_car_src_car_masks[:, i, :, :]
                    train_data_car_src_rgbs[:, i, :, :, :] = train_data_car_src_rgbs[:, i, :, :, :] * (
                                1.0 - src_car_mask) + black_background * (src_car_mask)
                    train_data_car['src_rgbs'][car_id][:, i, :, :, :] = train_data_car_src_rgbs[:, i, :, :, :]

                ray_sampler_car = RaySamplerSingleImage(train_data_car, device)
                ray_batch_car = ray_sampler_car.get_all()

                'LinGaoyuan_operation_20240929: create featmap and feature volume for car object'
                if args.use_retr_feature_extractor is False:
                    featmaps_car = model_car.feature_net(
                        ray_batch_car["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
                else:
                    featmaps_car, fpn_car = model_car.retr_feature_extractor(
                        ray_batch_car["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
                if args.use_volume_car is True and args.use_retr_feature_extractor is True:
                    feature_volume_car = model_car.retr_feature_volume(fpn_car, ray_batch_car)
                else:
                    feature_volume_car = None

                latent_shape_code_car = latent_shape_code_car_list[car_id]
                latent_appearance_code_car = latent_appearance_code_car_list[car_id]

                ret_car = render_single_image(
                    args,
                    ray_sampler=ray_sampler_car,
                    ray_batch=ray_batch_car,
                    projector=projector,
                    chunk_size=args.chunk_size,
                    N_samples=args.N_samples,
                    model=model_car,
                    inv_uniform=args.inv_uniform,
                    det=True,
                    N_importance=args.N_importance,
                    white_bkgd=args.white_bkgd,
                    render_stride=args.render_stride,
                    featmaps=featmaps_car,
                    ret_alpha=ret_alpha,
                    single_net=args.single_net,
                    sky_style_code=z,
                    # sky_style_model=sky_style_model,
                    sky_model=sky_model,
                    feature_volume=feature_volume_car,
                    mode='train',
                    use_updated_prior_depth=use_updated_prior_depth,
                    train_depth_prior=train_depth_prior,
                    data_mode='train',
                    # latent_code=latent_code_car,
                    deformation_network=clip_deformation_car,
                    pos_embedder=clip_embedder,
                    latent_shape_code=latent_shape_code_car,
                    latent_appearance_code=latent_appearance_code_car,
                    # model_building=model_building,
                    # model_street=model_street,
                    # featmaps_building=featmaps_building,
                    # featmaps_street=featmaps_street,
                    # feature_volume_building=feature_volume_building,
                    # feature_volume_street=feature_volume_street,
                    # latent_code_building=latent_code_building,
                    # latent_code_street=latent_code_street,
                )

                ret_cars_list.update({car_id:ret_car})


            'LinGaoyuan_operation_20240906: (optional) use cov of depth from Uncle SLAM formular 5 to determine whether update depth prior or not'
            'LinGaoyuan_20240925: uncomment following code during the clip test process'
            # if args.cov_criteria is True:
            #     pred_depth_cov = ret["outputs_coarse"]["depth_cov"]
            #     if pred_depth_cov > args.preset_depth_cov:
            #         use_updated_prior_depth = True

            'LinGaoyuan_operation_20240925: set two seperated model to zero_grad()'
            # # compute loss
            # model.optimizer.zero_grad()
            model_building.optimizer.zero_grad()
            model_street.optimizer.zero_grad()

            'LinGaoyuan_operation_20240929: set car_model to zero_grad()'
            model_car.optimizer.zero_grad()

            'set sky model and style model to zero_grad'
            if args.sky_model_type == 'mlp':
                sky_model.sky_optimizer.zero_grad()
                sky_model.sky_style_optimizer.zero_grad()
            else:
                sky_model.optimizer.zero_grad()
            # sky_optimizer.zero_grad()
            # sky_style_optimizer.zero_grad()

            '''
            LinGaoyuan_operation_20240925: adjust loss to fit the two seperated model
            TO DO: maybe need to calculate the seperated loss for seperated model?
            '''
            # loss, scalars_to_log = criterion(ret["outputs_coarse"], ray_batch, scalars_to_log)
            'LinGaoyuan_20240926:combined loss'
            # rendered_image = ret_building["outputs_coarse"]["rgb"] * building_mask + ret_street["outputs_coarse"]["rgb"] * street_mask
            # loss, scalars_to_log = criterion.forward_clip_rgb(rendered_image, input_img, scalars_to_log)
            'LinGaoyuan_20240926: seperated loss'
            loss_building, _ = criterion.forward_clip_rgb(ret_building["outputs_coarse"]["rgb"], input_img_building,
                                                          scalars_to_log)
            loss_street, _ = criterion.forward_clip_rgb(ret_street["outputs_coarse"]["rgb"], input_img_street,
                                                          scalars_to_log)

            'LinGaoyuan_operation_20240929: add loss of each car object'
            loss_cars_list = {}
            for car_id in train_data['car_id']:
                ret_car=ret_cars_list['car_id']
                input_img_car = input_img_car_list['car_id']
                loss_car, _ = criterion.forward_clip_rgb(ret_car["outputs_coarse"]["rgb"], input_img_car,
                                                            scalars_to_log)

                loss_cars_list.update({car_id:loss_car})




            '''
            LinGaoyuan_operation_20240924: create clip mapper loss for shape code mapper and appearance code mapper
            To Do: compute different clip loss for shape code mapper and appearance code according to the formular 11 and formular 12 of clip-NeRF
            LinGaoyuan_20240925: TO DO: a complete rendering image with the same shape of input_img must be generated in order to calculate the clip loss
            '''
            # loss_clip = Loss_clip(clip_model, clip_preprocess, ray_batch["rgb"], ret["outputs_coarse"]["rgb"], device)
            loss_clip_building = Loss_clip_version_2(clip_model, clip_preprocess, clip_features_building,
                                                   ret_building["outputs_coarse"]["rgb"], device)
            loss_clip_street = Loss_clip_version_2(clip_model, clip_preprocess, clip_features_street,
                                                   ret_street["outputs_coarse"]["rgb"], device)

            loss_mapper_building = loss_building + args.lambda_shape_code*loss_clip_building
            loss_mapper_street = loss_street + args.lambda_appearance_code * loss_clip_street

            'LinGaoyuan_operation_20240929: add mapper loss of each car object'
            loss_shape_mapper_cars_list = {}
            loss_appearance_mapper_cars_list = {}
            for car_id in train_data['car_id']:
                ret_car=ret_cars_list['car_id']
                input_img_car = input_img_car_list['car_id']
                '''
                LinGaoyuan_20240929: TODO: we must considerate the formular 11 and 12 in Clip-NeRF. whether the lambda_n and z_n should be added or not. 
                if use formular 11 and 12, we need to generate two rendering images, one for loss_shape_mapper_car and one for loss_appearance_mapper_car.
                the approach must be discussed with supervisor!!!
                '''
                loss_shape_mapper_car = Loss_clip(clip_model, clip_preprocess, input_img_car, ret_car["outputs_coarse"]["rgb"], device)
                loss_appearance_mapper_car = Loss_clip(clip_model, clip_preprocess, input_img_car, ret_car["outputs_coarse"]["rgb"], device)

                loss_shape_mapper_cars_list.update({car_id:loss_shape_mapper_car})
                loss_appearance_mapper_cars_list.update({car_id:loss_appearance_mapper_car})


            'loss of sky area, add by LinGaoyuan'
            loss_sky_rgb = criterion.sky_loss_rgb(ret_sky["outputs_coarse"], ray_batch)

            'LinGaoyuan_20240925: uncomment following code during clip test process'
            # 'loss of sky area, add by LinGaoyuan'
            # loss_sky_rgb = criterion.sky_loss_rgb(ret["outputs_coarse"], ray_batch)
            #
            # 'LinGaoyuan_operation_20240907: the depth loss will not be added to total loss after epoch reach a preset value'
            # if ret_alpha is True:
            #     loss_depth_value = criterion.depth_loss(ret["outputs_coarse"], ray_batch, train_depth_prior)
            #     if epoch < args.update_prior_depth_epochs:
            #         loss = args.lambda_rgb * loss + args.lambda_depth * loss_depth_value
            #
            # 'LinGaoyuan_operation_20240920: (optional) use depth loss to determine whether update depth prior or not'
            # if args.depth_loss_criteria is True:
            #     if loss_depth_value > args.preset_depth_loss:
            #         use_updated_prior_depth = True
            #
            # if ret["outputs_fine"] is not None:
            #     fine_loss, scalars_to_log = criterion(
            #         ret["outputs_fine"], ray_batch, scalars_to_log
            #     )
            #     loss += fine_loss

            'LinGaoyuan_operation_20240905: update prior depth with depth prediction if epoch reach preset value'
            'LinGaoyuan_operation_20240920: add a indicator(args.update_prior_depth) to determine whether update depth prior or not'
            'LinGaoyuan_operation_20240925: depth will not be update during the clip test process, so we uncomment following code'
            # if use_updated_prior_depth and args.update_prior_depth is True:
            #     train_depth_pred = ret["outputs_coarse"]["depth"].detach()
            #     train_depth_prior[ray_batch["selected_inds"]] = train_depth_pred[...,None]
            #     if args.resize_image is True:
            #         train_depth_prior = train_depth_prior.reshape(1, args.image_resize_H, args.image_resize_W)
            #     else:
            #         train_depth_prior = train_depth_prior.reshape(1,args.image_H, args.image_W)
            #     train_prior_depth_values[ray_batch["idx"], ...] = train_depth_prior
            #     # print('finish update train prior depth value in epoch: {}'.format(epoch), 'step: {}'.format(global_step))


            # loss.backward()
            # scalars_to_log["loss"] = loss.item()
            # model.optimizer.step()
            # model.scheduler.step()

            'LinGaoyuan_operation_20240926: seperated loss for seperated model'
            loss_building.backward()
            loss_street.backward()
            scalars_to_log["loss_building"] = loss_building.item()
            scalars_to_log["loss_street"] = loss_street.item()
            model_building.optimizer.step()
            model_building.scheduler.step()
            model_street.optimizer.step()
            model_street.scheduler.step()


            'LinGaoyuan_operation_20240926: clip loss for mapper'
            loss_mapper_building.backward()
            loss_mapper_street.backward()

            clip_mapper_building.step()
            clip_mapper_street.step()


            'LinGaoyuan_20240929: update optimizer of car model and car shape&appearance mapper'
            for car_id in train_data['car_id']:
                loss_car = loss_cars_list[car_id]
                loss_car.backward()
                model_car.optimizer.step()
                model_car.scheduler.step()

                loss_shape_mapper_car = loss_shape_mapper_cars_list[car_id]
                loss_appearance_mapper_car = loss_appearance_mapper_cars_list[car_id]
                loss_shape_mapper_car.backward()
                loss_appearance_mapper_car.backward()
                clip_shape_mapper_car.step()
                clip_appearance_mapper_car.step()


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

            'LinGaoyuan_operation_20240925: use new code for log'
            # scalars_to_log["lr"] = model.scheduler.get_last_lr()[0]
            scalars_to_log["lr_building"] = model_building.scheduler.get_last_lr()[0]
            scalars_to_log["lr_street"] = model_street.scheduler.get_last_lr()[0]


            # end of core optimization loop
            dt = time.time() - time0

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
                'LinGaoyuan_operation_20240925: rewrite output log to fit seperated model'
                # if global_step % args.i_print == 0 or global_step < 10:
                #     # write mse and psnr stats
                #     mse_error = img2mse(ret["outputs_coarse"]["rgb"], ray_batch["rgb"]).item()
                #     scalars_to_log["train/coarse-loss"] = mse_error
                #     scalars_to_log["train/coarse-psnr-training-batch"] = mse2psnr(mse_error)
                #     if ret["outputs_fine"] is not None:
                #         mse_error = img2mse(ret["outputs_fine"]["rgb"], ray_batch["rgb"]).item()
                #         scalars_to_log["train/fine-loss"] = mse_error
                #         scalars_to_log["train/fine-psnr-training-batch"] = mse2psnr(mse_error)
                #
                #     logstr = "{} Epoch: {}  step: {} ".format(args.expname, epoch, global_step)
                #     for k in scalars_to_log.keys():
                #         logstr += " {}: {:.6f}".format(k, scalars_to_log[k])
                #     print(logstr)
                #
                #     print("depth loss: {}".format(loss_depth_value), "depth cov: {}".format(ret["outputs_coarse"]["depth_cov"]))
                #
                #     print("sky model lr: {}".format(sky_model_lr), "sky_style_lr: {}".format(sky_style_lr), "sky_loss: {}".format(loss_sky_rgb))
                #     print("each iter time {:.05f} seconds".format(dt))
                if global_step % args.i_print == 0 or global_step < 10:
                    # write mse and psnr stats
                    mse_error_building = img2mse(ret_building["outputs_coarse"]["rgb"], ray_batch_building["rgb"]).item()
                    scalars_to_log["train/coarse-loss_building"] = mse_error_building
                    scalars_to_log["train/coarse-psnr-training-batch_building"] = mse2psnr(mse_error_building)

                    mse_error_street = img2mse(ret_street["outputs_coarse"]["rgb"], ray_batch_street["rgb"]).item()
                    scalars_to_log["train/coarse-loss_street"] = mse_error_street
                    scalars_to_log["train/coarse-psnr-training-batch_street"] = mse2psnr(mse_error_street)


                    logstr = "{} Epoch: {}  step: {} ".format(args.expname, epoch, global_step)
                    for k in scalars_to_log.keys():
                        logstr += " {}: {:.6f}".format(k, scalars_to_log[k])
                    print(logstr)

                    'LinGaoyuan_20240925: depth loss will not be output during clip test process, so uncomment following code'
                    # print("depth loss: {}".format(loss_depth_value), "depth cov: {}".format(ret["outputs_coarse"]["depth_cov"]))

                    print("sky model lr: {}".format(sky_model_lr), "sky_style_lr: {}".format(sky_style_lr), "sky_loss: {}".format(loss_sky_rgb))
                    print("each iter time {:.05f} seconds".format(dt))

                'LinGaoyuan_20240925: we will not save checkpoint during clip test process, so uncomment the following code'
                # if global_step % args.i_weights == 0:
                #     print("Saving checkpoints at {} to {}...".format(global_step, out_folder))
                #     fpath = os.path.join(out_folder, "model_{:06d}.pth".format(global_step))
                #     model.save_model(fpath)
                #
                #     print("Saving checkpoints of sky model at {} to {}...".format(global_step, out_folder))
                #     fpath = os.path.join(out_folder, "sky_model_{:06d}.pth".format(global_step))
                #     sky_model.save_model(fpath)
                #
                #     'LinGaoyuan_operation_20240920: save train_prior_depth_values as .pt file when args.save_prior_depth is True'
                #     if args.save_prior_depth is True:
                #         print("Saving train_prior_depth_values at {} to {}...".format(global_step, out_folder))
                #         fpath = os.path.join(out_folder, "train_prior_depth_values.pt")
                #         torch.save(train_prior_depth_values, fpath)





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
                    # tmp_ray_sampler = RaySamplerSingleImage(
                    #     val_data, device, render_stride=args.render_stride
                    # )
                    # H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
                    # gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)

                    input_img_val = val_data['rgb'].squeeze()
                    black_background = torch.zeros((3,)).cuda()
                    'LinGaoyuan_20240925: TO DO: building_mask should be added in the future'
                    building_mask_val = None
                    input_img_building = input_img * (1.0 - building_mask) + black_background * (building_mask)
                    street_mask_val = None
                    input_img_street = input_img * (1.0 - street_mask) + black_background * (street_mask)
                    with torch.no_grad():
                        clip_features_building = clip_model.encode_image(input_img_building)
                        clip_features_street = clip_model.encode_image(input_img_street)
                    'LinGaoyuan_20240925: (optional) normalize for the clip feature'
                    clip_features_building /= clip_features_building.norm(dim=-1, keepdim=True).to(torch.float32)
                    clip_features_street /= clip_features_street.norm(dim=-1, keepdim=True).to(torch.float32)

                    'LinGaoyuan_operation_20240925: use MLP mapper or attention mapper to create delta latent code'
                    delta_latent_code_building = clip_mapper_building(clip_features_building)
                    delta_latent_code_street = clip_mapper_street(clip_features_street)

                    latent_code_building_val = latent_code_building + delta_latent_code_building
                    latent_code_street_val = latent_code_street + delta_latent_code_street

                    val_data_building = val_data.copy()
                    val_data_building['rgb'] = input_img_building[None, ...]

                    'LinGaoyuan_operation_20241001: extract building area of src image according to building mask'
                    'LinGaoyuan_20241001: TO DO: the building masks of src image should be add in the future, here I assume building masks of src image is exist'
                    val_data_building_src_rgbs = val_data_building['src_rgbs']  # shape: (batch, num_src, H, W, 3)
                    val_data_building_src_building_masks = val_data_building[
                        'src_building_masks']  # shape: (batch, num_src, H, W)
                    num_src = val_data_building_src_rgbs.shape[1]
                    for i in range(num_src):
                        src_building_mask = val_data_building_src_building_masks[:, i, :, :]
                        val_data_building_src_rgbs[:, i, :, :, :] = val_data_building_src_rgbs[:, i, :, :, :] * (
                                    1.0 - src_building_mask) + black_background * (src_building_mask)
                        val_data_building['src_rgbs'][:, i, :, :, :] = val_data_building_src_rgbs[:, i, :, :, :]

                    ray_sampler_building_val = RaySamplerSingleImage(val_data_building, device, render_stride=args.render_stride)


                    val_data_street = val_data.copy()
                    val_data_street['rgb'] = input_img_street[None, ...]

                    'LinGaoyuan_operation_20241001: extract street area of src image according to street mask'
                    'LinGaoyuan_20241001: TO DO: the street masks of src image should be add in the future, here I assume street masks of src image is exist'
                    val_data_street_src_rgbs = val_data_street['src_rgbs']  # shape: (batch, num_src, H, W, 3)
                    val_data_street_src_street_masks = val_data_street[
                        'src_street_masks']  # shape: (batch, num_src, H, W)
                    num_src = val_data_street_src_rgbs.shape[1]
                    for i in range(num_src):
                        src_street_mask = val_data_street_src_street_masks[:, i, :, :]
                        val_data_street_src_rgbs[:, i, :, :, :] = val_data_street_src_rgbs[:, i, :, :, :] * (
                                    1.0 - src_street_mask) + black_background * (src_street_mask)
                        val_data_street['src_rgbs'][:, i, :, :, :] = val_data_street_src_rgbs[:, i, :, :, :]

                    ray_sampler_street_val = RaySamplerSingleImage(val_data_street, device, render_stride=args.render_stride)


                    'LinGaoyuan_operation_20240929: create data from multiple cars during eval process'

                    input_img_car_list_val = {}
                    car_mask_list_val = {}
                    latent_shape_code_car_list_val = {}
                    latent_appearance_code_car_list_val = {}
                    for car_id in val_data['car_id']:
                        car_mask_val = None  # car area = 0, other area = 1
                        car_mask_list_val.update({car_id: car_mask_val})
                        input_img_car_val = input_img * (1.0 - car_mask_val) + black_background * (car_mask_val)
                        input_img_car_list_val.update({car_id: input_img_car_val})

                        with torch.no_grad():
                            clip_features_car_val = clip_model.encode_image(input_img_car_val)

                        'normalization'
                        clip_features_car_val /= clip_features_car_val.norm(dim=-1, keepdim=True).to(torch.float32)

                        delta_latent_shape_code_car_val = clip_shape_mapper_car(clip_features_car_val)
                        delta_latent_appearance_code_car_val = clip_appearance_mapper_car(clip_features_car_val)

                        latent_shape_code_car = latent_shape_code_car_list[car_id]
                        latent_appearance_code_car = latent_appearance_code_car_list[car_id]

                        latent_shape_code_car_val = latent_shape_code_car + delta_latent_shape_code_car_val
                        latent_appearance_code_car_val = latent_appearance_code_car + delta_latent_appearance_code_car_val

                        latent_shape_code_car_list_val[car_id] = latent_shape_code_car_val
                        latent_appearance_code_car_list_val[car_id] = latent_appearance_code_car_val





                    'LinGaoyuan_operation_20240830: set create depth image by default even N_inportance is 0 in order to create depth image'
                    'LinGaoyuan_operation_20240920: set data_mode to val when use val dataset in val process'
                    'LinGaoyuan_operation_20240925: rewrite log_view for clip test process'
                    log_view(
                        global_step,
                        args,
                        # model,
                        # tmp_ray_sampler,
                        projector,
                        input_img_val,
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
                        model_building=model_building,
                        model_street=model_street,
                        latent_code_building=latent_code_building_val,
                        latent_code_street=latent_code_street_val,
                        ray_sampler_building=ray_sampler_building_val,
                        ray_sampler_street=ray_sampler_street_val,
                        building_mask=building_mask_val,
                        street_mask=street_mask_val,
                        model_car=model_car,
                        latent_shape_code_car_list=latent_shape_code_car_list_val,
                        latent_appearance_code_car_list=latent_appearance_code_car_list_val,
                        car_mask_list=car_mask_list_val,
                        input_img_car_list=input_img_car_list_val,
                        data=val_data,
                    )
                    torch.cuda.empty_cache()

                    print("Logging current training view...")
                    'LinGaoyuan_20240925: uncomment following code and rewrite the log_view()'
                    # tmp_ray_train_sampler = RaySamplerSingleImage(
                    #     train_data, device, render_stride=1
                    # )
                    # H, W = tmp_ray_train_sampler.H, tmp_ray_train_sampler.W
                    # gt_img = tmp_ray_train_sampler.rgb.reshape(H, W, 3)

                    # filename_gt = os.path.join(out_folder, 'img_gt.png')
                    # torchvision.io.write_png(torch.tensor(gt_img * 255).to(torch.uint8).permute(2, 0, 1), filename_gt)

                    'LinGaoyuan_operation_20240830: set create depth image by default even N_inportance is 0'
                    '''
                    LinGaoyuan_operation_20240918: use updated prior depth if use train sampler and use_updated_prior_depth is True. 
                    set data_mode=train in this situation
                    
                    '''
                    'LinGaoyuan_operation_20240925: rewrite log_view for clip test process'
                    '----------------------------------------------dividing line------------------------------------------------------'

                    log_view(
                        global_step,
                        args,
                        # model,
                        # tmp_ray_train_sampler,
                        projector,
                        input_img,
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
                        model_building=model_building,
                        model_street=model_street,
                        latent_code_building=latent_code_building,
                        latent_code_street=latent_code_street,
                        ray_sampler_building=ray_sampler_building,
                        ray_sampler_street=ray_sampler_street,
                        building_mask=building_mask,
                        street_mask=street_mask,
                        model_car=model_car,
                        latent_shape_code_car_list=latent_shape_code_car_list,
                        latent_appearance_code_car_list=latent_appearance_code_car_list,
                        car_mask_list=car_mask_list,
                        input_img_car_list=input_img_car_list,
                        data=train_data,
                    )
                    '----------------------------------------------dividing line------------------------------------------------------'

            global_step += 1

            epoch_step += 1

            'LinGaoyuan_operation_20240925: use model_building to define global_step during clip test process'
            # if global_step > model.start_step + args.n_iters + 1:
            if global_step > model_building.start_step + args.n_iters + 1:
                break
        # epoch += 1


'LinGaoyuan_operation_20240925: rewrite log_view for clip test process'
@torch.no_grad()
def log_view(
    global_step,
    args,
    # model,
    # ray_sampler,
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
    model_building=None,
    model_street=None,
    latent_code_building=None,
    latent_code_street=None,
    ray_sampler_building=None,
    ray_sampler_street=None,
    building_mask=None,
    street_mask=None,
    model_car=None,
    latent_shape_code_car_list=None,
    latent_appearance_code_car_list=None,
    car_mask_list=None,
    input_img_car_list=None,
    pos_embedder=None,
    clip_deformation_car=None,
    data=None,
):
    model_building.switch_to_eval()
    model_street.switch_to_eval()

    with torch.no_grad():
        'LinGaoyuan_operation_20240925: rewrite log_view to fit the situation with two seperated model for clip test process'
        # ray_batch = ray_sampler.get_all()
        ray_batch_building = ray_sampler_building.get_all()
        ray_batch_street = ray_sampler_street.get_all()

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
            train_depth_prior = (train_prior_depth_values[ray_batch_building["idx"], ...]).reshape(-1, 1)
        else:
            train_depth_prior = None

        '''LinGaoyuan_operation_20240925: seperately generate rendering image of building area and street area'''
        '----------------------------------------------dividing line------------------------------------------------------'
        '------------------------------------ generate rendering image of building ----------------------------------'
        if args.use_retr_feature_extractor is False:
            featmaps_building = model_building.feature_net(ray_batch_building["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
        else:
            featmaps_building, fpn_building = model_building.retr_feature_extractor(ray_batch_building["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
        if args.use_volume_feature is True and args.use_retr_feature_extractor is True:
            feature_volume_building = model_building.retr_feature_volume(fpn_building, ray_batch_building)
        else:
            feature_volume_building = None

        ret_building = render_single_image(
            args,
            ray_sampler=ray_sampler_building,
            ray_batch=ray_batch_building,
            # model=model,
            projector=projector,
            chunk_size=args.chunk_size,
            N_samples=args.N_samples,
            model=model_building,
            inv_uniform=args.inv_uniform,
            det=True,
            N_importance=args.N_importance,
            white_bkgd=args.white_bkgd,
            render_stride=render_stride,
            featmaps=featmaps_building,
            ret_alpha=ret_alpha,
            single_net=single_net,
            sky_style_code=sky_style_code,
            # sky_style_model=sky_style_model,
            sky_model=sky_model,
            feature_volume=feature_volume_building,
            use_updated_prior_depth=use_updated_prior_depth,
            train_depth_prior=train_depth_prior,
            data_mode=data_mode,
            latent_code=latent_code_building,
        )

        '------------------------------------ generate rendering image of street ----------------------------------'
        '''
         LinGaoyuan_operation_20240918: In val process, if the data is from training dataset(which mean data_mode is train) 
         and use_updated_prior_depth is True, the updated prior depth will be used.
         '''
        # if data_mode == 'train' and use_updated_prior_depth is True:
        #     print('use updated prior depth for training dataset in val process')
        #     train_depth_prior = (train_prior_depth_values[ray_batch_street["idx"], ...]).reshape(-1, 1)
        # else:
        #     train_depth_prior = None

        if args.use_retr_feature_extractor is False:
            featmaps_street = model_street.feature_net(ray_batch_street["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
        else:
            featmaps_street, fpn_street = model_street.retr_feature_extractor(ray_batch_street["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
        if args.use_volume_feature is True and args.use_retr_feature_extractor is True:
            feature_volume_street = model_street.retr_feature_volume(fpn_street, ray_batch_street)
        else:
            feature_volume_street = None

        ret_street = render_single_image(
            args,
            ray_sampler=ray_sampler_street,
            ray_batch=ray_batch_street,
            # model=model,
            projector=projector,
            chunk_size=args.chunk_size,
            N_samples=args.N_samples,
            model=model_street,
            inv_uniform=args.inv_uniform,
            det=True,
            N_importance=args.N_importance,
            white_bkgd=args.white_bkgd,
            render_stride=render_stride,
            featmaps=featmaps_street,
            ret_alpha=ret_alpha,
            single_net=single_net,
            sky_style_code=sky_style_code,
            # sky_style_model=sky_style_model,
            sky_model=sky_model,
            feature_volume=feature_volume_street,
            use_updated_prior_depth=use_updated_prior_depth,
            train_depth_prior=train_depth_prior,
            data_mode=data_mode,
            latent_code=latent_code_street,
        )

        '----------------------------------------------dividing line------------------------------------------------------'

        '------------------------------------ generate rendering image of cars ----------------------------------'
        ret_cars_list = {}
        device = "cuda:{}".format(args.local_rank)
        black_background = torch.zeros((3,)).cuda()
        for car_id in data['car_id']:
            'LinGaoyuan_operation_20240929: create ray_batch for car object'
            data_car = data.copy()
            input_img_car = input_img_car_list[car_id]
            data_car['rgb'] = input_img_car[None, ...]

            'LinGaoyuan_operation_20241001: extract car area of src image according to car mask'
            'LinGaoyuan_20241001: TO DO: the car masks of src image should be add in the future, here I assume car masks of src image is exist'
            data_car_src_rgbs = data_car['src_rgbs'][car_id]  # shape: (batch, num_src, H, W, 3)
            data_car_src_car_masks = data_car[
                'src_car_masks'][car_id]  # shape: (batch, num_src, H, W)
            num_src = data_car_src_rgbs.shape[1]
            for i in range(num_src):
                src_car_mask = data_car_src_car_masks[:, i, :, :]
                data_car_src_rgbs[:, i, :, :, :] = data_car_src_rgbs[:, i, :, :, :] * (
                        1.0 - src_car_mask) + black_background * (src_car_mask)
                data_car['src_rgbs'][car_id][:, i, :, :, :] = data_car_src_rgbs[:, i, :, :, :]

            ray_sampler_car = RaySamplerSingleImage(data_car, device)
            ray_batch_car = ray_sampler_car.get_all()

            'LinGaoyuan_operation_20240929: create featmap and feature volume for car object'
            if args.use_retr_feature_extractor is False:
                featmaps_car = model_car.feature_net(
                    ray_batch_car["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
            else:
                featmaps_car, fpn_car = model_car.retr_feature_extractor(
                    ray_batch_car["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
            if args.use_volume_car is True and args.use_retr_feature_extractor is True:
                feature_volume_car = model_car.retr_feature_volume(fpn_car, ray_batch_car)
            else:
                feature_volume_car = None

            latent_shape_code_car = latent_shape_code_car_list[car_id]
            latent_appearance_code_car = latent_appearance_code_car_list[car_id]

            ret_car = render_single_image(
                args,
                ray_sampler=ray_sampler_car,
                ray_batch=ray_batch_car,
                projector=projector,
                chunk_size=args.chunk_size,
                N_samples=args.N_samples,
                model=model_car,
                inv_uniform=args.inv_uniform,
                det=True,
                N_importance=args.N_importance,
                white_bkgd=args.white_bkgd,
                render_stride=args.render_stride,
                featmaps=featmaps_car,
                ret_alpha=ret_alpha,
                single_net=args.single_net,
                sky_style_code=sky_style_code,
                # sky_style_model=sky_style_model,
                sky_model=sky_model,
                feature_volume=feature_volume_car,
                use_updated_prior_depth=use_updated_prior_depth,
                train_depth_prior=train_depth_prior,
                data_mode=data_mode,
                # latent_code=latent_code_car,
                deformation_network=clip_deformation_car,
                pos_embedder=pos_embedder,
                latent_shape_code=latent_shape_code_car,
                latent_appearance_code=latent_appearance_code_car,
            )

            ret_cars_list.update({car_id: ret_car})

            '----------------------------------------------dividing line------------------------------------------------------'


    'LinGaoyuan_20240925: average_im is useless in clip test process, so uncomment next line'
    # average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))

    'get the mask of sky area'
    H = ray_sampler_street.H
    W = ray_sampler_street.W
    sky_mask = ray_sampler_street.sky_mask.reshape(H, W, 1)
    # sky_mask = sky_mask.permute(1, 2, 0)

    'LinGaoyuan_operation_20240926: create a totally black image as the initial target image'
    rgb_pred = torch.zeros(3,H,W)

    'LinGaoyuan_20240925: we do not care about depth during clip test process, so uncomment following code'
    # depth_value = ray_sampler.depth_value.reshape(H, W, 1).squeeze().cpu()
    # # plt.imshow(depth_value)
    # # plt.show()

    if args.render_stride != 1:
        print(gt_img.shape)
        gt_img = gt_img[::render_stride, ::render_stride]
        # average_im = average_im[::render_stride, ::render_stride]

        sky_mask = sky_mask[::render_stride, ::render_stride]

    rgb_gt = img_HWC2CHW(gt_img)
    # average_im = img_HWC2CHW(average_im)

    'LinGaoyuan_operation_20240925: generate rendering image of building and street and recombine to complete image'
    # rgb_pred = img_HWC2CHW(ret["outputs_coarse"]["rgb"].detach().cpu())
    rgb_pred_building = img_HWC2CHW(ret_building["outputs_coarse"]["rgb"].detach().cpu())
    rgb_pred_street = img_HWC2CHW(ret_street["outputs_coarse"]["rgb"].detach().cpu())

    '''
    LinGaoyuan_20240930: the model of sky area is sky_model, the input of sky_model are only the sky_style_code and ray_d.  
    so in the theory, we can use the rgb_sky from ret_building or ret_street or ret_car
    '''
    rgb_sky = img_HWC2CHW(ret_street["outputs_coarse"]["rgb_sky"].detach().cpu())

    print("sky rgb max value: {}".format(rgb_sky.max()), "sky rgb min value: {}".format(rgb_sky.min()))
    # plt.imshow((rgb_sky*255).permute(1, 2, 0))
    # plt.show()

    rgb_sky = torch.clamp(rgb_sky, 0, 1)

    # softmax_func = torch.nn.Softmax(dim=2)
    # rgb_sky = softmax_func(rgb_sky)

    sky_mask = sky_mask.permute(2,0,1)
    'the sky color should be considered to different color combination'

    rgb_pred = rgb_pred * sky_mask + rgb_sky * (1 - sky_mask)
    'LinGaoyuan_operation_20240926: use the zhengzhisheng method, replace the relevant area of rgb_pred with rendering image'
    # rgb_pred = rgb_pred_building * building_mask + rgb_pred_street * street_mask + rgb_sky * (1 - sky_mask)
    # rgb_pred[building_mask] +=  rgb_pred_building*building_mask
    # rgb_pred[street_mask] += rgb_pred_street*street_mask.
    rgb_building = img_HWC2CHW(ret_building["outputs_coarse"]["rgb"].detach().cpu())
    rgb_street = img_HWC2CHW(ret_street["outputs_coarse"]["rgb"].detach().cpu())
    rgb_pred = rgb_pred * building_mask + rgb_building * (1 - building_mask)
    rgb_pred = rgb_pred * street_mask + rgb_street * (1 - street_mask)

    'LinGaoyuan_operation_20240929: recombine car rendering rgb to rgb_pred'
    for car_id in data['car_id']:
        ret_car = ret_cars_list[car_id]
        rgb_car = img_HWC2CHW(ret_car["outputs_coarse"]["rgb"].detach().cpu())
        car_mask = car_mask_list[car_id]
        rgb_pred = rgb_pred * car_mask + rgb_car * (1 - car_mask)


    'LinGaoyuan_operation_20240925: do not considerated average_im during clip test process, so redefine the code of h_max and w_max'
    # h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], average_im.shape[-2])
    # w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], average_im.shape[-1])
    h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2])
    w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1])

    # rgb_im = torch.zeros(3, h_max, 3 * w_max)
    # rgb_im[:, : average_im.shape[-2], : average_im.shape[-1]] = average_im
    # rgb_im[:, : rgb_gt.shape[-2], w_max : w_max + rgb_gt.shape[-1]] = rgb_gt
    # rgb_im[:, : rgb_pred.shape[-2], 2 * w_max : 2 * w_max + rgb_pred.shape[-1]] = rgb_pred
    rgb_im = torch.zeros(3, h_max, 2 * w_max)
    rgb_im[:, : rgb_gt.shape[-2], : rgb_gt.shape[-1]] = rgb_gt
    rgb_im[:, : rgb_pred.shape[-2], w_max : w_max + rgb_pred.shape[-1]] = rgb_pred

    'LinGaoyuan_20240925: do not considerat depth during clip test process, so uncomment following code'
    '----------------------------------------------dividing line------------------------------------------------------'
    # depth_matrix = {}
    #
    # if "depth" in ret["outputs_coarse"].keys():
    #     depth_pred = ret["outputs_coarse"]["depth"].detach().cpu()
    #
    #     # print(torch.max(depth_pred), torch.min(depth_pred))
    #     print('depth predict max value: {}'.format(torch.max(depth_pred)), 'depth predict min value: {}'.format(torch.min(depth_pred)))
    #
    #     'this depth prediction should be save as .json file'
    #     depth_matrix.update({'depth_value_pred': np.array(depth_pred).tolist()})
    #
    #     depth_sky = torch.full_like(depth_pred, 200)
    #     depth_pred_replace_sky = (depth_pred * sky_mask + depth_sky * (1 - sky_mask)).squeeze()
    #
    #     depth_replace_sky = ((depth_pred_replace_sky - depth_pred_replace_sky.min()) / (depth_pred_replace_sky.max() - depth_pred_replace_sky.min())) * 255.0
    #     depth_replace_sky = depth_replace_sky.cpu().numpy().astype(np.uint8)
    #
    #     # plt.imshow(depth_replace_sky)
    #     # plt.title('Depth img replaced sky area')
    #     # plt.show()
    #
    #     # plt.imshow(depth_pred)
    #     # plt.show()
    #     depth = ((depth_pred - depth_pred.min()) / (depth_pred.max() - depth_pred.min())) * 255.0
    #     depth = depth.cpu().numpy().astype(np.uint8)
    #     # plt.imshow(depth)
    #     # plt.title('normalized depth image')
    #     # plt.show()
    #     # print(torch.min(depth_pred),torch.max(depth_pred))
    #     # depth_range = [torch.min(depth_pred), torch.max(depth_pred)]
    #
    #     depth_im = img_HWC2CHW(colorize(depth_pred, cmap_name="jet"))
    #     # print(torch.max(depth_im), torch.min(depth_im))
    #     # plt.title('RGB depth image')
    #     # plt.imshow(depth_im.permute(1,2,0))
    #     # plt.show()
    #
    #
    # else:
    #     depth_im = None
    '----------------------------------------------dividing line------------------------------------------------------'

    'LinGaoyuan_20240925: do not considerate fine network during clip test process, so uncomment following code'
    '----------------------------------------------dividing line------------------------------------------------------'
    # if ret["outputs_fine"] is not None:
    #     rgb_fine = img_HWC2CHW(ret["outputs_fine"]["rgb"].detach().cpu())
    #     rgb_fine_ = torch.zeros(3, h_max, w_max)
    #     rgb_fine_[:, : rgb_fine.shape[-2], : rgb_fine.shape[-1]] = rgb_fine
    #
    #     'update sky area for rgb_fine'
    #     rgb_fine_ = rgb_fine_ * sky_mask + rgb_sky * (1 - sky_mask)
    #
    #     rgb_im = torch.cat((rgb_im, rgb_fine_), dim=-1)
    #     depth_pred = torch.cat((depth_pred, ret["outputs_fine"]["depth"].detach().cpu()), dim=-1)
    #     depth_im = img_HWC2CHW(colorize(depth_pred, cmap_name="jet"))
    '----------------------------------------------dividing line------------------------------------------------------'

    rgb_im = rgb_im.permute(1, 2, 0).detach().cpu().numpy()
    rgb_im = np.array(Image.fromarray((rgb_im * 255).astype(np.uint8)))
    filename = os.path.join(out_folder, prefix[:-1] + "_{:03d}.png".format(global_step))

    torchvision.io.write_png(torch.tensor(rgb_im).permute(2,0,1), filename)
    # imageio.imwrite(filename, rgb_im)

    'LinGaoyuan_20240925: do not considerat depth during clip test process, so uncomment following code'
    '----------------------------------------------dividing line------------------------------------------------------'
    # if depth_im is not None:
    #     depth_im = depth_im.permute(1, 2, 0).detach().cpu().numpy()
    #     filename = os.path.join(out_folder, prefix[:-1] + "depth_{:03d}.png".format(global_step))
    #     depth_im = np.array(Image.fromarray((depth_im * 255).astype(np.uint8)))
    #
    #     # R = depth_im[:,:,0]
    #     # G = depth_im[:, :, 1]
    #     # B = depth_im[:, :, 2]
    #     # normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
    #     #
    #     # print(np.max(normalized), np.min(normalized))
    #     #
    #     # plt.imshow(normalized)
    #     # plt.show()
    #     # plt.imsave(filename, normalized, cmap='gray')
    #
    #     imageio.imwrite(filename, depth_replace_sky)
    #     # imageio.imwrite(filename, depth_im)
    #
    #     'save depth prediction matrix as .json file'
    #     # depth_matrix_filename = os.path.join(out_folder, prefix[:-1] + "_depth_{:03d}.json".format(global_step))
    #     # with open(depth_matrix_filename,'w') as f:
    #     #     json.dump(depth_matrix, f)
    '----------------------------------------------dividing line------------------------------------------------------'

    'LinGaoyuan_20240925: temporary do not considerat psnr during clip test process, so uncomment following code'
    '----------------------------------------------dividing line------------------------------------------------------'
    # # write scalar
    # pred_rgb = (
    #     ret["outputs_fine"]["rgb"]
    #     if ret["outputs_fine"] is not None
    #     else ret["outputs_coarse"]["rgb"]
    # )
    # psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    # print(prefix + "psnr_image: ", psnr_curr_img)
    '----------------------------------------------dividing line------------------------------------------------------'

    'LinGaoyuan_operation_20240925: rewrite log_view to fit the situation with two seperated model for clip test process'
    # model.switch_to_train()
    model_building.switch_to_train()
    model_street.switch_to_train()


if __name__ == "__main__":
    parser = config.config_parser()
    args = parser.parse_args()

    # print(args.aliasing_filter, args.contraction_type, args.aliasing_filter_type)

    if args.distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.local_rank = int(os.environ.get("LOCAL_RANK"))
        torch.cuda.set_device(args.local_rank)

    train(args)
