import os
import numpy as np
import shutil
import torch
import torch.utils.data.distributed

from torch.utils.data import DataLoader

from model_and_model_component.data_loaders import dataset_dict
from ZYW_model.render_image_ZYW import render_single_image
from ZYW_model.model_ZYW import Model
from model_and_model_component.sample_ray_LinGaoyuan import RaySamplerSingleImage
from utils import img_HWC2CHW, colorize, img2psnr, lpips, ssim
import config
import torch.distributed as dist
from model_and_model_component.projection import Projector
from model_and_model_component.data_loaders.create_training_dataset_LinGaoyuan import create_training_dataset, create_eval_dataset
import imageio
from PIL import Image

from LinGaoyuan_function.sky_network import SKYMLP, StyleMLP, SkyModel
from LinGaoyuan_function.sky_transformer_network import SkyTransformer, SkyTransformerModel

import json


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


@torch.no_grad()
def eval(args):

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

    if args.run_val == False:
        # # create training dataset
        # dataset, sampler = create_training_dataset(args)
        'LinGaoyuan_operation_20240927: create val dataset'
        dataset, sampler = create_eval_dataset(args)

        # currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
        # please use distributed parallel on multiple GPUs to train multiple target views per batch
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            worker_init_fn=lambda _: np.random.seed(),
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=True if sampler is None else False,
        )
        iterator = iter(loader)
    else:
        # create validation dataset
        dataset = dataset_dict[args.eval_dataset](args, "validation", scenes=args.eval_scenes)
        loader = DataLoader(dataset, batch_size=1)
        iterator = iter(loader)

    # Create GNT model
    model = Model(
        args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler
    ) # Zhenyi Wan [2025/4/22] BRDF new model

    'LinGaoyuan_operation_20240927: add sky model to eval_LinGaoyuan'
    # Create sky model
    if args.sky_model_type == 'mlp':
        sky_model = SkyModel(args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler)
    else:
        sky_model = SkyTransformerModel(args, ray_d_input=3, style_dim=128, dim_embed=256, num_head=2, input_embedding=False,
                                   dropout=0.05, num_layers=5).to(device)

    'LinGaoyuan_20240927: create sky style code'
    style_dims = 128
    batch_size = 1
    if sky_model.start_step > 0:
        print('load pretrained sky style code')
        z = sky_model.sky_style_code
    else:
        print('create initial sky style code')
        z = torch.randn(batch_size, style_dims, dtype=torch.float32, device=device)


    # create projector
    projector = Projector(device=device)

    indx = 0
    psnr_scores = []
    lpips_scores = []
    ssim_scores = []
    while True:
        try:
            data = next(iterator)
        except:
            break
        if args.local_rank == 0:
            print('the {}st val data loaded'.format(indx))
            tmp_ray_sampler = RaySamplerSingleImage(data, device, render_stride=args.render_stride)
            H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
            gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
            'LinGaoyuan_operation_20240927: add new parameter to log_view, set ret_alpha == True to always return depth map, set prefix == val'
            psnr_curr_img, lpips_curr_img, ssim_curr_img = log_view(
                indx,
                args,
                model,
                tmp_ray_sampler,
                projector,
                gt_img,
                render_stride=args.render_stride,
                prefix="val/",
                out_folder=out_folder,
                ret_alpha=True,
                single_net=args.single_net,
                sky_style_code=z,
                sky_model=sky_model,
                data_mode='val',
            )
            psnr_scores.append(psnr_curr_img)
            lpips_scores.append(lpips_curr_img)
            ssim_scores.append(ssim_curr_img)
            torch.cuda.empty_cache()
            indx += 1
    print("Average PSNR: ", np.mean(psnr_scores))
    print("Average LPIPS: ", np.mean(lpips_scores))
    print("Average SSIM: ", np.mean(ssim_scores))

    'LinGaoyuan_operation_20240927: save PSNR, LPIPS, SSIM as .txt file'

    np.savetxt(os.path.join(out_folder, 'psnr_scores.txt'), psnr_scores, delimiter=',')
    np.savetxt(os.path.join(out_folder, 'ssim_scores.txt'), ssim_scores, delimiter=',')
    np.savetxt(os.path.join(out_folder, 'lpips_scores.txt'), lpips_scores, delimiter=',')




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
    sky_model=None,
    data_mode=None,
):
    model.switch_to_eval()
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()

        'LinGaoyuan_operation_20240927'
        # if model.feature_net is not None:
        #     featmaps = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
        # else:
        #     featmaps = [None, None]
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
            data_mode=data_mode,
            use_updated_prior_depth=True
        )

    'LinGaoyuan_20240927: average_im is seem that useless in eval process'
    average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))

    color_NeRO = None
    color_NeILF = None
    roughness_map = None
    metallic_map = None
    albedo_map = None
    normals_map = None
    BRDF_im = None

    'LinGaoyuan_operation_20240927: get the mask of sky area'
    H = ray_sampler.H
    W = ray_sampler.W
    sky_mask = ray_sampler.sky_mask.reshape(H, W, 1)

    if args.render_stride != 1:
        gt_img = gt_img[::render_stride, ::render_stride]
        average_im = average_im[::render_stride, ::render_stride]

        'LinGaoyuan_operation_20240927: sample the sky_mask according to the render_stride'
        sky_mask = sky_mask[::render_stride, ::render_stride]

    rgb_gt = img_HWC2CHW(gt_img)
    average_im = img_HWC2CHW(average_im)


    'LinGaoyuan_operation_20240927: get the rendering image of sky area '
    rgb_sky = img_HWC2CHW(ret["outputs_coarse"]["rgb_sky"].detach().cpu())
    print("sky rgb max value: {}".format(rgb_sky.max()), "sky rgb min value: {}".format(rgb_sky.min()))
    rgb_sky = torch.clamp(rgb_sky, 0, 1)
    sky_mask = sky_mask.permute(2, 0, 1)

    rgb_coarse = img_HWC2CHW(ret["outputs_coarse"]["rgb"].detach().cpu())

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

    'LinGaoyuan_operation_20240927: replace the sky area of rgb_coarse with rgb_sky'
    rgb_coarse = rgb_coarse * sky_mask + rgb_sky * (1 - sky_mask)

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


    if "depth" in ret["outputs_coarse"].keys():
        depth_pred = ret["outputs_coarse"]["depth"].detach().cpu()
        depth_coarse = img_HWC2CHW(colorize(depth_pred, cmap_name="jet"))
    else:
        depth_coarse = None

    if ret["outputs_fine"] is not None:
        rgb_fine = img_HWC2CHW(ret["outputs_fine"]["rgb"].detach().cpu())
        if "depth" in ret["outputs_fine"].keys():
            depth_pred = ret["outputs_fine"]["depth"].detach().cpu()
            depth_fine = img_HWC2CHW(colorize(depth_pred, cmap_name="jet"))
    else:
        rgb_fine = None
        depth_fine = None

    'LinGaoyuan_operation_20240927: combined output img_gt and img_pred as train_LinGaoyuan_epoch_version_test_ReTR.py has done'
    rgb_im = torch.zeros(3, H, 2 * W)
    rgb_im[:, : rgb_gt.shape[-2], : rgb_gt.shape[-1]] = rgb_gt
    rgb_im[:, : rgb_coarse.shape[-2], W : W + rgb_coarse.shape[-1]] = rgb_coarse

    if color_NeRO is not None:
        color_NeRO_ = torch.zeros(3, H, W)
        color_NeRO_[:, : color_NeRO.shape[-2], : color_NeRO.shape[-1]] = color_NeRO
        rgb_im = torch.cat((rgb_im, color_NeRO_), dim=-1)
    if color_NeILF is not None:
        color_NeILF_ = torch.zeros(3, H, W)
        color_NeILF_[:, : color_NeILF.shape[-2], : color_NeILF.shape[-1]] = color_NeILF
        rgb_im = torch.cat((rgb_im, color_NeILF_), dim=-1)

    if roughness_map is not None and metallic_map is not None and albedo_map is not None and normals_map is not None:
        BRDF_im = torch.zeros(3, H, 4 * W)  # Zhenyi Wan [2025/4/18] create a black background[3,H_max,4*max]
        BRDF_im[:, : roughness_map.shape[-2], : roughness_map.shape[-1]] = roughness_map.repeat(3, 1,
                                                                                                1)  # Zhenyi Wan [2025/4/18] The third fig is roughness_map after converting to 3-channel
        BRDF_im[:, : metallic_map.shape[-2], W: W + metallic_map.shape[-1]] = metallic_map.repeat(3, 1,
                                                                                                          1)  # Zhenyi Wan [2025/4/18] The fourth fig is metallic_map after converting to 3-channel
        BRDF_im[:, : albedo_map.shape[-2],
        2 * W: 2 * W + albedo_map.shape[-1]] = albedo_map  # Zhenyi Wan [2025/4/18] The fifth fig is albedo map
        BRDF_im[:, : normals_map.shape[-2], 3 * W: 3 * W + normals_map.shape[-1]] = normals_map

    # rgb_coarse = rgb_coarse.permute(1, 2, 0).detach().cpu().numpy()
    # filename = os.path.join(out_folder, prefix[:-1] + "_{:03d}_coarse.png".format(global_step))
    # rgb_coarse = np.array(Image.fromarray((rgb_coarse * 255).astype(np.uint8)))
    # imageio.imwrite(filename, rgb_coarse)

    'LinGaoyuan_operation_20240927: save rgb_im replace original rgb_coarse'
    rgb_im = rgb_im.permute(1, 2, 0).detach().cpu().numpy()
    filename = os.path.join(out_folder, prefix[:-1] + "_{:03d}_coarse.png".format(global_step))
    rgb_im = np.array(Image.fromarray((rgb_im * 255).astype(np.uint8)))
    imageio.imwrite(filename, rgb_im)

    if BRDF_im is not None:
        BRDF_im = BRDF_im.permute(1, 2, 0).detach().cpu().numpy()
        BRDF_im = np.array(Image.fromarray((BRDF_im * 255).astype(np.uint8)))
        filename = os.path.join(out_folder, prefix[:-1] + "BRDF_{:03d}.png".format(global_step))
        imageio.imwrite(filename, BRDF_im)


    if depth_coarse is not None:
        # depth_coarse = depth_coarse.permute(1, 2, 0).detach().cpu().numpy()
        # depth_coarse = np.array(Image.fromarray((depth_coarse * 255).astype(np.uint8)))


        'LinGaoyuan_20240927: set sky area of depth image to 200'
        depth_sky = torch.full_like(depth_pred, 200)
        depth_pred_replace_sky = (depth_pred * sky_mask + depth_sky * (1 - sky_mask)).squeeze()
        depth_replace_sky = ((depth_pred_replace_sky - depth_pred_replace_sky.min()) / (
                    depth_pred_replace_sky.max() - depth_pred_replace_sky.min())) * 255.0
        depth_coarse = depth_replace_sky.cpu().numpy().astype(np.uint8)


        filename = os.path.join(
            out_folder, prefix[:-1] + "_{:03d}_coarse_depth.png".format(global_step)
        )
        imageio.imwrite(filename, depth_coarse)

    if rgb_fine is not None:
        rgb_fine = rgb_fine.permute(1, 2, 0).detach().cpu().numpy()
        rgb_fine = np.array(Image.fromarray((rgb_fine * 255).astype(np.uint8)))
        filename = os.path.join(out_folder, prefix[:-1] + "_{:03d}_fine.png".format(global_step))
        imageio.imwrite(filename, rgb_fine)

    if depth_fine is not None:
        depth_fine = depth_fine.permute(1, 2, 0).detach().cpu().numpy()
        depth_fine = np.array(Image.fromarray((depth_fine * 255).astype(np.uint8)))
        filename = os.path.join(
            out_folder, prefix[:-1] + "_{:03d}_fine_depth.png".format(global_step)
        )
        imageio.imwrite(filename, depth_fine)

    # write scalar
    pred_rgb = (
        ret["outputs_fine"]["rgb"]
        if ret["outputs_fine"] is not None
        else ret["outputs_coarse"]["rgb"]
    )

    'LinGaoyuan_operation_20240927: replace the sky area of pred_rgb with rgb_sky'
    pred_rgb = img_HWC2CHW(pred_rgb.detach().cpu())
    pred_rgb = pred_rgb * sky_mask + rgb_sky * (1 - sky_mask)
    pred_rgb = pred_rgb.permute(1, 2, 0).detach()


    pred_rgb = torch.clip(pred_rgb, 0.0, 1.0)
    lpips_curr_img = lpips(pred_rgb, gt_img, format="HWC").item()
    ssim_curr_img = ssim(pred_rgb, gt_img, format="HWC").item()
    psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    print(prefix + "psnr_image: ", psnr_curr_img)
    print(prefix + "lpips_image: ", lpips_curr_img)
    print(prefix + "ssim_image: ", ssim_curr_img)
    if color_NeRO is not None:
        NeRO_rgb = torch.clip(ret["color_NeRO"], 0.0, 1.0)
        lpips_NeRO_img = lpips(NeRO_rgb, gt_img, format="HWC").item()
        print(prefix + "lpips_image_NeROPBR: ", lpips_NeRO_img)
        ssim_NeRO_img = ssim(NeRO_rgb, gt_img, format="HWC").item()
        print(prefix + "ssim_image_NeROPBR: ", ssim_NeRO_img)
        psnr_NeRO_img = img2psnr(ret["color_NeRO"].detach().cpu(), gt_img)
        print(prefix + "psnr_image_NeROPBR: ", psnr_NeRO_img)
    if color_NeILF is not None:
        NeILF_rgb = torch.clip(ret["color_NeILF"], 0.0, 1.0)
        lpips_NeILF_img = lpips(NeILF_rgb, gt_img, format="HWC").item()
        print(prefix + "lpips_image_NeILFPBR: ", lpips_NeILF_img)
        ssim_NeILF_img = ssim(NeILF_rgb, gt_img, format="HWC").item()
        print(prefix + "ssim_image_NeILFPBR: ", ssim_NeILF_img)
        psnr_NeILF_img = img2psnr(ret["color_NeILF"].detach().cpu(), gt_img)
        print(prefix + "psnr_image_NeILFPBR: ", psnr_NeILF_img)

    return psnr_curr_img, lpips_curr_img, ssim_curr_img


if __name__ == "__main__":
    parser = config.config_parser()
    parser.add_argument("--run_val", action="store_true", help="run on val set")
    args = parser.parse_args()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    eval(args)
