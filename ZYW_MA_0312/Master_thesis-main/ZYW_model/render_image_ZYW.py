import torch
from collections import OrderedDict
from ZYW_model.render_ray_ZYW import render_rays

from ZYW_PBR_functions.field_ZYW import AppShadingNetwork
from ZYW_PBR_functions.PBR_MI_ZYW import NeILFPBR


def render_single_image(
    args,
    ray_sampler,
    ray_batch,
    model,
    projector,
    chunk_size,
    N_samples,
    inv_uniform=False,
    N_importance=0,
    det=False,
    white_bkgd=False,
    render_stride=1,
    featmaps=None,
    ret_alpha=False,
    single_net=False,
    sky_style_code=None,
    sky_style_model=None,
    sky_model=None,
    feature_volume = None,
    mode='val',
    use_updated_prior_depth=False,
    train_depth_prior=None,
    data_mode=None,
):
    """
    :param ray_sampler: RaySamplingSingleImage for this view
    :param model:  {'net_coarse': , 'net_fine': , ...}
    :param chunk_size: number of rays in a chunk
    :param N_samples: samples along each ray (for both coarse and fine model)
    :param inv_uniform: if True, uniformly sample inverse depth for coarse model
    :param N_importance: additional samples along each ray produced by importance sampling (for fine model)
    :param ret_alpha: if True, will return learned 'density' values inferred from the attention maps
    :param single_net: if True, will use single network, can be cued with both coarse and fine points
    :return: {'outputs_coarse': {'rgb': numpy, 'depth': numpy, ...}, 'outputs_fine': {}}
    """
    # Zhenyi Wan [2025/4/17] Render rays into an image

    all_ret = OrderedDict([("outputs_coarse", OrderedDict()), ("outputs_fine", OrderedDict())])

    default_cfg = {
        # shader network for NeRO method
        'shader_config': {},
    }
    cfg = {**default_cfg}

    NeRO_PBR = AppShadingNetwork(cfg['shader_config'])
    NeILF_PBR = NeILFPBR()

    # Zhenyi Wan [2025/4/1] get the number of rays
    N_rays = ray_batch["ray_o"].shape[0]  # 360000 in train, 1440000 in eval

    # Zhenyi Wan [2025/4/1] Use chunk_size to control the number of rays each time to calculate
    for i in range(0, N_rays, chunk_size):# Zhenyi Wan [2025/4/1] chunk_size works as stride
        chunk = OrderedDict()
        for k in ray_batch:
            if k in ["camera", "depth_range", "src_rgbs", "src_cameras"]:
                chunk[k] = ray_batch[k]
            elif ray_batch[k] is not None:
                chunk[k] = ray_batch[k][i : i + chunk_size]
            else:
                chunk[k] = None

        '''
        LinGaoyuan_operation_20240918: In val process, if the data is from training dataset and use_updated_prior_depth is True,
        the updated prior depth will be used
        '''# Zhenyi Wan [2025/4/8] Validation
        if train_depth_prior is not None and use_updated_prior_depth is True:
            train_depth_prior_chunk = train_depth_prior[i : i + chunk_size]
        else:
            train_depth_prior_chunk = None

        ret, _ = render_rays(
            args,
            chunk,
            model,
            featmaps,
            projector=projector,
            N_samples=N_samples,
            inv_uniform=inv_uniform,
            N_importance=N_importance,
            det=det,
            white_bkgd=white_bkgd,
            ret_alpha=ret_alpha,
            single_net=single_net,
            sky_style_code=sky_style_code,
            # sky_style_model=sky_style_model,
            sky_model=sky_model,
            mode = 'val',
            feature_volume=feature_volume,
            use_updated_prior_depth=use_updated_prior_depth,
            train_depth_prior=train_depth_prior_chunk,
            data_mode = data_mode,
        )

        # handle both coarse and fine outputs
        # cache chunk results on cpu
        # Zhenyi Wan [2025/4/17] set the list to be zero at first
        if i == 0:
            for k in ret["outputs_coarse"]:
                if ret["outputs_coarse"][k] is not None:
                    all_ret["outputs_coarse"][k] = []

            if ret["outputs_roughness"] is None:
                all_ret["outputs_roughness"] = None
            else:
                for k in ret["outputs_roughness"]:
                    if ret["outputs_roughness"][k] is not None:
                        all_ret["outputs_roughness"][k] = []

            if ret["outputs_metallic"] is None:
                all_ret["outputs_metallic"] = None
            else:
                for k in ret["outputs_metallic"]:
                    if ret["outputs_metallic"][k] is not None:
                        all_ret["outputs_metallic"][k] = []

            if ret["outputs_albedo"] is None:
                all_ret["outputs_albedo"] = None
            else:
                for k in ret["outputs_albedo"]:
                    if ret["outputs_albedo"][k] is not None:
                        all_ret["outputs_albedo"][k] = []

            if ret["outputs_normals"] is None:
                all_ret["outputs_normals"] = None
            else:
                for k in ret["outputs_normals"]:
                    if ret["outputs_normals"][k] is not None:
                        all_ret["outputs_normals"][k] = []

            if ret["outputs_fine"] is None:
                all_ret["outputs_fine"] = None
            else:
                for k in ret["outputs_fine"]:
                    if ret["outputs_fine"][k] is not None:
                        all_ret["outputs_fine"][k] = []

        for k in ret["outputs_coarse"]:
            if ret["outputs_coarse"][k] is not None:
                all_ret["outputs_coarse"][k].append(ret["outputs_coarse"][k].cpu())

        if ret["outputs_roughness"] is not None:
            for k in ret["outputs_roughness"]:
                if ret["outputs_roughness"][k] is not None:
                    all_ret["outputs_roughness"][k].append(ret["outputs_roughness"][k].cpu())

        if ret["outputs_metallic"] is not None:
            for k in ret["outputs_metallic"]:
                if ret["outputs_metallic"][k] is not None:
                    all_ret["outputs_metallic"][k].append(ret["outputs_metallic"][k].cpu())

        if ret["outputs_albedo"] is not None:
            for k in ret["outputs_albedo"]:
                if ret["outputs_albedo"][k] is not None:
                    all_ret["outputs_albedo"][k].append(ret["outputs_albedo"][k].cpu())

        if ret["outputs_normals"] is not None:
            for k in ret["outputs_normals"]:
                if ret["outputs_normals"][k] is not None:
                    all_ret["outputs_normals"][k].append(ret["outputs_normals"][k].cpu())

        if ret["outputs_fine"] is not None:
            for k in ret["outputs_fine"]:
                if ret["outputs_fine"][k] is not None:
                    all_ret["outputs_fine"][k].append(ret["outputs_fine"][k].cpu())

        points = torch.cat(all_ret["outputs_coarse"]["points"], dim=0)
        ray_d = ray_batch["ray_d"]
        view_d = -ray_d

        if all_ret["outputs_albedo"] is not None and all_ret["outputs_normals"] is not None and all_ret["outputs_roughness"] is not None and all_ret["outputs_metallic"] is not None:
            pred_roughness = torch.cat(all_ret["outputs_roughness"]["roughness"], dim=0)
            pred_metallic = torch.cat(all_ret["outputs_metallic"]["metallic"], dim=0)
            pred_albedo = torch.cat(all_ret["outputs_albedo"]["albedo"], dim=0)
            pred_normals = torch.cat(all_ret["outputs_normals"]["normals"], dim=0)
            if args.use_NeROPBR is True:
                color_NeRO, _ = NeRO_PBR(points, pred_roughness, pred_metallic, pred_albedo, pred_normals, ray_d)
            if args.use_NeILFPBR is True:
                color_NeILF = NeILF_PBR(points, pred_roughness, pred_metallic, pred_albedo, pred_normals, view_d)

    rgb_strided = torch.ones(ray_sampler.H, ray_sampler.W, 3)[::render_stride, ::render_stride, :] # Zhenyi Wan [2025/4/17] [H//3,W//s,3]
    # Zhenyi Wan [2025/4/17] add material stride, the channel is 1 for roughness and metallic
    material_strided = torch.ones(ray_sampler.H, ray_sampler.W, 1)[::render_stride, ::render_stride, :]# Zhenyi Wan [2025/4/17] [H//3,W//s,1]
    # merge chunk results and reshape
    for k in all_ret["outputs_coarse"]:
        if k == "random_sigma":
            continue

        'LinGaoyuan_operation_20240906: do not cat depth_cov in validate phase'
        if k == "depth_cov":
            continue

        tmp = torch.cat(all_ret["outputs_coarse"][k], dim=0).reshape(
            (rgb_strided.shape[0], rgb_strided.shape[1], -1)
        )
        all_ret["outputs_coarse"][k] = tmp.squeeze()

    if ret["outputs_roughness"] is not None:
        for k in all_ret["outputs_roughness"]:
            tmp = torch.cat(all_ret["outputs_roughness"][k], dim=0).reshape(
                (material_strided.shape[0], material_strided.shape[1], -1)
            )
            all_ret["outputs_roughness"][k] = tmp  # Zhenyi Wan [2025/4/17] the roughness channel chould not be deleted[H//s, W//s, 1],[H//s, W//s, N_samples]

    if ret["outputs_metallic"] is not None:
        for k in all_ret["outputs_metallic"]:
            tmp = torch.cat(all_ret["outputs_metallic"][k], dim=0).reshape(
                (material_strided.shape[0], material_strided.shape[1], -1)
            )
            all_ret["outputs_metallic"][k] = tmp


    if ret["outputs_albedo"] is not None:
        for k in all_ret["outputs_albedo"]:
            tmp = torch.cat(all_ret["outputs_albedo"][k], dim=0).reshape(
                (rgb_strided.shape[0], rgb_strided.shape[1], -1)
            )
            all_ret["outputs_albedo"][k] = tmp

    if ret["outputs_normals"] is not None:
        for k in all_ret["outputs_normals"]:
            tmp = torch.cat(all_ret["outputs_normals"][k], dim=0).reshape(
                (rgb_strided.shape[0], rgb_strided.shape[1], -1)
            )
            all_ret["outputs_normals"][k] = tmp

    if color_NeRO is not None:
        tmp = color_NeRO.reshape((rgb_strided.shape[0], rgb_strided.shape[1], -1))
        all_ret["color_NeRO"] = tmp

    if color_NeILF is not None:
        tmp = color_NeILF.reshape((rgb_strided.shape[0], rgb_strided.shape[1], -1))
        all_ret["color_NeILF"] = tmp

    # TODO: if invalid: replace with white
    # all_ret["outputs_coarse"]["rgb"][all_ret["outputs_coarse"]["mask"] == 0] = 1.0
    if all_ret["outputs_fine"] is not None:
        for k in all_ret["outputs_fine"]:
            if k == "random_sigma":
                continue

            'LinGaoyuan_operation_20240906: do not cat depth_cov in validate phase'
            if k == "depth_cov":
                continue

            tmp = torch.cat(all_ret["outputs_fine"][k], dim=0).reshape(
                (rgb_strided.shape[0], rgb_strided.shape[1], -1)
            )

            all_ret["outputs_fine"][k] = tmp.squeeze()

    return all_ret
