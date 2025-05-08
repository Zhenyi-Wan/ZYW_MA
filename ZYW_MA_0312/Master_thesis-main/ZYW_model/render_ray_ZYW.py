import torch
from collections import OrderedDict
from LinGaoyuan_function.unbounded2bounded import (SceneContraction, contract_to_unisphere_LinGaoyuan,
                                                   contract_to_unisphere_LinGaoyuan_xuyan)
from model_and_model_component.ReTR_model_LinGaoyuan import LinGaoyuan_ReTR_model

from ZYW_PBR_functions.field_ZYW import AppShadingNetwork
from ZYW_PBR_functions.PBR_MI_ZYW import NeILFPBR
# import imaginaire.model_utils.gancraft.voxlib as voxlib

########################################################################################################################
# helper functions for nerf ray rendering
########################################################################################################################


def sample_pdf(bins, weights, N_samples, det=False):
    """
    :param bins: tensor of shape [N_rays, M+1], M is the number of bins
    :param weights: tensor of shape [N_rays, M]
    :param N_samples: number of samples along each ray
    :param det: if True, will perform deterministic sampling
    :return: [N_rays, N_samples]
    """

    M = weights.shape[1]
    weights += 1e-5
    # Get pdf
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)  # [N_rays, M]
    cdf = torch.cumsum(pdf, dim=-1)  # [N_rays, M]
    cdf = torch.cat([torch.zeros_like(cdf[:, 0:1]), cdf], dim=-1)  # [N_rays, M+1]

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, N_samples, device=bins.device)
        u = u.unsqueeze(0).repeat(bins.shape[0], 1)  # [N_rays, N_samples]
    else:
        u = torch.rand(bins.shape[0], N_samples, device=bins.device)

    # Invert CDF
    above_inds = torch.zeros_like(u, dtype=torch.long)  # [N_rays, N_samples]
    for i in range(M):
        above_inds += (u >= cdf[:, i : i + 1]).long()

    # random sample inside each bin
    below_inds = torch.clamp(above_inds - 1, min=0)
    inds_g = torch.stack((below_inds, above_inds), dim=2)  # [N_rays, N_samples, 2]

    cdf = cdf.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
    cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    bins = bins.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
    bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    # t = (u-cdf_g[:, :, 0]) / (cdf_g[:, :, 1] - cdf_g[:, :, 0] + TINY_NUMBER)  # [N_rays, N_samples]
    # fix numeric issue
    denom = cdf_g[:, :, 1] - cdf_g[:, :, 0]  # [N_rays, N_samples]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[:, :, 0]) / denom

    samples = bins_g[:, :, 0] + t * (bins_g[:, :, 1] - bins_g[:, :, 0])

    return samples


def sample_along_camera_ray(ray_o, ray_d, depth_range, N_samples, inv_uniform=False, det=False):
    """
    :param ray_o: origin of the ray in scene coordinate system; tensor of shape [N_rays, 3]
    :param ray_d: homogeneous ray direction vectors in scene coordinate system; tensor of shape [N_rays, 3]
    :param depth_range: [near_depth, far_depth]
    :param inv_uniform: if True, uniformly sampling inverse depth
    :param det: if True, will perform deterministic sampling
    :return: tensor of shape [N_rays, N_samples, 3]
    """
    # will sample inside [near_depth, far_depth]
    # assume the nearest possible depth is at least (min_ratio * depth)
    near_depth_value = depth_range[0, 0]
    far_depth_value = depth_range[0, 1]
    assert near_depth_value > 0 and far_depth_value > 0 and far_depth_value > near_depth_value

    near_depth = near_depth_value * torch.ones_like(ray_d[..., 0])# Zhenyi Wan [2025/4/8] [N_rays]

    far_depth = far_depth_value * torch.ones_like(ray_d[..., 0])# Zhenyi Wan [2025/4/8] [N_rays]

    # Zhenyi Wan [2025/4/8] Get the z_vals inverse or uniformly
    if inv_uniform:
        start = 1.0 / near_depth  # [N_rays,]
        step = (1.0 / far_depth - start) / (N_samples - 1)
        inv_z_vals = torch.stack(
            [start + i * step for i in range(N_samples)], dim=1
        )  # [N_rays, N_samples]
        z_vals = 1.0 / inv_z_vals
    else:
        start = near_depth
        step = (far_depth - near_depth) / (N_samples - 1)
        z_vals = torch.stack(
            [start + i * step for i in range(N_samples)], dim=1
        )  # [N_rays, N_samples]

    # Zhenyi Wan [2025/4/8] add some random disturbance. Random sampling within the interval
    if not det:
        # get intervals between samples
        mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])# Zhenyi Wan [2025/4/8] [N_rays, N_samples - 1]
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)# Zhenyi Wan [2025/4/8] [N_rays, N_samples]
        lower = torch.cat([z_vals[:, 0:1], mids], dim=-1)# Zhenyi Wan [2025/4/8] [N_rays, N_samples]
        # uniform samples in those intervals
        t_rand = torch.rand_like(z_vals)# Zhenyi Wan [2025/4/8] [N_rays, N_samples]
        z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]

    ray_d = ray_d.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, 3]
    ray_o = ray_o.unsqueeze(1).repeat(1, N_samples, 1)
    pts = z_vals.unsqueeze(2) * ray_d + ray_o  # [N_rays, N_samples, 3]
    return pts, z_vals


def sample_prior_depth_perturb(ray_o, ray_d, depth_prior, depth_offset_ratio = 0.05, N_samples_d = 32, inv_uniform=False, det=False):
    """
    :param ray_o: origin of the ray in scene coordinate system; tensor of shape [N_rays, 3]
   :param ray_d: homogeneous ray direction vectors in scene coordinate system; tensor of shape [N_rays, 3]
   :param depth_prior: [N_rays, 1]
   :param depth_offset_ratio: scale of depth perturbation, default is 0.05, indicates the offset of the depth value
   :param N_samples_d: numbers of points along a ray, default is 32
   :return: tensor of shape [N_rays, N_samples, 3]
    """
    mids = depth_prior # Zhenyi Wan [2025/4/8] [N_rays, 1]
    uppers = mids*(1+depth_offset_ratio)
    lowers = mids*(1-depth_offset_ratio)

    batch_size = depth_prior.size(0)# Zhenyi Wan [2025/4/8] N_rays
    z_vals = torch.zeros((batch_size, N_samples_d)).cuda()# Zhenyi Wan [2025/4/8] [N_rays, N_samples_d]

    for i in range(batch_size):
        upper = uppers[i, 0]
        lower = lowers[i, 0]
        z_vals[i] = torch.linspace(lower, upper, N_samples_d)


    # z_vals = []
    # for i in range(len(mids)):
    #     upper = uppers[i,0]
    #     lower = lowers[i,0]
    #     z_val = torch.linspace(lower, upper, N_samples_d).unsqueeze(0)
    #     if i==0:
    #         z_vals = z_val
    #     else:
    #         z_vals = torch.cat((z_vals, z_val),dim=0)

    # ray_d = ray_d.unsqueeze(1).repeat(1, N_samples_d, 1)  # [N_rays, N_samples, 3]
    # ray_o = ray_o.unsqueeze(1).repeat(1, N_samples_d, 1)
    ray_d = ray_d[..., None,:].repeat(1, N_samples_d, 1)  # [N_rays, N_samples, 3]
    ray_o = ray_o[..., None,:].repeat(1, N_samples_d, 1)

    pts = ray_o + ray_d * z_vals[..., :, None]

    return pts, z_vals

def sample_pts_with_z_vals(ray_o, ray_d, z_vals):
    N_samples = z_vals.size(1)

    ray_d = ray_d[..., None,:].repeat(1, N_samples, 1)  # [N_rays, N_samples, 3]
    ray_o = ray_o[..., None,:].repeat(1, N_samples, 1)

    pts = ray_o + ray_d * z_vals[..., :, None]

    return pts


########################################################################################################################
# ray rendering of nerf
########################################################################################################################


def raw2outputs(raw, z_vals, mask, white_bkgd=False):
    """
    :param raw: raw network output; tensor of shape [N_rays, N_samples, 4]
    :param z_vals: depth of point samples along rays; tensor of shape [N_rays, N_samples]
    :param ray_d: [N_rays, 3]
    :return: {'rgb': [N_rays, 3], 'depth': [N_rays,], 'weights': [N_rays,], 'depth_std': [N_rays,]}
    """
    rgb = raw[:, :, :3]  # [N_rays, N_samples, 3]
    sigma = raw[:, :, 3]  # [N_rays, N_samples]

    # note: we did not use the intervals here, because in practice different scenes from COLMAP can have
    # very different scales, and using interval can affect the model's generalization ability.
    # Therefore we don't use the intervals for both training and evaluation.
    sigma2alpha = lambda sigma, dists: 1.0 - torch.exp(-sigma)

    # point samples are ordered with increasing depth
    # interval between samples
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat((dists, dists[:, -1:]), dim=-1)  # [N_rays, N_samples]

    alpha = sigma2alpha(sigma, dists)  # [N_rays, N_samples]

    # Eq. (3): T
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)[:, :-1]  # [N_rays, N_samples-1]
    T = torch.cat((torch.ones_like(T[:, 0:1]), T), dim=-1)  # [N_rays, N_samples]

    # maths show weights, and summation of weights along a ray, are always inside [0, 1]
    weights = alpha * T  # [N_rays, N_samples]
    rgb_map = torch.sum(weights.unsqueeze(2) * rgb, dim=1)  # [N_rays, 3]

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - torch.sum(weights, dim=-1, keepdim=True))

    mask = (
        mask.float().sum(dim=1) > 8
    )  # should at least have 8 valid observation on the ray, otherwise don't consider its loss
    depth_map = torch.sum(weights * z_vals, dim=-1)  # [N_rays,]

    ret = OrderedDict(
        [
            ("rgb", rgb_map),
            ("depth", depth_map),
            ("weights", weights),  # used for importance sampling of fine samples
            ("mask", mask),
            ("alpha", alpha),
            ("z_vals", z_vals),
        ]
    )

    return ret


def sample_fine_pts(inv_uniform, N_importance, det, N_samples, ray_batch, weights, z_vals):
    if inv_uniform:
        inv_z_vals = 1.0 / z_vals
        inv_z_vals_mid = 0.5 * (inv_z_vals[:, 1:] + inv_z_vals[:, :-1])  # [N_rays, N_samples-1]
        weights = weights[:, 1:-1]  # [N_rays, N_samples-2]
        inv_z_vals = sample_pdf(
            bins=torch.flip(inv_z_vals_mid, dims=[1]),
            weights=torch.flip(weights, dims=[1]),
            N_samples=N_importance,
            det=det,
        )  # [N_rays, N_importance]
        z_samples = 1.0 / inv_z_vals
    else:
        # take mid-points of depth samples
        z_vals_mid = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])  # [N_rays, N_samples-1]
        weights = weights[:, 1:-1]  # [N_rays, N_samples-2]
        z_samples = sample_pdf(
            bins=z_vals_mid, weights=weights, N_samples=N_importance, det=det
        )  # [N_rays, N_importance]

    z_vals = torch.cat((z_vals, z_samples), dim=-1)  # [N_rays, N_samples + N_importance]

    # samples are sorted with increasing depth
    z_vals, _ = torch.sort(z_vals, dim=-1)
    N_total_samples = N_samples + N_importance

    viewdirs = ray_batch["ray_d"].unsqueeze(1).repeat(1, N_total_samples, 1)
    ray_o = ray_batch["ray_o"].unsqueeze(1).repeat(1, N_total_samples, 1)
    pts = z_vals.unsqueeze(2) * viewdirs + ray_o  # [N_rays, N_samples + N_importance, 3]
    return pts, z_vals


def render_rays(
    args,
    ray_batch,
    model,
    featmaps,
    projector,
    N_samples,
    inv_uniform=False,
    N_importance=0,
    det=False,
    white_bkgd=False,
    ret_alpha=False,
    single_net=True,
    sky_style_code = None,
    sky_style_model = None,
    sky_model = None,
    mode = 'train',
    use_updated_prior_depth = False,
    train_depth_prior = None,
    feature_volume = None,
    data_mode = None,
    # retr_model = None,
):
    """
    :param ray_batch: {'ray_o': [N_rays, 3] , 'ray_d': [N_rays, 3], 'view_dir': [N_rays, 2]}
    :param model:  {'net_coarse':  , 'net_fine': }
    :param N_samples: samples along each ray (for both coarse and fine model)
    :param inv_uniform: if True, uniformly sample inverse depth for coarse model
    :param N_importance: additional samples along each ray produced by importance sampling (for fine model)
    :param det: if True, will deterministicly sample depths
    :param ret_alpha: if True, will return learned 'density' values inferred from the attention maps
    :param single_net: if True, will use single network, can be cued with both coarse and fine points
    :return: {'outputs_coarse': {}, 'outputs_fine': {}}
    """
    # Initialize shader networks (equivalent to Criterion's initialization)
    shader_config = {}  # Default shader configuration
    NeRO_PBR = AppShadingNetwork(shader_config)
    NeILF_PBR = NeILFPBR()

    ret = {"outputs_coarse": None, "outputs_fine": None}
    ray_o, ray_d = ray_batch["ray_o"], ray_batch["ray_d"]

    sky_mask = ray_batch["sky_mask"]  # sky area = 0, other area = 1

    # pts: [N_rays, N_samples, 3]
    # z_vals: [N_rays, N_samples]


    # Zhenyi Wan [2025/4/8] Part1: Get the randomly chosen z_vals and the corresponding pts
    pts, z_vals = sample_along_camera_ray(
        ray_o=ray_o,
        ray_d=ray_d,
        depth_range=ray_batch["depth_range"],
        N_samples=N_samples,
        inv_uniform=inv_uniform,
        det=det,
    )

    'LinGaoyuan_operation_20240920: add a new if condition: when data_mode is val use depth_value from ray_batch as depth_prior'
    if use_updated_prior_depth is False or data_mode == 'val':
        depth_prior = ray_batch["depth_value"]
    elif mode == 'train':
        depth_prior = train_depth_prior[ray_batch['selected_inds']]
    else:
        depth_prior = train_depth_prior

    N_samples_d = args.N_samples_depth

    'LinGaoyuan_operation_20240907: the uniform sampling will be used before training epoch reach preset value, after that the prior depth guided sampling is used'
    if args.sample_with_prior_depth is True and use_updated_prior_depth is True:
        pts_with_prior_depth, z_vals_with_prior_depth = sample_prior_depth_perturb(ray_o, ray_d, depth_prior, depth_offset_ratio=0.2,
                                             N_samples_d=N_samples_d, inv_uniform=inv_uniform, det=det)

        z_vals_total = torch.cat((z_vals, z_vals_with_prior_depth), dim=-1)# Zhenyi Wan [2025/4/8] [N_rays, N_samples+N_samples_d]
        z_vals_total, z_vals_total_indices = torch.sort(z_vals_total, dim=-1)

        pts_total = sample_pts_with_z_vals(ray_o, ray_d, z_vals_total)

        pts = pts_with_prior_depth
        z_vals = z_vals_with_prior_depth
        # pts = pts_total
        # z_vals = z_vals_total


    # Zhenyi Wan [2025/4/8] Decide a contraction type for the points
    'the unbounded function should be used for the pts after pts is generated from sample_along_camera_ray()'
    contraction_type = args.contraction_type  # 'zhengzhisheng' of 'nerfstudio'

    if contraction_type == 'nerfstudio':
        scene_contraction = SceneContraction(order=float("inf"))
        pts = scene_contraction(pts)
    elif contraction_type == 'zhengzhisheng':
        pts = contract_to_unisphere_LinGaoyuan(pts)
    elif contraction_type == 'xuyan':
        pts = contract_to_unisphere_LinGaoyuan_xuyan(pts) # Zhenyi Wan [2025/5/4] [N_rays, N_samples, 3]

    N_rays, N_samples = pts.shape[:2]

    # Zhenyi Wan [2025/4/8] Use the model,change the ReTR part. add additonal BRDF outputs
    if args.use_retr_model is True:
        if args.use_retr_feature_extractor is True:
            if args.use_volume_feature is not True:
                'LinGaoyuan_20240930: retr model + retr feature extractor'
                rgb_feat, ray_diff, mask = projector.compute(
                    pts,
                    ray_batch["camera"],
                    ray_batch["src_rgbs"],
                    ray_batch["src_cameras"],
                    featmaps=featmaps,
                )
                rgb = model.net_coarse(pts, ray_batch, rgb_feat, z_vals, mask, ray_d, ray_diff, ret_alpha=ret_alpha)
                if args.BRDF_model is True:
                    BRDF_Buffer = model.net_coarse.forward_BDRF(pts, ray_batch, rgb_feat, z_vals, mask, ray_d, ray_diff,
                                                                ret_alpha=ret_alpha)
                else:
                    BRDF_Buffer = None

                # rgb = model.net_coarse.forward_retr(pts, ray_batch, featmaps, z_vals, ret_alpha=ret_alpha)
            else:
                'LinGaoyuan_20240930: retr model + retr feature extractor + feature volume'
                rgb_feat, ray_diff, mask = projector.compute(
                    pts,
                    ray_batch["camera"],
                    ray_batch["src_rgbs"],
                    ray_batch["src_cameras"],
                    featmaps=featmaps,
                )
                rgb = model.net_coarse.forward_retr(pts, ray_batch, rgb_feat, z_vals, mask, ray_d, ray_diff, ret_alpha=ret_alpha, fea_volume=feature_volume)
                if args.BRDF_model is True:
                    BRDF_Buffer = model.net_coarse.forward_BDRF(pts, ray_batch, rgb_feat, z_vals, mask, ray_d, ray_diff,
                                                                ret_alpha=ret_alpha)
                else:
                    BRDF_Buffer = None
                # rgb = model.net_coarse.forward_retr(pts, ray_batch, featmaps, z_vals, fea_volume=feature_volume, ret_alpha=ret_alpha)
        else:
            'LinGaoyuan_20240930: retr model + model_and_model_component feature extractor'
            rgb_feat, ray_diff, mask = projector.compute(
                pts,
                ray_batch["camera"],
                ray_batch["src_rgbs"],
                ray_batch["src_cameras"],
                featmaps=featmaps[0],
            )  # [N_rays, N_samples, N_views, x]
            # TODO: include pixel mask in ray transformer
            # pixel_mask = (
            #     mask[..., 0].sum(dim=2) > 1
            # )  # [N_rays, N_samples], should at least have 2 observations

            rgb = model.net_coarse(pts, ray_batch, rgb_feat, z_vals, mask, ray_d, ray_diff, ret_alpha=ret_alpha)
            if args.BRDF_model is True:
                BRDF_Buffer = model.net_coarse.forward_BDRF(pts, ray_batch, rgb_feat, z_vals, mask, ray_d, ray_diff,
                                                            ret_alpha=ret_alpha)
            else:
                BRDF_Buffer = None
    else:
        'LinGaoyuan_20240930: model_and_model_component model + model_and_model_component feature extractor'
        # Zhenyi Wan [2025/4/8] GNT model
        rgb_feat, ray_diff, mask = projector.compute(
            pts,
            ray_batch["camera"],
            ray_batch["src_rgbs"],
            ray_batch["src_cameras"],
            featmaps=featmaps[0],
        )
        rgb = model.net_coarse(rgb_feat, ray_diff, mask, pts, ray_d)

    if ret_alpha:
        rgb, weights = rgb[:, 0:3], rgb[:, 3:]
        depth_map = torch.sum(weights * z_vals, dim=-1)# Zhenyi Wan [2025/4/9](N_rand,)
        PBR_pts = depth_map.unsqueeze(-1) * ray_d + ray_o # Zhenyi Wan [2025/4/10] (N_rand,3)
        # Zhenyi Wan [2025/4/8] Extract BRDF values and weight
        if BRDF_Buffer is not None:
            metallic, metallic_weights = BRDF_Buffer["metallic"][:, 0:1], BRDF_Buffer["metallic"][:, 1:]# Zhenyi Wan [2025/4/9] [N_rand, 1]; [N_rand, N_samples]
            roughness, roughness_weights = BRDF_Buffer["roughness"][:, 0:1], BRDF_Buffer["roughness"][:, 1:]
            albedo, albedo_weights = BRDF_Buffer["albedo"][:, 0:3], BRDF_Buffer["albedo"][:, 3:]# Zhenyi Wan [2025/4/9] [N_rand, 3]; [N_rand, N_samples]
            normals, normals_weights = BRDF_Buffer["normals"][:, 0:3], BRDF_Buffer["normals"][:, 3:]

            normals = torch.nn.functional.normalize(normals, dim=-1) # Zhenyi Wan [2025/4/22]
            metallic = torch.sigmoid(metallic)
            roughness = torch.sigmoid(roughness)
            albedo = torch.sigmoid(albedo)

        'LinGaoyuan_operation_20240906: add cov of depth prediction based on Uncle SLAM formular 5'
        depth_pred = depth_map[..., None]
        depth_cov = torch.sqrt(torch.sum(weights*(z_vals-depth_pred)*(z_vals-depth_pred)))

        # depth_img = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0

        # depth_sky = depth_map*(1-sky_mask)
        depth_sky = None
    else:
        weights = None
        depth_map = None
        PBR_pts = None

        if BRDF_Buffer is not None:
            metallic = BRDF_Buffer["metallic"]
            roughness = BRDF_Buffer["roughness"]
            albedo = BRDF_Buffer["albedo"]
            normals = BRDF_Buffer["normals"]
            metallic_weights = None
            roughness_weights = None
            albedo_weights = None
            normals_weights = None

        depth_cov = None
        depth_sky = None

    'operation of sky'
    rgb_sky, sky_style_code = sky_model(ray_d, sky_style_code.cuda(), sky_mask)

    z = sky_style_code.detach()

    ret["color_NeRO"] = None
    ret["color_NeILF"] = None
    if args.use_NeROPBR is True:
        # Zhenyi Wan [2025/5/8] NeRO PBR, which uses traditional BRDF render method.
        start_time_NeRO = time.time()  # Start timing
        color_NeRO, _ = NeRO_PBR(PBR_pts, roughness, metallic, albedo, normals, ray_d)
        end_time_NeRO = time.time()  # End timing
        time_NeRO = end_time_NeRO - start_time_NeRO
        ret["color_NeRO"] = {"color_NeRO":color_NeRO, "time_NeRO":time_NeRO}

    if args.use_NeILFPBR is True:
        start_time_NeILF = time.time()  # Start timing
        view_d = -ray_d # Zhenyi Wan [2025/4/15] View direction is the opposite to ray_d
        # Zhenyi Wan [2025/54/8] NeILF PBR, which uses traditional BRDF render method.
        color_NeILF, _ = NeILF_PBR(PBR_pts, roughness, metallic, albedo, normals, view_d)
        end_time_NeILF = time.time()  # End timing
        time_NeILF = end_time_NeILF - start_time_NeILF
        ret["color_NeILF"] = {"color_NeILF":color_NeILF, "time_NeILF":time_NeILF}



    # sky_style_code = sky_style_model(sky_style_code.cuda())
    #
    # # sky_raydirs_in = voxlib.positional_encoding(ray_d, 5, -1, True)
    #
    # skynet_out_color = sky_model(ray_d, sky_style_code)
    #
    # 'set the color of no sky area to black in order to do not affect the rendering rgb img'
    # black_background = torch.zeros((3,)).cuda()
    # # sky_mask = ray_batch["sky_mask"].cuda()  # sky area = 0, other area = 1
    # skynet_out_c = skynet_out_color * (1.0 - sky_mask) + black_background * (sky_mask)
    # # rgb_sky = torch.clamp(skynet_out_c, 0, 1)
    #
    # # softmax_func = torch.nn.Softmax(dim=1)
    # # rgb_sky = softmax_func(skynet_out_c)
    #
    # # rgb_sky = (skynet_out_c-skynet_out_c.min())
    # # rgb_sky = rgb_sky/rgb_sky.max()
    #
    # rgb_sky = skynet_out_c
    #
    # # 'combine rgb_sky and rendering rgb img to one img'
    # # if mode == 'train':
    # #     rgb = rgb + rgb_sky

    # Zhenyi Wan [2025/4/10] add the pts output for PBR rendering
    ret["outputs_coarse"] = {"rgb": rgb, "weights": weights, "depth": depth_map, "rgb_sky": rgb_sky, "depth_sky": depth_sky, "depth_cov": depth_cov, "points":PBR_pts}
    # Zhenyi Wan [2025/4/9] return BRDF_Buffer
    if BRDF_Buffer is not None:
        ret["outputs_roughness"] = {"roughness":roughness, "roughness_weights": roughness_weights}
        ret["outputs_metallic"] = {"metallic":metallic, "metallic_weights": metallic_weights}
        ret["outputs_albedo"] = {"albedo":albedo, "albedo_weights": albedo_weights}
        ret["outputs_normals"] = {"normals":normals, "normals_weights": normals_weights}
    else:
        ret["outputs_roughness"] = None
        ret["outputs_metallic"] = None
        ret["outputs_albedo"] = None
        ret["outputs_normals"] = None

    if N_importance > 0:
        # detach since we would like to decouple the coarse and fine networks
        weights = ret["outputs_coarse"]["weights"].clone().detach()  # [N_rays, N_samples]
        'LinGaoyuan_20240830: return N_samples+N_importance sampled point in each ray'
        pts, z_vals = sample_fine_pts(
            inv_uniform, N_importance, det, N_samples, ray_batch, weights, z_vals
        )

        rgb_feat_sampled, ray_diff, mask = projector.compute(
            pts,
            ray_batch["camera"],
            ray_batch["src_rgbs"],
            ray_batch["src_cameras"],
            featmaps=featmaps[1],
        )

        # TODO: Include pixel mask in ray transformer
        # pixel_mask = (
        #     mask[..., 0].sum(dim=2) > 1
        # )  # [N_rays, N_samples]. should at least have 2 observations

        if single_net:
            rgb = model.net_coarse(rgb_feat_sampled, ray_diff, mask, pts, ray_d)
        else:
            rgb = model.net_fine(rgb_feat_sampled, ray_diff, mask, pts, ray_d)
        rgb, weights = rgb[:, 0:3], rgb[:, 3:]
        depth_map = torch.sum(weights * z_vals, dim=-1)
        ret["outputs_fine"] = {"rgb": rgb, "weights": weights, "depth": depth_map}

    return ret, z
