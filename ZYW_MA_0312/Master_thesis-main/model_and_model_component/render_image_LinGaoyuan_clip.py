import torch
from collections import OrderedDict
from render_ray_LinGaoyuan_clip import render_rays


def render_single_image(
    args,
    ray_sampler,
    ray_batch,
    # model,
    projector,
    chunk_size,
    N_samples,
    model=None,
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
    latent_code=None,
    deformation_network=None,
    pos_embedder=None,
    latent_shape_code=None,
    latent_appearance_code=None,
    # model_building=None,
    # model_street=None,
    # featmaps_building=None,
    # featmaps_street=None,
    # feature_volume_building=None,
    # feature_volume_street=None,
    # latent_code_building=None,
    # latent_code_street=None,
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

    all_ret = OrderedDict([("outputs_coarse", OrderedDict()), ("outputs_fine", OrderedDict())])

    N_rays = ray_batch["ray_o"].shape[0]  # 360000 in train, 1440000 in eval

    '''
    LinGaoyuan_operation_20241001: if we want to generate the rendering image of car, the input variable latent_code is None and shape code&appearance code is exist.
    we need to cat them together as latent code
    '''
    if len(latent_shape_code.shape) == 1:
        latent_shape_code = latent_shape_code.unsqueeze(0)
    if len(latent_appearance_code.shape) == 1:
        latent_appearance_code = latent_appearance_code.unsqueeze(0)
    latent_code = torch.cat((latent_shape_code,latent_appearance_code), dim=0)

    for i in range(0, N_rays, chunk_size):
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
        '''
        if train_depth_prior is not None and use_updated_prior_depth is True:
            train_depth_prior_chunk = train_depth_prior[i : i + chunk_size]
        else:
            train_depth_prior_chunk = None

        ret, _ = render_rays(
            args,
            chunk,
            # featmaps,
            projector=projector,
            N_samples=N_samples,
            model=model,
            featmaps=featmaps,
            inv_uniform=inv_uniform,
            N_importance=N_importance,
            det=det,
            white_bkgd=white_bkgd,
            ret_alpha=ret_alpha,
            single_net=single_net,
            sky_style_code=sky_style_code,
            # sky_style_model=sky_style_model,
            sky_model=sky_model,
            mode = mode,
            feature_volume=feature_volume,
            use_updated_prior_depth=use_updated_prior_depth,
            train_depth_prior=train_depth_prior_chunk,
            data_mode = data_mode,
            latent_code=latent_code,
        )

        # handle both coarse and fine outputs
        # cache chunk results on cpu
        if i == 0:
            for k in ret["outputs_coarse"]:
                if ret["outputs_coarse"][k] is not None:
                    all_ret["outputs_coarse"][k] = []

            if ret["outputs_fine"] is None:
                all_ret["outputs_fine"] = None
            else:
                for k in ret["outputs_fine"]:
                    if ret["outputs_fine"][k] is not None:
                        all_ret["outputs_fine"][k] = []

        for k in ret["outputs_coarse"]:
            if ret["outputs_coarse"][k] is not None:
                all_ret["outputs_coarse"][k].append(ret["outputs_coarse"][k].cpu())

        if ret["outputs_fine"] is not None:
            for k in ret["outputs_fine"]:
                if ret["outputs_fine"][k] is not None:
                    all_ret["outputs_fine"][k].append(ret["outputs_fine"][k].cpu())

    rgb_strided = torch.ones(ray_sampler.H, ray_sampler.W, 3)[::render_stride, ::render_stride, :]
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
