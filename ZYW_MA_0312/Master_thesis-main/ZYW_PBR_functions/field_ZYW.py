import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import nvdiffrast.torch as dr
import mcubes

from utils.base_utils import az_el_to_points, sample_sphere
from utils.raw_utils import linear_to_srgb
from utils.ref_utils import generate_ide_fn

# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

def make_predictor(feats_dim: object, output_dim: object, weight_norm: object = True, activation='sigmoid', exp_max=0.0) -> object:
    if activation == 'sigmoid':
        activation = nn.Sigmoid()
    elif activation=='exp':
        activation = ExpActivation(max_light=exp_max)
    elif activation=='none':
        activation = IdentityActivation()
    elif activation=='relu':
        activation = nn.ReLU()
    else:
        raise NotImplementedError

    run_dim = 256#隐藏层维度
    if weight_norm:
        module=nn.Sequential(
            nn.utils.weight_norm(nn.Linear(feats_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, output_dim)),
            activation,
        )
    else:
        module=nn.Sequential(
            nn.Linear(feats_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, output_dim),
            activation,
        )

    return module

def get_camera_plane_intersection(pts, dirs, poses):
    #计算射线与相机 XoY 平面的交点
    """
    compute the intersection between the rays and the camera XoY plane
    :param pts:      pn,3 点云
    :param dirs:     pn,3 方向
    :param poses:    pn,3,4 姿态矩阵
    :return:
    """
    R, t = poses[:,:,:3], poses[:,:,3:]#分解旋转和平移矩阵

    # transfer into human coordinate
    pts_ = (R @ pts[:,:,None] + t)[..., 0] # pn,3
    dirs_ = (R @ dirs[:,:,None])[..., 0]   # pn,3

    hits = torch.abs(dirs_[..., 2]) > 1e-4#哪些射线的 z 方向分量的绝对值大于 1e-4
    dirs_z = dirs_[:, 2]#提取 dirs_ 的 z 方向分量
    dirs_z[~hits] = 1e-4#将未命中的射线的 z 方向分量设置为 1e-4，以避免后续计算中的除零错误
    dist = -pts_[:, 2] / dirs_z#计算从点到平面的距离 dist
    inter = pts_ + dist.unsqueeze(-1) * dirs_
    return inter, dist, hits

def expected_sin(mean, var):
  """Compute the mean of sin(x), x ~ N(mean, var)."""
  return torch.exp(-0.5 * var) * torch.sin(mean)  # large var -> small value.

def IPE(mean,var,min_deg,max_deg):
    scales = 2**torch.arange(min_deg, max_deg)#傅里叶变换的频率
    shape = mean.shape[:-1] + (-1,)
    scaled_mean = torch.reshape(mean[..., None, :] * scales[:, None], shape)
    scaled_var = torch.reshape(var[..., None, :] * scales[:, None]**2, shape)
    return expected_sin(torch.concat([scaled_mean, scaled_mean + 0.5 * np.pi], dim=-1), torch.concat([scaled_var] * 2, dim=-1))

def offset_points_to_sphere(points):
    """
    points: [N_rays, 3]
    """
    points_norm = torch.norm(points, dim=-1)
    mask = points_norm > 0.999
    if torch.sum(mask)>0:
        points = torch.clone(points)
        points[mask] /= points_norm[mask].unsqueeze(-1)
        points[mask] *= 0.999
        # points[points_norm>0.999] = 0
    return points#点现在都位于单位球面内，或者严格位于球面上

def get_sphere_intersection(pts, dirs):
    # Zhenyi Wan [2025/4/10] Get teh distance between a point with the unit circle
    dtx = torch.sum(pts*dirs,dim=-1,keepdim=True) # rn,1
    xtx = torch.sum(pts**2,dim=-1,keepdim=True) # rn,1
    dist = dtx ** 2 - xtx + 1
    assert torch.sum(dist<0)==0
    dist = -dtx + torch.sqrt(dist+1e-6) # rn,1
    return dist

# this function is borrowed from NeuS
def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def get_weights(sdf_fun, inv_fun, z_vals, origins, dirs):
    points = z_vals.unsqueeze(-1) * dirs.unsqueeze(-2) + origins.unsqueeze(-2) # pn,sn,3
    inv_s = inv_fun(points[:, :-1, :])[..., 0]  # pn,sn-1
    sdf = sdf_fun(points)[..., 0]  # pn,sn

    prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]  # pn,sn-1
    prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
    mid_sdf = (prev_sdf + next_sdf) * 0.5
    cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)  # pn,sn-1
    surface_mask = (cos_val < 0)  # pn,sn-1
    cos_val = torch.clamp(cos_val, max=0)

    dist = next_z_vals - prev_z_vals  # pn,sn-1
    prev_esti_sdf = mid_sdf - cos_val * dist * 0.5  # pn, sn-1
    next_esti_sdf = mid_sdf + cos_val * dist * 0.5
    prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
    next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5) * surface_mask.float()
    weights = alpha * torch.cumprod(torch.cat([torch.ones([alpha.shape[0], 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
    mid_sdf[~surface_mask]=-1.0
    return weights, mid_sdf

def get_intersection(sdf_fun, inv_fun, pts, dirs, sn0=128, sn1=9):
    """
    :param sdf_fun:
    :param inv_fun:
    :param pts:    pn,3
    :param dirs:   pn,3
    :param sn0:
    :param sn1:
    :return:
    """
    inside_mask = torch.norm(pts, dim=-1) < 0.999 # left some margin
    pn, _ = pts.shape
    hit_z_vals = torch.zeros([pn, sn1-1])
    hit_weights = torch.zeros([pn, sn1-1])
    hit_sdf = -torch.ones([pn, sn1-1])
    if torch.sum(inside_mask)>0:
        pts = pts[inside_mask]
        dirs = dirs[inside_mask]
        max_dist = get_sphere_intersection(pts, dirs) # pn,1
        with torch.no_grad():
            z_vals = torch.linspace(0, 1, sn0) # sn0
            z_vals = max_dist * z_vals.unsqueeze(0) # pn,sn0
            weights, mid_sdf = get_weights(sdf_fun, inv_fun, z_vals, pts, dirs) # pn,sn0-1
            z_vals_new = sample_pdf(z_vals, weights, sn1, True) # pn,sn1
            weights, mid_sdf = get_weights(sdf_fun, inv_fun, z_vals_new, pts, dirs) # pn,sn1-1
            z_vals_mid = (z_vals_new[:,1:] + z_vals_new[:,:-1]) * 0.5

        hit_z_vals[inside_mask] = z_vals_mid
        hit_weights[inside_mask] = weights
        hit_sdf[inside_mask] = mid_sdf
    return hit_z_vals, hit_weights, hit_sdf


class AppShadingNetwork(nn.Module):
    default_cfg={
        'human_light': False,
        'sphere_direction': False,
        'light_pos_freq': 8,
        'inner_init': -0.95,
        'roughness_init': 0.0,
        'metallic_init': 0.0,
        'light_exp_max': 0.0,
    }
    def __init__(self, cfg):
        super().__init__()
        self.cfg={**self.default_cfg, **cfg}
        feats_dim = 256

        FG_LUT = torch.from_numpy(np.fromfile('assets/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2))
        self.register_buffer('FG_LUT', FG_LUT)#缓冲区

        self.sph_enc = generate_ide_fn(5)#使用5阶的球面谐波
        self.dir_enc, dir_dim = get_embedder(6, 3)
        self.pos_enc, pos_dim = get_embedder(self.cfg['light_pos_freq'], 3)
        exp_max = self.cfg['light_exp_max']
        # outer lights are direct lights
        if self.cfg['sphere_direction']:
            self.outer_light = make_predictor(72*2, 3, activation='exp', exp_max=exp_max)
        else:
            self.outer_light = make_predictor(72, 3, activation='exp', exp_max=exp_max)
        nn.init.constant_(self.outer_light[-2].bias,np.log(0.5))#初始一个较小的值

        # inner lights are indirect lights
        self.inner_light = make_predictor(pos_dim + 72, 3, activation='exp', exp_max=exp_max)
        nn.init.constant_(self.inner_light[-2].bias, np.log(0.5))
        self.inner_weight = make_predictor(pos_dim + dir_dim, 1, activation='none')
        nn.init.constant_(self.inner_weight[-2].bias, self.cfg['inner_init'])

        # human lights are the lights reflected from the photo capturer
        if self.cfg['human_light']:
            self.human_light_predictor = make_predictor(2 * 2 * 6, 4, activation='exp')
            nn.init.constant_(self.human_light_predictor[-2].bias, np.log(0.01))


    def predict_human_light(self,points, reflective, human_poses, roughness):
        #通过计算光线与相机平面的交点来确定人类光照的影响区域，并根据距离和粗糙度计算光照的均值和方差
        inter, dists, hits = get_camera_plane_intersection(points, reflective, human_poses)
        scale_factor = 0.3
        mean = inter[..., :2] * scale_factor
        var = roughness * (dists[:, None] * scale_factor) ** 2#方差
        hits = hits & (torch.norm(mean, dim=-1) < 1.5) & (dists > 0)#只有当 mean 的范数（即 x 和 y 的距离）小于 1.5 时才视为有效
        hits = hits.float().unsqueeze(-1)
        mean = mean * hits
        var = var * hits

        var = var.expand(mean.shape[0], 2)
        pos_enc = IPE(mean, var, 0, 6)  # 2*2*6
        human_lights = self.human_light_predictor(pos_enc)
        human_lights = human_lights * hits
        human_lights, human_weights = human_lights[..., :3], human_lights[..., 3:]
        human_weights = torch.clamp(human_weights, max=1.0, min=0.0)
        return human_lights, human_weights

    def predict_specular_lights(self, points, reflective, roughness, human_poses, step):
        human_light, human_weight = 0, 0
        ref_roughness = self.sph_enc(reflective, roughness) # Zhenyi Wan [2025/4/10] [N_rays, 72]
        pts = self.pos_enc(points) # Zhenyi Wan [2025/4/10] [N_rays, 27]
        if self.cfg['sphere_direction']:
            sph_points = offset_points_to_sphere(points)#[N_rays,3]
            sph_points = F.normalize(sph_points + reflective * get_sphere_intersection(sph_points, reflective), dim=-1)#[N_rays,3]
            sph_points = self.sph_enc(sph_points, roughness) # Zhenyi Wan [2025/4/10] [N_rays, 72]
            direct_light = self.outer_light(torch.cat([ref_roughness, sph_points], -1)) # Zhenyi Wan [2025/4/10] [N_rays, 3]
        else:
            direct_light = self.outer_light(ref_roughness) # Zhenyi Wan [2025/4/10] [N_rays, 3]

        indirect_light = self.inner_light(torch.cat([pts, ref_roughness], -1))#[N_rays, 3]
        ref_ = self.dir_enc(reflective) # Zhenyi Wan [2025/4/10] [...,15]
        occ_prob = self.inner_weight(torch.cat([pts.detach(), ref_.detach()], -1)) # this is occlusion prob[N_rays,1]
        occ_prob = occ_prob*0.5 + 0.5
        occ_prob_ = torch.clamp(occ_prob,min=0,max=1)

        if self.cfg['human_light']:
            human_light, human_weight = self.predict_human_light(points, reflective, human_poses, roughness)
        else:
            human_light = torch.zeros_like(direct_light)
            human_weight = torch.zeros_like(occ_prob_)

        #计算最终光照
        light = indirect_light * occ_prob_ + (human_light * human_weight + direct_light * (1 - human_weight)) * (1 - occ_prob_)
        indirect_light = indirect_light * occ_prob_
        return light, occ_prob, indirect_light, human_light * human_weight

    def predict_diffuse_lights(self, points, normals):
        roughness = torch.ones([normals.shape[0],1]) # Zhenyi Wan [2025/4/10] [N_rays,1], for diffuse light, all the roughness is set to 1
        ref = self.sph_enc(normals, roughness) # Zhenyi Wan [2025/4/10] [N_rays, 72]
        if self.cfg['sphere_direction']: # Zhenyi Wan [2025/4/10] default cfg in this class, false as default
            sph_points = offset_points_to_sphere(points) # Zhenyi Wan [2025/4/10] [N_rays,3]
            sph_points = F.normalize(sph_points + normals * get_sphere_intersection(sph_points, normals), dim=-1) # Zhenyi Wan [2025/4/10] [N_rays, 3]
            sph_points = self.sph_enc(sph_points, roughness) # Zhenyi Wan [2025/4/10] [N_rays, 72]
            light = self.outer_light(torch.cat([ref, sph_points], -1)) # Zhenyi Wan [2025/4/10] [N_rays, 3]
        else:
            light = self.outer_light(ref) # Zhenyi Wan [2025/4/10] [N_rays, 3]
        return light

    def forward(self, points, roughness, metallic, albedo, normals, ray_d, feature_vectors=None, human_poses=None, inter_results=False, step=None):
        """
        roughness: [N_rays, 1] - Roughness value for each ray (0: smooth, 1: rough).
        metallic: [N_rays, 1] - Metallic factor for each ray (0: non-metallic, 1: metallic).
        albedo: [N_rays, 3] - Albedo (color) for each ray in RGB.
        normals: [N_rays, 3] - Normal vectors for each ray (surface orientation).
        points: [N_rays, 3] - 3D surface points for each ray.
        ray_d: [N_rays, 3]
        """
        # Zhenyi Wan [2025/4/10] Calculate the ray direction, normals, reflection and also NOV for later PBR rendering
        normals = F.normalize(normals, dim=-1)# Zhenyi Wan [2025/4/10] [N_rays, 3]
        view_dirs = -F.normalize(ray_d, dim=-1)
        reflective = torch.sum(view_dirs * normals, -1, keepdim=True) * normals * 2 - view_dirs # Zhenyi Wan [2025/4/10] reflection direction
        NoV = torch.sum(normals * view_dirs, -1, keepdim=True)

        # diffuse light
        diffuse_albedo = (1 - metallic) * albedo # Zhenyi Wan [2025/4/10] [N_rays, 3]
        diffuse_light = self.predict_diffuse_lights(points, normals) # Zhenyi Wan [2025/4/10] [N_rays, 3]
        diffuse_color = diffuse_albedo * diffuse_light # Zhenyi Wan [2025/4/10] [N_rays, 3]

        # specular light
        specular_albedo = 0.04 * (1 - metallic) + metallic * albedo#F0  [N_rays, 3]
        specular_light, occ_prob, indirect_light, human_light = self.predict_specular_lights(points, reflective, roughness, human_poses, step)

        fg_uv = torch.cat([torch.clamp(NoV, min=0.0, max=1.0), torch.clamp(roughness,min=0.0,max=1.0)],-1) # Zhenyi Wan [2025/4/10] [N_rays,2]
        pn, bn = points.shape[0], 1 # Zhenyi Wan [2025/4/10] N_rays,1
        fg_lookup = dr.texture(self.FG_LUT, fg_uv.reshape(1, pn//bn, bn, -1).contiguous(), filter_mode='linear', boundary_mode='clamp').reshape(pn, 2)# Zhenyi Wan [2025/4/10] [N_rays,2]
        specular_ref = (specular_albedo * fg_lookup[:,0:1] + fg_lookup[:,1:2]) # Zhenyi Wan [2025/4/10] [N_rays,1]
        specular_color = specular_ref * specular_light # Zhenyi Wan [2025/4/10] [N_rays,3]

        # integrated together
        color = diffuse_color + specular_color# Zhenyi Wan [2025/4/10] [N_rays,3]

        # gamma correction
        diffuse_color = linear_to_srgb(diffuse_color) # Zhenyi Wan [2025/4/10] [N_rays,3]
        specular_color = linear_to_srgb(specular_color)
        color = linear_to_srgb(color)# Zhenyi Wan [2025/4/10] [N_rays,3]
        color = torch.clamp(color, min=0.0, max=1.0)

        occ_info = {
            'reflective': reflective,
            'occ_prob': occ_prob,
        }

        if inter_results:
            intermediate_results={
                'specular_albedo': specular_albedo,
                'specular_ref': torch.clamp(specular_ref, min=0.0, max=1.0),
                'specular_light': torch.clamp(linear_to_srgb(specular_light), min=0.0, max=1.0),
                'specular_color': torch.clamp(specular_color, min=0.0, max=1.0),

                'diffuse_albedo': diffuse_albedo,
                'diffuse_light': torch.clamp(linear_to_srgb(diffuse_light), min=0.0, max=1.0),
                'diffuse_color': torch.clamp(diffuse_color, min=0.0, max=1.0),

                'metallic': metallic,
                'roughness': roughness,

                'occ_prob': torch.clamp(occ_prob, max=1.0, min=0.0),
                'indirect_light': indirect_light,
            }
            if self.cfg['human_light']:
                intermediate_results['human_light'] = linear_to_srgb(human_light)
            return color, occ_info, intermediate_results
        else:
            return color, occ_info