import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np

from utils.ray_sampling import fibonacci_sphere_sampling
from utils.nn_arch import NeILFMLP

EPS = 1e-7


class NeILFPBR(nn.Module):

    def __init__(self):
        super().__init__()
        # Zhenyi Wan [2025/4/15] In NeILF, for dateset blendedemvs and DTU, outputdim is 1, synthesis dataset is 3. So in our case,
        #I set the out_dims = 1.
        self.neilf_nn = NeILFMLP(
            in_dims=6,
            out_dims=1,
            dims=[128] * 8,
            skip_connection=[4],
            position_insertion=[4],
            weight_norm=False,
            multires_view=6
        )
        self.S = 128
        self.S_evel = 256

    def sample_incident_rays(self, normals, is_training=True):

        if is_training:
            incident_dirs, incident_areas = fibonacci_sphere_sampling(
                normals, self.S, random_rotate=True)
        else:
            incident_dirs, incident_areas = fibonacci_sphere_sampling(
                normals, self.S_evel, random_rotate=False)

        return incident_dirs, incident_areas  # [N, 128, 3], [N, 128, 1]

    def sample_incident_lights(self, points, incident_dirs):

        # point: [N, 3], incident_dirs: [N, 128, 3]
        N, S, _ = incident_dirs.shape # Zhenyi Wan [2025/4/15] N_rays, 128
        points = points.unsqueeze(1).repeat([1, S, 1])  # [N, 128, 3]
        nn_inputs = torch.cat([points, -incident_dirs], axis=2)  # [N, 128, 6]
        nn_inputs = nn_inputs.reshape([-1, 6])  # [N * 128, 6]
        incident_lights = self.neilf_nn(nn_inputs)  # [N * 128, 1]
        # use mono light
        if incident_lights.shape[1] == 1:
            incident_lights = incident_lights.repeat([1, 3]) # Zhenyi Wan [2025/4/15] We use mono light, but it should be appropriate with RGB
        return incident_lights.reshape([N, S, 3])  # [N, 128, 3]


    def rendering_equation(self,
                           output_dirs,
                           normals,
                           base_color,
                           roughness,
                           metallic,
                           incident_lights,
                           incident_dirs,
                           incident_areas):

        # extend all inputs into shape [N, 1, 1/3] for multiple incident lights
        output_dirs = output_dirs.unsqueeze(dim=1)  # [N, 1, 3]
        normal_dirs = normals.unsqueeze(dim=1)  # [N, 1, 3]
        base_color = base_color.unsqueeze(dim=1)  # [N, 1, 3]
        roughness = roughness.unsqueeze(dim=1).reshape(-1, 1)  # [N, 1, 1]
        metallic = metallic.unsqueeze(dim=1).reshape(-1, 1)  # [N, 1, 1]

        def _dot(a, b):
            return (a * b).sum(dim=-1, keepdim=True)  # [N, 1, 1]

        def _f_diffuse(h_d_o, n_d_i, n_d_o, base_color, metallic, roughness):
            return (1 - metallic) * base_color / np.pi  # [N, 1, 1]

        def _f_specular(h_d_n, h_d_o, n_d_i, n_d_o, base_color, roughness, metallic):
            # used in SG, wrongly normalized
            def _d_sg(r, cos):
                r2 = (r * r).clamp(min=EPS)
                amp = 1 / (r2 * np.pi)
                sharp = 2 / r2
                return amp * torch.exp(sharp * (cos - 1))

            D = _d_sg(roughness, h_d_n)

            # Fresnel term F
            F_0 = 0.04 * (1 - metallic) + base_color * metallic  # [N, 1, 3]
            F = F_0 + (1.0 - F_0) * ((1.0 - h_d_o) ** 5)  # [N, S, 1]

            # geometry term V, we use V = G / (4 * cos * cos) here
            def _v_schlick_ggx(r, cos):
                r2 = ((1 + r) ** 2) / 8
                return 0.5 / (cos * (1 - r2) + r2).clamp(min=EPS)

            V = _v_schlick_ggx(roughness, n_d_i) * _v_schlick_ggx(roughness, n_d_o)

            return D * F * V  # [N, S, 1]

        # half vector and all cosines
        half_dirs = incident_dirs + output_dirs  # [N, S, 3]
        half_dirs = Func.normalize(half_dirs, dim=-1)  # [N, S, 3]
        h_d_n = _dot(half_dirs, normal_dirs).clamp(min=0)  # [N, S, 1]
        h_d_o = _dot(half_dirs, output_dirs).clamp(min=0)  # [N, S, 1]
        n_d_i = _dot(normal_dirs, incident_dirs).clamp(min=0)  # [N, S, 1]
        n_d_o = _dot(normal_dirs, output_dirs).clamp(min=0)  # [N, 1, 1]

        f_d = _f_diffuse(h_d_o, n_d_i, n_d_o, base_color, metallic, roughness)  # [N, 1, 3]
        f_s = _f_specular(h_d_n, h_d_o, n_d_i, n_d_o, base_color, roughness, metallic)  # [N, S, 3]

        rgb = ((f_d + f_s) * incident_lights * incident_areas * n_d_i).mean(dim=1)  # [N, 3]

        return rgb

    def forward(self, points, roughness, metallic, albedo, normals, view_dirs, is_training=None):
        """
                roughness: [N_rays, 1] - Roughness value for each ray (0: smooth, 1: rough).
                metallic: [N_rays, 1] - Metallic factor for each ray (0: non-metallic, 1: metallic).
                albedo: [N_rays, 3] - Albedo (color) for each ray in RGB.
                normals: [N_rays, 3] - Normal vectors for each ray (surface orientation).
                points: [N_rays, 3] - 3D surface points for each ray.
                ray_d: [N_rays, 3]
        """
        # Zhenyi Wan [2025/4/15] During training, self.training = True; during Evaluation, self.training = False
        if is_training is None:
            is_training = self.training

        # sample incident rays for the input point
        # Zhenyi Wan [2025/4/15] Sample incoming light directions and weights for shading integration.
        incident_dirs, incident_areas = self.sample_incident_rays(
            normals, is_training)  # [N, 128, 3], [N, 128, 1]

        # sample incident lights for the input point
        incident_lights = self.sample_incident_lights(points, incident_dirs)  # [N, 128, 3]

        # the rendering equation, first pass
        rgb = self.rendering_equation(
            view_dirs, normals, albedo, roughness, metallic,
            incident_lights, incident_dirs, incident_areas)  # [N, 3]

        return rgb