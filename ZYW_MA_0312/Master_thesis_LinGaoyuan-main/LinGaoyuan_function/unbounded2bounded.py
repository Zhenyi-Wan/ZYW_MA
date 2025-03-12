# import jax
# import jax.numpy as jnp
# from nerfacc import ContractionType
from typing import Optional, Union
import torch
import torch.nn as nn
# from jaxtyping import Bool, Float
from torch import Tensor
import abc
from dataclasses import dataclass


'mip-nerf360中对应论文公式10将远距离的点投影为一定范围内的点的对应代码'
# def contract(x):
#   """Contracts points towards the origin (Eq 10 of arxiv.org/abs/2111.12077)."""
#   eps = jnp.finfo(jnp.float32).eps
#   # Clamping to eps prevents non-finite gradients when x == 0.
#   x_mag_sq = jnp.maximum(eps, jnp.sum(x**2, axis=-1, keepdims=True))
#   z = jnp.where(x_mag_sq <= 1, x, ((2 * jnp.sqrt(x_mag_sq) - 1) / x_mag_sq) * x)
#   return z

'zheng zhisheng的对应论文公式10的代码'
# def contract_to_unisphere(x, inp_scale, tgt_scale, contraction_type):
#   if contraction_type == ContractionType.AABB:
#     x = scale_anything(x, inp_scale, tgt_scale)
#   elif contraction_type == ContractionType.UN_BOUNDED_SPHERE:
#     x = scale_anything(x, inp_scale, tgt_scale)
#     x = x * 2 - 1  # aabb is at [-1, 1]
#     mag = x.norm(dim=-1, keepdim=True)
#     mask = mag.squeeze(-1) > 1
#     x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
#     x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
#   else:
#     raise NotImplementedError
#   return x

'xu yan的对应论文公式10的代码'
# def contract_to_unisphere_neus(x, radius, contraction_type):
#   if contraction_type == ContractionType.AABB:
#     x = scale_anything(x, (-radius, radius), (0, 1))
#   elif contraction_type == ContractionType.UN_BOUNDED_SPHERE:
#     x = scale_anything(x, (-radius, radius), (0, 1))
#     x = x * 2 - 1  # aabb is at [-1, 1]
#     mag = x.norm(dim=-1, keepdim=True)
#     mask = mag.squeeze(-1) > 1
#     x = x.clone()
#     x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
#     x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
#   else:
#     raise NotImplementedError
#   return x

' my customization function based on xu yan and zheng zhisheng'
def contract_to_unisphere_LinGaoyuan(x, inp_scale=None, tgt_scale=None, contraction_type=None):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  if inp_scale == None:
    range = [250,250,200]
    inp_scale = torch.as_tensor(
      [[-range[0], -range[1], -range[2]], [range[0], range[1], range[2]]], dtype=torch.float32).reshape(2,3)
    inp_scale = inp_scale.to(device)

  if tgt_scale == None:
    tgt_scale = torch.torch.as_tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int8).reshape(2, 3)
    tgt_scale = tgt_scale.to(device)

  x = scale_anything(x.to(device), inp_scale, tgt_scale)
  x = x * 2 - 1  # aabb is at [-1, 1]
  mag = x.norm(dim=-1, keepdim=True)

  # print('mag = {}'.format(mag))
  # print(torch.min(mag))
  # print(mag.shape)

  mask = mag.squeeze(-1) > 1
  x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
  x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]

  # if contraction_type == ContractionType.AABB:
  #   x = scale_anything(x, inp_scale, tgt_scale)
  # elif contraction_type == ContractionType.UN_BOUNDED_SPHERE:
  #   x = scale_anything(x, inp_scale, tgt_scale)
  #   x = x * 2 - 1  # aabb is at [-1, 1]
  #   mag = x.norm(dim=-1, keepdim=True)
  #   mask = mag.squeeze(-1) > 1
  #   x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
  #   x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
  # else:
  #   raise NotImplementedError
  return x


def contract_to_unisphere_LinGaoyuan_xuyan(x, radius = 200, contraction_type = None):

  x = scale_anything(x, (-radius, radius), (0, 1))
  x = x * 2 - 1  # aabb is at [-1, 1]
  mag = x.norm(dim=-1, keepdim=True)

  # print('mag = {}'.format(mag))
  # print(torch.min(mag))
  # print(mag.shape)

  mask = mag.squeeze(-1) > 1
  x = x.clone()
  x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
  x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]

  return x


def scale_anything(dat, inp_scale, tgt_scale):
  if inp_scale is None:
    inp_scale = [dat.min(), dat.max()]
  dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
  dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
  return dat

'NeRF-studio中关于unbounded to bounded的相关代码, 一般初始化时设置order=float("inf")'

class SceneContraction(nn.Module):
  """Contract unbounded space using the contraction was proposed in MipNeRF-360.
      We use the following contraction equation:

      .. math::

          f(x) = \\begin{cases}
              x & ||x|| \\leq 1 \\\\
              (2 - \\frac{1}{||x||})(\\frac{x}{||x||}) & ||x|| > 1
          \\end{cases}

      If the order is not specified, we use the Frobenius norm, this will contract the space to a sphere of
      radius 2. If the order is L_inf (order=float("inf")), we will contract the space to a cube of side length 4.
      If using voxel based encodings such as the Hash encoder, we recommend using the L_inf norm.

      Args:
          order: Order of the norm. Default to the Frobenius norm. Must be set to None for Gaussians.

  """

  def __init__(self, order: Optional[Union[float, int]] = None) -> None:
    super().__init__()
    self.order = order

  def forward(self, positions):
    def contract(x):
      mag = torch.linalg.norm(x, ord=self.order, dim=-1)[..., None]
      # print('mag = {}'.format(mag))
      # print(torch.min(mag))
      # print(mag.shape)
      return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))

    # if isinstance(positions, Gaussians):
    #   means = contract(positions.mean.clone())

      # def contract_gauss(x):
      #   return (2 - 1 / torch.linalg.norm(x, ord=self.order, dim=-1, keepdim=True)) * (
      #           x / torch.linalg.norm(x, ord=self.order, dim=-1, keepdim=True)
      #   )
      #
      # jc_means = vmap(jacrev(contract_gauss))(positions.mean.view(-1, positions.mean.shape[-1]))
      # jc_means = jc_means.view(list(positions.mean.shape) + [positions.mean.shape[-1]])
      #
      # # Only update covariances on positions outside the unit sphere
      # mag = positions.mean.norm(dim=-1)
      # mask = mag >= 1
      # cov = positions.cov.clone()
      # cov[mask] = jc_means[mask] @ positions.cov[mask] @ torch.transpose(jc_means[mask], -2, -1)
      #
      # return Gaussians(mean=means, cov=cov)

    return contract(positions)


