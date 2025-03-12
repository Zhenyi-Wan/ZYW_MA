import numpy as np
import torch
import torch.nn.functional as F


rng = np.random.RandomState(234)

########################################################################################################################
# ray batch sampling
########################################################################################################################


def parse_camera(params):
    H = params[:, 0]
    W = params[:, 1]
    intrinsics = params[:, 2:18].reshape((-1, 4, 4))
    c2w = params[:, 18:34].reshape((-1, 4, 4))
    return W, H, intrinsics, c2w


def dilate_img(img, kernel_size=20):
    import cv2

    assert img.dtype == np.uint8
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilation = cv2.dilate(img / 255, kernel, iterations=1) * 255
    return dilation


class RaySamplerSingleImage(object):
    def __init__(self, data, device, resize_factor=1, render_stride=1):
        super().__init__()
        self.render_stride = render_stride
        self.rgb = data["rgb"] if "rgb" in data.keys() else None
        self.camera = data["camera"]
        self.rgb_path = data["rgb_path"]
        self.depth_range = data["depth_range"]
        self.device = device
        W, H, self.intrinsics, self.c2w_mat = parse_camera(self.camera)
        self.batch_size = len(self.camera)

        self.H = int(H[0])
        self.W = int(W[0])

        self.sky_mask = data["sky_mask"] if "sky_mask" in data.keys() else None

        self.depth_value = data["depth_value"] if "depth_value" in data.keys() else None

        self.idx = data["idx"] if "idx" in data.keys() else None

        # half-resolution output
        if resize_factor != 1:
            self.W = int(self.W * resize_factor)
            self.H = int(self.H * resize_factor)
            self.intrinsics[:, :2, :3] *= resize_factor
            if self.rgb is not None:
                self.rgb = F.interpolate(
                    self.rgb.permute(0, 3, 1, 2), scale_factor=resize_factor
                ).permute(0, 2, 3, 1)

        self.rays_o, self.rays_d = self.get_rays_single_image(
            self.H, self.W, self.intrinsics, self.c2w_mat
        )
        if self.rgb is not None:
            self.rgb = self.rgb.reshape(-1, 3)


        if self.sky_mask is not None:
            self.sky_mask = self.sky_mask.reshape(-1, 1)

        if self.depth_value is not None:
            self.depth_value = self.depth_value.reshape(-1,1)

        if "src_rgbs" in data.keys():
            self.src_rgbs = data["src_rgbs"]
        else:
            self.src_rgbs = None
        if "src_cameras" in data.keys():
            self.src_cameras = data["src_cameras"]
        else:
            self.src_cameras = None

        if "src_sky_masks" in data.keys():
            self.src_sky_masks = data["src_sky_masks"]
        else:
            self.src_sky_masks = None

    def get_rays_single_image(self, H, W, intrinsics, c2w):
        """
        :param H: image height
        :param W: image width
        :param intrinsics: 4 by 4 intrinsic matrix
        :param c2w: 4 by 4 camera to world extrinsic matrix
        :return:
        """
        u, v = np.meshgrid(
            np.arange(W)[:: self.render_stride], np.arange(H)[:: self.render_stride]
        )
        u = u.reshape(-1).astype(dtype=np.float32)  # + 0.5    # add half pixel
        v = v.reshape(-1).astype(dtype=np.float32)  # + 0.5
        pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)
        pixels = torch.from_numpy(pixels)
        batched_pixels = pixels.unsqueeze(0).repeat(self.batch_size, 1, 1)

        rays_d = (
            c2w[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(batched_pixels)
        ).transpose(1, 2)
        rays_d = rays_d.reshape(-1, 3)  # shape: ([1440000, 3])
        rays_o = (
            c2w[:, :3, 3].unsqueeze(1).repeat(1, rays_d.shape[0], 1).reshape(-1, 3)
        )  # B x HW x 3

        return rays_o, rays_d

    def get_all(self):
        ret = {
            "ray_o": self.rays_o.cuda(),
            "ray_d": self.rays_d.cuda(),
            "depth_range": self.depth_range.cuda(),
            "camera": self.camera.cuda(),
            "depth_value": self.depth_value.cuda(),
            "rgb": self.rgb.cuda() if self.rgb is not None else None,
            "src_rgbs": self.src_rgbs.cuda() if self.src_rgbs is not None else None,
            "src_cameras": self.src_cameras.cuda() if self.src_cameras is not None else None,
            "sky_mask": self.sky_mask.cuda() if self.sky_mask is not None else None,
            "src_sky_mask": self.src_sky_masks.cuda() if self.src_sky_masks is not None else None,
            "idx": self.idx.cuda() if self.idx is not None else None,
        }
        return ret

    def sample_random_pixel(self, N_rand, sample_mode, center_ratio=0.8):
        if sample_mode == "center":
            border_H = int(self.H * (1 - center_ratio) / 2.0)
            border_W = int(self.W * (1 - center_ratio) / 2.0)

            # pixel coordinates
            u, v = np.meshgrid(
                np.arange(border_H, self.H - border_H), np.arange(border_W, self.W - border_W)
            )
            u = u.reshape(-1)
            v = v.reshape(-1)

            select_inds = rng.choice(u.shape[0], size=(N_rand,), replace=False)
            select_inds = v[select_inds] + self.W * u[select_inds]

        elif sample_mode == "uniform":
            # Random from one image
            select_inds = rng.choice(self.H * self.W, size=(N_rand,), replace=False)
        else:
            raise Exception("unknown sample mode!")

        return select_inds

    def random_sample(self, N_rand, sample_mode, center_ratio=0.8):
        """
        :param N_rand: number of rays to be casted
        :return:
        """

        select_inds = self.sample_random_pixel(N_rand, sample_mode, center_ratio)

        rays_o = self.rays_o[select_inds]
        rays_d = self.rays_d[select_inds]

        if self.rgb is not None:
            rgb = self.rgb[select_inds]
        else:
            rgb = None

        if self.sky_mask is not None:
            sky_mask = self.sky_mask[select_inds]
        else:
            sky_mask = None

        if self.depth_value is not None:
            depth_value = self.depth_value[select_inds]
        else:
            depth_value= None

        ret = {
            "ray_o": rays_o.cuda(),
            "ray_d": rays_d.cuda(),
            "camera": self.camera.cuda(),
            "depth_range": self.depth_range.cuda(),
            "rgb": rgb.cuda() if rgb is not None else None,
            "src_rgbs": self.src_rgbs.cuda() if self.src_rgbs is not None else None,
            "src_cameras": self.src_cameras.cuda() if self.src_cameras is not None else None,
            "selected_inds": select_inds,
            "sky_mask": sky_mask.cuda(),
            "depth_value": depth_value.cuda(),
            "idx": self.idx.cuda() if self.idx is not None else None,
        }
        return ret


    'get the sky image with the rgb and sky_mask'
    def get_sky_image(self, rgb, sky_mask):
        '''

        Args:
            rgb: H,W,3
            sky_mask: H,W,1

        Returns:
            sky_rgb: H,W,3

        '''
        black_background = torch.zeros((3,))
        sky_area = rgb * sky_mask[..., None] + black_background * (1 - sky_mask[..., None])
