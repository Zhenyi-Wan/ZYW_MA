import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import sys
import json
import torchvision

import matplotlib.pyplot as plt

sys.path.append("../")
from .data_utils import rectify_inplane_rotation, get_nearest_pose_ids

def read_cameras(pose_file):
    ''
    '获取上一层文件目录'
    basedir = os.path.dirname(pose_file)
    with open(pose_file, "r") as fp:
        images_info_dictionary = json.load(fp)

    # camera_angle_x = float(meta["camera_angle_x"])
    rgb_files = []
    c2w_mats = []
    sky_mask_files = []
    depth_value_files = []

    # img = imageio.imread(os.path.join(basedir, meta["frames"][0]["file_path"] + ".png"))
    # H, W = img.shape[:2]
    # focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    # intrinsics = get_intrinsics_from_hwf(H, W, focal)

    intrinsics_mats = []

    # for i, frame in enumerate(meta["frames"]):
    RGB_path = os.path.join(basedir, 'RGB')
    sky_mask_path = os.path.join(basedir, 'depth_sky_mask_v2')
    deyth_img_path = os.path.join(basedir, 'depth_value_metric_v2')
    for key, single_images_info in (images_info_dictionary).items():
        rgb_file = os.path.join(RGB_path, key + ".png")
        rgb_files.append(rgb_file)

        sky_mask_file = os.path.join(sky_mask_path, key + "_sky_mask.json")
        sky_mask_files.append(sky_mask_file)

        depth_value_file = os.path.join(deyth_img_path, key + "_depth_value_pred.json")
        depth_value_files.append(depth_value_file)


        # c2w = np.array(frame["transform_matrix"])
        # w2c_blender = np.linalg.inv(c2w)
        # w2c_opencv = w2c_blender
        # w2c_opencv[1:3] *= -1
        # c2w_opencv = np.linalg.inv(w2c_opencv)

        c2w_opencv = np.array(single_images_info['c2w_opencv'])
        c2w_mats.append(c2w_opencv)

        intrinsics = np.array(single_images_info['intrinsic'])
        intrinsics = np.concatenate((intrinsics, np.array([[0,0,0]]).T), axis=1)
        intrinsics = np.concatenate((intrinsics, [[0, 0, 0, 1]]), axis=0)
    c2w_mats = np.array(c2w_mats)
    return rgb_files, np.array([intrinsics] * len(images_info_dictionary)), c2w_mats, sky_mask_files, depth_value_files


class NusceneDataset_train_val(Dataset):
    def __init__(
        self,
        args,
        mode = "train",
        # scenes=('chair', 'drum', 'lego', 'hotdog', 'materials', 'mic', 'ship'),
        scenes= 'scene-0033',
        **kwargs
    ):
        self.folder_path = os.path.join(args.rootdir, "data/Nuscene/")
        self.rectify_inplane_rotation = args.rectify_inplane_rotation
        if mode == "validation":
            mode = "val"
        assert mode in ["train", "val", "test"]
        self.mode = mode  # train / test / val
        self.num_source_views = args.num_source_views
        self.testskip = args.testskip

        # all_scenes = ("chair", "drums", "lego", "hotdog", "materials", "mic", "ship")
        if len(scenes) > 0:
            if isinstance(scenes, str):
                scenes = [scenes]
        else:
            scenes = ['scene-0007']

        print("loading {} for {}".format(scenes, mode))
        self.render_rgb_files = []
        self.render_poses = []
        self.render_intrinsics = []
        self.sky_mask_files = []
        self.depth_value_files = []

        'LinGaoyuan_operation_20240906: add 3 variable for image resize, but the debug process is failed, need follow optimization'
        self.image_resize_H = args.image_resize_H
        self.image_resize_W = args.image_resize_W
        self.resize_image = args.resize_image
        self.resize_fun = torchvision.transforms.Resize((self.image_resize_H,self.image_resize_W))

        for scene in scenes:
            self.scene_path = os.path.join(self.folder_path, scene)
            pose_file = os.path.join(self.scene_path, "images_info_dictionary_{}.json".format(mode))
            # pose_file = os.path.join(pose_file)

            rgb_files, intrinsics, poses, sky_mask_files, depth_value_files = read_cameras(pose_file)
            if self.mode != "train":
                'if mode is not train, just select some of image from val dataset as the val data'
                rgb_files = rgb_files[:: self.testskip]
                intrinsics = intrinsics[:: self.testskip]
                poses = poses[:: self.testskip]
                sky_mask_files = sky_mask_files[:: self.testskip]
                depth_value_files = depth_value_files[:: self.testskip]
            self.render_rgb_files.extend(rgb_files)
            self.render_poses.extend(poses)
            self.render_intrinsics.extend(intrinsics)
            self.sky_mask_files.extend(sky_mask_files)
            self.depth_value_files.extend(depth_value_files)

            # print('end loading data')

    def __len__(self):
        return len(self.render_rgb_files)

    ' when train.py run the for loop of train_loader, it will randomly choose a data of train dataset and this getitem() will be called'
    def __getitem__(self, idx):
        # print('run getitem with idx{}'.format(idx))
        rgb_file = self.render_rgb_files[idx]
        render_pose = self.render_poses[idx]
        render_intrinsics = self.render_intrinsics[idx]
        sky_mask_file = self.sky_mask_files[idx]
        depth_value_file = self.depth_value_files[idx]

        # train_pose_file = os.path.join("/".join(rgb_file.split("/")[:-2]), "images_info_dictionary_train.json")
        # train_rgb_files, train_intrinsics, train_poses, train_sky_mask_files, train_depth_value_files = read_cameras(train_pose_file)

        if self.mode == "train":
            id_render = int(os.path.basename(rgb_file)[:-4].split("_")[1])
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.3, 0.5, 0.2])

            train_pose_file = os.path.join("/".join(rgb_file.split("/")[:-2]), "images_info_dictionary_train.json")
            rgb_files, intrinsics, poses, sky_mask_files, depth_value_files = read_cameras(
                train_pose_file)

        else:
            id_render = -1
            subsample_factor = 1

            val_pose_file = os.path.join("/".join(rgb_file.split("/")[:-2]), "images_info_dictionary_val.json")
            rgb_files, intrinsics, poses, sky_mask_files, depth_value_files = read_cameras(
                val_pose_file)

        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.0

        self.sky_color = np.zeros((3,))
        'change color of sky pixels to black'
        with open(sky_mask_file, 'r') as f:
            sky_mask_dictionary = json.load(f)
        sky_mask_0_1 = np.array(sky_mask_dictionary['sky_mask'])
        # rgb = rgb * sky_mask_0_1[..., None] + self.sky_color * (1 - sky_mask_0_1[..., None])
        #
        # plt.imshow(rgb)
        # plt.show()

        with open(depth_value_file, 'r') as f:
            depth_value_dictionary = json.load(f)
        depth_value = np.array(depth_value_dictionary['depth_value_pred'])

        'LinGaoyuan_operation_20240906: resize input image if resize_image is set to True'
        if self.resize_image is True:
            rgb = np.array(resize_img(torch.tensor(rgb).permute(2,0,1), self.image_resize_H, self.image_resize_W, self.resize_fun).permute(1,2,0))
            sky_mask_0_1 = resize_img(torch.tensor(sky_mask_0_1[None, ...]), self.image_resize_H, self.image_resize_W, self.resize_fun).squeeze()
            depth_value = resize_img(torch.tensor(depth_value[None, ...]), self.image_resize_H, self.image_resize_W, self.resize_fun).squeeze()

        'the next code will add some indistinct effect to the whole image'
        # rgb = rgb[..., [-1]] * rgb[..., :3] + 1 - rgb[..., [-1]]

        img_size = rgb.shape[:2]
        camera = np.concatenate(
            (list(img_size), render_intrinsics.flatten(), render_pose.flatten())
        ).astype(np.float32)

        nearest_pose_ids = get_nearest_pose_ids(
            render_pose,
            poses,
            int(self.num_source_views * subsample_factor)+1,
            tar_id=id_render,
            angular_dist_method="vector",
        )

        'remove target img from nearest_pose_ids if it is exist in nearest_pose_ids'
        nearest_pose_ids_remove_target_img= None
        for i in range(len(nearest_pose_ids)):
            if nearest_pose_ids[i] == idx:
                nearest_pose_ids_remove_target_img = np.delete(nearest_pose_ids, i)
        if nearest_pose_ids_remove_target_img is not None:
            nearest_pose_ids = nearest_pose_ids_remove_target_img


        nearest_pose_ids = np.random.choice(nearest_pose_ids, self.num_source_views, replace=False)

        assert id_render not in nearest_pose_ids
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == "train":
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        src_rgbs = []
        src_cameras = []
        src_sky_masks = []
        src_depth_values = []
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(rgb_files[id]).astype(np.float32) / 255.0


            'change color of sky pixels to black'
            with open(sky_mask_files[id], 'r') as f:
                src_sky_mask_dictionary = json.load(f)
            src_sky_mask_0_1 = np.array(src_sky_mask_dictionary['sky_mask'])
            # src_sky_masks.append(src_sky_mask_0_1)
            # src_rgb = src_rgb * src_sky_mask_0_1[..., None] + self.sky_color * (1 - sky_mask_0_1[..., None])
            #
            # plt.imshow(src_rgb)
            # plt.show()

            with open(depth_value_files[id], 'r') as f:
                src_depth_value_dictionary = json.load(f)
            src_depth_value = np.array(src_depth_value_dictionary['depth_value_pred'])
            # src_depth_values.append(src_depth_value)

            'the next code will add some indistinct effect to the whole image'
            # src_rgb = src_rgb[..., [-1]] * src_rgb[..., :3] + 1 - src_rgb[..., [-1]]

            pose = poses[id]
            intrinsics_ = intrinsics[id]
            if self.rectify_inplane_rotation:
                pose, src_rgb = rectify_inplane_rotation(pose, render_pose, src_rgb)

            'LinGaoyuan_operation_20240906: resize input image if resize_image is set to True'
            if self.resize_image is True:
                src_rgb = resize_img(torch.tensor(src_rgb).permute(2,0,1), self.image_resize_H, self.image_resize_W, self.resize_fun).permute(1,2,0)
                src_sky_mask_0_1 = resize_img(torch.tensor(src_sky_mask_0_1[None, ...]), self.image_resize_H, self.image_resize_W, self.resize_fun).squeeze()
                src_depth_value = resize_img(torch.tensor(src_depth_value[None, ...]), self.image_resize_H, self.image_resize_W, self.resize_fun).squeeze()

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate(
                (list(img_size), intrinsics_.flatten(), pose.flatten())
            ).astype(np.float32)
            src_cameras.append(src_camera)

            src_sky_masks.append(src_sky_mask_0_1)
            src_depth_values.append(src_depth_value)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        src_sky_masks = np.stack(src_sky_masks, axis=0)

        src_depth_values = np.stack(src_depth_values, axis=0)

        # print(type(src_rgbs))

        near_depth = 1.0
        far_depth = 200.0

        depth_range = torch.tensor([near_depth, far_depth])

        return {
            "rgb": torch.from_numpy(rgb[..., :3]).to(dtype=torch.float),
            "sky_mask": sky_mask_0_1,
            "depth_value": depth_value,
            "camera": torch.from_numpy(camera),
            "rgb_path": rgb_file,
            "sky_mask_path": sky_mask_file,
            "depth_value_path": depth_value_file,
            "src_rgbs": torch.from_numpy(src_rgbs[..., :3]).to(dtype=torch.float),
            "src_cameras": torch.from_numpy(src_cameras),
            "src_sky_masks": torch.from_numpy(src_sky_masks),
            "src_depth_values": torch.from_numpy(src_depth_values),
            "depth_range": depth_range,
            "idx": idx,
        }



def resize_img(image, target_H, target_W, resize_fun):
    if len(image.shape) == 2:
        image = image[None, ...]
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = image.permute(2,0,1)
    return resize_fun(image)
