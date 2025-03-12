# Copyright (c) OpenMMLab. All rights reserved.
import os
from collections import OrderedDict
from os import path as osp
from typing import List, Tuple, Union

import mmcv
import mmengine
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

from mmdet3d.datasets.convert_utils import NuScenesNameMapping
from mmdet3d.structures import points_cam2img
import cv2
import torch
import matplotlib.pyplot as plt
import json
from nuscenes.nuscenes import NuScenes


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

'based on mmdetection3d'


def get_2d_boxes(nusc,
                 sample_data_token: str,
                 visibilities: List[str],
                 camera=None,
                 mono3d=True,
                 ):
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token (str): Sample data token belonging to a camera
            keyframe.
        visibilities (list[str]): Visibility filter.
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                      'pedestrian.moving', 'pedestrian.standing',
                      'pedestrian.sitting_lying_down', 'vehicle.moving',
                      'vehicle.parked', 'vehicle.stopped', 'None')

    human_list = ['human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
                  'human.pedestrian.personal_mobility',
                  'human.pedestrian.police_officer', 'human.pedestrian.stroller', 'human.pedestrian.wheelchair'
                  ]

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec[
               'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
                                               ' for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [
        nusc.get('sample_annotation', token) for token in s_rec['anns']
    ]
    ann_recs = [
        ann_rec for ann_rec in ann_recs
        if (ann_rec['visibility_token'] in visibilities)
    ]

    repro_recs = []

    for ann_rec in ann_recs:

        if len(ann_rec['attribute_tokens']) > 0:

            current_attr = nusc.get('attribute', ann_rec['attribute_tokens'][0])['name']

            # if ann_rec['category_name'] == 'vehicle.car' or ann_rec['category_name'] == 'vehicle.truck':
            # if 'vehicle' in ann_rec['category_name']:

            'if condition for creating car bbox'
            # if ann_rec['category_name'] in vehicle_list and current_attr != 'vehicle.parked':
            # if ann_rec['category_name'] in vehicle_list and current_attr == 'vehicle.moving' or current_attr == 'vehicle.stopped':

            'if condition for creating human bbox'
            if ann_rec['category_name'] in human_list:

                # Augment sample_annotation with token information.
                ann_rec['sample_annotation_token'] = ann_rec['token']
                ann_rec['sample_data_token'] = sample_data_token

                # Get the box in global coordinates.
                box = nusc.get_box(ann_rec['token'])

                # Move them to the ego-pose frame.
                box.translate(-np.array(pose_rec['translation']))
                box.rotate(Quaternion(pose_rec['rotation']).inverse)

                # Move them to the calibrated sensor frame.
                box.translate(-np.array(cs_rec['translation']))
                box.rotate(Quaternion(cs_rec['rotation']).inverse)

                # Filter out the corners that are not in front of the calibrated
                # sensor.
                corners_3d = box.corners()
                in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
                corners_3d = corners_3d[:, in_front]

                # Project 3d box to 2d.
                corner_coords = view_points(corners_3d, camera_intrinsic,
                                            True).T[:, :2].tolist()

                # Keep only corners that fall within the image.
                final_coords = post_process_coords(corner_coords)

                # Skip if the convex hull of the re-projected corners
                # does not intersect the image canvas.
                if final_coords is None:
                    continue
                else:
                    min_x, min_y, max_x, max_y = final_coords

                # Generate dictionary record to be included in the .json file.
                repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                            sample_data_token, sd_rec['filename'])

                # If mono3d=True, add 3D annotations in camera coordinates
                if mono3d and (repro_rec is not None):
                    loc = box.center.tolist()

                    dim = box.wlh
                    dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
                    dim = dim.tolist()

                    rot = box.orientation.yaw_pitch_roll[0]
                    rot = [-rot]  # convert the rot to our cam coordinate

                    global_velo2d = nusc.box_velocity(box.token)[:2]
                    global_velo3d = np.array([*global_velo2d, 0.0])
                    e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
                    c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
                    cam_velo3d = global_velo3d @ np.linalg.inv(
                        e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
                    velo = cam_velo3d[0::2].tolist()

                    repro_rec['bbox_cam3d'] = loc + dim + rot
                    repro_rec['velo_cam3d'] = velo

                    center3d = np.array(loc).reshape([1, 3])
                    center2d = points_cam2img(
                        center3d, camera_intrinsic, with_depth=True)
                    repro_rec['center2d'] = center2d.squeeze().tolist()
                    # normalized center2D + depth
                    # if samples with depth < 0 will be removed
                    if repro_rec['center2d'][2] <= 0:
                        continue

                    ann_token = nusc.get('sample_annotation',
                                         box.token)['attribute_tokens']
                    if len(ann_token) == 0:
                        attr_name = 'None'
                    else:
                        attr_name = nusc.get('attribute', ann_token[0])['name']
                    attr_id = nus_attributes.index(attr_name)
                    repro_rec['attribute_name'] = attr_name
                    repro_rec['attribute_id'] = attr_id
                    repro_rec['camera_type'] = camera
                    repro_rec['object_annotation_token'] = ann_rec['token']

                repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str) -> OrderedDict:
    """Generate one 2D annotation record given various information on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): file name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                      'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                      'barrier')

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    if repro_rec['category_name'] not in NuScenesNameMapping:
        return None
    cat_name = NuScenesNameMapping[repro_rec['category_name']]
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = nus_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2, y2]
    coco_rec['iscrowd'] = 0

    return coco_rec



def main():
    human_bbox_save_path = '/media/lingaoyuan/SATA/Dataset/NuScene/data/nuscenes/human_bbox'
    nuscene_data_path = '/media/lingaoyuan/SATA/Dataset/NuScene/data/nuscenes'

    nusc = NuScenes(version='v1.0-trainval', dataroot=nuscene_data_path,
                    verbose=True)


    cameras_list = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    for scene_number in range(len(nusc.scene)):
        scene_name = nusc.scene[scene_number]['name']
        print(f"number_{scene_number}: {scene_name} start")

        my_scene = nusc.scene[scene_number]

        # create sample token list
        sample_token = my_scene['first_sample_token']
        my_sample = nusc.get('sample', sample_token)
        token_list = []
        token_list.append(my_sample['token'])
        while not my_sample['next'] == "":
            my_sample = nusc.get('sample', sample_token)
            sample_token = my_sample['next']
            token_list.append(sample_token)

        # create bbox
        coco_infos_arrays = {}
        for token in token_list:
            if token != '':
                my_sample = nusc.get('sample', token)
                for camera in cameras_list:
                    # cam_data = nusc.get('sample_data', my_sample['data'][camera])
                    from shapely.geometry import MultiPoint, box
                    coco_infos = get_2d_boxes(
                        nusc,
                        my_sample['data'][camera],
                        visibilities=['2', '3', '4'],
                        # visibilities=['3', '4'],
                        camera=camera,
                        mono3d=True,
                    )
                    boxes = []

                    if len(coco_infos) > 0 and coco_infos[0] is not None:
                        for coco_infos_number in range(len(coco_infos)):
                            if coco_infos[coco_infos_number] is not None:
                                box = np.array(coco_infos[coco_infos_number]['bbox'])
                                boxes.append(box)
                        boxes = np.array(boxes)

                        key = "token_{token_number}_camera_{camera_type}".format(token_number=token, camera_type=camera)
                        coco_infos_arrays.update({key: coco_infos})

        # save bbox in each scene
        if len(coco_infos_arrays) > 0:
            print(f"length of bbox array is {len(coco_infos_arrays)}")

            save_name = f"human_bbox_{scene_name}.json"

            print(f"{save_name} has complete the bbox")

            bbox_path = os.path.join(human_bbox_save_path, save_name)
            with open(bbox_path, "w") as outfile:
                json.dump(coco_infos_arrays, outfile)
        print(
            '---------------------------------------------------------------------------------------------------------------------')


if __name__ == '__main__':
    main()