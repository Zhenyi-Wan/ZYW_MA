import cv2
from segment_anything import SamAutomaticMaskGenerator
import torch
from segment_anything import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from os import listdir
from os.path import isfile, join, isdir
from pycocotools import mask

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def load_SAM_checkpoint(CHECKPOINT_PATH):

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    MODEL_TYPE = "vit_h"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)

    mask_predictor = SamPredictor(sam)

    return mask_predictor

def get_all_bbox_files(bbox_files_path):

    bbox_file_name_array = [f for f in listdir(bbox_files_path) if isfile(join(bbox_files_path, f))]

    return bbox_file_name_array

def create_mask(mask_predictor, bbox_files_path, bbox_file_name):

    print(f"{bbox_file_name} start")

    bbox_path = os.path.join(bbox_files_path, bbox_file_name)

    with open(bbox_path) as f:
        bbox_array = json.load(f)

    for key, bbox_info in bbox_array.items():
        for bbox_single_dictionary in bbox_info:
            if bbox_single_dictionary is not None:
                image_path = os.path.join('/media/lingaoyuan/SATA/Dataset/NuScene/data/nuscenes',
                                          bbox_single_dictionary['file_name'])
                bbox = bbox_single_dictionary['bbox']
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                mask_predictor.set_image(image)
                masks, scores, logits = mask_predictor.predict(
                    box=np.array(bbox),
                    multimask_output=True
                )
                # bbox_single_dictionary.update({'mask': (masks[0] * 1)})
                bimask = (masks[0] * 1).astype(np.uint8)

                mask_encode = mask.encode(np.asfortranarray(bimask))

                'if save as .json file, uncomment this code.'
                mask_encode['counts'] = mask_encode['counts'].decode('utf-8')

                bbox_single_dictionary.update({'mask': mask_encode})


    return bbox_array

def save_mask_file(mask_array, mask_files_path, bbox_file_name):

    'save as .pt file'
    # mask_save_path = os.path.join(mask_files_path, bbox_file_name.replace('json', 'pt'))
    # torch.save(mask_array, mask_save_path)

    'save as .npy file'
    # mask_save_path = os.path.join(mask_files_path, bbox_file_name.replace('json', 'npy'))
    # with open(mask_save_path, 'wb') as f:
    #     np.save(f, mask_array)

    'save as .json file'
    mask_save_path = os.path.join(mask_files_path, bbox_file_name.replace('bbox', 'mask'))
    with open(mask_save_path, "w") as outfile:
        json.dump(mask_array, outfile)

    print(f"{bbox_file_name} has completed the mask")
    print('---------------------------------------------------------------------------------------------------------------------')


def main():
    bbox_files_path = '/media/lingaoyuan/SATA/Dataset/NuScene/data/nuscenes/human_bbox'
    mask_files_path = '/media/lingaoyuan/SATA/Dataset/NuScene/data/nuscenes/human_mask'
    CHECKPOINT_PATH = '/media/lingaoyuan/SATA/Segmention_anything_checkpoint/sam_vit_h.pth'

    torch.cuda.empty_cache()

    mask_predictor = load_SAM_checkpoint(CHECKPOINT_PATH)

    bbox_file_name_array = get_all_bbox_files(bbox_files_path)

    for bbox_file_name in bbox_file_name_array[0:2]:
        mask_array = create_mask(mask_predictor, bbox_files_path, bbox_file_name)
        save_mask_file(mask_array, mask_files_path, bbox_file_name)


if __name__ == '__main__':
    main()
