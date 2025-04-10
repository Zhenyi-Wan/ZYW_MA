import torch
from torch.utils.data import Dataset
import sys
import json
import torchvision


def resize_img(image, resize_fun):
    if len(image.shape) == 2:
        image = image[None, ...]
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = image.permute(2,0,1)
    return resize_fun(image)