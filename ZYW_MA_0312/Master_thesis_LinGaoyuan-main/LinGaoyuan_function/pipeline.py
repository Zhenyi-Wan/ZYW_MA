import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
import torchvision
import clip
from PIL import Image
import config

from mapper_for_clip_feature import mapper
from sky_transformer_network import mapper_attention

from aliasing import single_gaussian_filter, gaussian_filter, Downsample_PASA_group_softmax

from unbounded2bounded import contract_to_unisphere


def run_clip_encoder(image, pretrained_clip_model, pretrained_clip_preprocess, device = "cuda"):
    '''
    :param image:
    :param pretrained_clip_model:
    :param pretrained_clip_preprocess:
    :return:
    '''
    '用clip预训练模型对图片进行预处理'
    image_clip = pretrained_clip_preprocess(image).unsqueeze(0).to(device)

    '获取利用clip的图片encoder获取clip特征'
    with torch.no_grad():
        clip_feature = pretrained_clip_model.encode_image(image_clip)

    return clip_feature

def run_clip_mapper(clip_feature, clip_shape_mapper, clip_appearacne_mapper):
    '''
    :param clip_feature:
    :param clip_shape_mapper:
    :param clip_appearacne_mapper:
    :return:
    '''
    shape_code = clip_shape_mapper(clip_feature)
    appearance_code = clip_appearacne_mapper(clip_feature)

    return shape_code, appearance_code

def run_clip_attention_mapper(clip_feature, clip_shape_attention_mapper, clip_appearacne_attention_mapper, pytorch_attention = False):
    '''
    :param clip_feature:
    :param clip_shape_attention_mapper:
    :param clip_appearacne_attention_mapper:
    :param pytorch_attention:
    :return:
    '''
    if pytorch_attention:
        x = clip_feature.unsqueeze(0)
        shape_attention_code = clip_shape_attention_mapper(x,x,x)
        appearance_attention_code = clip_appearacne_attention_mapper(x,x,x)
    else:
        shape_attention_code = clip_shape_attention_mapper(clip_feature)
        appearance_attention_code = clip_appearacne_attention_mapper(clip_feature)

    return shape_attention_code, appearance_attention_code

def get_filter_bank(kernel_size = 5, mean = [0.,0.], cov = [[4.,0.],[0.,4.]], rotation_angle = 30, scale = 0.3, eliptical = 0.5, device = "cuda"):
    '''
    :param kernel_size:
    :param rotation_angle:
    :param scale:
    :param eliptical:
    :return:
    '''
    'standard_gaus_filter对应论文Blending Anti-Aliasing into Vision Transformer的Figure 4中第四个高斯滤波器, 对应index为3'
    standard_gaus_filter = gaussian_filter._get_multivariate_gaussian(kernel_size, mean, cov, dtype = torch.float32, device = device)
    filter_3 = standard_gaus_filter
    'filter_index对应图中从左往右数第(index+1)个滤波器, 左边第一个滤波器的index为0'
    filter_4 = gaussian_filter._get_multivariate_gaussian(kernel_size, mean, cov * scale, dtype = torch.float32, device = device)
    filter_5 = gaussian_filter._get_multivariate_gaussian(kernel_size, mean, cov * torch.square(scale), dtype=torch.float32, device=device)

    '基于将cov中y方向的方差减小为x方向的一半来创建filter_2, 对应图片中的左边往右第3个滤波器'
    sigma_x = cov[0,0]
    sigma_y = cov[1,1]
    new_sigma_y = sigma_y/2
    cov_index_2 = [[sigma_x, 0], [0, new_sigma_y]]
    filter_2 = gaussian_filter._get_multivariate_gaussian(kernel_size, mean, cov_index_2, dtype=torch.float32,
                                                          device=device)
    '计算filter_2对应的椭圆几何参数'
    angle_2, width, height = gaussian_filter.covariance_ellipse(cov_index_2, deviations=1)

    '基于filter_2的椭圆几何参数和输入的角度获得filter_0和filter_1的cov矩阵'
    angle_index_1 = 90 + rotation_angle
    angle_index_0 = 90 - rotation_angle

    cov_index_1 = gaussian_filter._calculate_covariance_matrix_from_elipse(width, height, angle_index_1, deviations=1)
    cov_index_0 = gaussian_filter._calculate_covariance_matrix_from_elipse(width, height, angle_index_0, deviations=1)

    '创建filter_1和filter_0'
    filter_1 = gaussian_filter._get_multivariate_gaussian(kernel_size, mean, cov_index_1, dtype=torch.float32, device=device)
    filter_0 = gaussian_filter._get_multivariate_gaussian(kernel_size, mean, cov_index_0, dtype=torch.float32, device=device)

    '创建DoG_0和DoG_1, 分别对应filter_6和filter_7'
    filter_6 = standard_gaus_filter - filter_4
    filter_7 = standard_gaus_filter - filter_5

    filter_bank = 0.125 * (filter_0 + filter_1 + filter_2 + filter_3 + filter_4 + filter_5 + filter_6 + filter_7)

    return filter_bank



def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    '执行clip预训练的模型, 之后这里要改为读取args里的信息而不是直接指定模型名字'
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    # clip_model, clip_preprocess = clip.load("ViT-L/14@336px", device=device)
    # clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device)  # ViT-B/32 或者 ViT-L/14@336px


    '读取图片, 可能需要换成pytorch的读取图片的函数'
    image_path = args.image_path
    image = Image.open(image_path)
    image_tensor = torch.tensor(image).clone().detach()

    '获取clip特征'
    clip_feature = run_clip_encoder(image, clip_model, clip_preprocess, device)
    clip_feature = clip_feature.to(torch.float32)

    '调用MLP层获取论文clip-NeRF中的latent code'
    clip_shape_mapper = mapper(input_dim = clip_feature.shape[1]).to(device)
    clip_appearacne_mapper = mapper(input_dim=clip_feature.shape[1]).to(device)
    shape_code_mlp, appearance_code_mlp = run_clip_mapper(clip_feature, clip_shape_mapper, clip_appearacne_mapper)

    '调用自己编写的attention模型获取论文clip-NeRF中的latent code'
    clip_shape_mapper_attention = mapper_attention(input_dim=clip_feature.shape[1]).to(device)
    clip_appearacne_mapper_attention = mapper_attention(input_dim=clip_feature.shape[1]).to(device)
    shape_attention_code, appearance_attention_code = run_clip_attention_mapper(clip_feature, clip_shape_mapper_attention, clip_appearacne_mapper_attention)

    '调用pytorch的multihead-attention模型获取论文clip-NeRF中的latent code'
    clip_shape_mapper_nn_attention = nn.MultiheadAttention(embed_dim = clip_feature.shape[1], num_heads = 1).to(device)
    clip_appearacne_mapper_nn_attention = nn.MultiheadAttention(embed_dim=clip_feature.shape[1], num_heads=1).to(device)
    shape_nn_attention_code, appearance_nn_attention_code = run_clip_attention_mapper(clip_feature,
                                                                                      clip_shape_mapper_nn_attention,
                                                                                      clip_appearacne_mapper_nn_attention,
                                                                                      pytorch_attention = True)

    '调用anti-aliasing模块'

    '单个高斯滤波器'
    single_filter = single_gaussian_filter(kernel_size=(5, 5), sigma=(2.0, 2.0))

    '多个高斯滤波器组成的filter bank'
    filter_bank = get_filter_bank(kernel_size=5, mean=[0.,0.], cov=[[4.,0.],[0.,4.]], rotation_angle=30, scale=0.3, eliptical=0.5, device="cuda")

    'CNN作为anti-aliasing模块: https://maureenzou.github.io/ddac/'
    # 注意这里初始化的输入维度需要另外考虑, 因为要结合经过transformer提取后的feature map的channel来考虑
    filter_CNN = Downsample_PASA_group_softmax(in_channels = 3, kernel_size = 3)


    



if __name__ == "__main__":
    parser = config.config_parser()
    args = parser.parse_args()

    run(args)