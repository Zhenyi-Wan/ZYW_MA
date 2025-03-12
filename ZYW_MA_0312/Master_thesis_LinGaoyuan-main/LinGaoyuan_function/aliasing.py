import torchvision
from PIL import Image
import numpy as np
import torch
# import matplotlib.pyplot as plt
# import filterpy.stats as stats
# from matplotlib import cm
import math
from torch import Tensor
import scipy
import torchvision.transforms._functional_tensor as F_t
import scipy.linalg as linalg
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import conv2d, grid_sample, interpolate, pad as torch_pad
from typing import List, Optional, Tuple, Union

'create a simple gaussian filter, use the same origin funtion as same as gaussian_filter_pytorch()'
def single_gaussian_filter(kernel_size = (5, 5), sigma=(1.0, 1.0), typical_gaussian_blur = False):

    if typical_gaussian_blur:
        filter = torch.tensor([[1/16, 2/16, 1/16],[2/16, 4/16, 2/16],[1/16, 2/16, 1/16]])

    else:
        filter = torchvision.transforms.GaussianBlur(kernel_size, sigma)

    return filter


class Gaussian_filter():
    def __init__(self, kernel_size = 5, mean = [0, 0], cov = [[3, 0],[0, 3]], dtype = torch.float32, device = 'cpu'):
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.device = device
        self.mean = mean
        self.cov = cov

    'create gaussian filter based on the function from pytorch, can not create gaussian matrix based on cov, Only STD can be used to generate ordinary circles or 0°&90° ellipses.'
    def gaussian_filter_pytorch(self, std= 3):
        if not isinstance(self.kernel_size, (int, list, tuple)):
            raise TypeError(f"kernel_size should be int or a sequence of integers. Got {type(self.kernel_size)}")
        if isinstance(self.kernel_size, int):
            kernel_size = [self.kernel_size, self.kernel_size]
        else:
            kernel_size = self.kernel_size
        if isinstance(std, int):
            std = [std, std]

        kernel = F_t._get_gaussian_kernel2d(kernel_size,[3,3], dtype = self.dtype, device = self.device)

        return kernel

    'calculate the direction, major axis and minor axis of the corresponding ellipse according to the input covariance matrix'
    def covariance_ellipse(self, cov, deviations=1):
        """
        Returns a tuple defining the ellipse representing the 2 dimensional
        covariance matrix P.

        Parameters
        ----------

        P : nd.array shape (2,2)
           covariance matrix

        deviations : int (optional, default = 1)
           # of standard deviations. Default is 1.

        Returns (angle_radians, width_radius, height_radius)
        """

        P = cov
        U, s, _ = linalg.svd(P)
        # The orientation value returned here is in rad, so for example, 90° will return 1.570796.., which is pi/2. Convert it to degrees when returning.
        orientation = math.atan2(U[1, 0], U[0, 0])
        width = deviations * math.sqrt(s[0])
        height = deviations * math.sqrt(s[1])

        if height > width:
            raise ValueError('width must be greater than height')

        return torch.tensor([np.degrees(orientation), width, height])

    'Function to calculate the covariance matrix based on the ellipse parameters (direction & major axis & minor axis)'
    def _calculate_covariance_matrix_from_elipse(self, semiMajorAxis, semiMinorAxis, angle, deviations=1):
        '''
        semiMajorAxis: the length of major axis
        semiMinorAxis: the length of minor axis
        angle: the angle of the orientation of ellipse
        deviations: standard deviation, corresponding to the function for creating ellipse from cov, set to 1 by default
        '''
        semiMajorAxis, semiMinorAxis, angle = semiMajorAxis / deviations, semiMinorAxis / deviations, angle / deviations
        phi = np.deg2rad(angle)
        varX1 = np.square(semiMajorAxis) * np.square(np.cos(phi)) + np.square(semiMinorAxis) * np.square(np.sin(phi))
        varX2 = np.square(semiMajorAxis) * np.square(np.sin(phi)) + np.square(semiMinorAxis) * np.square(np.cos(phi))
        cov12 = (np.square(semiMajorAxis) - np.square(semiMinorAxis)) * np.sin(phi) * np.cos(phi)

        return [[varX1, cov12], [cov12, varX2]]

    'the funtion for calculation of gaussian distribution of ellipse from scipy'
    def _get_multivariate_gaussian_scipy(self, kernel_size: int, mean: list, cov: list, dtype: torch.dtype,
                                         device: torch.device) -> Tensor:
        ksize_half = (kernel_size - 1) * 0.5

        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, dtype=dtype, device=device)
        xv, yv = torch.meshgrid(x, x, indexing='xy')
        pos = np.dstack((xv, yv))
        # print(pos.shape)

        rv = scipy.stats.multivariate_normal(mean, cov)

        zs_torch = rv.pdf(pos)

        return zs_torch

    'the function calculation of gaussian distribution of ellipse, this version is adjusted by LinGaoyuan based on the origin code'
    # code origin:
    # https://github.com/rlabbe/filterpy/blob/3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33/filterpy/stats/stats.py#L321-L397
    def multivariate_gaussian(self, x, mu, cov):

        x = torch.tensor(x).flatten()
        mu = torch.tensor(mu).flatten()
        cov = torch.tensor(cov)

        nx = len(mu)

        norm_coeff = nx * math.log(2 * math.pi) + torch.linalg.slogdet(cov)[1]

        err = (x - mu).to(torch.float32)

        numerator = torch.linalg.solve(cov.to(torch.float32), err).T.dot(err)

        return math.exp(-0.5 * (norm_coeff + numerator))

    'generate sampled point and call the funtion: self.multivariate_gaussian() to generate gaussian distribution with the shape of ellipse'
    def _get_multivariate_gaussian(self, kernel_size: int, mean : list, cov: list, dtype = torch.float32,
                                   device = 'cpu') -> Tensor:

        # if self.dtype is not None:
        #     dtype = self.dtype
        # if self.device is not None:
        #     device = self.device

        ksize_half = (kernel_size - 1) * 0.5

        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, dtype=dtype, device=device)
        xv, yv = torch.meshgrid(x, x, indexing='xy')

        zs_torch = torch.tensor([self.multivariate_gaussian([xx, yy], mean, cov) \
                                 for xx, yy in zip(torch.ravel(xv), torch.ravel(yv))]).to(device)

        # zs_torch = np.array([100.* stats.multivariate_gaussian(np.array([xx, yy]), mean, cov) \
        #            for xx, yy in zip(np.ravel(xv), np.ravel(yv))])

        zs_torch = zs_torch.reshape(xv.shape)

        return zs_torch

    'generate difference of gaussian'
    def DoG(self, gaussian_filter_1, gaussian_filter_2):
        return gaussian_filter_1 - gaussian_filter_2

    'normalize data to 0~1'
    def NormalizeData(self, data):
        return (data - torch.min(data)) / (torch.max(data) - torch.min(data))






'code origin: cnn-anti-aliasing: https://maureenzou.github.io/ddac/'
def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                 padding=1, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


'code origin: https://github.com/MaureenZOU/Adaptive-anti-Aliasing/blob/909f3daf9c6df44d67657c225942e674126d7ca1/models_lpf/layers/pasa.py#L73C1-L106C93'
class Downsample_PASA_group_softmax(nn.Module):

    def __init__(self, in_channels, kernel_size, stride=1, pad_type='reflect', group=2):
        super(Downsample_PASA_group_softmax, self).__init__()

        kernel_size = int(kernel_size)
        group = int(group)

        self.pad = get_pad_layer(pad_type)(kernel_size // 2)
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(in_channels, group*kernel_size*kernel_size, kernel_size=kernel_size, stride=1, bias=False)
        # self.conv = nn.Conv2d(1536, 96 * 4 * 4, kernel_size=kernel_size, stride=1,
        #                       bias=False)

        self.bn = nn.BatchNorm2d(group*kernel_size*kernel_size)
        # self.bn = nn.BatchNorm2d(1536)

        self.softmax = nn.Softmax(dim=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        sigma = self.conv(self.pad(x))

        if len(sigma.shape) == 3:
            sigma = sigma.unsqueeze(0)

        sigma = self.bn(sigma)
        sigma = self.softmax(sigma)

        n,c,h,w = sigma.shape

        sigma = sigma.reshape(n,1,c,h*w)


        n,c,h,w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape((n,c,self.kernel_size*self.kernel_size,h*w))

        n,c1,p,q = x.shape
        x = x.permute(1,0,2,3).reshape(self.group, c1//self.group, n, p, q).permute(2,0,1,3,4)

        n,c2,p,q = sigma.shape
        sigma = sigma.permute(2,0,1,3).reshape((p//(self.kernel_size*self.kernel_size), self.kernel_size*self.kernel_size,n,c2,q)).permute(2,0,3,1,4)

        x = torch.sum(x*sigma, dim=3).reshape(n,c1,h,w)
        return x[:,:,torch.arange(h)%self.stride==0,:][:,:,:,torch.arange(w)%self.stride==0]

def get_filter_bank(kernel_size = 5, mean = [0.,0.], cov = [[4.,0.],[0.,4.]], rotation_angle = 30, scale = 0.3, eliptical = 0.5, device = "cuda",
                    gaussian_filter = Gaussian_filter()):
    '''
    :param kernel_size:
    :param rotation_angle:
    :param scale:
    :param eliptical:
    :return:
    '''
    'standard_gaus_filter corresponding to the 4th gaussian filter of figute in paper: Blending Anti-Aliasing into Vision Transformer, we set index to 3'
    kernel_size = torch.tensor(kernel_size)
    mean = torch.tensor(mean)
    cov = torch.tensor(cov)
    scale = torch.tensor(scale)
    eliptical = torch.tensor(eliptical)

    standard_gaus_filter = gaussian_filter._get_multivariate_gaussian(kernel_size = kernel_size, mean = mean, cov = cov, dtype = torch.float32, device = device)
    filter_3 = standard_gaus_filter

    'filter_index corresponding to the (index+1)th filter from left to right, the first filter in the left has index=0'
    filter_4 = gaussian_filter._get_multivariate_gaussian(kernel_size, mean, cov * scale, dtype = torch.float32, device = device)
    filter_5 = gaussian_filter._get_multivariate_gaussian(kernel_size, mean, cov * torch.square(scale), dtype=torch.float32, device=device)

    'reduce the value of y-direction of cov to halft to createfilter_2, corresponding to the 3rd filter from left to right'
    sigma_x = cov[0,0]
    sigma_y = cov[1,1]
    new_sigma_y = sigma_y/2
    cov_index_2 = [[sigma_x, 0], [0, new_sigma_y]]
    filter_2 = gaussian_filter._get_multivariate_gaussian(kernel_size, mean, cov_index_2, dtype=torch.float32,
                                                          device=device)
    'calculate the geometric parameters of filter_2'
    angle_2, width, height = gaussian_filter.covariance_ellipse(cov_index_2, deviations=1)

    'generate the cov matrix of filter_0 and filter_1 based on the geometric parameters of filter_2 and input angle'
    angle_index_1 = 90 + rotation_angle
    angle_index_0 = 90 - rotation_angle

    cov_index_1 = gaussian_filter._calculate_covariance_matrix_from_elipse(width, height, angle_index_1, deviations=1)
    cov_index_0 = gaussian_filter._calculate_covariance_matrix_from_elipse(width, height, angle_index_0, deviations=1)

    'create filter_1 and filter_0'
    filter_1 = gaussian_filter._get_multivariate_gaussian(kernel_size, mean, cov_index_1, dtype=torch.float32, device=device)
    filter_0 = gaussian_filter._get_multivariate_gaussian(kernel_size, mean, cov_index_0, dtype=torch.float32, device=device)

    'create DoG_0 and DoG_1, corresponding to filter_6 and filter_7 respectively'
    filter_6 = standard_gaus_filter - filter_4
    filter_7 = standard_gaus_filter - filter_5

    filter_bank = 0.125 * (filter_0 + filter_1 + filter_2 + filter_3 + filter_4 + filter_5 + filter_6 + filter_7)

    return filter_bank

def aliasing_filter(filter_type = 'filter bank',kernel_size=5):
    '''
    filter_type = 'filter bank' or 'single filter' or 'CNN'
    '''

    if filter_type == 'filter bank':
        filter = get_filter_bank(kernel_size=kernel_size, mean=[0., 0.], cov=[[4.,0.],[0.,4.]], rotation_angle=30,
                                          scale=0.3, eliptical=0.5, device="cuda")
    elif filter_type == 'single filter':
        kernel_size = (kernel_size, kernel_size)
        filter = single_gaussian_filter(kernel_size=kernel_size, sigma=(2.0, 2.0))
    elif filter_type == 'CNN':
        kernel_size_cnn = torch.tensor(4)
        filter = Downsample_PASA_group_softmax(in_channels=64, kernel_size=kernel_size_cnn,
                                               group=1536/(kernel_size_cnn*kernel_size_cnn))

    return filter

def exercute_aliasing_filter(img: Tensor, filter_type = 'filter bank', kernel_size = 5, device="cuda") -> Tensor:
    if not (isinstance(img, torch.Tensor)):
        raise TypeError(f"img should be Tensor. Got {type(img)}")

    kernel_size = torch.tensor(kernel_size).to(dtype=torch.uint8)
    kernel_size_tensor = kernel_size.clone().detach().to(device)
    img = img.float().to(device)

    dtype = img.dtype if torch.is_floating_point(img) else torch.float32

    filter = aliasing_filter(filter_type, kernel_size_tensor).to(device)

    if filter_type == 'filter bank':

        filter = filter.expand(img.shape[-3], 1, filter.shape[0], filter.shape[1])

        img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [filter.dtype])

        # padding = (left, right, top, bottom)
        kernel_size_tensor = torch.tensor((kernel_size_tensor,kernel_size_tensor))
        padding = [kernel_size_tensor[0] // 2, kernel_size_tensor[0] // 2, kernel_size_tensor[1] // 2, kernel_size_tensor[1] // 2]

        img = torch_pad(img, padding, mode="reflect")

        img = conv2d(img, filter, groups=img.shape[-3])

        img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    elif filter_type == 'single filter':
        img = filter(img)
    elif filter_type == 'CNN':
        # TODO: the dimension of in_channels should be debug in the future
        'LinGaoyuan_20241104: if user want to use CNN as the ARM, the sampled pixels during training process must be integral multiple of kerner_size^2'
        kernel_size_cnn = torch.tensor(5)
        cnn_model = Downsample_PASA_group_softmax(in_channels=img.shape[-3], kernel_size=kernel_size_cnn,
                                                  group=img.shape[-3]/(kernel_size_cnn*kernel_size_cnn)
                                                  ).cuda()
        img = cnn_model(img).squeeze()
    return img

def _cast_squeeze_out(img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: torch.dtype) -> Tensor:
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            # it is better to round before cast
            img = torch.round(img)
        img = img.to(out_dtype)

    return img

def _cast_squeeze_in(img: Tensor, req_dtypes: List[torch.dtype]) -> Tuple[Tensor, bool, bool, torch.dtype]:
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype