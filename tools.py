import os
import math
import random
import torch
import torch.nn.functional as F
from torchvision import transforms as tvt
from torchvision.transforms import Normalize as VisionNormalize
import numpy as np
from itertools import tee
from pynvml import *
import psutil
from datetime import datetime, timezone, timedelta
from shapely.geometry import Polygon, Point

# initializing here!!
nvmlInit() 


# unfold 4D tensor by splitting dim 2 and 3, with padding
# [batch, channel, heigh, width] -> num_patch_in_h * num_patch_in_w * batch, channel, kernal, kernal]
def unfold4D_with_padding(x, side, padding_value):
    # x = [batch, channel, heigh, width]
    pad2_right = (x.size(2) // side * side + side) - x.size(2)
    pad3_right = (x.size(3) // side * side + side) - x.size(3)
    x = F.pad(x, (0, pad3_right, 0, pad2_right, 0, 0, 0, 0), value = padding_value)
    x = x.unfold(2, side, side).unfold(3, side, side).permute(2, 3, 0, 1, 4, 5).reshape(-1, x.shape[1], side, side)
    return x


# fold a list of 4D tensors into one 4D large tensor, and keep padding.
# num_patch_in_h * num_patch_in_w * batch, channel, kernal, kernal] -> [batch, channel, heigh(with padding), width(with padding)]
def fold4D_depadding(x, side, real_h, real_w):
    N, c, _, _ = x.shape
    nh = int(math.ceil(real_h / side))
    nw = int(math.ceil(real_w / side))
    n = int(N / nh / nw) 
    x = x.reshape(nh, nw, n, c, side, side).permute(2, 3, 0, 4, 1, 5).reshape(n, c, nh*side, nw*side)
    x = x[:, :, :real_h, :real_w]
    return x # [batch, c, realheight, realwidth]


def image_norm(t):
    t = t / 255.
    t = VisionNormalize(mean = [0.7913, 0.8069, 0.8089], 
                        std = [0.2094, 0.2274, 0.2478])(t)
    return t
    
    
def image_augmentation(t):
    # t : float32
    fn =  tvt.Compose([tvt.ColorJitter(contrast = (0.8, 1.5), brightness = [0.7, 1.3]), \
                            tvt.GaussianBlur(kernel_size = (1, 11), sigma = (3, 7)), \
                            tvt.RandomAdjustSharpness(3, p = 0.5)])
    return fn(t)

    
def is_lake(poly):
    # https://gis.stackexchange.com/questions/295874/getting-polygon-breadth-in-shapely
    box = poly.minimum_rotated_rectangle
    x, y = box.exterior.coords.xy
    edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))
    length = max(edge_length)
    width = min(edge_length)
    return True if length / width >= 10.0 else False

    
def mean(x):
    if x == []:
        return 0.0
    return sum(x) / len(x)


def std(x):
    return np.std(x)


def minmax_norm(v, minv, maxv):
    return (v-minv) / (maxv-minv) + 1


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def set_seed(seed = -1):
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dump_config_to_strs(file_path):
    # return list
    lines = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('from') or line.startswith('import') or line.startswith('#') \
                    or line.strip() == '':
                continue
            lines.append(line.strip())
    return lines


def log_file_name():
    dt = datetime.now(timezone(timedelta(hours=8)))
    return dt.strftime("%Y%m%d_%H%M%S") + '.log'


class GPUInfo:
    _h = nvmlDeviceGetHandleByIndex(0)

    @classmethod
    def mem(cls, handle = None):
        _h = handle if handle else cls._h
        info = nvmlDeviceGetMemoryInfo(_h)
        return info.used // 1048576, info.total // 1048576 # in MB
    
    @classmethod
    def power(cls, handle = None):
        _h = handle if handle else cls._h
        return nvmlDeviceGetPowerUsage(_h) / 1e3


class RAMInfo:
    @classmethod
    def mem(cls):
        return int(psutil.Process(os.getpid()).memory_info().rss / 1048576) # in MB

    @classmethod
    def mem_global(cls):
        return int(psutil.virtual_memory()[3] / 1048576) # in MB

