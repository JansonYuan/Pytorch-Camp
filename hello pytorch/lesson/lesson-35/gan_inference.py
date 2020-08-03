# -*- coding: utf-8 -*-
"""
# @file name  : gan_inference.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2019-12-05
# @brief      : gan inference
"""
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import imageio
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

import sys
hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+".."+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)

from tools.common_tools import set_seed
from torch.utils.data import DataLoader
from tools.my_dataset import CelebADataset
from tools.dcgan import Discriminator, Generator
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def remove_module(state_dict_g):
    # remove module.
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict_g.items():
        namekey = k[7:] if k.startswith('module.') else k
        new_state_dict[namekey] = v

    return new_state_dict

set_seed(1)  # 设置随机种子

# config
path_checkpoint = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "dcgan_checkpoint_14_epoch.pkl"))
if not os.path.exists(path_checkpoint):
    raise Exception("\n{} 不存在，请下载 08-04-数据-20200724.zip  放到\n{}  下，并解压即可".format(
        path_checkpoint, os.path.dirname(path_checkpoint)))

image_size = 64
num_img = 64
nc = 3
nz = 100
ngf = 128
ndf = 128

d_transforms = transforms.Compose([transforms.Resize(image_size),
                   transforms.CenterCrop(image_size),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
               ])

# step 1: data
fixed_noise = torch.randn(num_img, nz, 1, 1, device=device)

flag = 0
# flag = 1
if flag:
    z_idx = 0
    single_noise = torch.randn(1, nz, 1, 1, device=device)
    for i in range(num_img):
        add_noise = single_noise
        add_noise = add_noise[0, z_idx, 0, 0] + i*0.01
        fixed_noise[i, ...] = add_noise


# step 2: model
net_g = Generator(nz=nz, ngf=ngf, nc=nc)
# net_d = Discriminator(nc=nc, ndf=ndf)
checkpoint = torch.load(path_checkpoint, map_location="cpu")

state_dict_g = checkpoint["g_model_state_dict"]
state_dict_g = remove_module(state_dict_g)
net_g.load_state_dict(state_dict_g)
net_g.to(device)
# net_d.load_state_dict(checkpoint["d_model_state_dict"])
# net_d.to(device)

# step3: inference
with torch.no_grad():
    fake_data = net_g(fixed_noise).detach().cpu()
img_grid = vutils.make_grid(fake_data, padding=2, normalize=True).numpy()
img_grid = np.transpose(img_grid, (1, 2, 0))
plt.imshow(img_grid)
plt.show()









