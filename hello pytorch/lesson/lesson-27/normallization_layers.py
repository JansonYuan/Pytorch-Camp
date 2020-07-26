# -*- coding: utf-8 -*-
"""
# @file name  : bn_and_initialize.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2019-11-03
# @brief      : pytorch中常见的 normalization layers
"""
import torch
import numpy as np
import torch.nn as nn
import sys, os
hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+".."+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)
from tools.common_tools import set_seed


set_seed(1)  # 设置随机种子

# ======================================== nn.layer norm
# flag = 1
flag = 0
if flag:
    batch_size = 8
    num_features = 6

    features_shape = (3, 4)

    feature_map = torch.ones(features_shape)  # 2D
    feature_maps = torch.stack([feature_map * (i + 1) for i in range(num_features)], dim=0)  # 3D
    feature_maps_bs = torch.stack([feature_maps for i in range(batch_size)], dim=0)  # 4D

    # feature_maps_bs shape is [8, 6, 3, 4],  B * C * H * W
    # ln = nn.LayerNorm(feature_maps_bs.size()[1:], elementwise_affine=True)
    # ln = nn.LayerNorm(feature_maps_bs.size()[1:], elementwise_affine=False)
    # ln = nn.LayerNorm([6, 3, 4])
    ln = nn.LayerNorm([6, 3])

    output = ln(feature_maps_bs)

    print("Layer Normalization")
    print(ln.weight.shape)
    print(feature_maps_bs[0, ...])
    print(output[0, ...])

# ======================================== nn.instance norm 2d
# flag = 1
flag = 0
if flag:

    batch_size = 3
    num_features = 3
    momentum = 0.3

    features_shape = (2, 2)

    feature_map = torch.ones(features_shape)    # 2D
    feature_maps = torch.stack([feature_map * (i + 1) for i in range(num_features)], dim=0)  # 3D
    feature_maps_bs = torch.stack([feature_maps for i in range(batch_size)], dim=0)  # 4D

    print("Instance Normalization")
    print("input data:\n{} shape is {}".format(feature_maps_bs, feature_maps_bs.shape))

    instance_n = nn.InstanceNorm2d(num_features=num_features, momentum=momentum)

    for i in range(1):
        outputs = instance_n(feature_maps_bs)

        print(outputs)
        # print("\niter:{}, running_mean.shape: {}".format(i, bn.running_mean.shape))
        # print("iter:{}, running_var.shape: {}".format(i, bn.running_var.shape))
        # print("iter:{}, weight.shape: {}".format(i, bn.weight.shape))
        # print("iter:{}, bias.shape: {}".format(i, bn.bias.shape))


# ======================================== nn.grop norm
flag = 1
# flag = 0
if flag:

    batch_size = 2
    num_features = 4
    num_groups = 4   # 3 Expected number of channels in input to be divisible by num_groups

    features_shape = (2, 2)

    feature_map = torch.ones(features_shape)    # 2D
    feature_maps = torch.stack([feature_map * (i + 1) for i in range(num_features)], dim=0)  # 3D
    feature_maps_bs = torch.stack([feature_maps * (i + 1) for i in range(batch_size)], dim=0)  # 4D

    gn = nn.GroupNorm(num_groups, num_features)
    outputs = gn(feature_maps_bs)

    print("Group Normalization")
    print(gn.weight.shape)
    print(outputs[0])






