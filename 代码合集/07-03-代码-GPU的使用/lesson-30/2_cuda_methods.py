# -*- coding: utf-8 -*-

import os
import numpy as np
import torch


# ========================== 选择 gpu
# flag = 0
flag = 1
if flag:
    gpu_id = 0
    gpu_str = "cuda:{}".format(gpu_id)
    device = torch.device(gpu_str if torch.cuda.is_available() else "cpu")

    x_cpu = torch.ones((3, 3))
    x_gpu = x_cpu.to(device)

    print("x_gpu:\ndevice: {} is_cuda: {} id: {}".format(x_gpu.device, x_gpu.is_cuda, id(x_gpu)))


# ========================== 查看 gpu数量/名称
# flag = 0
flag = 1
if flag:
    device_count = torch.cuda.device_count()
    print("\ndevice_count: {}".format(device_count))

    device_name = torch.cuda.get_device_name(0)
    print("\ndevice_name: {}".format(device_name))









