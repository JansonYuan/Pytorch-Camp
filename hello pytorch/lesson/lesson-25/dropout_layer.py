# -*- coding:utf-8 -*-
"""
@file name  : dropout_layer.py
# @author   : TingsongYu https://github.com/TingsongYu
@date       : 2019-10-31
@brief      : dropout 使用实验
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys, os
hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+".."+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)

from tools.common_tools import set_seed
from torch.utils.tensorboard import SummaryWriter

# set_seed(1)  # 设置随机种子


class Net(nn.Module):
    def __init__(self, neural_num, d_prob=0.5):
        super(Net, self).__init__()

        self.linears = nn.Sequential(

            nn.Dropout(d_prob),
            nn.Linear(neural_num, 1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.linears(x)

input_num = 10000
x = torch.ones((input_num, ), dtype=torch.float32)

net = Net(input_num, d_prob=0.5)
net.linears[1].weight.detach().fill_(1.)

net.train()
y = net(x)
print("output in training mode", y)

net.eval()
y = net(x)
print("output in eval mode", y)

















