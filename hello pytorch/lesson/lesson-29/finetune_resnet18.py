# -*- coding: utf-8 -*-
"""
# @file name  : finetune_resnet18.py
# @author     :  TingsongYu https://github.com/TingsongYu
# @date       : 2019-11-05
# @brief      : 模型finetune方法
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt

import sys
hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+".."+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)

from tools.my_dataset import AntsDataset
from tools.common_tools import set_seed
import torchvision.models as models
import torchvision
BASEDIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use device :{}".format(device))

set_seed(1)  # 设置随机种子
label_name = {"ants": 0, "bees": 1}

# 参数设置
MAX_EPOCH = 25
BATCH_SIZE = 16
LR = 0.001
log_interval = 10
val_interval = 1
classes = 2
start_epoch = -1
lr_decay_step = 7


# ============================ step 1/5 数据 ============================
data_dir = os.path.abspath(os.path.join(BASEDIR, "..", "..", "data", "hymenoptera_data"))
if not os.path.exists(data_dir):
    raise Exception("\n{} 不存在，请下载 07-02-数据-模型finetune.zip  放到\n{} 下，并解压即可".format(
        data_dir, os.path.dirname(data_dir)))

train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "val")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

# 构建MyDataset实例
train_data = AntsDataset(data_dir=train_dir, transform=train_transform)
valid_data = AntsDataset(data_dir=valid_dir, transform=valid_transform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

# ============================ step 2/5 模型 ============================

# 1/3 构建模型
resnet18_ft = models.resnet18()

# 2/3 加载参数
# flag = 0
flag = 1
if flag:
    path_pretrained_model = os.path.join(BASEDIR, "..", "..", "data", "finetune_resnet18-5c106cde.pth")
    if not os.path.exists(path_pretrained_model):
        raise Exception("\n{} 不存在，请下载 07-02-数据-模型finetune.zip\n放到 {}下，并解压即可".format(
            path_pretrained_model, os.path.dirname(path_pretrained_model)))
    state_dict_load = torch.load(path_pretrained_model)
    resnet18_ft.load_state_dict(state_dict_load)

# 法1 : 冻结卷积层
flag_m1 = 0
# flag_m1 = 1
if flag_m1:
    for param in resnet18_ft.parameters():
        param.requires_grad = False
    print("conv1.weights[0, 0, ...]:\n {}".format(resnet18_ft.conv1.weight[0, 0, ...]))


# 3/3 替换fc层
num_ftrs = resnet18_ft.fc.in_features
resnet18_ft.fc = nn.Linear(num_ftrs, classes)


resnet18_ft.to(device)
# ============================ step 3/5 损失函数 ============================
criterion = nn.CrossEntropyLoss()                                                   # 选择损失函数

# ============================ step 4/5 优化器 ============================
# 法2 : conv 小学习率
# flag = 0
flag = 1
if flag:
    fc_params_id = list(map(id, resnet18_ft.fc.parameters()))     # 返回的是parameters的 内存地址
    base_params = filter(lambda p: id(p) not in fc_params_id, resnet18_ft.parameters())
    optimizer = optim.SGD([
        {'params': base_params, 'lr': LR*0},   # 0
        {'params': resnet18_ft.fc.parameters(), 'lr': LR}], momentum=0.9)

else:
    optimizer = optim.SGD(resnet18_ft.parameters(), lr=LR, momentum=0.9)               # 选择优化器

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.1)     # 设置学习率下降策略


# ============================ step 5/5 训练 ============================
train_curve = list()
valid_curve = list()

for epoch in range(start_epoch + 1, MAX_EPOCH):

    loss_mean = 0.
    correct = 0.
    total = 0.

    resnet18_ft.train()
    for i, data in enumerate(train_loader):

        # forward
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = resnet18_ft(inputs)

        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()

        # update weights
        optimizer.step()

        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().cpu().sum().numpy()

        # 打印训练信息
        loss_mean += loss.item()
        train_curve.append(loss.item())
        if (i+1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
            loss_mean = 0.

            # if flag_m1:
            print("epoch:{} conv1.weights[0, 0, ...] :\n {}".format(epoch, resnet18_ft.conv1.weight[0, 0, ...]))

    scheduler.step()  # 更新学习率

    # validate the model
    if (epoch+1) % val_interval == 0:

        correct_val = 0.
        total_val = 0.
        loss_val = 0.
        resnet18_ft.eval()
        with torch.no_grad():
            for j, data in enumerate(valid_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = resnet18_ft(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).squeeze().cpu().sum().numpy()

                loss_val += loss.item()

            loss_val_mean = loss_val/len(valid_loader)
            valid_curve.append(loss_val_mean)
            print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val_mean, correct_val / total_val))

train_x = range(len(train_curve))
train_y = train_curve

train_iters = len(train_loader)
valid_x = np.arange(1, len(valid_curve)+1) * train_iters*val_interval # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
valid_y = valid_curve

plt.plot(train_x, train_y, label='Train')
plt.plot(valid_x, valid_y, label='Valid')

plt.legend(loc='upper right')
plt.ylabel('loss value')
plt.xlabel('Iteration')
plt.show()







