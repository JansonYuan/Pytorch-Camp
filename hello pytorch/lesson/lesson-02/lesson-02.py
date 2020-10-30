# -*- coding:utf-8 -*-
"""
@file name  : lesson-02.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2018-08-26
@brief      : 张量的创建
"""
import torch
import numpy as np
torch.manual_seed(1)

# ===============================  example 1 ===============================
# 通过torch.tensor创建张量
#
# flag = True
flag = False
if flag:
    arr = np.ones((3, 3))
    print("ndarray的数据类型：", arr.dtype)

    t = torch.tensor(arr, device='cuda')
    # t = torch.tensor(arr)

    print(t)
# ===============================  example 2 ===============================
# 通过torch.from_numpy创建张量
# flag = True
flag = False
if flag:
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    t = torch.from_numpy(arr)
    # print("numpy array: ", arr)
    # print("tensor : ", t)

    # print("\n修改arr")
    # arr[0, 0] = 0
    # print("numpy array: ", arr)
    # print("tensor : ", t)

    print("\n修改tensor")
    t[0, 0] = -1
    print("numpy array: ", arr)
    print("tensor : ", t)

# ===============================  example 3 ===============================
# 通过torch.zeros创建张量
# flag = True
flag = False
if flag:
    out_t = torch.tensor([1])

    t = torch.zeros((3, 3), out=out_t)

    print(t, '\n', out_t)
    print(id(t), id(out_t), id(t) == id(out_t))

# ===============================  example 4 ===============================
# 通过torch.full创建全1张量
flag = True
# flag = False
if flag:
    t = torch.full((3, 3), 1.)  # 1.6之后若不指定dtype，就需要传入浮点数
    print(t)


# ===============================  example 5 ===============================
# 通过torch.arange创建等差数列张量
# flag = True
flag = False
if flag:
    t = torch.arange(2, 10, 2)
    print(t)

# ===============================  example 6 ===============================
# 通过torch.linspace创建均分数列张量
# flag = True
flag = False
if flag:
    # t = torch.linspace(2, 10, 5)
    t = torch.linspace(2, 10, 6)
    print(t)

# ===============================  example 7 ===============================
# 通过torch.normal创建正态分布张量
# flag = True
flag = False
if flag:

    # mean：张量 std: 张量
    # mean = torch.arange(1, 5, dtype=torch.float)
    # std = torch.arange(1, 5, dtype=torch.float)
    # t_normal = torch.normal(mean, std)
    # print("mean:{}\nstd:{}".format(mean, std))
    # print(t_normal)

    # mean：标量 std: 标量
    # t_normal = torch.normal(0., 1., size=(4,))
    # print(t_normal)

    # mean：张量 std: 标量
    mean = torch.arange(1, 5, dtype=torch.float)
    std = 1
    t_normal = torch.normal(mean, std)
    print("mean:{}\nstd:{}".format(mean, std))
    print(t_normal)












