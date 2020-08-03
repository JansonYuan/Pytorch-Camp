# -*- coding: utf-8 -*-
"""
# @file name  : resnet_inference.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2019-11-16
# @brief      : inference demo
"""

import os
import time
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.models as models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# config
vis = True
# vis = False
vis_row = 4

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

classes = ["ants", "bees"]


def img_transform(img_rgb, transform=None):
    """
    将数据转换为模型读取的形式
    :param img_rgb: PIL Image
    :param transform: torchvision.transform
    :return: tensor
    """

    if transform is None:
        raise ValueError("找不到transform！必须有transform对img进行处理")

    img_t = transform(img_rgb)
    return img_t


def get_img_name(img_dir, format="jpg"):
    """
    获取文件夹下format格式的文件名
    :param img_dir: str
    :param format: str
    :return: list
    """
    file_names = os.listdir(img_dir)
    img_names = list(filter(lambda x: x.endswith(format), file_names))

    if len(img_names) < 1:
        raise ValueError("{}下找不到{}格式数据".format(img_dir, format))
    return img_names


def get_model(m_path, vis_model=False):

    resnet18 = models.resnet18()
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 2)

    checkpoint = torch.load(m_path)
    resnet18.load_state_dict(checkpoint['model_state_dict'])

    if vis_model:
        from torchsummary import summary
        summary(resnet18, input_size=(3, 224, 224), device="cpu")

    return resnet18


if __name__ == "__main__":

    BASEDIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(BASEDIR, "..", "..", "data", "hymenoptera_data"))
    if not os.path.exists(data_dir):
        raise Exception("\n{} 不存在，请下载 07-02-数据-模型finetune.zip  放到\n{}  下，并解压即可".format(
            data_dir, os.path.dirname(data_dir)))

    img_dir = os.path.join(data_dir, "val", "bees")
    model_path = os.path.abspath(os.path.join(BASEDIR, "..", "..", "data", "resnet_checkpoint_14_epoch.pkl"))
    if not os.path.exists(model_path):
        raise Exception("\n{} 不存在，请下载 08-01-数据-20200724.zip  放到\n{}  下，并解压即可".format(
            model_path, os.path.dirname(model_path)))

    time_total = 0
    img_list, img_pred = list(), list()

    # 1. data
    img_names = get_img_name(img_dir)
    num_img = len(img_names)

    # 2. model
    resnet18 = get_model(model_path, True)
    resnet18.to(device)
    resnet18.eval()

    with torch.no_grad():
        for idx, img_name in enumerate(img_names):

            path_img = os.path.join(img_dir, img_name)

            # step 1/4 : path --> img
            img_rgb = Image.open(path_img).convert('RGB')

            # step 2/4 : img --> tensor
            img_tensor = img_transform(img_rgb, inference_transform)
            img_tensor.unsqueeze_(0)
            img_tensor = img_tensor.to(device)

            # step 3/4 : tensor --> vector
            time_tic = time.time()
            outputs = resnet18(img_tensor)
            time_toc = time.time()

            # step 4/4 : visualization
            _, pred_int = torch.max(outputs.data, 1)
            pred_str = classes[int(pred_int)]

            if vis:
                img_list.append(img_rgb)
                img_pred.append(pred_str)

                if (idx+1) % (vis_row*vis_row) == 0 or num_img == idx+1:
                    for i in range(len(img_list)):
                        plt.subplot(vis_row, vis_row, i+1).imshow(img_list[i])
                        plt.title("predict:{}".format(img_pred[i]))
                    plt.show()
                    plt.close()
                    img_list, img_pred = list(), list()

            time_s = time_toc-time_tic
            time_total += time_s

            print('{:d}/{:d}: {} {:.3f}s '.format(idx + 1, num_img, img_name, time_s))

    print("\ndevice:{} total time:{:.1f}s mean:{:.3f}s".
          format(device, time_total, time_total/num_img))
    if torch.cuda.is_available():
        print("GPU name:{}".format(torch.cuda.get_device_name()))

