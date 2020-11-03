# -*- coding: utf-8 -*-
"""
# @file name  : 1_split_dataset.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-07-24 10:08:00
# @brief      : 将数据集划分为训练集，验证集，测试集
"""

import os
import random
import shutil
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


if __name__ == '__main__':

    dataset_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "RMB_data"))
    split_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "rmb_split"))
    train_dir = os.path.join(split_dir, "train")
    valid_dir = os.path.join(split_dir, "valid")
    test_dir = os.path.join(split_dir, "test")

    if not os.path.exists(dataset_dir):
        raise Exception("\n{} 不存在，请下载 02-01-数据-RMB_data.rar 放到\n{} 下，并解压即可".format(
            dataset_dir, os.path.dirname(dataset_dir)))

    train_pct = 0.8
    valid_pct = 0.1
    test_pct = 0.1

    for root, dirs, files in os.walk(dataset_dir):
        for sub_dir in dirs:

            imgs = os.listdir(os.path.join(root, sub_dir))
            imgs = list(filter(lambda x: x.endswith('.jpg'), imgs))
            random.shuffle(imgs)
            img_count = len(imgs)

            train_point = int(img_count * train_pct)
            valid_point = int(img_count * (train_pct + valid_pct))

            if img_count == 0:
                print("{}目录下，无图片，请检查".format(os.path.join(root, sub_dir)))
                import sys
                sys.exit(0)
            for i in range(img_count):
                if i < train_point:
                    out_dir = os.path.join(train_dir, sub_dir)
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sub_dir)
                else:
                    out_dir = os.path.join(test_dir, sub_dir)

                makedir(out_dir)

                target_path = os.path.join(out_dir, imgs[i])
                src_path = os.path.join(dataset_dir, sub_dir, imgs[i])

                shutil.copy(src_path, target_path)

            print('Class:{}, train:{}, valid:{}, test:{}'.format(sub_dir, train_point, valid_point-train_point,
                                                                 img_count-valid_point))
            print("已在 {} 创建划分好的数据\n".format(out_dir))
