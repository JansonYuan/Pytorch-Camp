# PyTorch框架班 

## 🎯Week 3

### 🛴【任务1】

**任务名称：**  
nn.Module与网络模型构建步骤；模型容器与AlexNet构建

**任务简介：**  
学习nn.Module类以及搭建网络模型步骤；熟悉搭建网络模型时常用的模型容器

**详细说明：**  
本节第一部分介绍网络模型的基本类nn.Module，nn.Module是所有网络层的基本类，它拥有8个有序字典，用于管理模型属性，本节课中将要学习如何构建一个Module。  
然后通过网络结构和计算图两个角度去观察搭建一个网络模型需要两个步骤：第一步，搭建子模块；第二步，拼接子模块。  

本节第二部分介绍搭建网络模型常用的容器，如Sequential，ModuleList, ModuleDict，然后学习pytorch提供的Alexnet网络模型结构加深对模型容器的认识  

**作业名称（详解）：**  
1. 采用步进(Step into)的调试方法从创建网络模型开始（**net = LeNet(classes=2)**）进入到每一个被调用函数，观察net的_modules字段何时被**构建**并且**赋值**，记录其中所有进入的类与函数   
例如：  
第一步：net = LeNet(classes=2)  
第二步：LeNet类，__init__()，super(LeNet, self).__init__()  
第三步: Module类, ......  
第n步：返回net  

2. 采用sequential容器，改写Alexnet，给features中每一个网络层增加名字，并通过下面这行代码打印出来  
print(alexnet._modules['features']._modules.keys())
- 本节代码下载：🥠[DataLoader与DataSet](https://github.com/JansonYuan/Pytorch-Camp/tree/master/%E4%BB%A3%E7%A0%81%E5%90%88%E9%9B%86/02-01-%E4%BB%A3%E7%A0%81-DataLoader%E4%B8%8EDataset/02-01-DataLoader%E4%B8%8EDataset)🍺[transforms与数据标准化](https://github.com/JansonYuan/Pytorch-Camp/blob/master/%E4%BB%A3%E7%A0%81%E5%90%88%E9%9B%86/02-02-%E4%BB%A3%E7%A0%81-transforms%E4%B8%8E%E6%95%B0%E6%8D%AE%E6%A0%87%E5%87%86%E5%8C%96/lesson-07-Logistic-Regression-norm.py)
- 本节课件下载：🥠[DataLoader与DataSet](https://github.com/JansonYuan/Pytorch-Camp/blob/master/%E8%AF%BE%E4%BB%B6%E5%90%88%E9%9B%86/02-01-ppt-DataLoader%E4%B8%8EDataSet.pdf)🍺[transforms与数据标准化](https://github.com/JansonYuan/Pytorch-Camp/blob/master/%E8%AF%BE%E4%BB%B6%E5%90%88%E9%9B%86/02-02-ppt-transforms%E4%B8%8E%E6%95%B0%E6%8D%AE%E6%A0%87%E5%87%86%E5%8C%96.pdf)

### 🛴【任务2】

**任务名称：**  
学习网络层中的卷积层，池化层，全连接层和激活函数层

**任务简介：**  
学习网络模型中采用的神经网络层，包括卷积层，池化层，全连接层和激活函数层，学会如何区分二维卷积和三维卷积；

**详细说明：**  
本节第一部分学习卷积神经网络中最重要的卷积层，了解卷积操作的过程与步骤，同时学会区分一维/二维/三维卷积，最后学习转置卷积（Transpose Convolution）的由来以及实现方法；  

本节第二部分学习池化层，全连接层和激活函数层，在池化层中有正常的最大值池化，均值池化，还有图像分割任务中常用的反池化——MaxUnpool，在激活函数中会学习Sigmoid,Tanh和Relu，以及Relu的各种变体，如LeakyReLU，PReLU， RReLU   

**作业名称（详解）：**  
1. 深入理解二维卷积，采用手算的方式实现以下卷积操作，然后**用代码验证**  
  1）采用2个尺寸为3*3的卷积核对3通道的5*5图像进行卷积，padding=0，stride=1，dilation=0
其中 input shape = （3， 5， 5），数据如下  
![](https://github.com/JansonYuan/Pytorch-Camp/blob/master/picture/Week3_3.jpg)
kernel size = 3*3，第一个卷积核所有权值均为1，第二个卷积核所有权值均为2，  
**计算输出的feature map尺寸以及所有像素值**  
  2）接1）题，上下左右四条边均采用padding，padding=1，填充值为0，计算输出的feature map尺寸以及所有像素值  

2. 对lena图进行3*3*33d卷积，提示：padding=（1， 0， 0）
```
# ================ 3d
# flag = 1
flag = 0
if flag:
    conv_layer = nn.Conv3d(3, 1, (1, 3, 3), padding=(1, 0, 0))
    nn.init.xavier_normal_(conv_layer.weight.data)
 
    # calculation
    img_tensor.unsqueeze_(dim=2)    # B*C*H*W to B*C*D*H*W
    img_conv = conv_layer(img_tensor)
```

- 本节代码下载：🍻[transforms（一）](https://github.com/JansonYuan/Pytorch-Camp/blob/master/%E4%BB%A3%E7%A0%81%E5%90%88%E9%9B%86/02-03-%E4%BB%A3%E7%A0%81-transforms%EF%BC%88%E4%B8%80%EF%BC%89/lesson-08-transforms_methods_1.py)🍻[transforms（二）](https://github.com/JansonYuan/Pytorch-Camp/tree/master/%E4%BB%A3%E7%A0%81%E5%90%88%E9%9B%86/02-04-%E4%BB%A3%E7%A0%81-transforms%EF%BC%88%E4%BA%8C%EF%BC%89)
- 本节课件下载：🍻[transforms（一）](https://github.com/JansonYuan/Pytorch-Camp/blob/master/%E8%AF%BE%E4%BB%B6%E5%90%88%E9%9B%86/02-03-ppt-transforms%EF%BC%88%E4%B8%80%EF%BC%89.pdf)🍻[transforms（二）](https://github.com/JansonYuan/Pytorch-Camp/blob/master/%E8%AF%BE%E4%BB%B6%E5%90%88%E9%9B%86/02-04-ppt-transforms%EF%BC%88%E4%BA%8C%EF%BC%89.pdf)



