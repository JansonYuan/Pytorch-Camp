# PyTorch框架班 

## 🎯Week 6

### 🛴【任务1】

**任务名称：**  
weight_decay；dropout

**任务简介：**  
了解正则化中L1和L2（weight decay）；了解dropout

**详细说明：**  
本节第一部分讲解正则化的概念，正则化方法是机器学习（深度学习）中重要的方法，它目的在于减小方差。常用的正则化方法有L1和L2正则化，其中L2正则化又称为weight decay。在pytorch的优化器中就提供了weight decay的实现，本节课将学习weight decay的pytorch实现。

本节第二部分讲解深度学习中常见的正则化方法——Dropout，Dropout是简洁高效的正则化方法，但需要注意其在实现过程中的权值数据尺度问题。本节课将详细介绍pytorch中Dropout的实现细节。

**作业名称（详解）：**  
1. weight decay在pytorch的SGD中实现代码是哪一行？它对应的数学公式为？

2. PyTorch中，Dropout在训练的时候权值尺度会进行什么操作？
- 本节代码下载：
🌭[正则化之weight_decay](https://github.com/JansonYuan/Pytorch-Camp/blob/master/%E4%BB%A3%E7%A0%81%E5%90%88%E9%9B%86/06-01-%E4%BB%A3%E7%A0%81-%E6%AD%A3%E5%88%99%E5%8C%96%E4%B9%8Bweight_decay/lesson-24/L2_regularization.py)
🍟[正则化-Dropout](https://github.com/JansonYuan/Pytorch-Camp/tree/master/%E4%BB%A3%E7%A0%81%E5%90%88%E9%9B%86/06-02-%E4%BB%A3%E7%A0%81-%E6%AD%A3%E5%88%99%E5%8C%96-Dropout/lesson-25)
- 本节课件下载：
🌭[正则化之weight_decay](https://github.com/JansonYuan/Pytorch-Camp/blob/master/%E8%AF%BE%E4%BB%B6%E5%90%88%E9%9B%86/06-01-ppt-%E6%AD%A3%E5%88%99%E5%8C%96%E4%B9%8Bweight_decay.pdf)
🍟[正则化-Dropout](https://github.com/JansonYuan/Pytorch-Camp/blob/master/%E8%AF%BE%E4%BB%B6%E5%90%88%E9%9B%86/06-02-ppt-%E6%AD%A3%E5%88%99%E5%8C%96-Dropout.pdf)
### 🛴【任务2】

**任务名称：**  
Batch Normalization；Layer Normalizatoin、Instance Normalizatoin和Group Normalizatoin

**任务简介：**  
学习深度学习中常见的标准化方法

**详细说明：**  
本节第一部分介绍深度学习中最重要的一个 Normalizatoin方法——Batch Normalization，并分析其计算方式，同时讲解PyTorch中nn.BatchNorm1d、nn.BatchNorm2d、nn.BatchNorm3d三种BN的计算方式及原理。

本节第二部分介绍2015年之后出现的常见的Normalization方法——Layer Normalizatoin、Instance Normalizatoin和Group Normalizatoin，分析各Normalization的由来与应用场景，同时对比分析BN，LN，IN和GN之间的计算差异。

**作业名称（详解）：**  
1. Batch Normalization 的4个重要参数分别是什么？BN层中采用这个四个参数对X做了什么操作？

2. 课程结尾的 “加减乘除”是什么意思？

- 本节代码下载：
🍘[Batch Normalization](https://github.com/JansonYuan/Pytorch-Camp/tree/master/%E4%BB%A3%E7%A0%81%E5%90%88%E9%9B%86/06-03-%E4%BB%A3%E7%A0%81-Batch%20Normalization/lesson-26)
🍙[Normalizaiton_layers](https://github.com/JansonYuan/Pytorch-Camp/blob/master/%E4%BB%A3%E7%A0%81%E5%90%88%E9%9B%86/06-04-%E4%BB%A3%E7%A0%81-Normalizaiton_layers/lesson-27/normallization_layers.py)
- 本节课件下载：
🍘[Batch Normalization](https://github.com/JansonYuan/Pytorch-Camp/blob/master/%E8%AF%BE%E4%BB%B6%E5%90%88%E9%9B%86/06-03-Batch%20Normalization.pdf)
🍙[Normalizaiton_layers](https://github.com/JansonYuan/Pytorch-Camp/blob/master/%E8%AF%BE%E4%BB%B6%E5%90%88%E9%9B%86/06-04-ppt-Normalizaiton_layers.pdf)
