# PyTorch框架班 

## 🎯Week 6
- 代码下载: ☕[autograd](https://github.com/JansonYuan/Pytorch-Camp/tree/master/hello%20pytorch)

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

