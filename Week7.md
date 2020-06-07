# PyTorch框架班 

## 🎯Week 7

### 🛴【任务1】

**任务名称：**  
模型保存与加载；Finetune

**任务简介：**  
了解序列化与反序列化；了解transfer learning 与 model finetune

**详细说明：**  
本节第一部分学习pytorch中的模型保存与加载，也常称为序列化与反序列化，本节将讲解序列化与反序列化的概念，进而对模型的保存与加载有深刻的认识，同时介绍pytorch中模型保存与加载的方法函数。

本节第二部分学习模型微调（Finetune）的方法，以及认识Transfer Learning（迁移学习）与Model Finetune之间的关系。

**作业名称（详解）：**  
1. 任意一个模型，将其内部权重赋值为0.123，然后保存state_dict；接着重新创建一个网络net_load，打印刚创建的net_load中的参数，接着加载保存好的state_dict，并且load到net_load中，再次打印net_load中的参数，观察是否成功将保存的参数重新加载进来。
**打卡要求**：赋值print的信息或截图重新加载进来的参数的数值

2. 一句话描述迁移学习研究内容。
- 本节代码下载：
🌭[正则化之weight_decay](https://github.com/JansonYuan/Pytorch-Camp/blob/master/%E4%BB%A3%E7%A0%81%E5%90%88%E9%9B%86/06-01-%E4%BB%A3%E7%A0%81-%E6%AD%A3%E5%88%99%E5%8C%96%E4%B9%8Bweight_decay/lesson-24/L2_regularization.py)
🍟[正则化-Dropout](https://github.com/JansonYuan/Pytorch-Camp/tree/master/%E4%BB%A3%E7%A0%81%E5%90%88%E9%9B%86/06-02-%E4%BB%A3%E7%A0%81-%E6%AD%A3%E5%88%99%E5%8C%96-Dropout/lesson-25)
- 本节课件下载：
🌭[正则化之weight_decay](https://github.com/JansonYuan/Pytorch-Camp/blob/master/%E8%AF%BE%E4%BB%B6%E5%90%88%E9%9B%86/06-01-ppt-%E6%AD%A3%E5%88%99%E5%8C%96%E4%B9%8Bweight_decay.pdf)
🍟[正则化-Dropout](https://github.com/JansonYuan/Pytorch-Camp/blob/master/%E8%AF%BE%E4%BB%B6%E5%90%88%E9%9B%86/06-02-ppt-%E6%AD%A3%E5%88%99%E5%8C%96-Dropout.pdf)
### 🛴【任务2】

**任务名称：**  
GPU的使用；PyTorch中常见报错
**任务简介：**  
学习使用GPU进行加速运算；学习常见报错信息，方便调试代码

**详细说明：**  
本节第一部分学习如何使用GPU进行加速模型运算，介绍Tensor和Module的to函数使用以及它们之间的差异，同时学习多GPU运算的分发并行机制

本节第二部分学习PyTorch中常见的报错信息及其解决方法，为大家提供一个快速的调试文档，同时希望大家一同贡献该文档

**作业名称（详解）：**  
1. 采用time函数与for 循环，对比cpu上运算时间以及gpu上运算时间。
**打卡要求**：文字描述cpu型号，gpu型号，运算时间对比。

2. 尝试贡献一个pytorch中的报错信息/存在的坑（选做）。

- 本节代码下载：
🍘[Batch Normalization](https://github.com/JansonYuan/Pytorch-Camp/tree/master/%E4%BB%A3%E7%A0%81%E5%90%88%E9%9B%86/06-03-%E4%BB%A3%E7%A0%81-Batch%20Normalization/lesson-26)
🍙[Normalizaiton_layers](https://github.com/JansonYuan/Pytorch-Camp/blob/master/%E4%BB%A3%E7%A0%81%E5%90%88%E9%9B%86/06-04-%E4%BB%A3%E7%A0%81-Normalizaiton_layers/lesson-27/normallization_layers.py)
- 本节课件下载：
🍘[Batch Normalization](https://github.com/JansonYuan/Pytorch-Camp/blob/master/%E8%AF%BE%E4%BB%B6%E5%90%88%E9%9B%86/06-03-Batch%20Normalization.pdf)
🍙[Normalizaiton_layers](https://github.com/JansonYuan/Pytorch-Camp/blob/master/%E8%AF%BE%E4%BB%B6%E5%90%88%E9%9B%86/06-04-ppt-Normalizaiton_layers.pdf)
