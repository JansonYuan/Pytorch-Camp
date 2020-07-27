# PyTorch框架班 

## 🎯Week 7
- 代码下载: ☕[autograd](https://github.com/JansonYuan/Pytorch-Camp/tree/master/hello%20pytorch)

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

