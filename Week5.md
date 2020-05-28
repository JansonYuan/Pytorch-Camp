# PyTorch框架班 

## 🎯Week 5

### 🛴【任务1】

**任务名称：**  
学习率调整；TensorBoard简介与安装

**任务简介：**  
熟悉pytorch的学习率调整策略；安装可视化工具TensorBoard

**详细说明：**  
本节第一部分讲解pytorch中提供的学习率调整策略，首先介绍基类_LRScheduler基本属性与方法，然后逐个学习率方法进行讲解使用，分别Step、MultiStep、Exponential、CosineAnnealing、ReduceLROnPleateau和Lambda，一共六种学习率调整策略；

本节第二部分讲解可视化工具TensorBoard的运行机制与安装过程，TensorBoard是强大的可视化工具，起初为TensorFlow的副产品，但目前PyTorch已支持TensorBoard的使用。目前，TensorBoard支持Scalars, Images, Audio, Graphs, Distrbutions, Histograms, Embeddings, Text等数据的可视化。

**作业名称（详解）：**  
1. 熟悉TensorBoard的运行机制，安装TensorBoard，并绘制曲线 y = 2*x。

- 本节代码下载：
🍄[学习率调整策略](https://github.com/JansonYuan/Pytorch-Camp/tree/master/%E4%BB%A3%E7%A0%81%E5%90%88%E9%9B%86/05-01-%E4%BB%A3%E7%A0%81-%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4%E7%AD%96%E7%95%A5/lesson-19)
🥑[TensorBoard简介与安装](https://github.com/JansonYuan/Pytorch-Camp/blob/master/%E4%BB%A3%E7%A0%81%E5%90%88%E9%9B%86/05-02-%E4%BB%A3%E7%A0%81-TensorBoard%E7%AE%80%E4%BB%8B%E4%B8%8E%E5%AE%89%E8%A3%85/lesson-20/test_tensorboard.py)
- 本节课件下载：
🍄[学习率调整策略](https://github.com/JansonYuan/Pytorch-Camp/blob/master/%E8%AF%BE%E4%BB%B6%E5%90%88%E9%9B%86/05-01-ppt-%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4%E7%AD%96%E7%95%A5.pdf)
🥑[TensorBoard简介与安装](https://github.com/JansonYuan/Pytorch-Camp/blob/master/%E8%AF%BE%E4%BB%B6%E5%90%88%E9%9B%86/05-02-ppt-TensorBoard%E7%AE%80%E4%BB%8B%E4%B8%8E%E5%AE%89%E8%A3%85.pdf)
### 🛴【任务2】

**任务名称：**  
TensorBoard使用(一)；TensorBoard使用(二)；

**任务简介：**  
学习TensorBoard中scalar与histogram的使用；学习TensorBoard中Image与PyTorch的make_grid使用。

**详细说明：**  
本节第一部分学习TensorBoard的SummaryWriter类的基本属性，然后学习add_scalar, add_scalars和add_histogram的使用，最后采用所学函数实现模型训练过程中的Loss曲线，Accuracy曲线的对比监控，同时对参数及其梯度的分布进行可视化。

本节第二部分学习TensorBoard的add_image方法，并学习PyTorch的make_grid函数构建网格图片，对批量图片进行可视化，最后采用所学函数对AlexNet网络卷积核与特征图进行可视化分析。


**作业名称（详解）：**  
1. 可视化任意网络模型训练的Loss，及Accuracy曲线图，Train与Valid必须在同一个图中。

2. 采用make_grid，对任意图像训练输入数据进行批量可视化。 

- 本节代码下载：
🍆[TensorBoard使用（一）](https://github.com/JansonYuan/Pytorch-Camp/tree/master/%E4%BB%A3%E7%A0%81%E5%90%88%E9%9B%86/05-03-%E4%BB%A3%E7%A0%81-TensorBoard%E4%BD%BF%E7%94%A8%EF%BC%88%E4%B8%80%EF%BC%89/lesson-21)
🥜[TensorBoard使用（二）](https://github.com/JansonYuan/Pytorch-Camp/tree/master/%E4%BB%A3%E7%A0%81%E5%90%88%E9%9B%86/05-04-%E4%BB%A3%E7%A0%81-TensorBoard%E4%BD%BF%E7%94%A8%EF%BC%88%E4%BA%8C%EF%BC%89/lesson-22)
- 本节课件下载：
🍆[TensorBoard使用（一）](https://github.com/JansonYuan/Pytorch-Camp/blob/master/%E8%AF%BE%E4%BB%B6%E5%90%88%E9%9B%86/05-03-ppt-TensorBoard%E4%BD%BF%E7%94%A8%EF%BC%88%E4%B8%80%EF%BC%89.pdf)
🥜[TensorBoard使用（二）](https://github.com/JansonYuan/Pytorch-Camp/blob/master/%E8%AF%BE%E4%BB%B6%E5%90%88%E9%9B%86/05-04-ppt-TensorBoard%E4%BD%BF%E7%94%A8%EF%BC%88%E4%BA%8C%EF%BC%89.pdf)

### 🛴【任务3】

**任务名称：**  
hook函数与CAM(class activation map, 类激活图)

**任务简介：**  
学习pytorch的hook函数机制以及CAM可视化算法

**详细说明：**
深入学习了解pytorch的hook函数运行机制，介绍pytorch中提供的4种hook函数，分别为：
1. torch.Tensor.register_hook(hook)
2. torch.nn.Module.register_forward_hook
3. torch.nn.Module.register_forward_pre_hook
4. torch.nn.Module.register_backward_hook
最后，介绍CAM可视化及其改进算法Grad-CAM  

**作业名称（详解）：** 
1. 采用torch.nn.Module.register_forward_hook机制实现AlexNet第一个卷积层输出特征图的可视化，并将/torchvision/models/alexnet.py中第28行改为：nn.ReLU(inplace=False)，观察
inplace=True与inplace=False的差异。

- 本节代码下载：
🌽[hook函数与CAM可视化](https://github.com/JansonYuan/Pytorch-Camp/tree/master/%E4%BB%A3%E7%A0%81%E5%90%88%E9%9B%86/05-05-%E4%BB%A3%E7%A0%81-hook%E5%87%BD%E6%95%B0%E4%B8%8ECAM%E5%8F%AF%E8%A7%86%E5%8C%96/lesson-23)
- 本节课件下载：
🌽[hook函数与CAM可视化](https://github.com/JansonYuan/Pytorch-Camp/blob/master/%E8%AF%BE%E4%BB%B6%E5%90%88%E9%9B%86/05-05-ppt-hook%E5%87%BD%E6%95%B0%E4%B8%8ECAM%E5%8F%AF%E8%A7%86%E5%8C%96.pdf)
