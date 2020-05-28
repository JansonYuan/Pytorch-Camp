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
🍀[权值初始化](https://github.com/JansonYuan/Pytorch-Camp/blob/master/代码合集/04-01-代码-权值初始化/lesson-14-grad_vanish_explod.py)
🌸[损失函数（一）](https://github.com/JansonYuan/Pytorch-Camp/tree/master/代码合集/04-02-代码-损失函数(一)/lesson-15)
- 本节课件下载：
🍀[权值初始化](https://github.com/JansonYuan/Pytorch-Camp/blob/master/课件合集/04-01-ppt-权值初始化.pdf)
🌸[损失函数（一）](https://github.com/JansonYuan/Pytorch-Camp/blob/master/课件合集/04-02-ppt-损失函数(一).pdf)
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
🏵[损失函数（二）](https://github.com/JansonYuan/Pytorch-Camp/blob/master/代码合集/04-03-代码-损失函数(二)/lesson-16-loss_function_2.py)
- 本节课件下载：
🏵[损失函数（二）](https://github.com/JansonYuan/Pytorch-Camp/blob/master/课件合集/04-03-ppt-损失函数(二).pdf)

### 🛴【任务3】

**任务名称：**  
torch.optim.SGD

**任务简介：**  
学习最常用的优化器， optim.SGD

**详细说明：**
深入了解学习率和momentum在梯度下降法中的作用，分析LR和Momentum这两个参数对优化过程的影响，最后学习optim.SGD以及pytorch的十种优化器简介。

**作业名称（详解）：** 
1. 优化器的作用是管理并更新参数组，请构建一个SGD优化器，通过add_param_group方法添加三组参数，三组参数的学习率分别为 0.01， 0.02， 0.03， momentum分别为0.9, 0.8, 0.7，构建好之后，并打印优化器中的param_groups属性中的每一个元素的key和value（提示：param_groups是list，其每一个元素是一个字典）

- 本节代码下载：
🍏[优化器（一）](https://github.com/JansonYuan/Pytorch-Camp/tree/master/代码合集/04-04-代码-优化器%EF%BC%88一%EF%BC%89/lesson-17)
🍎[优化器（二）](https://github.com/JansonYuan/Pytorch-Camp/tree/master/代码合集/04-05-代码-优化器%EF%BC%88二%EF%BC%89/lesson-18)
- 本节课件下载：
🍏[优化器（一）](https://github.com/JansonYuan/Pytorch-Camp/blob/master/课件合集/04-04-ppt-优化器%EF%BC%88一%EF%BC%89.pdf)
🍎[优化器（二）](https://github.com/JansonYuan/Pytorch-Camp/blob/master/课件合集/04-05-ppt-优化器%EF%BC%88二%EF%BC%89.pdf)
