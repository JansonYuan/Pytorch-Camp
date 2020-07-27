# PyTorch框架班 

## 🎯Week 2
- 代码下载: ☕[autograd](https://github.com/JansonYuan/Pytorch-Camp/tree/master/hello%20pytorch)

### 🛴【任务1】

**任务名称：**  
PyTorch数据读取机制Dataloader与Dataset；数据预处理transforms模块机制

**任务简介：**  
学习PyTorch数据读取机制中的两个重要模块Dataloader与Dataset；熟悉数据预处理transforms方法的运行机制

**详细说明：**  
本节第一部分介绍pytorch的数据读取机制，通过一个人民币分类实验来学习pytorch是如何从硬盘中读取数据的，并且深入学习数据读取过程中涉及到的两个模块Dataloader与Dataset    
第二部分介绍数据的预处理模块transforms的运行机制，数据在读取到pytorch之后通常都需要对数据进行预处理，包括尺寸缩放、转换张量、数据中心化或标准化等等，这些操作都是通过transforms进行的，所以本节重点学习transforms的运行机制  
并介绍数据标准化(Normalize)的使用原理

**作业名称（详解）：**  
1. 采用步进(Step into)的调试方法从 for i, data in enumerate(train_loader) 这一行代码开始，进入到每一个被调用函数，直到进入RMBDataset类中的__getitem__函数，记录从 for循环到RMBDataset的__getitem__所设计的类与函数？  
例如：  
第一步：for i, data in enumerate(train_loader)  
第二步：DataLoader类，__iter__函数  
第三步：***类， ***函数  
第n步：RMBDataset类，__getitem__函数  

2. 训练RMB二分类模型，熟悉数据读取机制，并且从kaggle中下载猫狗二分类训练数据，自己编写一个DogCatDataset，使得pytorch可以对猫狗二分类训练集进行读取。数据下载：https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

### 🛴【任务2】

**任务名称：**  
学习二十二种transforms数据预处理方法；学会自定义transforms方法

**任务简介：**  
pytorch提供了大量的transforms预处理方法，在这里归纳总结为四大类共二十二种方法进行一一学习；学会自定义transforms方法以兼容实际项目

**详细说明：**  
本节将介绍张量的基本操作，如张量拼接切分、索引和变换，同时学习张量的数学运算，并基于所学习的知识，实现线性回归模型的训练，以加深知识点的认识。  
本节第二部分介绍pytorch最大的特性——动态图机制，动态图机制是pytorch与tensorflow最大的区别，该部分首先介绍计算图的概念，并通过演示动态图与静态图的搭建过程来理解动态图与静态图的差异。

**作业名称（详解）：**  
1. 将介绍的transforms方法一一地，单独地实现对图片的变换，并且通过plt.savefig将图片保存下来（不少于10张不一样的数据增强变换的图片，如裁剪，缩放，平移，翻转，色彩变换，错切，遮挡等等）

2. 自定义一个增加椒盐噪声的transforms方法，使得其能正确运行（复制YourTransforms类的代码）



