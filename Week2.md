# PyTorch框架班 

## Week 2

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
第三步: ***类， ***函数  
第n步：RMBDataset类，__getitem__函数  

2. 训练RMB二分类模型，熟悉数据读取机制，并且从kaggle中下载猫狗二分类训练数据，自己编写一个DogCatDataset，使得pytorch可以对猫狗二分类训练集进行读取。数据下载：https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
- 本节代码下载：
- 本节课件下载:


