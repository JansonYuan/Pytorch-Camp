
为保证代码少报错，本项目严格按照以下目录树进行组织
若工具包无法导入，请检查文件目录结构，或自行学习python的包导入相关知识点

hello pytorch
│  readme.txt
├─data
│      02-01-数据-RMB_data.rar
│      07-02-数据-模型finetune.zip
│      08-01-数据-20200724.zip
│      08-02-数据-PortraitDataset-20200724.zip
│      08-03-数据-20200724.zip
│      08-04-数据-20200724.zip
│      08-05-数据-20200724.zip
│      
├─lesson
│  ├─lesson-01
│  │      hello pytorch.py
│  │      
│  ├─lesson-02
│  │      lesson-02.py
│  │      
│  ├─lesson-03
│  │      lesson-03-Linear-Regression.py
│  │      lesson-03.py
│  │      
│  ├─lesson-04
│  │      lesson-04-Computational-Graph.py
│  │      
│  ├─lesson-05
│  │      lesson-05-autograd.py
│  │      lesson-05-Logistic-Regression.py
│  │      
│  ├─lesson-06
│  │  │  1_split_dataset.py
│  │  │  2_train_lenet.py
│  │  │  
│  │  └─test_data
│  │      └─100
│  │              100.jpg
│  │              
│  ├─lesson-07
│  │      Logistic-Regression-norm.py
│  │      
│  ├─lesson-08
│  │      transforms_methods_1.py
│  │      
│  ├─lesson-09
│  │  │  my_transforms.py
│  │  │  RMB_data_augmentation.py
│  │  │  transforms_methods_2.py
│  │  │  
│  │  └─test_data
│  │      └─100
│  │              100.jpg
│  │              1001.jpg
│  │              1002.jpg
│  │              1003.jpg
│  │              
│  ├─lesson-10
│  │      create_module.py
│  │      
│  ├─lesson-11
│  │      module_containers.py
│  │      
│  ├─lesson-12
│  │      lena.png
│  │      nn_layers_convolution.py
│  │      
│  ├─lesson-13
│  │      lena.png
│  │      nn_layers_others.py
│  │      
│  ├─lesson-14
│  │      grad_vanish_explod.py
│  │      
│  ├─lesson-15
│  │      ce_loss.py
│  │      loss_function_1.py
│  │      
│  ├─lesson-16
│  │      loss_function_2.py
│  │      
│  ├─lesson-17
│  │      create_optimizer.py
│  │      optimizer_methods.py
│  │      
│  ├─lesson-18
│  │      learning_rate.py
│  │      momentum.py
│  │      
│  ├─lesson-19
│  │      create_scheduler.py
│  │      lr_decay_scheduler.py
│  │      
│  ├─lesson-20
│  │      test_tensorboard.py
│  │      
│  ├─lesson-21
│  │      loss_acc_weights_grad.py
│  │      tensorboard_methods.py
│  │      
│  ├─lesson-22
│  │      lena.png
│  │      tensorboard_methods_2.py
│  │      weight_fmap_visualization.py
│  │      
│  ├─lesson-23
│  │      hook_fmap_vis.py
│  │      hook_methods.py
│  │      lena.png
│  │      
│  ├─lesson-24
│  │      L2_regularization.py
│  │      
│  ├─lesson-25
│  │      dropout_layer.py
│  │      dropout_regularization.py
│  │      
│  ├─lesson-26
│  │      bn_and_initialize.py
│  │      bn_application.py
│  │      bn_in_123_dim.py
│  │      
│  ├─lesson-27
│  │      normallization_layers.py
│  │      
│  ├─lesson-28
│  │      checkpoint_resume.py
│  │      model_load.py
│  │      model_save.py
│  │      readme.txt
│  │      save_checkpoint.py
│  │      
│  ├─lesson-29
│  │      finetune_resnet18.py
│  │      
│  ├─lesson-30
│  │      1_cuda_use.py
│  │      2_cuda_methods.py
│  │      3_multi_gpu.py
│  │      4_model_load_in_gpu.py
│  │      model_in_gpu_0.pkl
│  │      model_in_multi_gpu.pkl
│  │      
│  ├─lesson-31
│  │      common_errors.py
│  │      foo_net.pkl
│  │      model_in_multi_gpu.pkl
│  │      
│  ├─lesson-32
│  │      resnet_inference.py
│  │      
│  ├─lesson-33
│  │      1_seg_demo.py
│  │      2_unet_portrait_matting.py
│  │      3_portrait_inference.py
│  │      
│  ├─lesson-34
│  │      detection_demo.py
│  │      fasterrcnn_demo.py
│  │      
│  ├─lesson-35
│  │      gan_demo.py
│  │      gan_inference.py
│  │      
│  └─lesson-36
│          rnn_demo.py
│          
├─model
│      lenet.py
│      
└─tools
        common_tools.py
        dcgan.py
        my_dataset.py
        unet.py
        
