## 文件结构：
```
  ├── model.py: ResNet模型搭建
  ├── train.py: 训练脚本
  ├── predict.py: 单张图像预测脚本
  └── batch_predict.py: 批量图像预测脚本
  
  └──PREresnet34是在源域上训练好的参数 准确率达到0.9
  
  
  model.py 在原来的resnet34的基础上加上了class FeatureWiseTransformation2d_fw(nn.BatchNorm2d):类
  并在初始化basicblock的时候加上film层
            self.film=FeatureWiseTransformation2d_fw(out_channel)
  
  train.py的63-80行是固定权重，优化FILM的实现
  
  将源域的300张图像放入GID_dataset中命名目标域再train 准确率仅有0.2
```