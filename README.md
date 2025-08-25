#  Deep Learning Projects Collection - 深度学习项目合集

> 一个全面的深度学习学习项目，涵盖从传统机器学习到现代深度神经网络的完整学习路径

## 项目概述

本项目是我在深度学习学习过程中的完整笔记和实践代码集合。从基础的机器学习算法到先进的深度学习架构，通过实际项目来理解和掌握各种模型和技术。


##  项目结构

###  传统机器学习
- **Titanic Survival Prediction.ipynb** - 泰坦尼克号生存预测
  - 数据清洗和预处理
  - 特征工程
  - 分类模型训练和评估
  
- **Prediction Regression.ipynb** - 回归预测分析
  - 线性回归模型
  - 数据分割和模型评估
  - 预测结果分析

###  计算机视觉与深度学习

#### 基础分类任务
- **FashionMNIST_AlexNet.ipynb** - FashionMNIST数据集分类
  - AlexNet架构实现
  - 卷积神经网络基础
  - 图像分类任务

- **Cifat-10_NiN.ipynb** - CIFAR-10数据集分类
  - Network in Network (NiN) 架构
  - 全局平均池化
  - 多类别图像分类

#### 高级计算机视觉
- **DogBreedIdentification_ResNet34.ipynb** - 狗品种识别
  - ResNet34预训练模型
  - 迁移学习技术
  - 图像数据增强
  - 微调策略

###  生成式模型

- **DigitRecognizer_VAE.ipynb** - 数字识别变分自编码器
  - VAE架构实现
  - 编码器-解码器结构
  - 潜在空间学习
  - 生成模型基础

- **FashionMNIST_UNet.ipynb** - UNet图像生成
  - UNet网络架构
  - 下采样和上采样
  - 图像生成和重建
  - 扩散模型基础

##  技术栈

### 核心框架
- **PyTorch** - 深度学习框架
- **TorchVision** - 计算机视觉工具包
- **NumPy** - 数值计算
- **Pandas** - 数据处理

### 机器学习
- **Scikit-learn** - 传统机器学习算法
- **Matplotlib** - 数据可视化
- **Seaborn** - 统计图表

### 数据集
- **MNIST** - 手写数字识别
- **FashionMNIST** - 时尚物品分类
- **CIFAR-10** - 彩色图像分类
- **Titanic** - 生存预测
- **Dog Breed** - 狗品种识别

##  快速开始

### 环境要求
```bash
Python >= 3.7
PyTorch >= 1.9.0
torchvision >= 0.10.0
scikit-learn >= 1.0.0
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

### 安装依赖
```bash
pip install torch torchvision
pip install scikit-learn pandas numpy matplotlib seaborn
```

### 运行示例
```bash
# 运行泰坦尼克号生存预测
jupyter notebook "Titanic Survival Prediction.ipynb"

# 运行FashionMNIST分类
jupyter notebook "FashionMNIST_AlexNet.ipynb"
```

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

⭐ 如果这个项目对您有帮助，请给它一个星标！感谢您！ 
