# Lasso for High-Dimensional Data Modeling
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本仓库是论文《Lasso Method: Core Value and Domain Contributions in High-Dimensional Data Modeling》的实验复现代码，核心验证Lasso在高维数据建模中的性能优势，对比岭回归、逐步回归的表现。

## 实验说明
### 1. 实验目标
验证Lasso在高维数据（含冗余特征）中的特征选择能力和预测准确率；
模拟“无Lasso”场景，对比逐步回归、岭回归的性能损失；
计算核心指标：测试集准确率、选中特征数、训练时间、核心特征命中率。

### 2. 数据集
数据源：UCI Heart Disease Dataset（与Kaggle《Heart Attack Analysis and Prediction Dataset》特征一致，无需手动下载）；
特征：13维原始生理特征 + 20维冗余特征（原始特征+高斯噪声，模拟高维场景）；
任务：二分类（预测是否有心脏病发作风险）；
样本数：303。


# 安装依赖包
pip install -r requirements.txt
