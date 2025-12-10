import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

def load_heart_dataset():
    """
    加载UCI心脏病数据集（替代Kaggle版本，保证可访问性）
    返回：
        X_original: 原始13维特征（标准化后）
        X_extended: 13维原始特征 + 20维冗余特征（模拟高维场景）
        y: 二值化标签（1=患病，0=健康）
        core_features_idx: 领域验证的8个核心特征索引
    """
    # 加载UCI Heart数据集（和Kaggle版本特征一致）
    heart = fetch_openml(name='heart', version=1, parser='auto', cache=False)
    X = heart.data
    y = heart.target
    
    # 标签二值化（统一为0/1）
    y = np.where(y == '1', 1, 0).astype(int)
    
    # 标准化特征（正则化模型必须标准化）
    scaler = StandardScaler()
    X_original = scaler.fit_transform(X)
    
    # 生成20维冗余特征（原始特征 + 高斯噪声，固定随机种子保证复现）
    np.random.seed(42)
    noise = np.random.normal(loc=0, scale=0.01, size=(X_original.shape[0], 20))
    X_extended = np.hstack([X_original, noise])
    
    # 领域知识验证的8个核心特征索引（对应原始13维特征）
    core_features_idx = [0, 3, 4, 7, 9, 10, 11, 12]
    
    return X_original, X_extended, y, core_features_idx

def calculate_core_hit_rate(selected_features, core_features_idx):
    """
    计算核心特征命中率：选中特征中属于核心特征的比例
    参数：
        selected_features: 模型选中的特征索引列表
        core_features_idx: 核心特征索引列表
    返回：
        core_hit_rate: 核心特征命中率（百分比，保留1位小数）
    """
    core_selected = [idx for idx in selected_features if idx in core_features_idx]
    if len(selected_features) == 0:
        return 0.0
    core_hit_rate = (len(core_selected) / len(selected_features)) * 100
    return round(core_hit_rate, 1)
