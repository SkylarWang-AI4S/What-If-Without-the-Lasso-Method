import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import Lasso, Ridge
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ===================== 1. 数据加载与预处理 =====================
def load_and_preprocess_data():
    """
    加载心脏病数据集（UCI替代Kaggle版本，保证可访问性），并生成冗余特征
    返回：X_original(原始特征), X_extended(含冗余特征), y(标签), core_features_idx(领域验证的核心特征索引)
    """
    # 加载UCI心脏病数据集（和Kaggle版本特征一致）
    heart = fetch_openml(name='heart', version=1, parser='auto')
    X = heart.data
    y = heart.target
    y = np.where(y == '1', 1, 0)  # 二值化标签（1=患病，0=健康）
    
    # 标准化原始特征（正则化模型必须标准化）
    scaler = StandardScaler()
    X_original = scaler.fit_transform(X)
    
    # 生成20个冗余特征（原始特征+高斯噪声）
    np.random.seed(42)  # 固定随机种子保证复现
    noise = np.random.normal(0, 0.01, (X_original.shape[0], 20))
    X_extended = np.hstack([X_original, noise])
    
    # 领域知识验证的8个核心特征索引（对应原始13个特征中的关键指标）
    core_features_idx = [0, 3, 4, 7, 9, 10, 11, 12]
    
    return X_original, X_extended, y, core_features_idx

# ===================== 2. 模型训练与指标计算 =====================
def train_model_and_calculate_metrics(X, y, core_features_idx, model_name):
    """
    训练模型并计算关键指标：准确率、选中特征数、训练时间、核心特征命中率
    参数：
        X: 特征矩阵
        y: 标签
        core_features_idx: 核心特征索引
        model_name: 模型名称（lasso/ridge/stepwise）
    返回：accuracy, selected_num, train_time, core_hit_rate
    """
    # 划分训练集/测试集（7:3）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 初始化模型
    if model_name == 'lasso':
        # Lasso超参数λ用5折交叉验证优化
        lasso = Lasso(random_state=42, max_iter=10000)
        param_grid = {'alpha': np.logspace(-4, 2, 100)}
        grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='accuracy')
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        train_time = time.time() - start_time
        best_lasso = grid_search.best_estimator_
        y_pred = best_lasso.predict(X_test)
        y_pred = np.where(y_pred >= 0.5, 1, 0)  # 二分类阈值
        # 选中特征数（系数非零）
        selected_features = np.nonzero(best_lasso.coef_)[0]
        selected_num = len(selected_features)
        
    elif model_name == 'ridge':
        ridge = Ridge(random_state=42, max_iter=10000)
        param_grid = {'alpha': np.logspace(-4, 2, 100)}
        grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='accuracy')
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        train_time = time.time() - start_time
        best_ridge = grid_search.best_estimator_
        y_pred = best_ridge.predict(X_test)
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        # 岭回归无特征选择，选中特征数=总特征数
        selected_features = np.arange(X.shape[1])
        selected_num = len(selected_features)
        
    elif model_name == 'stepwise':
        # 逐步回归（基于逻辑回归，前向选择）
        log_reg = LogisticRegression(max_iter=10000, random_state=42)
        sfs = SequentialFeatureSelector(
            log_reg, direction='forward', n_features_to_select='auto', 
            tol=0.001, cv=5, scoring='accuracy'
        )
        start_time = time.time()
        sfs.fit(X_train, y_train)
        train_time = time.time() - start_time
        X_train_sfs = sfs.transform(X_train)
        X_test_sfs = sfs.transform(X_test)
        log_reg.fit(X_train_sfs, y_train)
        y_pred = log_reg.predict(X_test_sfs)
        # 选中特征数
        selected_features = np.where(sfs.get_support())[0]
        selected_num = len(selected_features)
        
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    # 核心特征命中率（选中特征中属于核心特征的比例）
    core_selected = [idx for idx in selected_features if idx in core_features_idx]
    core_hit_rate = len(core_selected) / len(selected_features) if selected_num > 0 else 0
    
    return round(accuracy*100, 1), selected_num, round(train_time, 1), round(core_hit_rate*100, 1)

# ===================== 3. 主实验流程 =====================
if __name__ == "__main__":
    # 加载数据
    X_original, X_extended, y, core_features_idx = load_and_preprocess_data()
    
    # 实验1：有Lasso场景（用扩展数据集）
    lasso_acc, lasso_selected, lasso_time, lasso_hit = train_model_and_calculate_metrics(
        X_extended, y, core_features_idx, 'lasso'
    )
    
    # 实验2：无Lasso场景（逐步回归+岭回归）
    stepwise_acc, stepwise_selected, stepwise_time, stepwise_hit = train_model_and_calculate_metrics(
        X_extended, y, core_features_idx, 'stepwise'
    )
    ridge_acc, ridge_selected, ridge_time, ridge_hit = train_model_and_calculate_metrics(
        X_extended, y, core_features_idx, 'ridge'
    )
    
    # 整理结果为表格
    results = pd.DataFrame({
        '场景': ['有Lasso', '无Lasso', '无Lasso'],
        '建模方法': ['Lasso', '逐步回归', '岭回归'],
        '测试集准确率(%)': [lasso_acc, stepwise_acc, ridge_acc],
        '选中特征数': [lasso_selected, stepwise_selected, ridge_selected],
        '训练时间(秒)': [lasso_time, stepwise_time, ridge_time],
        '核心特征命中率(%)': [lasso_hit, stepwise_hit, ridge_hit]
    })
    
    # 输出结果
    print("===== 实验结果汇总 =====")
    print(results)
    
    # 保存结果到CSV（便于论文图表复用）
    results.to_csv('experiment_results.csv', index=False, encoding='utf-8')
    print("\n结果已保存至 experiment_results.csv")
