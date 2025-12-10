import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score
from utils import load_heart_dataset, calculate_core_hit_rate

def train_model(X, y, core_features_idx, model_type):
    """
    训练模型并计算关键指标：准确率、选中特征数、训练时间、核心特征命中率
    参数：
        X: 特征矩阵（原始/扩展）
        y: 标签
        core_features_idx: 核心特征索引
        model_type: 模型类型（lasso/ridge/stepwise）
    返回：
        acc: 测试集准确率（百分比，保留1位小数）
        selected_num: 选中特征数
        train_time: 训练时间（秒，保留1位小数）
        core_hit_rate: 核心特征命中率（百分比，保留1位小数）
    """
    # 划分训练集/测试集（7:3，固定随机种子保证复现）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    if model_type == "lasso":
        # Lasso模型 + 5折交叉验证优化超参数alpha
        lasso = Lasso(max_iter=10000, random_state=42)
        param_grid = {"alpha": np.logspace(-4, 2, 100)}  # alpha搜索范围
        grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring="neg_mean_squared_error")
        
        # 记录训练时间
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        train_time = round(time.time() - start_time, 1)
        
        # 最优模型预测
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred = np.where(y_pred >= 0.5, 1, 0)  # 二分类阈值
        
        # 选中特征数（系数非零）
        selected_features = np.nonzero(best_model.coef_)[0]
        selected_num = len(selected_features)
        
    elif model_type == "ridge":
        # 岭回归模型 + 5折交叉验证优化超参数alpha
        ridge = Ridge(max_iter=10000, random_state=42)
        param_grid = {"alpha": np.logspace(-4, 2, 100)}
        grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring="neg_mean_squared_error")
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        train_time = round(time.time() - start_time, 1)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        
        # 岭回归无特征选择，选中特征数=总特征数
        selected_features = np.arange(X.shape[1])
        selected_num = len(selected_features)
        
    elif model_type == "stepwise":
        # 逐步回归（前向选择，基于逻辑回归）
        log_reg = LogisticRegression(max_iter=10000, random_state=42)
        sfs = SequentialFeatureSelector(
            log_reg, direction="forward", n_features_to_select="auto",
            tol=0.001, cv=5, scoring="accuracy"
        )
        
        start_time = time.time()
        sfs.fit(X_train, y_train)
        train_time = round(time.time() - start_time, 1)
        
        # 用选中的特征训练最终模型
        X_train_sfs = sfs.transform(X_train)
        X_test_sfs = sfs.transform(X_test)
        log_reg.fit(X_train_sfs, y_train)
        y_pred = log_reg.predict(X_test_sfs)
        
        # 选中特征数
        selected_features = np.where(sfs.get_support())[0]
        selected_num = len(selected_features)
    
    # 计算核心指标
    acc = round(accuracy_score(y_test, y_pred) * 100, 1)
    core_hit_rate = calculate_core_hit_rate(selected_features, core_features_idx)
    
    return acc, selected_num, train_time, core_hit_rate

if __name__ == "__main__":
    # 1. 加载数据
    print("===== 加载数据集 =====")
    X_original, X_extended, y, core_features_idx = load_heart_dataset()
    print(f"原始特征维度：{X_original.shape[1]}，扩展特征维度（含冗余）：{X_extended.shape[1]}")
    print(f"样本数：{X_original.shape[0]}，核心特征数：{len(core_features_idx)}")
    
    # 2. 训练模型（有Lasso/无Lasso场景）
    print("\n===== 训练模型 =====")
    # 有Lasso场景（Lasso模型）
    lasso_acc, lasso_selected, lasso_time, lasso_hit = train_model(
        X_extended, y, core_features_idx, "lasso"
    )
    # 无Lasso场景（逐步回归+岭回归）
    stepwise_acc, stepwise_selected, stepwise_time, stepwise_hit = train_model(
        X_extended, y, core_features_idx, "stepwise"
    )
    ridge_acc, ridge_selected, ridge_time, ridge_hit = train_model(
        X_extended, y, core_features_idx, "ridge"
    )
    
    # 3. 整理结果
    results = pd.DataFrame({
        "场景": ["有Lasso", "无Lasso", "无Lasso"],
        "建模方法": ["Lasso", "逐步回归", "岭回归"],
        "测试集准确率(%)": [lasso_acc, stepwise_acc, ridge_acc],
        "选中特征数": [lasso_selected, stepwise_selected, ridge_selected],
        "训练时间(秒)": [lasso_time, stepwise_time, ridge_time],
        "核心特征命中率(%)": [lasso_hit, stepwise_hit, ridge_hit]
    })
    
    # 4. 输出结果
    print("\n===== 实验结果汇总 =====")
    print(results)
    
    # 5. 保存结果到CSV
    results.to_csv("experiment_results.csv", index=False, encoding="utf-8")
    print("\n实验结果已保存至 experiment_results.csv")
