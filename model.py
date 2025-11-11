#!/usr/bin/env python3
"""
Simple Baseline Model (LOF k=50) for Kaggle Submission

- 加载由 2_build_features_advanced.py 生成的 'scaled' 特征。
- 使用 baseline.py 中表现最好的基线模型 (LOF k=50, n_jobs=1)。
- 仅在正常 (target==0) 训练数据上训练。
- 在测试集上预测分数。
- 将分数缩放到 [0, 1] 范围并保存为 submission.csv。
"""

import numpy as np
import pandas as pd
import os
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler # 用于将分数缩放到 [0, 1]

# --- 1. 定义路径 ---
DATA_PATH = "./data/feature_advanced/"
OUTPUT_FILE = "./data/submission/submission_simple_model_lof.csv"

def load_data(path: str) -> tuple:
    """
    加载 .npy 文件 (scaled 版)
    """
    print(f"从 '{path}' 加载特征文件...")
    
    try:
        # 加载用于 LOF 的 scaled (非PCA) 特征
        X_train_scaled = np.load(os.path.join(path, "train_scaled.npy"))
        X_test_scaled = np.load(os.path.join(path, "test_scaled.npy"))
        
        # 加载标签和ID
        y_train_full_labels = np.load(os.path.join(path, "train_labels.npy"))
        test_ids = np.load(os.path.join(path, "test_ids.npy"))
        
    except FileNotFoundError as e:
        print(f"错误：找不到文件 {e.filename}。")
        print(f"请确保你已经运行了 '2_build_features_advanced.py'。")
        return None, None, None

    # 关键步骤：只选择 target==0 的训练样本用于拟合 (fit)
    normal_mask = (y_train_full_labels == 0)
    X_train_normal = X_train_scaled[normal_mask]

    print(f" - 训练集 (正常) shape: {X_train_normal.shape}")
    print(f" - 测试集 shape: {X_test_scaled.shape}")
    print(f" - 测试集 IDs shape: {test_ids.shape}")
    
    return X_train_normal, X_test_scaled, test_ids

def main():
    """
    主执行函数：加载、训练、预测、保存。
    """
    print("--- 启动简单基线模型 (LOF k=50) 预测 ---")
    
    # --- 2. 加载数据 ---
    X_train, X_test, test_ids = load_data(DATA_PATH)
    if X_train is None:
        return

    # --- 3. 训练 LOF 模型 ---
    # 使用 baseline.py 中相同的参数 (k=50, n_jobs=1)
    print(f"正在训练 LOF (k=50, n_jobs=1) 模型...")
    lof = LocalOutlierFactor(n_neighbors=15, metric='manhattan', novelty=True, n_jobs=1)
    
    # 只在正常数据上训练
    lof.fit(X_train)

    # --- 4. 预测测试集分数 ---
    print("正在预测测试集异常分数...")
    # LOF 的分数越低越异常, 所以取负使其越高越异常
    raw_scores = -lof.decision_function(X_test)

    # --- 5. 缩放分数到 [0, 1] (Kaggle 要求) ---
    print("正在将分数缩放到 [0, 1] 范围...")
    scaler = MinMaxScaler()
    # .reshape(-1, 1) 是 scaler 的标准输入要求
    scaled_scores = scaler.fit_transform(raw_scores.reshape(-1, 1))

    # --- 6. 创建并保存提交文件 ---
    print(f"正在创建提交通知 '{OUTPUT_FILE}'...")
    
    # 使用 'Id' 和 'target' 列名
    submission_df = pd.DataFrame({
        'Id': test_ids,
        'target': scaled_scores.flatten() # .flatten() 将其转回一维数组
    })
    
    submission_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n--- 预测完成 ---")
    print(f"已成功保存提交文件到: {OUTPUT_FILE}")
    print("文件预览:")
    print(submission_df.head())

if __name__ == "__main__":
    main()