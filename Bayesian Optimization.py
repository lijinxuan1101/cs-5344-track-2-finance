#!/usr/bin/env python3
"""
高级参数搜索 (Optuna / 贝叶斯优化)

- 使用 Optuna (一个贝叶斯优化库) 来智能地搜索 LocalOutlierFactor 的最佳参数。
- 这是比网格搜索 (Grid Search) 或随机搜索 (Random Search) 更高效的方法。
- 目标：最大化 AUPRC 分数。
- 修正：始终为 LOF 设置 n_jobs=1 以避免多进程崩溃。
"""

import numpy as np
import pandas as pd
import os
import optuna  # 导入 Optuna
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score, roc_auc_score
import time
import warnings

# 抑制 Optuna 的试验日志 (如果需要)
# optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- 1. 定义路径 ---
DATA_PATH = "./data/feature_advanced/"
OUTPUT_PATH = "./baseline_models/results/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

#
# ==========================================================
#  函数定义 (必须在 main() 之前)
# ==========================================================
#

def load_data(path: str) -> tuple:
    """
    加载由 2_build_features_advanced.py 生成的 .npy 文件。
    """
    print(f"从 '{path}' 加载特征文件...")
    
    try:
        X_train_scaled = np.load(os.path.join(path, "train_scaled.npy"))
        X_valid_scaled = np.load(os.path.join(path, "valid_scaled.npy"))
        
        y_train_full_labels = np.load(os.path.join(path, "train_labels.npy"))
        y_valid = np.load(os.path.join(path, "valid_labels.npy"))

    except FileNotFoundError as e:
        print(f"错误：找不到文件 {e.filename}。")
        print(f"请先运行 '2_build_features_advanced.py'。")
        return None, None, None

    # 关键步骤：只选择 target==0 的训练样本用于拟合 (fit)
    normal_mask = (y_train_full_labels == 0)
    X_train_scaled_normal = X_train_scaled[normal_mask]

    print(f" - 训练集 (正常) shape (Scaled): {X_train_scaled_normal.shape}")
    print(f" - 验证集 shape (Scaled): {X_valid_scaled.shape}")
    print(f" - 验证集标签 shape: {y_valid.shape}")
    
    return X_train_scaled_normal, X_valid_scaled, y_valid

def objective(trial, X_train_fit, X_valid_eval, y_valid):
    """
    Optuna 的目标函数，它会尝试最大化 AUPRC。
    """
    # 1. 定义要搜索的参数
    # 在 5 到 100 之间智能搜索 k (n_neighbors)
    k = trial.suggest_int('n_neighbors', 5, 100) 
    
    # 在这个列表中智能选择 metric
    metric = trial.suggest_categorical('metric', ['minkowski', 'cosine', 'manhattan', 'chebyshev'])
    
    params = {'n_neighbors': k, 'metric': metric}
    print(f"  [Trial {trial.number}] 正在尝试: {params}...")

    # 2. 训练模型
    # 必须设置 novelty=True 才能调用 .decision_function
    # 必须设置 n_jobs=1 来避免 Windows 上的崩溃
    model = LocalOutlierFactor(novelty=True, n_jobs=1, **params)
    
    try:
        # 步骤 3a: 只在正常数据上训练 (Fit)
        model.fit(X_train_fit)
        
        # 步骤 3b: 在验证集上评估 (Score)
        scores = -model.decision_function(X_valid_eval)

        # 步骤 3c: 计算指标
        auprc = average_precision_score(y_valid, scores)
        
        # 处理可能的NaN/Inf
        if np.isnan(auprc) or np.isinf(auprc):
            print(f"  -> Trial {trial.number} 结果: NaN (返回 0.0)")
            return 0.0 # 返回一个低分，让 Optuna 知道这是不好的参数

        print(f"  -> Trial {trial.number} 结果: AUPRC = {auprc:.4f}")
        return auprc # 4. 返回要最大化的值 (AUPRC)

    except Exception as e:
        print(f"  -> Trial {trial.number} 失败: {e}")
        return 0.0 # 失败的尝试返回0分

def main():
    """
    主执行函数：加载数据，运行 Optuna 优化。
    """
    # --- 2. 加载数据 ---
    # !! 错误发生在这里，因为 load_data 在此之前没有被定义 !!
    X_train_fit, X_valid_eval, y_valid = load_data(DATA_PATH)
    if X_train_fit is None:
        return

    print("\n--- 开始 LOF 贝叶斯优化 (Optuna) ---")
    start_time = time.time()

    # --- 4. 运行优化 ---
    # direction="maximize" 告诉 Optuna 我们想让 AUPRC 越高越好
    study = optuna.create_study(direction="maximize")
    
    # n_trials=50 意味着 Optuna 会智能地尝试 50 种组合
    study.optimize(
        lambda trial: objective(trial, X_train_fit, X_valid_eval, y_valid), 
        n_trials=50
    )

    # --- 5. 总结结果 ---
    total_time = time.time() - start_time
    print(f"\n--- 优化完成 (耗时: {total_time:.1f} 秒) ---")
    
    print("\n" + "="*30)
    print(" 最佳参数组合 ")
    print("="*30)
    print(study.best_params)
    print(f"最佳 AUPRC: {study.best_value:.4f}")

    # 保存结果
    results_df = study.trials_dataframe().sort_values(by="value", ascending=False)
    output_filename = os.path.join(OUTPUT_PATH, "lof_optuna_search_results.csv")
    results_df.to_csv(output_filename, index=False)
    print(f"\n完整搜索结果已保存到: {output_filename}")


if __name__ == "__main__":
    main()