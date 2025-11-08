#!/usr/bin/env python3
"""
Baseline Model Evaluation (Phase 2) - 修正版

- 加载由 2_build_features_advanced.py 生成的特征。
- 严格遵守项目规范：只在 target==0 (正常) 的训练数据上训练。
- 在验证集上评估基线模型的 AUPRC 和 AUROC。
- 修正：根据 README.md，基线模型应在 "no PCA" (即 X_scaled) 数据上运行。
"""

import numpy as np
import pandas as pd
import os
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope # 新增
from sklearn.metrics import average_precision_score, roc_auc_score

# --- 1. 定义路径 ---
DATA_PATH = "./data/feature_advanced/"
OUTPUT_PATH = "./baseline_models/results/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_data(path: str) -> tuple:
    """
    加载由 2_build_features_advanced.py 生成的 .npy 文件。
    """
    print(f"从 '{path}' 加载特征文件...")
    
    try:
        # X_train_embed = np.load(os.path.join(path, "train_embed.npy")) # 基线不需要embed
        # X_valid_embed = np.load(os.path.join(path, "valid_embed.npy"))
        
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
    print(f" - 验证集异常率: {y_valid.mean()*100:.2f}%")
    
    # 基线脚本只需要 scaled 数据
    return X_train_scaled_normal, X_valid_scaled, y_valid

def main():
    """
    主执行函数：加载数据，训练并评估基线模型。
    """
    # --- 2. 加载数据 ---
    X_train_fit, X_valid_eval, y_valid = load_data(DATA_PATH)
    if X_train_fit is None:
        return
# --- 3. 定义基线模型 ---
    models = {
        "Local Outlier Factor (k=50)": {
            # 关键修正：添加 n_jobs=1 来禁用多进程
            "model": LocalOutlierFactor(n_neighbors=15, metric='manhattan', novelty=True, n_jobs=1),
            "data": "scaled",
            "score_type": "decision_function"
        }
        # "Isolation Forest": {
        #     "model": IsolationForest(n_estimators=200, random_state=42),
        #     "data": "scaled",
        #     "score_type": "decision_function"
        # },
        # "One-Class SVM (nu=0.05)": {
        #     "model": OneClassSVM(kernel="rbf", gamma="scale", nu=0.05),
        #     "data": "scaled",
        #     "score_type": "decision_function"
        # },
        # "PCA Reconstruction": {
        #     "model": PCA(n_components=30, random_state=42), #
        #     "data": "scaled",
        #     "score_type": "reconstruction"
        # },
        # "Elliptic Envelope (Mahalanobis)": { # 新增
        #     #  contamination 设为已知的验证集异常率
        #     "model": EllipticEnvelope(contamination=0.1261, random_state=42),
        #     "data": "scaled",
        #     "score_type": "decision_function"
        # },
        # "KNN Distance (k=50)": { # 新增
        #     "model": NearestNeighbors(n_neighbors=50, n_jobs=1), # 匹配 LOF k
        #     "data": "scaled",
        #     "score_type": "kneighbors",
        #     "k": 50 # 用于在评估循环中获取距离
        # }
    }
    results = []

    print("\n--- 开始评估基线模型 (已修正) ---")

    # --- 4. 训练和评估循环 ---
    for name, config in models.items():
        print(f"\n正在运行: {name}...")
        model = config["model"]
        
        try:
            # 步骤 4a: 只在正常数据上训练 (Fit)
            model.fit(X_train_fit)

            # 步骤 4b: 在验证集上评估 (Score)
            if config["score_type"] == "decision_function":
                # LOF, IF, OCSVM: 分数越低越异常, 所以取负
                scores = -model.decision_function(X_valid_eval)
            
            elif config["score_type"] == "reconstruction":
                # PCA: 重构误差越大越异常
                X_valid_rec = model.inverse_transform(model.transform(X_valid_eval))
                scores = np.mean((X_valid_eval - X_valid_rec) ** 2, axis=1)

            # 步骤 4c: 计算指标
            auprc = average_precision_score(y_valid, scores)
            auroc = roc_auc_score(y_valid, scores)
            
            print(f"  -> AUPRC: {auprc:.4f}, AUROC: {auroc:.4f}")
            results.append({
                "Model": name,
                "AUPRC": auprc,
                "AUROC": auroc
            })

        except Exception as e:
            print(f"  -> 失败: {e}")
            results.append({"Model": name, "AUPRC": np.nan, "AUROC": np.nan}) # 设为 NaN

    # --- 5. 总结和保存结果 ---
    print("\n--- 基线模型评估完成 ---")
    results_df = pd.DataFrame(results).sort_values(by="AUPRC", ascending=False)
    
    print("\n最终基线性能 (按AUPRC排序):")
    print(results_df.to_string(index=False))
    
    # 保存结果到 CSV
    output_filename = os.path.join(OUTPUT_PATH, "baseline_results.csv")
    results_df.to_csv(output_filename, index=False)
    print(f"\n结果已保存到: {output_filename}")


if __name__ == "__main__":
    main()