#!/usr/bin/env python3
"""
Single Feature Effectiveness Analysis
-------------------------------------
评估每个单独特征在异常检测任务中的有效性。

功能：
- 自动加载 2_build_features_advanced.py 导出的特征文件 (.npy)
- 每次仅用一个特征训练 IsolationForest，并在验证集上评估 AUPRC/AUROC
- 输出排序表，显示每个特征的异常检测能力

运行：
    python single_feature_analysis.py
"""

import numpy as np
import pandas as pd
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

# === 配置 ===
DATA_PATH = "./data/feature_advanced/"
OUTPUT_PATH = "./feature_tests/"
os.makedirs(OUTPUT_PATH, exist_ok=True)


def load_data(path):
    print(f"加载特征自: {path}")
    X_train = np.load(os.path.join(path, "train_scaled.npy"))
    X_valid = np.load(os.path.join(path, "valid_scaled.npy"))
    y_train = np.load(os.path.join(path, "train_labels.npy"))
    y_valid = np.load(os.path.join(path, "valid_labels.npy"))
    normal_mask = (y_train == 0)
    X_train_normal = X_train[normal_mask]
    print(f"Train(normal)={X_train_normal.shape}, Valid={X_valid.shape}")
    return X_train_normal, X_valid, y_valid


def evaluate_single_feature(X_train, X_valid, y_valid, idx, name):
    """
    评估单一特征的 AUPRC/AUROC。
    """
    X_train_col = X_train[:, [idx]]
    X_valid_col = X_valid[:, [idx]]

    # 再缩放
    scaler = RobustScaler().fit(X_train_col)
    X_train_scaled = scaler.transform(X_train_col)
    X_valid_scaled = scaler.transform(X_valid_col)

    model = IsolationForest(n_estimators=200, random_state=42)
    model.fit(X_train_scaled)

    scores = -model.decision_function(X_valid_scaled)
    ap = average_precision_score(y_valid, scores)
    roc = roc_auc_score(y_valid, scores)
    return ap, roc


def main():
    X_train, X_valid, y_valid = load_data(DATA_PATH)

    # 从保存的列名文件中读取 feature 列顺序
    cols_file = os.path.join(DATA_PATH, "feature_names.txt")
    if not os.path.exists(cols_file):
        print("⚠️ 未找到 feature_names.txt，使用索引代替。")
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]
    else:
        feature_names = [x.strip() for x in open(cols_file).readlines()]

    n_features = len(feature_names)
    print(f"共 {n_features} 个特征，将逐列测试...")

    results = []
    for i, name in enumerate(tqdm(feature_names, desc="特征评估中")):
        try:
            ap, roc = evaluate_single_feature(X_train, X_valid, y_valid, i, name)
            results.append({"Feature": name, "Index": i, "AUPRC": ap, "AUROC": roc})
        except Exception as e:
            results.append({"Feature": name, "Index": i, "AUPRC": np.nan, "AUROC": np.nan})
            print(f"⚠️ 特征 {name} 评估失败: {e}")

    df = pd.DataFrame(results).sort_values(by="AUPRC", ascending=False)
    print("\n--- 单特征评估完成 ---")
    print(df.head(20).to_string(index=False))

    output_file = os.path.join(OUTPUT_PATH, "single_feature_results.csv")
    df.to_csv(output_file, index=False)
    print(f"\n✅ 已保存完整结果到: {output_file}")


if __name__ == "__main__":
    main()
