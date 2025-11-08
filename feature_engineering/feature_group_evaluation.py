#!/usr/bin/env python3
"""
Feature Group Evaluation Script
--------------------------------
用来判断不同特征组是否有实际贡献。

功能：
- 自动加载 2_build_features_advanced.py 导出的特征
- 按特征组（static / cross / temporal / amort / LTV_Change）分批评估
- 计算每组特征的增量 AUPRC / AUROC
- 输出表格，帮助判断哪些特征可以保留
"""

import numpy as np
import pandas as pd
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

# === 配置 ===
DATA_PATH = "./data/feature_advanced/"
OUTPUT_PATH = "./feature_tests/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# === 加载数据 ===
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


# === 评估函数 ===
def evaluate_subset(X_train, X_valid, y_valid, feature_idx, group_name):
    """
    使用 IsolationForest 评估某个特征子集。
    """
    X_train_sub = X_train[:, feature_idx]
    X_valid_sub = X_valid[:, feature_idx]

    # 再次缩放子集，避免比例失衡
    scaler = RobustScaler().fit(X_train_sub)
    X_train_sub = scaler.transform(X_train_sub)
    X_valid_sub = scaler.transform(X_valid_sub)

    model = IsolationForest(n_estimators=200, random_state=42)
    model.fit(X_train_sub)
    scores = -model.decision_function(X_valid_sub)
    ap = average_precision_score(y_valid, scores)
    roc = roc_auc_score(y_valid, scores)
    print(f"  {group_name:<20} -> AUPRC={ap:.4f}, AUROC={roc:.4f}")
    return ap, roc


# === 主逻辑 ===
def main():
    X_train, X_valid, y_valid = load_data(DATA_PATH)

    # 从保存的列名文件中读取 feature 列顺序（由 feature_generator.py 保存）
    cols_file = os.path.join(DATA_PATH, "feature_names.txt")
    if not os.path.exists(cols_file):
        print("⚠️ 未找到 feature_names.txt，使用索引代替。")
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]
    else:
        feature_names = [x.strip() for x in open(cols_file).readlines()]
    n_features = len(feature_names)
    print(f"共 {n_features} 个特征")

    # === 自动按关键字分组 ===
    groups = {
        "Static": [i for i, f in enumerate(feature_names) if not any(x in f for x in ["_trend","_vol","_dmean","_dstd","amort","LTV_Change"])],
        "Temporal": [i for i, f in enumerate(feature_names) if any(x in f for x in ["_trend","_vol","_dmean","_dstd"])],
        "Amortization": [i for i, f in enumerate(feature_names) if "amort" in f or "io_payment" in f],
        "LTV_Change": [i for i, f in enumerate(feature_names) if "LTV_Change" in f],
        "Cross_Static": [i for i, f in enumerate(feature_names) if f in ["LTV_x_DTI","UPB_per_CreditScore","InterestRate_x_LTV"]],
        "All_Features": list(range(n_features))
    }

    # === 逐组评估 ===
    results = []
    print("\n--- 特征组性能评估开始 ---")
    for name, idx in groups.items():
        if len(idx) == 0:
            print(f"  {name:<20} -> ❌ 无特征，跳过")
            continue
        ap, roc = evaluate_subset(X_train, X_valid, y_valid, idx, name)
        results.append({"Group": name, "Num_Features": len(idx), "AUPRC": ap, "AUROC": roc})

    # === 输出结果 ===
    df = pd.DataFrame(results).sort_values(by="AUPRC", ascending=False)
    print("\n--- 特征组贡献汇总 ---")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUTPUT_PATH, "feature_group_results.csv"), index=False)
    print(f"\n✅ 已保存结果到: {OUTPUT_PATH}/feature_group_results.csv")


if __name__ == "__main__":
    main()
