#!/usr/bin/env python3
"""
Final Model - 融合 (Fusion) + Submission 生成脚本
--------------------------------------------------------
融合以下检测器的分数：
1️⃣ Early_Delinquency_Flag
2️⃣ amort_short_mean
3️⃣ Zero_Payment_Streak
4️⃣ LOF(k=50)

输出：
- 验证集性能评估（fusion_metrics.csv）
- 测试集提交文件（submission.csv）
"""

import os
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor

# =========================
# 路径定义
# =========================
DATA_PATH = "./data/feature_advanced/"
RESULT_PATH = "./final_models/results/"
SUB_PATH = "./final_models/submission/"
os.makedirs(RESULT_PATH, exist_ok=True)
os.makedirs(SUB_PATH, exist_ok=True)


# =========================
# 工具函数
# =========================
def safe_load_npy(path: str, fallback_len: int = None):
    """安全加载 npy 文件，如果不存在则返回零数组"""
    if os.path.exists(path):
        return np.load(path)
    else:
        print(f"⚠️ 警告: 文件缺失 {path}，使用零向量代替。")
        return np.zeros(fallback_len)


def load_feature_data():
    """加载训练/验证/测试特征与标签"""
    X_train = np.load(os.path.join(DATA_PATH, "train_scaled.npy"))
    X_valid = np.load(os.path.join(DATA_PATH, "valid_scaled.npy"))
    X_test = np.load(os.path.join(DATA_PATH, "test_scaled.npy"))
    y_train_full = np.load(os.path.join(DATA_PATH, "train_labels.npy"))
    y_valid = np.load(os.path.join(DATA_PATH, "valid_labels.npy"))
    
    # 加载特征名称
    feature_names_path = os.path.join(DATA_PATH, "feature_names.txt")
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines() if line.strip()]
    else:
        raise FileNotFoundError(f"未找到特征名称文件: {feature_names_path}")
    
    normal_mask = (y_train_full == 0)
    X_train_normal = X_train[normal_mask]
    print(f"Train(normal)={X_train_normal.shape}, Valid={X_valid.shape}, Test={X_test.shape}")
    return X_train_normal, X_valid, X_test, y_valid, feature_names


def extract_feature_from_matrix(X_scaled, feature_name, feature_names):
    """从特征矩阵中提取指定特征"""
    if feature_name not in feature_names:
        raise ValueError(f"特征 {feature_name} 不在特征名称列表中")
    idx = feature_names.index(feature_name)
    return X_scaled[:, idx]


def run_lof_detector(X_train, X_eval, k=50):
    """运行 LOF 检测器"""
    print(f"运行 LOF(k={k}) 检测器 ...")
    t0 = time.time()
    lof = LocalOutlierFactor(n_neighbors=k, novelty=True, n_jobs=-1)
    lof.fit(X_train)
    scores = -lof.decision_function(X_eval)
    print(f"  -> 完成 ({time.time() - t0:.2f}s)")
    return scores


# =========================
# 主函数
# =========================
def main():
    print("=" * 70)
    print("Final Model - Fusion Ensemble")
    print("=" * 70)

    # 1️⃣ 加载数据
    X_train_normal, X_valid, X_test, y_valid, feature_names = load_feature_data()

    # 2️⃣ 加载 3 个特征检测器分数（验证集）
    scores_file = os.path.join(RESULT_PATH, "final_model_scores.csv")
    if not os.path.exists(scores_file):
        raise FileNotFoundError(f"未找到 {scores_file}，请先运行 generate_ensemble_scores.py")
    scores_df = pd.read_csv(scores_file)
    print(f"加载已有检测器分数: {list(scores_df.columns)}")

    # 3️⃣ 运行 LOF(k=50)
    lof_valid = run_lof_detector(X_train_normal, X_valid)
    lof_test = run_lof_detector(X_train_normal, X_test)
    scores_df["LOF_k50"] = lof_valid

    # 4️⃣ 从测试集中提取特征值
    print("\n从测试集中提取特征值...")
    test_scores = {}
    for col in scores_df.columns:
        if col == "LOF_k50":
            test_scores[col] = lof_test
        else:
            # 从 test_scaled.npy 中提取特征
            test_scores[col] = extract_feature_from_matrix(X_test, col, feature_names)
    test_scores_df = pd.DataFrame(test_scores, columns=scores_df.columns)

    # 5️⃣ 标准化
    print("标准化分数到 [0,1] ...")
    scaler = MinMaxScaler()
    # 先在验证集上 fit
    scaled_valid = pd.DataFrame(scaler.fit_transform(scores_df), columns=scores_df.columns)
    # 然后在测试集上 transform（使用相同的 scaler）
    scaled_test = pd.DataFrame(scaler.transform(test_scores_df), columns=test_scores_df.columns)

    # 6️⃣ 融合权重（经验加权）
    # 经验值：强信号权重大一些
    weights = {
        "Early_Delinquency_Flag": 0.4,
        "amort_short_mean": 0.3,
        "Zero_Payment_Streak": 0.1,
        "LOF_k50": 0.2
    }

    # 确保所有权重对应的列都存在
    available_cols = set(scaled_valid.columns)
    weights = {col: w for col, w in weights.items() if col in available_cols}
    # 归一化权重
    total_weight = sum(weights.values())
    weights = {col: w / total_weight for col, w in weights.items()}
    
    print(f"\n融合权重: {weights}")

    # 加权融合
    final_valid = sum(w * scaled_valid[col] for col, w in weights.items())
    final_test = sum(w * scaled_test[col] for col, w in weights.items())

    # 7️⃣ 评估验证集性能
    auprc = average_precision_score(y_valid, final_valid)
    auroc = roc_auc_score(y_valid, final_valid)
    print("\n📊 Final Fusion Performance:")
    print(f"  AUPRC = {auprc:.6f}")
    print(f"  AUROC = {auroc:.6f}")

    # 保存性能结果
    metrics_df = pd.DataFrame({
        "Metric": ["AUPRC", "AUROC"],
        "Value": [auprc, auroc]
    })
    metrics_path = os.path.join(RESULT_PATH, "fusion_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✅ 性能指标已保存到: {metrics_path}")

    # 8️⃣ 生成提交文件
    # 加载测试集ID
    test_ids = np.load(os.path.join(DATA_PATH, "test_ids.npy"))
    
    submission = pd.DataFrame({
        "Id": test_ids.astype(int),
        "target": np.clip(final_test, 0, 1)
    })
    sub_path = os.path.join(SUB_PATH, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"\n✅ 已生成提交文件: {sub_path}")
    print(f"   行数: {len(submission)}")
    print(f"   分数范围: [{final_test.min():.6f}, {final_test.max():.6f}]")
    print("\n文件预览:")
    print(submission.head(10))
    
    # 保存融合权重摘要
    summary_df = pd.DataFrame([{
        "AUPRC": auprc,
        "AUROC": auroc,
        **{f"Weight_{col}": w for col, w in weights.items()}
    }])
    summary_path = os.path.join(RESULT_PATH, "fusion_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ 融合摘要已保存到: {summary_path}")


if __name__ == "__main__":
    main()
