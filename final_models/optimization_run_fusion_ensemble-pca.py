#!/usr/bin/env python3
"""
Final Model - PCA+LOF Fusion + Submission Generation Script
-------------------------------------------------------------
Fuse the following detectors:
1. Early_Delinquency_Flag
2. amort_short_mean
3. Zero_Payment_Streak
4. PCA → LOF_k50

Output:
- fusion_metrics.csv
- submission.csv
- fusion_summary.csv
"""

import os
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

# =========================
# Path Definitions
# =========================
DATA_PATH = "./data/feature_advanced/"
RESULT_PATH = "./final_models/results/"
SUB_PATH = "./final_models/submission/"
os.makedirs(RESULT_PATH, exist_ok=True)
os.makedirs(SUB_PATH, exist_ok=True)


# =========================
# Utility Functions
# =========================
def load_feature_data():
    """Load feature matrix + labels + feature names"""
    X_train = np.load(os.path.join(DATA_PATH, "train_scaled.npy"))
    X_valid = np.load(os.path.join(DATA_PATH, "valid_scaled.npy"))
    X_test = np.load(os.path.join(DATA_PATH, "test_scaled.npy"))

    y_train_full = np.load(os.path.join(DATA_PATH, "train_labels.npy"))
    y_valid = np.load(os.path.join(DATA_PATH, "valid_labels.npy"))

    with open(os.path.join(DATA_PATH, "feature_names.txt"), "r") as f:
        feature_names = [line.strip() for line in f.readlines()]

    normal_mask = (y_train_full == 0)
    X_train_normal = X_train[normal_mask]

    print(f"Train(normal)={X_train_normal.shape}, Valid={X_valid.shape}, Test={X_test.shape}")
    return X_train_normal, X_valid, X_test, y_valid, feature_names


def extract_feature(X, feature, feature_names):
    """Extract column from scaled matrix"""
    if feature not in feature_names:
        raise ValueError(f"❌ Feature {feature} not found")
    idx = feature_names.index(feature)
    return X[:, idx]


def run_pca_lof(X_train_normal, X_eval):
    """PCA → LOF(k=50)"""
    print("\n➡️ Running PCA+LOF detector ...")

    # 1️⃣ PCA dimensionality reduction
    t0 = time.time()
    pca = PCA(n_components=0.95)  # Automatically retain 95% variance
    pca.fit(X_train_normal)

    X_train_pca = pca.transform(X_train_normal)
    X_eval_pca = pca.transform(X_eval)
    print(f"   PCA completed: dimension {X_train_normal.shape[1]} → {X_train_pca.shape[1]} ({time.time() - t0:.2f}s)")

    # 2️⃣ LOF(k=50)
    t1 = time.time()
    lof = LocalOutlierFactor(n_neighbors=50, novelty=True, n_jobs=-1)
    lof.fit(X_train_pca)
    scores = -lof.decision_function(X_eval_pca)
    print(f"   LOF completed: ({time.time() - t1:.2f}s)")

    return scores


# =========================
# Main Function
# =========================
def main():
    print("=" * 70)
    print("Final Model - PCA+LOF Fusion Ensemble")
    print("=" * 70)

    # 1️⃣ Load data
    X_train_normal, X_valid, X_test, y_valid, feature_names = load_feature_data()

    # 2️⃣ Load previously computed 3 detectors (validation set)
    score_file = os.path.join(RESULT_PATH, "final_model_scores.csv")
    score_df = pd.read_csv(score_file)
    print(f"\nLoaded feature detectors: {list(score_df.columns)}")

    # 3️⃣ PCA+LOF scores
    lof_valid = run_pca_lof(X_train_normal, X_valid)
    lof_test  = run_pca_lof(X_train_normal, X_test)

    # Add to validation scores
    score_df["PCA_LOF_k50"] = lof_valid

    # 4️⃣ Build test_scores_df
    test_scores = {}
    for col in score_df.columns:
        if col == "PCA_LOF_k50":
            test_scores[col] = lof_test
        else:
            test_scores[col] = extract_feature(X_test, col, feature_names)

    test_scores_df = pd.DataFrame(test_scores, columns=score_df.columns)

    # 5️⃣ Standardize
    print("\n➡️ Standardizing all scores to [0,1] ...")
    scaler = MinMaxScaler()
    scaled_valid = pd.DataFrame(scaler.fit_transform(score_df), columns=score_df.columns)
    scaled_test = pd.DataFrame(scaler.transform(test_scores_df), columns=test_scores_df.columns)

    # 6️⃣ Fusion weights (empirical values)
    weights = {
        "Early_Delinquency_Flag": 0.4,
        "amort_short_mean": 0.3,
        "Zero_Payment_Streak": 0.1,
        "PCA_LOF_k50": 0.2
    }

    # Normalize
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}
    print(f"➡️ Using fusion weights: {weights}")

    # 7️⃣ Weighted fusion
    final_valid = sum(w * scaled_valid[col] for col, w in weights.items())
    final_test  = sum(w * scaled_test[col] for col, w in weights.items())

    # 8️⃣ Evaluate validation set performance
    auprc = average_precision_score(y_valid, final_valid)
    auroc = roc_auc_score(y_valid, final_valid)
    print("\n✅ Final Fusion Performance:")
    print(f"   AUPRC = {auprc:.6f}")
    print(f"   AUROC = {auroc:.6f}")

    # Save performance
    pd.DataFrame({
        "Metric": ["AUPRC", "AUROC"],
        "Value": [auprc, auroc]
    }).to_csv(os.path.join(RESULT_PATH, "fusion_metrics_pca.csv"), index=False)

    # 9️⃣ Generate submission file
    test_ids = np.load(os.path.join(DATA_PATH, "test_ids.npy"))
    submission = pd.DataFrame({
        "Id": test_ids.astype(int),
        "target": np.clip(final_test, 0, 1)
    })
    sub_path = os.path.join(SUB_PATH, "submission_pca.csv")
    submission.to_csv(sub_path, index=False)
    print(f"\n✅ Submission file generated: {sub_path}")
    print(f"   Rows: {len(submission)}")

    # 10️⃣ Save summary
    summary = {
        "AUPRC": auprc,
        "AUROC": auroc
    }
    for col, w in weights.items():
        summary[f"Weight_{col}"] = w

    pd.DataFrame([summary]).to_csv(
        os.path.join(RESULT_PATH, "fusion_summary_pca.csv"), index=False
    )

    print("\n✅ All results generated successfully!")


if __name__ == "__main__":
    main()

