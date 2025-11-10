#!/usr/bin/env python3
"""
Final Model - Fusion + Automatic Weight Optimization (Optuna)
--------------------------------------------------------
Fuse scores from the following detectors:
1️⃣ Early_Delinquency_Flag
2️⃣ amort_short_mean
3️⃣ Zero_Payment_Streak
4️⃣ LOF(k=50)

Use Optuna to automatically find the best fusion weights to maximize AUPRC.

Output:
- Best fusion weights and performance (fusion_summary.csv)
- Test set submission file (submission.csv)
"""

import os
import numpy as np
import pandas as pd
import time
import optuna  # Import Optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from typing import Tuple, List, Dict, Any

# =========================
# Path Definitions
# =========================
DATA_PATH = "./data/feature_advanced/"
RESULT_PATH = "./final_models/results/"
SUB_PATH = "./final_models/submission/"
os.makedirs(RESULT_PATH, exist_ok=True)
os.makedirs(SUB_PATH, exist_ok=True)

# =========================
# Model Configuration
# =========================
LOF_K = 50       # Baseline LOF(k=50)
OPTUNA_TRIALS = 100 # Run 100 trials to find best weights

# =========================
# Utility Functions
# =========================
def safe_load_npy(path: str, fallback_len: int = None):
    """Safely load npy file, return zero array if not exists"""
    if os.path.exists(path):
        return np.load(path)
    else:
        print(f"⚠️ Warning: File missing {path}, using zero vector instead.")
        return np.zeros(fallback_len)


def load_feature_data():
    """Load training/validation/test features and labels"""
    X_train = np.load(os.path.join(DATA_PATH, "train_scaled.npy"))
    X_valid = np.load(os.path.join(DATA_PATH, "valid_scaled.npy"))
    X_test = np.load(os.path.join(DATA_PATH, "test_scaled.npy"))
    y_train_full = np.load(os.path.join(DATA_PATH, "train_labels.npy"))
    y_valid = np.load(os.path.join(DATA_PATH, "valid_labels.npy"))
    
    feature_names_path = os.path.join(DATA_PATH, "feature_names.txt")
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines() if line.strip()]
    else:
        raise FileNotFoundError(f"Feature names file not found: {feature_names_path}")
    
    normal_mask = (y_train_full == 0)
    X_train_normal = X_train[normal_mask]
    print(f"Train(normal)={X_train_normal.shape}, Valid={X_valid.shape}, Test={X_test.shape}")
    return X_train_normal, X_valid, X_test, y_valid, feature_names


def extract_feature_from_matrix(X_scaled, feature_name, feature_names):
    """Extract specified feature from feature matrix"""
    if feature_name not in feature_names:
        raise ValueError(f"Feature {feature_name} not in feature names list")
    idx = feature_names.index(feature_name)
    return X_scaled[:, idx]


def run_lof_detector(X_train, X_eval, k=50):
    """Run LOF detector"""
    print(f"Running LOF(k={k}) detector ...")
    t0 = time.time()
    lof = LocalOutlierFactor(n_neighbors=k, novelty=True, n_jobs=-1)
    lof.fit(X_train)
    scores = -lof.decision_function(X_eval)
    print(f"  -> Completed ({time.time() - t0:.2f}s)")
    return scores

# =========================
# Optuna Objective Function
# =========================
def objective(trial, scaled_valid_df, y_valid):
    """
    Optuna objective function: find best weights
    """
    # 1. Define search space for 4 detectors
    # Use suggest_float to get raw weights between 0 and 1
    w1 = trial.suggest_float("w_Early_Delinquency_Flag", 0, 1)
    w2 = trial.suggest_float("w_amort_short_mean", 0, 1)
    w3 = trial.suggest_float("w_LOF_k50", 0, 1) # LOF baseline
    w4 = trial.suggest_float("w_Zero_Payment_Streak", 0, 1)

    # 2. Normalize weights (make them sum to 1)
    total_w = w1 + w2 + w3 + w4
    if total_w == 0: # Avoid division by zero
        return 0.0

    norm_w1 = w1 / total_w
    norm_w2 = w2 / total_w
    norm_w3 = w3 / total_w
    norm_w4 = w4 / total_w

    # 3. Calculate weighted fusion scores
    final_score = (
        norm_w1 * scaled_valid_df["Early_Delinquency_Flag"] +
        norm_w2 * scaled_valid_df["amort_short_mean"] +
        norm_w3 * scaled_valid_df["LOF_k50"] +
        norm_w4 * scaled_valid_df["Zero_Payment_Streak"]
    )
    
    # 4. Return AUPRC
    auprc = average_precision_score(y_valid, final_score)
    return auprc

# =========================
# Main Function
# =========================
def main():
    print("=" * 70)
    print("Final Model - Automatic Weight Optimization Fusion (Optuna)")
    print("=" * 70)

    # 1️⃣ Load data
    X_train_normal, X_valid, X_test, y_valid, feature_names = load_feature_data()

    # 2️⃣ Load 3 feature detector scores (validation set)
    scores_file = os.path.join(RESULT_PATH, "final_model_scores.csv")
    if not os.path.exists(scores_file):
        raise FileNotFoundError(f"File not found: {scores_file}, please run generate_ensemble_scores.py first")
    scores_df = pd.read_csv(scores_file)
    print(f"Loaded existing detector scores: {list(scores_df.columns)}")

    # 3️⃣ Run LOF(k=50)
    lof_valid = run_lof_detector(X_train_normal, X_valid)
    lof_test = run_lof_detector(X_train_normal, X_test)
    scores_df["LOF_k50"] = lof_valid

    # 4️⃣ Extract feature values from test set
    print("\nExtracting feature values from test set...")
    test_scores = {}
    for col in scores_df.columns:
        if col == "LOF_k50":
            test_scores[col] = lof_test
        else:
            test_scores[col] = extract_feature_from_matrix(X_test, col, feature_names)
    test_scores_df = pd.DataFrame(test_scores, columns=scores_df.columns)

    # 5️⃣ Standardize
    print("Standardizing scores to [0,1] ...")
    scaler = MinMaxScaler()
    scaled_valid = pd.DataFrame(scaler.fit_transform(scores_df), columns=scores_df.columns)
    scaled_test = pd.DataFrame(scaler.transform(test_scores_df), columns=test_scores_df.columns)

    # 6️⃣ Use Optuna to find best fusion weights
    print("\n" + "=" * 70)
    print(f"Step 6: Starting Optuna weight optimization ({OPTUNA_TRIALS} trials)")
    print("=" * 70)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, scaled_valid, y_valid),
        n_trials=OPTUNA_TRIALS
    )
    
    print(f"--- Optimization completed ---")

    # 7️⃣ Evaluate validation set performance (using best weights found by Optuna)
    print("\n" + "=" * 70)
    print("📊 Final Fusion Performance (Optimized)")
    print("=" * 70)
    
    best_weights_raw = study.best_params
    total_w = sum(best_weights_raw.values())
    
    # Normalize best weights
    best_weights = {name: w / total_w for name, w in best_weights_raw.items()}

    final_valid = sum(best_weights[f"w_{col}"] * scaled_valid[col] for col in scores_df.columns)
    auprc = average_precision_score(y_valid, final_valid)
    auroc = roc_auc_score(y_valid, final_valid)

    print(f"  Best AUPRC = {auprc:.6f}")
    print(f"  Best AUROC = {auroc:.6f}")
    
    print("\nBest weight combination:")
    for name, w in best_weights.items():
        print(f"  - {name.replace('w_', '')}: {w*100:.2f}%")

    # 8️⃣ Generate submission file (using best weights)
    print("\n" + "=" * 70)
    print("Step 8: Generating submission file")
    print("=" * 70)
    
    final_test = sum(best_weights[f"w_{col}"] * scaled_test[col] for col in scores_df.columns)
    
    test_ids = np.load(os.path.join(DATA_PATH, "test_ids.npy"))
    submission = pd.DataFrame({
        "Id": test_ids.astype(int),
        "target": np.clip(final_test, 0, 1)
    })
    sub_path = os.path.join(SUB_PATH, "submission_optuna.csv") # New file name
    submission.to_csv(sub_path, index=False)
    print(f"\n✅ Optimized submission file generated: {sub_path}")
    print(f"   Rows: {len(submission)}")
    print(f"   Score range: [{final_test.min():.6f}, {final_test.max():.6f}]")
    
    # Save fusion weight summary
    summary_df = pd.DataFrame([{
        "AUPRC": auprc,
        "AUROC": auroc,
        **{f"Weight_{name.replace('w_', '')}": w for name, w in best_weights.items()}
    }])
    summary_path = os.path.join(RESULT_PATH, "fusion_summary_optuna.csv") # New file name
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ Fusion summary saved to: {summary_path}")


if __name__ == "__main__":
    main()