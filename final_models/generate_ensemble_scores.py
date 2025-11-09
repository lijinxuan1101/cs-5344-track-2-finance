#!/usr/bin/env python3
"""
Final Model - 三个基于原始特征值的检测器

本脚本实现了集成策略的第一步：
将 AUPRC 最高的、且“多样化”的三个特征视为独立的异常检测器。

检测器 1 (王牌)： Early_Delinquency_Flag
检测器 2 (二号王牌)： amort_short_mean
检测器 3 (三号王牌)： Zero_Payment_Streak

脚本会分别评估这三个检测器的性能，并将它们各自的异常分数保存到
`final_model_scores.csv`，以便下一步进行“融合” (Fusion)。
"""

import numpy as np
import pandas as pd
import os
from sklearn.metrics import average_precision_score, roc_auc_score
from typing import Tuple, List, Dict, Any
from abc import ABC, abstractmethod

# --- 路径定义 ---
DATA_PATH = "./data/feature_advanced/"
OUTPUT_PATH = "./final_models/results/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# --- 检测器基类 ---

class BaseDetector(ABC):
    """
    检测器基类
    所有检测器都应该继承这个类
    """
    def __init__(self, name: str, feature_name: str):
        self.name = name
        self.feature_name = feature_name # 用于在 main 循环中查找数据

    @abstractmethod
    def fit(self, X_train_col: np.ndarray) -> None:
        """
        在训练数据上拟合检测器（仅使用正常样本）
        
        参数:
            X_train_col (np.ndarray): 形状为 (n_samples, 1) 的训练特征列
        """
        pass
    
    @abstractmethod
    def predict_score(self, X_col: np.ndarray) -> np.ndarray:
        """
        对验证集/测试集预测异常分数

        参数:
            X_col (np.ndarray): 形状为 (n_samples, 1) 的特征列
        返回:
            np.ndarray: 形状为 (n_samples,) 的异常分数数组（分数越高表示越异常）
        """
        pass

# --- 特征检测器实现 ---

class RawFeatureDetector(BaseDetector):
    """
    一个通用的“原始特征”检测器。
    它直接使用（已缩放的）特征值作为异常分数。
    
    假设：特征值越大，异常的可能性越高。
    这适用于 Early_Delinquency_Flag, amort_short_mean, 和 Zero_Payment_Streak。
    """
    def fit(self, X_train_col: np.ndarray) -> None:
        """
        这是一个“无状态”检测器，它不需要从训练集中学习任何东西。
        它只依赖于特征本身的值。
        """
        pass
        
    def predict_score(self, X_col: np.ndarray) -> np.ndarray:
        """
        直接返回（已缩放的）特征值作为异常分数。
        值越大表示风险越高。
        """
        # .flatten() 将 (n_samples, 1) 数组转为 (n_samples,)
        return X_col.flatten()

# --- 辅助函数 ---

def load_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    加载特征数据和特征名称
    """
    print(f"从 '{path}' 加载特征文件...")
    
    try:
        X_train_scaled = np.load(os.path.join(path, "train_scaled.npy"))
        X_valid_scaled = np.load(os.path.join(path, "valid_scaled.npy"))
        y_train_full_labels = np.load(os.path.join(path, "train_labels.npy"))
        y_valid = np.load(os.path.join(path, "valid_labels.npy"))
        
        # 加载特征名称
        feature_names_path = os.path.join(DATA_PATH, "feature_names.txt")
        if not os.path.exists(feature_names_path):
            raise FileNotFoundError(f"未找到 {feature_names_path}")
            
        with open(feature_names_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines() if line.strip()]
        
    except FileNotFoundError as e:
        print(f"错误：找不到文件 {e}。")
        print(f"请确保你已经成功运行了 feature_generator.py 脚本。")
        return None, None, None, None
    
    # 只选择 target==0 的训练样本用于拟合
    normal_mask = (y_train_full_labels == 0)
    X_train_scaled_normal = X_train_scaled[normal_mask]
    
    print(f" - 训练集 (正常) shape: {X_train_scaled_normal.shape}")
    print(f" - 验证集 shape: {X_valid_scaled.shape}")
    print(f" - 验证集标签 shape: {y_valid.shape}")
    print(f" - 验证集异常率: {y_valid.mean()*100:.2f}%")
    print(f" - 特征数量: {len(feature_names)}")
    
    return X_train_scaled_normal, X_valid_scaled, y_valid, feature_names


def evaluate_detector(detector: BaseDetector, 
                      X_train: np.ndarray, X_valid: np.ndarray, 
                      y_valid: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
    """
    评估单个检测器
    """
    print(f"\n正在运行: {detector.name}...")
    
    try:
        # 1. 动态查找特征索引
        if detector.feature_name not in feature_names:
            raise ValueError(f"特征 {detector.feature_name} 不在特征名称列表中")
        idx = feature_names.index(detector.feature_name)
        
        # 2. 提取数据列 (保持2D以便 fit/predict)
        train_col = X_train[:, [idx]]
        valid_col = X_valid[:, [idx]]
        
        # 3. 在训练数据上拟合（只使用正常样本）
        detector.fit(train_col)
        
        # 4. 在验证集上预测
        scores = detector.predict_score(valid_col)
        
        # 5. 计算指标
        auprc = average_precision_score(y_valid, scores)
        auroc = roc_auc_score(y_valid, scores)
        
        print(f"  -> AUPRC: {auprc:.4f}, AUROC: {auroc:.4f}")
        print(f"  -> (使用已缩放的值) 分数统计: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
        
        return {
            "Detector": detector.name,
            "Feature": detector.feature_name,
            "AUPRC": auprc,
            "AUROC": auroc,
            "Scores": scores
        }
        
    except Exception as e:
        print(f"  -> 失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            "Detector": detector.name,
            "Feature": detector.feature_name,
            "AUPRC": np.nan,
            "AUROC": np.nan,
            "Scores": None
        }

# --- 主执行函数 ---

def main():
    """
    主执行函数
    """
    print("=" * 60)
    print("Final Model - 三个核心特征检测器评估")
    print("=" * 60)
    
    # 1. 加载数据
    X_train_normal, X_valid, y_valid, feature_names = load_data(DATA_PATH)
    if X_train_normal is None:
        return
    
    # 2. 创建三个检测器实例
    # 我们重用 RawFeatureDetector 类，因为它完全符合我们的需求
    detectors: List[BaseDetector] = [
        RawFeatureDetector(
            name="Early_Delinquency_Flag Detector (王牌)",
            feature_name="Early_Delinquency_Flag"
        ),
        RawFeatureDetector(
            name="amort_short_mean Detector (二号王牌)",
            feature_name="amort_short_mean"
        ),
        RawFeatureDetector(
            name="Zero_Payment_Streak Detector (三号王牌)",
            feature_name="Zero_Payment_Streak"
        )
    ]
    
    # 3. 评估所有检测器
    results = []
    all_scores = {}
    
    for detector in detectors:
        result = evaluate_detector(detector, X_train_normal, X_valid, y_valid, feature_names)
        results.append(result)
        if result["Scores"] is not None:
            # 使用简短的特征名作为 key
            all_scores[detector.feature_name] = result["Scores"]
    
    # 4. 总结结果
    print("\n" + "=" * 60)
    print("检测器评估结果")
    print("=" * 60)
    
    results_df = pd.DataFrame([
        {
            "Detector": r["Detector"],
            "Feature": r["Feature"],
            "AUPRC": r["AUPRC"],
            "AUROC": r["AUROC"]
        }
        for r in results
    ]).sort_values(by="AUPRC", ascending=False)
    
    print("\n最终性能 (按AUPRC排序):")
    print(results_df.to_string(index=False))
    
    # 5. 保存结果
    output_filename = os.path.join(OUTPUT_PATH, "final_model_results.csv")
    results_df.to_csv(output_filename, index=False)
    print(f"\n结果已保存到: {output_filename}")
    
    # 保存所有检测器的分数
    if all_scores:
        scores_df = pd.DataFrame(all_scores)
        scores_output = os.path.join(OUTPUT_PATH, "final_model_scores.csv")
        scores_df.to_csv(scores_output, index=False)
        print(f"检测器分数已保存到: {scores_output}")
    
    print("\n" + "=" * 60)
    print("评估完成！")
    print("下一步：使用 'final_model_scores.csv' 和你的 LOF 基线分数进行融合。")
    print("=" * 60)


if __name__ == "__main__":
    main()