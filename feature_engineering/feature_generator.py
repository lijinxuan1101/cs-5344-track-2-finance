import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
import os
import re

# ---------- utilities ----------
def safe_div(a, b, eps=1e-9):
    return a / (b + eps)

def pct_change(a, axis=1, eps=1e-9):
    return (a[:, 1:] - a[:, :-1]) / (np.abs(a[:, :-1]) + eps)

# ---------- feature builder class  ---------- # 
class FeatureBuilderAdvanced:
    """
    Train-only feature builder:
      - sentinel -> NaN + missing flags (static)
      - label-encode categorical with UNKNOWN (train-only fit)
      - temporal engineering on multiple early windows (trend/vol/first-diff stats)
      - amortization residuals (FRM & not IO/balloon if possible; otherwise masked zeros)
      - ONE global scaler (train-only) across the final tabular feature matrix
    """
    def __init__(
        self,
        keep_month_idx_main=(0,3,6,9,12),
        keep_month_idx_alt1=(0,2,4,6,8,10,12),
        keep_month_idx_alt2=(0,3,6,9),
        scaler_type="robust",  # "standard" 或 "robust"
    ):
        # 定义要删除的特征（低价值或噪声特征）
        self.features_to_drop = [
            # ---- 全部 Missing Flag ----
            "CreditScore_missing", "OriginalDTI_missing", "OriginalLTV_missing",
            "FirstTimeHomebuyerFlag_missing", "OccupancyStatus_missing",
            "LoanPurpose_missing", "Channel_missing", "PropertyType_missing",
            
            # ---- 明显噪声类 (AUPRC≈0.126, AUROC≈0.5) ----
            "LoanAge_w_main_trend", "LoanAge_w_main_vol", "LoanAge_w_main_dmean", "LoanAge_w_main_dstd",
            "LoanAge_w_alt1_trend", "LoanAge_w_alt1_vol", "LoanAge_w_alt1_dmean", "LoanAge_w_alt1_dstd",
            "LoanAge_w_alt2_trend", "LoanAge_w_alt2_vol", "LoanAge_w_alt2_dmean", "LoanAge_w_alt2_dstd",
            "MonthlyReportingPeriod_w_main_trend", "MonthlyReportingPeriod_w_main_vol",
            "MonthlyReportingPeriod_w_main_dmean", "MonthlyReportingPeriod_w_main_dstd",
            "MonthlyReportingPeriod_w_alt1_trend", "MonthlyReportingPeriod_w_alt1_vol",
            "MonthlyReportingPeriod_w_alt1_dmean", "MonthlyReportingPeriod_w_alt1_dstd",
            "MonthlyReportingPeriod_w_alt2_trend", "MonthlyReportingPeriod_w_alt2_vol",
            "MonthlyReportingPeriod_w_alt2_dmean", "MonthlyReportingPeriod_w_alt2_dstd",
            "CurrentInterestRate_w_main_trend", "CurrentInterestRate_w_main_vol",
            "CurrentInterestRate_w_main_dmean", "CurrentInterestRate_w_main_dstd",
            "CurrentInterestRate_w_alt1_trend", "CurrentInterestRate_w_alt1_vol",
            "CurrentInterestRate_w_alt1_dmean", "CurrentInterestRate_w_alt1_dstd",
            "CurrentInterestRate_w_alt2_trend", "CurrentInterestRate_w_alt2_vol",
            "CurrentInterestRate_w_alt2_dmean", "CurrentInterestRate_w_alt2_dstd",
            "CurrentNonInterestBearingUPB_w_main_trend", "CurrentNonInterestBearingUPB_w_main_vol",
            "CurrentNonInterestBearingUPB_w_main_dmean", "CurrentNonInterestBearingUPB_w_main_dstd",
            "CurrentNonInterestBearingUPB_w_alt1_trend", "CurrentNonInterestBearingUPB_w_alt1_vol",
            "CurrentNonInterestBearingUPB_w_alt1_dmean", "CurrentNonInterestBearingUPB_w_alt1_dstd",
            "CurrentNonInterestBearingUPB_w_alt2_trend", "CurrentNonInterestBearingUPB_w_alt2_vol",
            "CurrentNonInterestBearingUPB_w_alt2_dmean", "CurrentNonInterestBearingUPB_w_alt2_dstd",
            "InterestBearingUPB_w_main_trend", "InterestBearingUPB_w_main_vol",
            "InterestBearingUPB_w_main_dmean", "InterestBearingUPB_w_main_dstd",
            "InterestBearingUPB_w_alt1_trend", "InterestBearingUPB_w_alt1_vol",
            "InterestBearingUPB_w_alt1_dmean", "InterestBearingUPB_w_alt1_dstd",
            "InterestBearingUPB_w_alt2_trend", "InterestBearingUPB_w_alt2_vol",
            "InterestBearingUPB_w_alt2_dmean", "InterestBearingUPB_w_alt2_dstd",
            "RemainingMonthsToLegalMaturity_w_main_trend", "RemainingMonthsToLegalMaturity_w_main_vol",
            "RemainingMonthsToLegalMaturity_w_main_dmean", "RemainingMonthsToLegalMaturity_w_main_dstd",
            "RemainingMonthsToLegalMaturity_w_alt1_trend", "RemainingMonthsToLegalMaturity_w_alt1_vol",
            "RemainingMonthsToLegalMaturity_w_alt1_dmean", "RemainingMonthsToLegalMaturity_w_alt1_dstd",
            "RemainingMonthsToLegalMaturity_w_alt2_trend", "RemainingMonthsToLegalMaturity_w_alt2_vol",
            "RemainingMonthsToLegalMaturity_w_alt2_dmean", "RemainingMonthsToLegalMaturity_w_alt2_dstd",
            
            # ---- 低信息类别特征 ----
            # "FirstTimeHomebuyerFlag", "PropertyType", "ProductType", "InterestOnlyFlag",
            # "PreHARP_Flag", "PPM_Flag", "SuperConformingFlag", "ReliefRefinanceIndicator",
            # "PropertyValMethod", "ServicerName", "SellerName", "MSA", "PostalCode",
            
            # # ---- 弱信号 LTV 系列 ----
            # "EstimatedLTV_max", "EstimatedLTV_std", "EstimatedLTV_last_val",
            # "EstimatedLTV_w_main_trend", "EstimatedLTV_w_main_vol", "EstimatedLTV_w_main_dmean", "EstimatedLTV_w_main_dstd",
            # "EstimatedLTV_w_alt1_trend", "EstimatedLTV_w_alt1_vol", "EstimatedLTV_w_alt1_dmean", "EstimatedLTV_w_alt1_dstd",
            # "EstimatedLTV_w_alt2_trend", "EstimatedLTV_w_alt2_vol", "EstimatedLTV_w_alt2_dmean", "EstimatedLTV_w_alt2_dstd",
            
            # # ---- 弱静态变量 ----
            # "OriginalLoanTerm", "OriginalDTI", "OriginalCLTV", "MI_Pct", "NumberOfUnits",
            # "PropertyState", "Channel", "LoanPurpose",
            
            # ---- 其他无效或平坦列 ----
            "amort_mask_not_applicable", "LTV_Change"
        ]
        self.keep_month_idx_main = set(keep_month_idx_main)
        self.keep_month_idx_alt1 = set(keep_month_idx_alt1)
        self.keep_month_idx_alt2 = set(keep_month_idx_alt2)
        self.scaler_type = scaler_type.lower()

        self.cat_encoders: Dict[str, LabelEncoder] = {}
        self.impute_vals: Dict[str, float] = {}
        self.scaler: Optional[Union[StandardScaler, RobustScaler]] = None

        self.static_cols: List[str] = []
        self.temporal_cols_all: List[str] = []
        self._fitted = False
        self.feature_names_: Optional[List[str]] = None  # 保存特征列名，用于确保transform时列顺序一致

        self._amort_slice = slice(0,0)
        self._temporal_slice = slice(0,0)
        self._static_interaction_slice = slice(0,0)

    # ---------- helpers ----------
    @staticmethod
    def _is_temporal(col: str) -> bool:
        if "_" not in col: return False
        left = col.split("_", 1)[0]
        return left.isdigit()

    def _sentinel_map(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Numeric sentinels
        SENT = {
            "CreditScore": 9999,
            "OriginalDTI": 999,
            "OriginalLTV": 999,
            "MI Pct": 999,
            "EstimatedLTV": 999,
        }
        for col, bad in SENT.items():
            if col in df.columns:
                miss = (df[col] == bad)
                df.loc[miss, col] = np.nan
                df[f"{col}_missing"] = miss.astype(int)

        # Flags '9'/'99' => missing
        FLAG_9 = ["FirstTimeHomebuyerFlag","OccupancyStatus","LoanPurpose","Channel","PropertyType"]
        for col in FLAG_9:
            if col in df.columns:
                x = df[col].astype(str).str.strip()
                miss = (x == "9") | (x == "99")
                df.loc[miss, col] = np.nan
                df[f"{col}_missing"] = miss.astype(int)

        return df

    def _collect_columns(self, df: pd.DataFrame):
        self.static_cols = []
        self.temporal_cols_all = []
        for c in df.columns:
            if c in ("index","Id","target"): continue
            if self._is_temporal(c):
                self.temporal_cols_all.append(c)
            else:
                self.static_cols.append(c)

    def _fit_cat_and_impute(self, df: pd.DataFrame):
        for c in self.static_cols:
            if df[c].dtype == "object":
                enc = LabelEncoder()
                vals = df[c].fillna("MISSING").astype(str).unique().tolist()
                if "UNKNOWN" not in vals: vals.append("UNKNOWN")
                enc.fit(vals)
                self.cat_encoders[c] = enc
        for c in self.static_cols:
            if c not in self.cat_encoders:
                v = df[c].dropna()
                self.impute_vals[c] = float(v.median() if len(v) else 0.0)

    def _transform_static(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df[self.static_cols].copy()
        for c, enc in self.cat_encoders.items():
            x = data[c].fillna("MISSING").astype(str)
            x = x.where(x.isin(enc.classes_), "UNKNOWN")
            data[c] = enc.transform(x)
        for c in self.static_cols:
            if c not in self.cat_encoders:
                data[c] = data[c].fillna(self.impute_vals.get(c, 0.0))
        
        # --- 新增：静态风险交叉特征 ---
        data['LTV_x_DTI'] = data['OriginalLTV'] * data['OriginalDTI']
        data['UPB_per_CreditScore'] = safe_div(data['OriginalUPB'], data['CreditScore'] + 1.0)
        data['InterestRate_x_LTV'] = data['OriginalInterestRate'] * data['OriginalLTV']
        # --- 结束新增 ---
        
        return data

    def _temporal_block_multiwindow(self, df: pd.DataFrame) -> pd.DataFrame:
        # 收集时序特征
        by_type: Dict[str, List[str]] = {}
        for c in self.temporal_cols_all:
            if "_" in c:
                by_type.setdefault(c.split("_",1)[1], []).append(c)
        if not by_type:
            return pd.DataFrame(index=df.index)

        feats = {}
        for ftype, cols in by_type.items():
            cols_sorted = sorted(cols, key=lambda x: int(x.split("_",1)[0]))
            M = df[cols_sorted].to_numpy(float)
            
            # --- 新增：在填充前计算的特征 ---
            with np.errstate(all='ignore'): # 抑制 "All-NaN slice" 警告
                if ftype == "EstimatedLTV":
                    feats[f"{ftype}_max"] = np.nanmax(M, axis=1) # Max_LTV
                    feats[f"{ftype}_std"] = np.nanstd(M, axis=1) # LTV_Volatility
                
                if ftype == "CurrentNonInterestBearingUPB":
                    # 计算 NIB_UPB_Trend (非计息本金余额趋势)
                    slopes = []
                    x_axis = np.arange(M.shape[1])
                    for row in M:
                        valid_mask = ~np.isnan(row)
                        if np.sum(valid_mask) >= 2:
                            slope, _ = np.polyfit(x_axis[valid_mask], row[valid_mask], 1)
                            slopes.append(slope)
                        else:
                            slopes.append(np.nan) # 稍后由全局填充器处理
                    feats[f"{ftype}_slope_full"] = np.array(slopes)
            # --- 结束新增 ---
            
            # 填充缺失值（向前和向后）
            mask = np.isnan(M)
            idx = np.where(~mask, np.arange(M.shape[1]), 0)
            np.maximum.accumulate(idx, axis=1, out=idx)
            M = M[np.arange(M.shape[0])[:,None], idx]
            mask = np.isnan(M)
            idx2 = np.where(~mask, np.arange(M.shape[1]), M.shape[1]-1)
            np.minimum.accumulate(idx2[:, ::-1], axis=1, out=idx2[:, ::-1])
            M = np.nan_to_num(M, nan=0.0)

            # --- 新增：在填充后计算的特征 ---
            if ftype == "EstimatedLTV":
                feats[f"{ftype}_last_val"] = M[:, -1] # 用于 LTV_Change 的辅助列
            # --- 结束新增 ---

            # 多窗口策略
            def select_steps(step_set):
                step_idx = []
                for c in cols_sorted:
                    m = int(c.split("_",1)[0])
                    if m in step_set:
                        step_idx.append(cols_sorted.index(c))
                if not step_idx:
                    return None
                return M[:, step_idx]

            for name, step_set in [
                ("w_main", self.keep_month_idx_main),
                ("w_alt1", self.keep_month_idx_alt1),
                ("w_alt2", self.keep_month_idx_alt2),
            ]:
                A = select_steps(step_set)
                if A is None or A.shape[1] < 2: 
                    continue
                first, last = A[:, 0], A[:, -1]
                trend = safe_div(last - first, np.abs(first) + 1.0) # 趋势
                vol   = safe_div(A.std(axis=1), np.abs(A.mean(axis=1)) + 1.0) # 波动性
                d = pct_change(A)
                d_mean = np.nanmean(d, axis=1) # 一阶差分均值
                d_std  = np.nanstd(d, axis=1) # 一阶差分标准差

                feats[f"{ftype}_{name}_trend"] = trend
                feats[f"{ftype}_{name}_vol"]   = vol
                feats[f"{ftype}_{name}_dmean"] = d_mean
                feats[f"{ftype}_{name}_dstd"]  = d_std

        return pd.DataFrame(feats, index=df.index)

    @staticmethod
    def _annuity_payment_for_state(upb_prev, rt, rem_m):
        # 计算年金支付
        rem_m = np.maximum(rem_m, 1.0)
        ok = (rt > 0) & (rem_m > 0)
        out = np.zeros_like(upb_prev)
        with np.errstate(over="ignore", invalid="ignore"):
            num = rt * np.power(1.0 + rt, rem_m)
            den = np.power(1.0 + rt, rem_m) - 1.0
            P = np.where((den > 0) & ok, upb_prev * (num/den), safe_div(upb_prev, rem_m))
        out = P
        return out

    def _amort_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # 计算摊销信号（Amortization Signals）
        tcols = [c for c in self.temporal_cols_all if c.endswith("_InterestBearingUPB")]
        if not tcols: 
            return pd.DataFrame(index=df.index)
        tcols_sorted = sorted(tcols, key=lambda x: int(x.split("_",1)[0]))
        IB = df[tcols_sorted].to_numpy(float)

        rcols = [c for c in self.temporal_cols_all if c.endswith("_CurrentInterestRate")]
        if rcols:
            rcols_sorted = sorted(rcols, key=lambda x: int(x.split("_",1)[0]))
            RT = df[rcols_sorted].to_numpy(float)/1200.0
        else:
            if "OriginalInterestRate" not in df.columns:
                return pd.DataFrame(index=df.index)
            RT = np.tile((df["OriginalInterestRate"].to_numpy(float)/1200.0)[:,None], (1, IB.shape[1]))

        mcols = [c for c in self.temporal_cols_all if c.endswith("_RemainingMonthsToLegalMaturity")]
        if mcols:
            mcols_sorted = sorted(mcols, key=lambda x: int(x.split("_",1)[0]))
            RM = df[mcols_sorted].to_numpy(float)
        else:
            if "OriginalLoanTerm" not in df.columns:
                return pd.DataFrame(index=df.index)
            T0 = df["OriginalLoanTerm"].to_numpy(float)
            RM = np.maximum(T0[:,None] - np.arange(IB.shape[1])[None,:], 1.0)

        is_frm = df.get("ProductType", pd.Series("FRM", index=df.index)).astype(str).str.upper().eq("FRM")
        is_io  = df.get("InterestOnlyFlag", pd.Series("N", index=df.index)).astype(str).str.upper().eq("Y")
        is_bal = df.get("BalloonIndicator", pd.Series("N", index=df.index)).astype(str).str.upper().eq("Y")
        use_mask = (is_frm & (~is_io) & (~is_bal)).to_numpy()

        n = IB.shape[0]; T = IB.shape[1]
        ExpPrin = np.zeros((n, T))
        for t in range(1, T):
            upb_prev = IB[:, t-1]
            rt = RT[:, t]
            rem = RM[:, t]
            P = self._annuity_payment_for_state(upb_prev, rt, rem)
            exp_prin = np.maximum(P - rt*upb_prev, 0.0)
            ExpPrin[:, t] = exp_prin

        ObsPrin = np.maximum(IB[:, :-1] - IB[:, 1:], 0.0)
        E = ExpPrin[:, 1:]
        S = (E - ObsPrin) / (np.abs(E) + 1e-9)
        S = np.clip(S, 0.0, 1.0)

        # --- 新增：摊销短缺波动性 和 仅付利息计数 ---
        with np.errstate(all='ignore'):
            short_std = np.nanstd(S, axis=1) # Amortization_Shortfall_Std
            
        # InterestOnly_Payment_Count
        # E = ExpPrin[:, 1:], O = ObsPrin
        io_count = np.sum((E > 1e-6) & (ObsPrin < 1e-6), axis=1)
        # --- 结束新增 ---
        
        short_mean = S.mean(axis=1)
        short_70   = (S > 0.70).mean(axis=1)
        short_50   = (S > 0.50).mean(axis=1)

        short_mean = np.where(use_mask, short_mean, 0.0)
        short_70   = np.where(use_mask, short_70, 0.0)
        short_50   = np.where(use_mask, short_50, 0.0)
        short_std = np.where(use_mask, short_std, 0.0) # 应用掩码
        io_count = np.where(use_mask, io_count, 0.0) # 应用掩码

        return pd.DataFrame({
            "amort_short_mean": short_mean,
            "amort_short_70": short_70,
            "amort_short_50": short_50,
            "amort_short_std": short_std, # 新增
            "io_payment_count": io_count, # 新增
            "amort_mask_not_applicable": (~use_mask).astype(int)
        }, index=df.index)

    # ---------- public API ----------
    def fit(self, df_train: pd.DataFrame):
        print("Fitting FeatureBuilderAdvanced...")
        df = self._sentinel_map(df_train.copy())
        self._collect_columns(df)
        self._fit_cat_and_impute(df)

        Xs = self._transform_static(df)
        Xt = self._temporal_block_multiwindow(df)

        # --- 新增：LTV_Change 特征 ---
        # 它必须在 Xs 和 Xt 都被计算之后
        if 'EstimatedLTV_last_val' in Xt.columns and 'OriginalLTV' in Xs.columns:
            Xs['LTV_Change'] = Xt['EstimatedLTV_last_val'] - Xs['OriginalLTV']
            # 确保在impute_vals中为LTV_Change添加一个中位数，以防万一
            self.impute_vals['LTV_Change'] = float(Xs['LTV_Change'].median())
        else:
            Xs['LTV_Change'] = 0.0 # 如果缺少列，则创建
            self.impute_vals['LTV_Change'] = 0.0
        # --- 结束新增 ---

        Xa = self._amort_signals(pd.concat([df, Xt], axis=1))

        # 填充在 _temporal_block_multiwindow 中可能产生的 NaN (例如 slope)
        Xt = Xt.fillna(self.impute_vals)

        X_full = pd.concat([Xs, Xt, Xa], axis=1).astype(float)
        
        # 删除低价值特征
        existing_drop_features = [f for f in self.features_to_drop if f in X_full.columns]
        if existing_drop_features:
            X_full = X_full.drop(columns=existing_drop_features)
            print(f"已删除 {len(existing_drop_features)} 个低价值特征")
            if len(existing_drop_features) < len(self.features_to_drop):
                missing_drop = set(self.features_to_drop) - set(existing_drop_features)
                print(f"注意: {len(missing_drop)} 个要删除的特征不存在于数据中")
        
        # 保存特征列名，确保transform时列顺序一致
        self.feature_names_ = X_full.columns.tolist()
        
        # 根据 scaler_type 选择标准化器
        if self.scaler_type == "robust":
            self.scaler = RobustScaler().fit(X_full.values)
            print("使用 RobustScaler (基于中位数和IQR，对异常值更鲁棒)")
        else:
            self.scaler = StandardScaler().fit(X_full.values)
            print("使用 StandardScaler (基于均值和标准差)")
        
        X_scaled = self.scaler.transform(X_full.values)

        # 记录切片
        start_temporal = Xs.shape[1]
        end_temporal   = start_temporal + Xt.shape[1]
        start_amort    = end_temporal
        end_amort      = start_amort + Xa.shape[1]
        
        # !! 重要：修正切片以包含静态交叉特征
        # 静态特征现在是 0 到 Xs.shape[1]
        self._static_interaction_slice = slice(
            len(self.static_cols), # 静态交叉特征的开始
            Xs.shape[1] # 静态特征的结束
        )
        self._temporal_slice = slice(start_temporal, end_temporal)
        self._amort_slice    = slice(start_amort, end_amort)

        self._fitted = True
        print("Fit complete.")

    def transform(self, df_any: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, slice]]:
        assert self._fitted, "Call fit() first."
        df = self._sentinel_map(df_any.copy())

        Xs = self._transform_static(df)
        Xt = self._temporal_block_multiwindow(df)

        # --- 新增：LTV_Change 特征 (应用) ---
        if 'EstimatedLTV_last_val' in Xt.columns and 'OriginalLTV' in Xs.columns:
            Xs['LTV_Change'] = Xt['EstimatedLTV_last_val'] - Xs['OriginalLTV']
        else:
            Xs['LTV_Change'] = 0.0 # 保持形状一致
        # --- 结束新增 ---

        Xa = self._amort_signals(pd.concat([df, Xt], axis=1))

        # 填充在 _temporal_block_multiwindow 中可能产生的 NaN (例如 slope)
        Xt = Xt.fillna(self.impute_vals)
        # 填充 LTV_Change (如果在fit时计算了中位数)
        if 'LTV_Change' in Xs.columns:
            Xs['LTV_Change'] = Xs['LTV_Change'].fillna(self.impute_vals.get('LTV_Change', 0.0))

        X_full = pd.concat([Xs, Xt, Xa], axis=1).astype(float)
        
        # 删除低价值特征（与 fit 时保持一致）
        existing_drop_features = [f for f in self.features_to_drop if f in X_full.columns]
        if existing_drop_features:
            X_full = X_full.drop(columns=existing_drop_features)
        
        # 确保列顺序与 fit 时一致
        if self.feature_names_ is not None:
            # 确保所有特征都存在
            missing_cols = set(self.feature_names_) - set(X_full.columns)
            if missing_cols:
                # 如果缺少某些列，用0填充
                for col in missing_cols:
                    X_full[col] = 0.0
            X_full = X_full[self.feature_names_]
        
        X_scaled = self.scaler.transform(X_full.values)

        slices = {
            "static": slice(0, Xs.shape[1]),
            "temporal": self._temporal_slice,
            "amort": self._amort_slice,
            "static_interactions": self._static_interaction_slice
        }
        return X_scaled, slices

# ---------- Main execution block ----------
def main():
    """
    主执行函数：加载原始数据，运行特征构建器，并保存输出。
    """
    print("--- 启动高级特征工程 (Feature Engineering) 脚本 (已更新) ---")
    
    # --- 1. 定义路径 ---
    RAW_DATA_PATH = "data/raw_data/"
    OUTPUT_PATH = "data/feature_advanced/"
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"输出目录: {OUTPUT_PATH}")

    # --- 2. 加载原始数据 ---
    print(f"从 {RAW_DATA_PATH} 加载原始数据...")
    try:
        train_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "loans_train.csv"))
        valid_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "loans_valid.csv"))
        test_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "loans_test.csv"))
    except FileNotFoundError:
        print(f"错误：无法在 {RAW_DATA_PATH} 找到原始CSV文件。请检查路径。")
        return
    
    print(f"原始数据: Train={train_df.shape}, Valid={valid_df.shape}, Test={test_df.shape}")

    # --- 3. 初始化并 Fit 特征构建器 ---
    # scaler_type: "robust" (推荐，对异常值更鲁棒)
    fb = FeatureBuilderAdvanced(scaler_type="robust")
    
    # 关键：只在 train_df (正常数据) 上 fit
    fb.fit(train_df)

    # --- 4. Transform 所有数据集 ---
    print("正在转换 训练集...")
    X_train_scaled, _ = fb.transform(train_df)
    
    print("正在转换 验证集...")
    X_valid_scaled, _ = fb.transform(valid_df)
    
    print("正在转换 测试集...")
    X_test_scaled, _ = fb.transform(test_df)

    print(f"\n特征矩阵形状 (Scaled): {X_train_scaled.shape}")

    # --- 5. 保存输出 ---
    print(f"正在保存特征到 {OUTPUT_PATH}...")
    
    # 保存 X_scaled (标准化后的完整特征)
    np.save(os.path.join(OUTPUT_PATH, "train_scaled.npy"), X_train_scaled)
    np.save(os.path.join(OUTPUT_PATH, "valid_scaled.npy"), X_valid_scaled)
    np.save(os.path.join(OUTPUT_PATH, "test_scaled.npy"), X_test_scaled)
    
    # 也保存标签
    np.save(os.path.join(OUTPUT_PATH, "train_labels.npy"), train_df['target'].values)
    np.save(os.path.join(OUTPUT_PATH, "valid_labels.npy"), valid_df['target'].values)
    
    # 保存测试集的ID
    test_ids = test_df["Id"] if "Id" in test_df.columns else test_df["index"]
    np.save(os.path.join(OUTPUT_PATH, "test_ids.npy"), test_ids.values)

    with open(os.path.join(OUTPUT_PATH, "feature_names.txt"), "w") as f:
        for col in fb.feature_names_:
            f.write(col + "\n")
    print(f"已保存特征列名到 {OUTPUT_PATH}/feature_names.txt")
    
    print("--- 特征工程执行完毕 ---")

if __name__ == "__main__":
    main()