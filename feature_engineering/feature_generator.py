import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Set
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
import os
import re
from abc import ABC, abstractmethod

# ---------- utilities ----------
def safe_div(a, b, eps=1e-9):
    return a / (b + eps)

def pct_change(a, axis=1, eps=1e-9):
    return (a[:, 1:] - a[:, :-1]) / (np.abs(a[:, :-1]) + eps)

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

@staticmethod
def _is_temporal(col: str) -> bool:
    if "_" not in col: return False
    left = col.split("_", 1)[0]
    return left.isdigit()

# ---------- 1. 特征构建器基类 (Interface) ----------

class FeatureBuilderBase(ABC):
    """
    所有特征构建器组件的抽象基类 (ABC)。
    定义了 fit 和 transform 接口。
    """
    def __init__(self):
        self.feature_names: List[str] = []
        self.impute_vals: Dict[str, float] = {}
        self._fitted = False

    @abstractmethod
    def fit(self, df: pd.DataFrame, context: Dict[str, pd.DataFrame]) -> 'FeatureBuilderBase':
        """
        学习数据中的统计信息 (例如中位数, 编码器)。
        'context' 可以用来从其他构建器获取依赖项 (例如静态特征)。
        """
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame, context: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        应用转换并生成特征。
        """
        pass
        
    def get_impute_values(self) -> Dict[str, float]:
        """返回此构建器学到的插补值"""
        return self.impute_vals

# ---------- 2. 组件类 (Components) ----------

class StaticFeatureBuilder(FeatureBuilderBase):
    """
    处理所有原始静态特征 (非时序)。
    - 拟合 LabelEncoders
    - 学习插补值 (中位数)
    - 应用编码和插补
    - 创建静态交叉特征
    """
    def __init__(self, static_cols: List[str]):
        super().__init__()
        self.static_cols = static_cols
        self.cat_encoders: Dict[str, LabelEncoder] = {}

    def fit(self, df: pd.DataFrame, context: Dict[str, pd.DataFrame]) -> 'StaticFeatureBuilder':
        # 1. 拟合类别编码器
        for c in self.static_cols:
            if df[c].dtype == "object":
                enc = LabelEncoder()
                vals = df[c].fillna("MISSING").astype(str).unique().tolist()
                if "UNKNOWN" not in vals: vals.append("UNKNOWN")
                enc.fit(vals)
                self.cat_encoders[c] = enc
        
        # 2. 学习数值插补值
        for c in self.static_cols:
            if c not in self.cat_encoders:
                v = df[c].dropna()
                self.impute_vals[c] = float(v.median() if len(v) else 0.0)
        
        # 3. 拟合交叉特征的插补值
        df_transformed = self.transform(df, context)
        for c in df_transformed.columns:
            if c not in self.impute_vals:
                v = df_transformed[c].dropna()
                self.impute_vals[c] = float(v.median() if len(v) else 0.0)

        self.feature_names = df_transformed.columns.tolist()
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, context: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        data = df[self.static_cols].copy()
        
        for c, enc in self.cat_encoders.items():
            x = data[c].fillna("MISSING").astype(str)
            x = x.where(x.isin(enc.classes_), "UNKNOWN")
            data[c] = enc.transform(x)
            
        for c in self.static_cols:
            if c not in self.cat_encoders:
                data[c] = data[c].fillna(self.impute_vals.get(c, 0.0))
        
        data['LTV_x_DTI'] = data['OriginalLTV'] * data['OriginalDTI']
        data['UPB_per_CreditScore'] = safe_div(data['OriginalUPB'], data['CreditScore'] + 1.0)
        data['InterestRate_x_LTV'] = data['OriginalInterestRate'] * data['OriginalLTV']
        
        # --- 新增：借款人结构风险特征 ---
        # Single_Borrower_Flag: 单人贷款标志（NumberOfBorrowers == 1）
        # 单人贷款的违约率通常更高，因为缺乏共同还款人的支持
        if 'NumberOfBorrowers' in data.columns:
            data['Single_Borrower_Flag'] = (data['NumberOfBorrowers'] == 1).astype(int)
        else:
            data['Single_Borrower_Flag'] = 0
        
        if self._fitted:
            for c in data.columns:
                 if c not in self.static_cols:
                    data[c] = data[c].fillna(self.impute_vals.get(c, 0.0))

        return data

class LeverageFeatureBuilder(FeatureBuilderBase):
    """
    杠杆风险 (Leverage Risk) 特征构建器
    - 静态 + 动态 + 波动 + 极端风险
    """
    def __init__(self, temporal_cols_all: List[str]):
        super().__init__()
        self.ltv_cols = sorted(
            [c for c in temporal_cols_all if c.endswith("_EstimatedLTV")],
            key=lambda x: int(x.split("_", 1)[0])
        )
        self.feature_names = [
            "EstimatedLTV_max", "EstimatedLTV_std", "EstimatedLTV_last_val",
            "LTV_Change", "LTV_Change_Rate",
            "LTV_Volatility_Rate",
            "High_LTV_Flag",  # 新增：高杠杆阈值标志
            "Negative_Equity_Flag", "Ever_Negative_Equity", "Negative_Equity_Share"
        ]
        self.impute_vals = {f: 0.0 for f in self.feature_names}

    def _calculate_features(self, df: pd.DataFrame, context: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)
        if not self.ltv_cols or "static_features" not in context:
            for f in self.feature_names:
                feats[f] = 0.0
            return feats

        M = df[self.ltv_cols].to_numpy(float)
        static_feats = context["static_features"]
        orig_ltv = static_feats["OriginalLTV"].to_numpy(float)
        EPS = 1e-9

        with np.errstate(all="ignore"):
            ltv_max = np.nanmax(M, axis=1)
            ltv_std = np.nanstd(M, axis=1)
            ltv_mean = np.nanmean(M, axis=1)
            feats["EstimatedLTV_max"] = ltv_max
            feats["EstimatedLTV_std"] = ltv_std

            mask = np.isnan(M)
            idx = np.where(~mask, np.arange(M.shape[1]), 0)
            np.maximum.accumulate(idx, axis=1, out=idx)
            M_filled = M[np.arange(M.shape[0])[:, None], idx]
            ltv_last = M_filled[:, -1]
            feats["EstimatedLTV_last_val"] = ltv_last

            feats["LTV_Change"] = ltv_last - orig_ltv
            feats["LTV_Change_Rate"] = (ltv_last - orig_ltv) / (np.abs(orig_ltv) + EPS)
            
            # --- 杠杆波动性：资产价值波动敏感度 ---
            feats["LTV_Volatility_Rate"] = ltv_std / (np.abs(ltv_mean) + EPS)
            
            # --- 杠杆极端化风险特征 ---
            # 1. High_LTV_Flag: 高杠杆阈值（OriginalLTV ≥ 95）
            feats["High_LTV_Flag"] = (orig_ltv >= 95).astype(int)
            
            # 2. Negative_Equity_Flag: 房贷倒挂风险（EstimatedLTV_last ≥ 100）
            feats["Negative_Equity_Flag"] = (ltv_last >= 100).astype(int)
            
            # 3. 其他负权益特征
            feats["Ever_Negative_Equity"] = (np.nanmax(M, axis=1) >= 100).astype(int)
            feats["Negative_Equity_Share"] = np.nanmean((M >= 100), axis=1)

        return feats

    def fit(self, df: pd.DataFrame, context: Dict[str, pd.DataFrame]) -> "LeverageFeatureBuilder":
        df_features = self._calculate_features(df, context)
        for c in df_features.columns:
            v = df_features[c].dropna()
            self.impute_vals[c] = float(v.median() if len(v) else 0.0)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, context: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        df_features = self._calculate_features(df, context)
        return df_features.fillna(self.impute_vals)

class DebtServicingFeatureBuilder(FeatureBuilderBase):
    """
    偿债压力 (Debt Servicing) 特征构建器
    - 基于 DTI 阈值风险
    - 基于借款人结构风险
    - 综合压力指数
    """
    def __init__(self):
        super().__init__()
        self.feature_names = [
            "DTI_HighRisk_Flag",
            "DTI_MidRisk_Flag",
            "Borrowers_Risk_Flag",
            "BorrowerCount_Ajusted_DTI",
            "Affordability_Index"
        ]
        self.impute_vals = {f: 0.0 for f in self.feature_names}

    def _calculate_features(self, df: pd.DataFrame, context: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)
        
        if "static_features" not in context:
            for f in self.feature_names:
                feats[f] = 0.0
            return feats

        static_feats = context["static_features"]
        
        dti = static_feats["OriginalDTI"]
        ltv = static_feats["OriginalLTV"]
        n_borrowers = static_feats["NumberOfBorrowers"]

        # 1. 静态阈值风险 (Threshold-based Risk)
        feats["DTI_HighRisk_Flag"] = (dti > 43).astype(int)
        feats["DTI_MidRisk_Flag"] = ((dti > 36) & (dti <= 43)).astype(int)

        # 2. 借款结构风险 (Household Composition)
        feats["Borrowers_Risk_Flag"] = (n_borrowers == 1).astype(int)
        safe_n_borrowers = np.maximum(n_borrowers, 1.0)
        feats["BorrowerCount_Ajusted_DTI"] = dti / np.sqrt(safe_n_borrowers)

        # 3. 综合压力指数 (Composite Affordability Index)
        feats["Affordability_Index"] = (dti / 43.0) * (ltv / 80.0)
        
        return feats

    def fit(self, df: pd.DataFrame, context: Dict[str, pd.DataFrame]) -> "DebtServicingFeatureBuilder":
        df_features = self._calculate_features(df, context)
        for c in df_features.columns:
            v = df_features[c].dropna()
            self.impute_vals[c] = float(v.median() if len(v) else 0.0)
        
        self.feature_names = df_features.columns.tolist() # 确保顺序
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, context: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        df_features = self._calculate_features(df, context)
        return df_features.fillna(self.impute_vals)

class BalanceFeatureBuilder(FeatureBuilderBase):
    """
    处理余额 (Balance) 相关的特定时序特征。
    - Non-Interest-Bearing UPB (NIB_UPB) 的趋势
    """
    def __init__(self, temporal_cols_all: List[str]):
        super().__init__()
        self.nib_cols = sorted(
            [c for c in temporal_cols_all if c.endswith("_CurrentNonInterestBearingUPB")],
            key=lambda x: int(x.split("_",1)[0])
        )
        self.feature_names = ["CurrentNonInterestBearingUPB_slope_full"]
        self.impute_vals = {f: 0.0 for f in self.feature_names}
        
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)

        if not self.nib_cols:
            feats[self.feature_names[0]] = 0.0
            return feats

        M = df[self.nib_cols].to_numpy(float)
        slopes = []
        x_axis = np.arange(M.shape[1])
        
        with np.errstate(all='ignore'):
            for row in M:
                valid_mask = ~np.isnan(row)
                if np.sum(valid_mask) >= 2:
                    slope, _ = np.polyfit(x_axis[valid_mask], row[valid_mask], 1)
                    slopes.append(slope)
                else:
                    slopes.append(np.nan) 
        
        feats["CurrentNonInterestBearingUPB_slope_full"] = np.array(slopes)
        return feats

    def fit(self, df: pd.DataFrame, context: Dict[str, pd.DataFrame]) -> 'BalanceFeatureBuilder':
        df_features = self._calculate_features(df)
        v = df_features["CurrentNonInterestBearingUPB_slope_full"].dropna()
        self.impute_vals["CurrentNonInterestBearingUPB_slope_full"] = float(v.median() if len(v) else 0.0)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, context: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        df_features = self._calculate_features(df)
        return df_features.fillna(self.impute_vals)


class MaturityRiskFeatureBuilder(FeatureBuilderBase):
    """
    处理到期风险 (Maturity Risk) 特征构建器
    - 到期压力指数：衡量剩余本金与剩余期限的关系
    """
    def __init__(self, temporal_cols_all: List[str]):
        super().__init__()
        self.upb_cols = sorted(
            [c for c in temporal_cols_all if c.endswith("_CurrentActualUPB")],
            key=lambda x: int(x.split("_",1)[0])
        )
        self.rem_cols = sorted(
            [c for c in temporal_cols_all if c.endswith("_RemainingMonthsToLegalMaturity")],
            key=lambda x: int(x.split("_",1)[0])
        )
        self.feature_names = ["MaturityPressure_Index"]
        self.impute_vals = {f: 0.0 for f in self.feature_names}
        
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)
        
        if not self.upb_cols or not self.rem_cols:
            feats["MaturityPressure_Index"] = 0.0
            return feats
        
        # 获取 CurrentActualUPB 的最后一个值
        UPB_M = df[self.upb_cols].to_numpy(float)
        mask_upb = np.isnan(UPB_M)
        idx_upb = np.where(~mask_upb, np.arange(UPB_M.shape[1]), 0)
        np.maximum.accumulate(idx_upb, axis=1, out=idx_upb)
        UPB_filled = UPB_M[np.arange(UPB_M.shape[0])[:, None], idx_upb]
        upb_last = UPB_filled[:, -1]
        
        # 获取 RemainingMonthsToLegalMaturity 的最后一个值
        REM_M = df[self.rem_cols].to_numpy(float)
        mask_rem = np.isnan(REM_M)
        idx_rem = np.where(~mask_rem, np.arange(REM_M.shape[1]), 0)
        np.maximum.accumulate(idx_rem, axis=1, out=idx_rem)
        REM_filled = REM_M[np.arange(REM_M.shape[0])[:, None], idx_rem]
        rem_last = REM_filled[:, -1]
        
        # 计算到期压力指数
        with np.errstate(all='ignore'):
            # MaturityPressure_Index = CurrentActualUPB_last / (RemainingMonthsToLegalMaturity_last + 1)
            maturity_pressure = upb_last / (np.abs(rem_last) + 1.0)
            feats["MaturityPressure_Index"] = maturity_pressure
        
        return feats

    def fit(self, df: pd.DataFrame, context: Dict[str, pd.DataFrame]) -> 'MaturityRiskFeatureBuilder':
        df_features = self._calculate_features(df)
        v = df_features["MaturityPressure_Index"].dropna()
        self.impute_vals["MaturityPressure_Index"] = float(v.median() if len(v) else 0.0)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, context: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        df_features = self._calculate_features(df)
        return df_features.fillna(self.impute_vals)


class AmortizationFeatureBuilder(FeatureBuilderBase):
    """
    处理摊销 (Amortization) 信号。
    - 预期本金 vs 实际本金支付
    - short_mean, short_std, io_payment_count 等
    """
    def __init__(self, temporal_cols_all: List[str]):
        super().__init__()
        self.temporal_cols_all = temporal_cols_all
        self.feature_names = [
            "amort_short_mean", "amort_short_70", "amort_short_50",
            "amort_short_std", "io_payment_count", "amort_mask_not_applicable",
            # 新增：摊销行为延伸特征
            "Cumulative_Shortfall", "Zero_Payment_Streak", "Shortfall_Volatility"
        ]
        self.impute_vals = {f: 0.0 for f in self.feature_names}

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        tcols = [c for c in self.temporal_cols_all if c.endswith("_InterestBearingUPB")]
        if not tcols: return pd.DataFrame({f: 0.0 for f in self.feature_names}, index=df.index)
        tcols_sorted = sorted(tcols, key=lambda x: int(x.split("_",1)[0]))
        IB = df[tcols_sorted].to_numpy(float)

        rcols = [c for c in self.temporal_cols_all if c.endswith("_CurrentInterestRate")]
        if rcols:
            rcols_sorted = sorted(rcols, key=lambda x: int(x.split("_",1)[0]))
            RT = df[rcols_sorted].to_numpy(float)/1200.0
        else:
            if "OriginalInterestRate" not in df.columns:
                return pd.DataFrame({f: 0.0 for f in self.feature_names}, index=df.index)
            RT = np.tile((df["OriginalInterestRate"].to_numpy(float)/1200.0)[:,None], (1, IB.shape[1]))

        mcols = [c for c in self.temporal_cols_all if c.endswith("_RemainingMonthsToLegalMaturity")]
        if mcols:
            mcols_sorted = sorted(mcols, key=lambda x: int(x.split("_",1)[0]))
            RM = df[mcols_sorted].to_numpy(float)
        else:
            if "OriginalLoanTerm" not in df.columns:
                return pd.DataFrame({f: 0.0 for f in self.feature_names}, index=df.index)
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
            P = _annuity_payment_for_state(upb_prev, rt, rem)
            exp_prin = np.maximum(P - rt*upb_prev, 0.0)
            ExpPrin[:, t] = exp_prin

        ObsPrin = np.maximum(IB[:, :-1] - IB[:, 1:], 0.0)
        E = ExpPrin[:, 1:]
        S = (E - ObsPrin) / (np.abs(E) + 1e-9)
        S = np.clip(S, 0.0, 1.0) 

        with np.errstate(all='ignore'):
            short_std = np.nanstd(S, axis=1)
            io_count = np.sum((E > 1e-6) & (ObsPrin < 1e-6), axis=1)
        
        short_mean = S.mean(axis=1)
        short_70   = (S > 0.70).mean(axis=1)
        short_50   = (S > 0.50).mean(axis=1)

        # --- 新增：摊销行为延伸特征 ---
        # 1. Cumulative_Shortfall: 累积短缺（衡量长期还款偏离程度）
        # 计算每个时间点的累积短缺比例总和
        cumulative_shortfall = np.nansum(S, axis=1)  # 对所有时间点求和
        
        # 2. Zero_Payment_Streak: 零支付连续月数（断供前兆）
        # 从后往前找，找出最近的连续零支付月数（更关注最近的支付行为）
        zero_payment_streak = np.zeros(n)
        for i in range(n):
            if use_mask[i]:
                obs_prin_row = ObsPrin[i, :]
                # 从后往前找连续零支付
                current_streak = 0
                for j in range(obs_prin_row.shape[0] - 1, -1, -1):
                    if obs_prin_row[j] < 1e-6:  # 接近零支付
                        current_streak += 1
                    else:
                        break  # 遇到非零支付就停止
                zero_payment_streak[i] = current_streak
        
        # 3. Shortfall_Volatility: 短缺波动性（捕捉还款不稳定性）
        # 使用变异系数（CV = std/mean）来标准化波动性，使其对不同水平的短缺都可比
        shortfall_mean_abs = np.abs(short_mean) + 1e-9
        shortfall_volatility = np.where(shortfall_mean_abs > 1e-6, short_std / shortfall_mean_abs, 0.0)
        # --- 结束新增 ---

        feats = pd.DataFrame({
            "amort_short_mean": np.where(use_mask, short_mean, 0.0),
            "amort_short_70": np.where(use_mask, short_70, 0.0),
            "amort_short_50": np.where(use_mask, short_50, 0.0),
            "amort_short_std": np.where(use_mask, short_std, 0.0),
            "io_payment_count": np.where(use_mask, io_count, 0.0),
            "amort_mask_not_applicable": (~use_mask).astype(int),
            # 新增特征
            "Cumulative_Shortfall": np.where(use_mask, cumulative_shortfall, 0.0),
            "Zero_Payment_Streak": np.where(use_mask, zero_payment_streak, 0.0),
            "Shortfall_Volatility": np.where(use_mask, shortfall_volatility, 0.0)
        }, index=df.index)
        
        return feats

    def fit(self, df: pd.DataFrame, context: Dict[str, pd.DataFrame]) -> 'AmortizationFeatureBuilder':
        df_features = self._calculate_features(df)
        for c in df_features.columns:
            v = df_features[c].dropna()
            self.impute_vals[c] = float(v.median() if len(v) else 0.0)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, context: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        df_features = self._calculate_features(df)
        return df_features.fillna(self.impute_vals)



class GenericTemporalBuilder(FeatureBuilderBase):
    """
    处理所有*剩余*的时序特征。
    - 计算通用的多窗口统计 (trend, vol, dmean, dstd)
    - 忽略已由其他构建器处理的特征。
    """
    def __init__(
        self,
        temporal_cols_all: List[str],
        ignore_col_suffixes: Set[str], # e.g., {"_EstimatedLTV", "_InterestBearingUPB", ...}
        keep_month_idx_main=(0,3,6,9,12),
        keep_month_idx_alt1=(0,2,4,6,8,10,12),
        keep_month_idx_alt2=(0,3,6,9),
    ):
        super().__init__()
        self.keep_month_idx_main = set(keep_month_idx_main)
        self.keep_month_idx_alt1 = set(keep_month_idx_alt1)
        self.keep_month_idx_alt2 = set(keep_month_idx_alt2)
        
        self.by_type: Dict[str, List[str]] = {}
        for c in temporal_cols_all:
            if "_" in c:
                suffix = "_" + c.split("_",1)[1]
                if suffix not in ignore_col_suffixes:
                    self.by_type.setdefault(suffix, []).append(c)
        
        print(f"[GenericTemporalBuilder] 正在处理 {len(self.by_type)} 个特征组。")

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.by_type:
            return pd.DataFrame(index=df.index)
            
        feats = {}
        for ftype_suffix, cols in self.by_type.items():
            ftype = ftype_suffix.lstrip('_') # e.g., "LoanAge"
            cols_sorted = sorted(cols, key=lambda x: int(x.split("_",1)[0]))
            M = df[cols_sorted].to_numpy(float)
            
            mask = np.isnan(M)
            idx = np.where(~mask, np.arange(M.shape[1]), 0)
            np.maximum.accumulate(idx, axis=1, out=idx)
            M = M[np.arange(M.shape[0])[:,None], idx]
            mask = np.isnan(M)
            idx2 = np.where(~mask, np.arange(M.shape[1]), M.shape[1]-1)
            np.minimum.accumulate(idx2[:, ::-1], axis=1, out=idx2[:, ::-1])
            M = np.nan_to_num(M, nan=0.0)

            def select_steps(step_set):
                step_idx = []
                for c in cols_sorted:
                    m = int(c.split("_",1)[0])
                    if m in step_set:
                        step_idx.append(cols_sorted.index(c))
                if not step_idx: return None
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
                trend = safe_div(last - first, np.abs(first) + 1.0)
                vol   = safe_div(A.std(axis=1), np.abs(A.mean(axis=1)) + 1.0)
                d = pct_change(A)
                d_mean = np.nanmean(d, axis=1)
                d_std  = np.nanstd(d, axis=1)

                feats[f"{ftype}_{name}_trend"] = trend
                feats[f"{ftype}_{name}_vol"]   = vol
                feats[f"{ftype}_{name}_dmean"] = d_mean
                feats[f"{ftype}_{name}_dstd"]  = d_std
        
        return pd.DataFrame(feats, index=df.index)

    def fit(self, df: pd.DataFrame, context: Dict[str, pd.DataFrame]) -> 'GenericTemporalBuilder':
        df_features = self._calculate_features(df)
        for c in df_features.columns:
            v = df_features[c].dropna()
            self.impute_vals[c] = float(v.median() if len(v) else 0.0)
            
        self.feature_names = df_features.columns.tolist()
        self._fitted = True
        return self
        
    def transform(self, df: pd.DataFrame, context: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        df_features = self._calculate_features(df)
        return df_features.fillna(self.impute_vals)


# ---------- 3. 主管道 (Orchestrator) ----------

class LoanFeaturePipeline:
    """
    主特征工程管道。
    - 拥有并按顺序运行所有 FeatureBuilderBase 组件。
    - 处理全局哨兵映射。
    - 处理全局特征删除。
    - 拟合和应用全局缩放器 (Scaler)。
    """
    def __init__(
        self,
        builders: List[FeatureBuilderBase],
        features_to_drop: List[str],
        scaler_type: str = "robust"
    ):
        self.builders = builders
        self.features_to_drop = features_to_drop
        self.scaler_type = scaler_type.lower()
        
        self.scaler: Optional[Union[StandardScaler, RobustScaler]] = None
        self.global_impute_vals: Dict[str, float] = {}
        self.feature_names_: List[str] = []
        self._fitted = False

    @staticmethod
    def _sentinel_map(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        SENT = {
            "CreditScore": 9999, "OriginalDTI": 999, "OriginalLTV": 999,
            "MI Pct": 999, "EstimatedLTV": 999,
        }
        for col, bad in SENT.items():
            if col in df.columns:
                miss = (df[col] == bad)
                df.loc[miss, col] = np.nan
                df[f"{col}_missing"] = miss.astype(int) 

        FLAG_9 = ["FirstTimeHomebuyerFlag","OccupancyStatus","LoanPurpose","Channel","PropertyType"]
        for col in FLAG_9:
            if col in df.columns:
                x = df[col].astype(str).str.strip()
                miss = (x == "9") | (x == "99")
                df.loc[miss, col] = np.nan
                df[f"{col}_missing"] = miss.astype(int) 

        return df

    def fit(self, df_train: pd.DataFrame):
        print("Fitting LoanFeaturePipeline...")
        df = self._sentinel_map(df_train.copy())
        
        context: Dict[str, pd.DataFrame] = {}
        all_features_list: List[pd.DataFrame] = []
        
        for builder in self.builders:
            print(f"Fitting {builder.__class__.__name__}...")
            builder.fit(df, context)
            X_builder = builder.transform(df, context)
            
            all_features_list.append(X_builder)
            self.global_impute_vals.update(builder.get_impute_values())
            
            if isinstance(builder, StaticFeatureBuilder):
                context['static_features'] = X_builder
        
        X_full = pd.concat(all_features_list, axis=1).astype(float)
        X_full = X_full.fillna(self.global_impute_vals) 

        existing_drop = [f for f in self.features_to_drop if f in X_full.columns]
        X_full = X_full.drop(columns=existing_drop)
        print(f"Dropped {len(existing_drop)} features.")

        self.feature_names_ = X_full.columns.tolist()
        
        if self.scaler_type == "robust":
            self.scaler = RobustScaler().fit(X_full.values)
            print("Using RobustScaler.")
        else:
            self.scaler = StandardScaler().fit(X_full.values)
            print("Using StandardScaler.")
            
        self._fitted = True
        print("Pipeline fit complete.")
        return self

    def transform(self, df_any: pd.DataFrame) -> np.ndarray:
        assert self.scaler is not None, "Call fit() first."
        
        df = self._sentinel_map(df_any.copy())
        
        context: Dict[str, pd.DataFrame] = {}
        all_features_list: List[pd.DataFrame] = []

        for builder in self.builders:
            X_builder = builder.transform(df, context)
            all_features_list.append(X_builder)
            
            if isinstance(builder, StaticFeatureBuilder):
                context['static_features'] = X_builder
                
        X_full = pd.concat(all_features_list, axis=1).astype(float)
        X_full = X_full.fillna(self.global_impute_vals)

        missing_cols = set(self.feature_names_) - set(X_full.columns)
        for col in missing_cols:
            X_full[col] = 0.0 
            
        extra_cols = set(X_full.columns) - set(self.feature_names_)
        if extra_cols:
            cols_to_drop = [f for f in extra_cols if f not in self.features_to_drop]
            X_full = X_full.drop(columns=list(cols_to_drop))

        X_full = X_full[self.feature_names_]
        
        X_scaled = self.scaler.transform(X_full.values)
        return X_scaled

# ---------- 4. Main execution block (更新后) ----------

def main():
    """
    主执行函数：加载原始数据，运行特征管道，并保存输出。
    """
    print("--- 启动 高级特征工程 (重构版) ---")
    
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

    # --- 3. 初始化管道和组件 ---
    
    static_cols = []
    temporal_cols_all = []
    for c in train_df.columns:
        if c in ("index","Id","target"): continue
        if _is_temporal(c):
            temporal_cols_all.append(c)
        else:
            static_cols.append(c)

    specialized_temporal_suffixes = {
        "_EstimatedLTV",
        "_CurrentNonInterestBearingUPB",
        "_InterestBearingUPB",
        "_CurrentInterestRate",
        "_RemainingMonthsToLegalMaturity",
    }
    
    # ==================================================================
    # === 使用你指定的 'features_to_drop' 列表 ===
    # ==================================================================
    features_to_drop = [
        # ---- 全部 Missing Flag ----
        "CreditScore_missing", "OriginalDTI_missing", "OriginalLTV_missing",
        "FirstTimeHomebuyerFlag_missing", "OccupancyStatus_missing",
        "LoanPurpose_missing", "Channel_missing", "PropertyType_missing",
        
        # ---- 明显噪声类 (来自旧代码) ----
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
        
        # ---- 其他无效或平坦列 ----
        "amort_mask_not_applicable",
        "EstimatedLTV_last_val", # 只是 'LTV_Change' 的辅助列
        
        # ---- 新增：事件特征 ----
        # 杠杆特征
        "Negative_Equity_Flag",
        "LTV_Change_Rate",
        "EstimatedLTV_std",
        "LTV_Volatility_Rate",
        "EstimatedLTV_max",
        "Ever_Negative_Equity",
        "Negative_Equity_Share",
        "EstimatedLTV_last_val" # (这个是重复的, 但不影响)
        
        # 偿债风险
        # "DTI_MidRisk_Flag",
        "Borrowers_Risk_Flag",
        
        # 摊销
        "Cumulative_Shortfall",
        "Shortfall_Volatility",
        # "Zero_Payment_Streak", -->有用

    ]


    # 按顺序定义构建器
    builders: List[FeatureBuilderBase] = [
        StaticFeatureBuilder(static_cols=static_cols),
        
        # --- 新增 ---
        DebtServicingFeatureBuilder(),
        
        LeverageFeatureBuilder(temporal_cols_all=temporal_cols_all),
        AmortizationFeatureBuilder(temporal_cols_all=temporal_cols_all),
        BalanceFeatureBuilder(temporal_cols_all=temporal_cols_all),
        MaturityRiskFeatureBuilder(temporal_cols_all=temporal_cols_all),  # 新增：到期风险特征
        GenericTemporalBuilder(
            temporal_cols_all=temporal_cols_all,
            ignore_col_suffixes=specialized_temporal_suffixes
        ),
    ]

    # 初始化主管道
    pipeline = LoanFeaturePipeline(
        builders=builders,
        features_to_drop=features_to_drop,
        scaler_type="robust"
    )
    
    # --- 4. 拟合 (Fit) 管道 ---
    pipeline.fit(train_df)

    # --- 5. 转换 (Transform) 所有数据集 ---
    print("正在转换 训练集...")
    X_train_scaled = pipeline.transform(train_df)
    
    print("正在转换 验证集...")
    X_valid_scaled = pipeline.transform(valid_df)
    
    print("正在转换 测试集...")
    X_test_scaled = pipeline.transform(test_df)

    print(f"\n特征矩阵形状 (Scaled): {X_train_scaled.shape}")

    # --- 6. 保存输出 ---
    print(f"正在保存特征到 {OUTPUT_PATH}...")
    
    np.save(os.path.join(OUTPUT_PATH, "train_scaled.npy"), X_train_scaled)
    np.save(os.path.join(OUTPUT_PATH, "valid_scaled.npy"), X_valid_scaled)
    np.save(os.path.join(OUTPUT_PATH, "test_scaled.npy"), X_test_scaled)
    
    np.save(os.path.join(OUTPUT_PATH, "train_labels.npy"), train_df['target'].values)
    np.save(os.path.join(OUTPUT_PATH, "valid_labels.npy"), valid_df['target'].values)
    
    test_ids = test_df["Id"] if "Id" in test_df.columns else test_df["index"]
    np.save(os.path.join(OUTPUT_PATH, "test_ids.npy"), test_ids.values)

    with open(os.path.join(OUTPUT_PATH, "feature_names.txt"), "w") as f:
        for col in pipeline.feature_names_:
            f.write(col + "\n")
    print(f"已保存特征列名到 {OUTPUT_PATH}/feature_names.txt")
    
    print("--- 特征工程执行完毕 ---")


if __name__ == "__main__":
    main()