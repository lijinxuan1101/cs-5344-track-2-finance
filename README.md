# CS5344 Track 2: Finance - Loan Default Prediction

A comprehensive machine learning project for predicting loan defaults using anomaly detection techniques on Freddie Mac single-family loan data.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Description](#data-description)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Results](#results)
- [Documentation](#documentation)
- [Contributors](#contributors)

## 🎯 Project Overview

This project implements an anomaly detection system to identify high-risk loans in the Freddie Mac single-family loan dataset. The system uses advanced feature engineering and ensemble methods to achieve state-of-the-art performance.

### Key Features

- **68 engineered features** from raw loan data
- **Modular feature engineering pipeline** with 7 specialized builders
- **Multiple baseline models** (Isolation Forest, LOF, One-Class SVM, etc.)
- **Fusion ensemble model** combining 4 specialized detectors
- **Comprehensive evaluation** with AUPRC and AUROC metrics

### Performance

- **Best Single Feature**: `Early_Delinquency_Flag` (AUPRC: 0.4999, AUROC: 0.8021)
- **Fusion Ensemble Model**: AUPRC: 0.5820, AUROC: 0.8203 (on validation set)

## 📁 Project Structure

```
cs-5344-track-2-finance/
├── data/
│   ├── raw_data/              # Original loan data files
│   │   ├── loans_train.csv
│   │   ├── loans_valid.csv
│   │   └── loans_test.csv
│   ├── feature_advanced/      # Processed features
│   │   ├── train_scaled.npy
│   │   ├── valid_scaled.npy
│   │   ├── test_scaled.npy
│   │   └── feature_names.txt
│   └── submission/            # Submission files
│
├── feature_engineering/       # Feature engineering scripts
│   ├── feature_generator.py  # Main feature engineering pipeline
│   ├── single_feature_analysis.py
│   └── feature_group_evaluation.py
│
├── baseline_models/           # Baseline model evaluation
│   └── results/              # Baseline results
│
├── final_models/             # Final production models
│   ├── run_fusion_ensemble.py
│   ├── optimization_run_fusion_ensemble.py
│   ├── results/              # Model evaluation results
│   └── submission/           # Final submission files
│
├── feature_tests/            # Feature analysis results
│   └── single_feature_results.csv
│
├── model.py                  # Simple baseline model for submission
├── baseline.py               # Baseline model evaluation script
├── Bayesian Optimization.py # Hyperparameter optimization
│
├── Column_Documentation.md   # Column descriptions (Chinese)
├── Feature_Documentation.md  # Feature engineering documentation (Chinese)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Setup

1. **Clone the repository** (if applicable):
```bash
git clone <repository-url>
cd cs-5344-track-2-finance
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Required Packages

- `pandas>=1.5.0,<2.0`
- `numpy>=1.23.0,<2.0`
- `scikit-learn>=1.1.0,<2.0`
- `scipy>=1.9.0,<2.0`
- `matplotlib>=3.5.0,<4.0`
- `seaborn>=0.12.0,<1.0`
- `joblib>=1.2.0`

## 📊 Data Description

### Dataset

The project uses **Freddie Mac Single-Family Loan Dataset**, which contains:

- **Origination Variables**: Static loan characteristics at origination (credit score, DTI, LTV, etc.)
- **Performance Panel Variables**: Monthly performance data over time (UPB, interest rates, etc.)

### Data Files

- `loans_train.csv`: Training set with labels
- `loans_valid.csv`: Validation set with labels
- `loans_test.csv`: Test set (no labels)

### Target Variable

- `0` = Normal loan (no default)
- `1` = Anomalous loan (default or anomalous event)

For detailed column descriptions, see `Column_Documentation.md`.

## 🔨 Feature Engineering

The feature engineering pipeline generates **68 features** organized into 7 categories:

### Feature Categories

1. **Static Features** (36 features, indices 0-35)
   - Original loan characteristics
   - Interaction features (LTV × DTI, etc.)
   - Borrower risk flags

2. **Debt Servicing Features** (5 features, indices 36-40)
   - DTI risk flags
   - Affordability indices

3. **Leverage Risk Features** (2 features, indices 41-42)
   - LTV changes
   - High leverage flags

4. **Amortization Signals** (10 features, indices 43-52)
   - **Top performers**: `Early_Delinquency_Flag` (AUPRC: 0.4999), `amort_short_mean` (AUPRC: 0.3713)
   - Shortfall ratios, payment streaks, early delinquency indicators

5. **Composite Features** (1 feature, index 53)
   - Balance trend × amortization shortfall

6. **Temporal Features** (13 features, indices 54-67)
   - CurrentActualUPB window statistics
   - Non-interest-bearing UPB trends

7. **Maturity Risk Features** (1 feature, index 55)
   - Maturity pressure index

### Feature Processing Pipeline

1. **Sentinel Value Handling**: Convert special codes (999, 9999) to NaN
2. **Categorical Encoding**: LabelEncoder for categorical variables
3. **Missing Value Imputation**: Median imputation for numerical features
4. **Feature Generation**: Modular builders generate engineered features
5. **Feature Selection**: Remove low-value features
6. **Scaling**: RobustScaler (robust to outliers)

For detailed feature documentation, see `Feature_Documentation.md`.

## 🤖 Model Architecture

### Baseline Models

The project evaluates multiple baseline models:

- **Isolation Forest**: Tree-based anomaly detection
- **Local Outlier Factor (LOF)**: Density-based detection
- **One-Class SVM**: Support vector machine for anomaly detection
- **Elliptic Envelope**: Gaussian distribution-based detection

**Best Baseline**: LOF (k=15, metric='manhattan') - AUPRC: ~0.35

### Final Fusion Ensemble Model

The final model combines 4 specialized detectors:

1. **Early_Delinquency_Flag Detector** (Weight: 40%)
   - AUPRC: 0.4999, AUROC: 0.8021
   - Captures early payment problems

2. **amort_short_mean Detector** (Weight: 30%)
   - AUPRC: 0.3713, AUROC: 0.6783
   - Captures long-term payment shortfalls

3. **Zero_Payment_Streak Detector** (Weight: 10%)
   - AUPRC: 0.2693, AUROC: 0.6029
   - Captures consecutive zero-payment periods

4. **LOF(k=50) Detector** (Weight: 20%)
   - Global anomaly pattern detection

**Fusion Performance**: AUPRC: 0.5820, AUROC: 0.8203

## 🚀 Usage

### Step 1: Feature Engineering

Generate features from raw data:

```bash
python feature_engineering/feature_generator.py
```

This will create:
- `data/feature_advanced/train_scaled.npy`
- `data/feature_advanced/valid_scaled.npy`
- `data/feature_advanced/test_scaled.npy`
- `data/feature_advanced/feature_names.txt`

### Step 2: Baseline Model Evaluation

Evaluate baseline models on validation set:

```bash
python baseline.py
```

Results will be saved to `baseline_models/results/baseline_results.csv`

### Step 3: Single Feature Analysis

Analyze individual feature performance:

```bash
python feature_engineering/single_feature_analysis.py
```

Results: `feature_tests/single_feature_results.csv`

### Step 4: Train Final Fusion Model

Train and evaluate the fusion ensemble model:

```bash
python final_models/run_fusion_ensemble.py
```

This generates:
- Validation metrics: `final_models/results/fusion_metrics.csv`
- Test predictions: `final_models/submission/submission.csv`

### Step 5: Generate Simple Submission

For a quick baseline submission:

```bash
python model.py
```

Output: `data/submission/submission_simple_model_lof.csv`

### Hyperparameter Optimization

Optimize fusion ensemble weights:

```bash
python final_models/optimization_run_fusion_ensemble.py
```

## 📈 Results

### Top 15 Features (by AUPRC)

| Rank | Feature | Index | AUPRC | AUROC | Category |
|------|---------|-------|-------|-------|----------|
| 1 | Early_Delinquency_Flag | 50 | 0.4999 | 0.8021 | Amortization |
| 2 | amort_short_mean | 43 | 0.3713 | 0.6783 | Amortization |
| 3 | Cumulative_Shortfall | 48 | 0.3713 | 0.6783 | Amortization |
| 4 | amort_short_50 | 45 | 0.3324 | 0.6362 | Amortization |
| 5 | amort_short_70 | 44 | 0.3278 | 0.6317 | Amortization |
| 6 | io_payment_count | 47 | 0.3198 | 0.6246 | Amortization |
| 7 | Zero_Payment_Streak | 49 | 0.2693 | 0.6029 | Amortization |
| 8 | CreditScore | 0 | 0.2106 | 0.6446 | Static |
| 9 | Amortization_Lag | 51 | 0.1534 | 0.5400 | Amortization |
| 10 | DTI_HighRisk_Flag | 36 | 0.1474 | 0.5712 | Debt servicing |

### Model Performance Summary

| Model | AUPRC | AUROC | Notes |
|-------|-------|-------|-------|
| **Fusion Ensemble** | **0.5820** | **0.8203** | Final model (4 detectors) |
| Early_Delinquency_Flag (single) | 0.4999 | 0.8021 | Best single feature |
| amort_short_mean (single) | 0.3713 | 0.6783 | Second best feature |
| LOF (k=15, manhattan) | ~0.35 | ~0.65 | Best baseline |

## 📚 Documentation

- **Column_Documentation.md**: Detailed description of all dataset columns
- **Feature_Documentation.md**: Complete feature engineering documentation (Chinese)
- **CS5344_Formal_Problem_Formulation.pdf**: Formal problem statement

## 🔍 Key Insights

### Most Important Features

1. **Early_Delinquency_Flag**: The single most powerful feature, capturing early payment problems
2. **Amortization Signals**: Payment shortfall features are highly predictive
3. **Credit Score**: Traditional credit metrics remain important
4. **DTI Risk Flags**: Debt-to-income ratios are strong risk indicators

### Model Design Decisions

1. **Anomaly Detection Approach**: Only train on normal samples (target==0)
2. **Robust Scaling**: Use RobustScaler instead of StandardScaler for outlier robustness
3. **Ensemble Fusion**: Combine specialized detectors rather than using a single model
4. **Feature Selection**: Remove low-value features (AUPRC < 0.13)

## 🛠️ Development

### Running Tests

```bash
# Feature engineering tests
python feature_engineering/single_feature_analysis.py

# Model evaluation
python baseline.py
```

### Code Structure

- **Modular Design**: Feature builders follow a consistent interface
- **Separation of Concerns**: Feature engineering, model training, and evaluation are separate
- **Reproducibility**: All random seeds are fixed

## 📝 Notes

- All models are trained **only on normal samples** (target==0) as per project requirements
- Features are scaled using **RobustScaler** for robustness to outliers
- The fusion model uses **weighted combination** of normalized detector scores
- Final predictions are scaled to [0, 1] range for submission

## 👥 Contributors

- Project developed for CS5344 Big-Data Analytics Technology course
- Track 2: Finance - Loan Default Prediction

## 📄 License

This project is for educational purposes as part of the CS5344 course.

## 🙏 Acknowledgments

- Freddie Mac for providing the loan dataset
- NUS CS5344 course instructors
- scikit-learn community for excellent ML tools

---

**Last Updated**: 2024

For questions or issues, please refer to the documentation files or contact the project maintainers.

