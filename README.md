# Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

> End-to-end fraud detection pipeline on the Kaggle Credit Card Fraud dataset — 284,807 transactions, 0.172% fraud rate.

---

## Problem

Detecting fraudulent transactions in a **highly imbalanced dataset** (fraud = 0.172%) where false negatives (missed fraud) carry far greater cost than false positives.

---

## Approach

| Step | Technique |
|---|---|
| Class imbalance | SMOTE oversampling on training split only |
| Feature scaling | StandardScaler (preserves PCA-transformed features V1–V28) |
| Baseline model | Logistic Regression |
| Primary model | XGBoost Classifier |
| Evaluation | Precision-Recall AUC (not accuracy — misleading on imbalanced data) |

---

## Results

| Model | PR-AUC | Recall (fraud) | Precision (fraud) |
|---|---|---|---|
| Logistic Regression | 0.71 | 0.88 | 0.06 |
| XGBoost | 0.87 | 0.86 | 0.74 |

XGBoost significantly reduces false positives while maintaining recall.

---

## Technical Stack

- **Language:** Python 3.10
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn, Matplotlib, Seaborn
- **Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## Key Learnings

- Using accuracy as a metric on imbalanced data is misleading — always report PR-AUC and F1 for fraud/anomaly tasks
- SMOTE must be applied **after** train-test split to prevent data leakage
- Threshold tuning on XGBoost probability scores allows precision-recall trade-off control

---

*Benson Moses Palaparthi — [linkedin.com/in/benson-moses-palaparthi](https://www.linkedin.com/in/benson-moses-palaparthi/)*
