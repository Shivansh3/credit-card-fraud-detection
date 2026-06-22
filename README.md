# Credit Card Fraud Detection

Production ML pipeline for credit card fraud detection using real European bank transaction data.

**Dataset:** 284,807 transactions | 492 fraud cases (0.172% fraud rate)

---

## Results

| Metric | Score |
|--------|-------|
| **Walk-Forward PR-AUC** | **0.7597** |
| Test Set PR-AUC | 0.8832 |
| ROC-AUC | 0.9851 |
| F2 Score | 0.8661 |
| Fraud Recall | 90% |
| False Positive Rate | 0.02% |

**Walk-Forward PR-AUC (0.7597) is the honest metric.** Random split (0.8832) is inflated by temporal data leakage.

---

## Business Impact

At $1M daily transaction volume:
- **Fraud prevented:** $1,610,400/month
- **Investigation costs:** $8,400/month
- **NET BENEFIT:** $1,602,000/month

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
# Download creditcard.csv from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place in same folder as fraud_detection_optimized.py

# 3. Run pipeline
python fraud_detection_optimized.py
```

**Execution time:** ~15 minutes (includes hyperparameter tuning + SHAP explanations)

**Output files created:**
- `models/fraud_model.pkl` — Trained LightGBM model
- `models/scaler.pkl` — StandardScaler
- `models/model_metadata.json` — Metadata (thresholds, metrics, features)

---

## Key Technical Decisions

### Metric Choice: PR-AUC
With 578:1 class imbalance, accuracy is useless (99.83% by predicting all normal). PR-AUC measures fraud detection performance, not inflated by true negatives.

### Walk-Forward Validation
Dataset spans 48 hours chronologically. Random split causes temporal leakage. Walk-forward validation (train past, test future) gives honest out-of-sample metric: 0.7597 PR-AUC.

### SMOTE Balancing
All models trained on SMOTE-balanced data (50:50 class split) for fair comparison. No special weighting tricks.

### Hyperparameter Tuning (Optuna)
LightGBM tuned on this dataset:
- `learning_rate: 0.0237`
- `num_leaves: 97`
- `min_child_samples: 38`
- `subsample: 0.8395`
- `colsample_bytree: 0.6624`

### Cost-Based Threshold
Cost analysis: missed fraud = $122, false alarm = $2. Ratio 61:1 → threshold 0.20 gives 90% recall.

---

## Model Comparison

| Model | PR-AUC | ROC-AUC | Notes |
|-------|--------|---------|-------|
| **LightGBM** | **0.8832** | **0.9851** | Best performance |
| XGBoost | 0.8810 | 0.9825 | Close second |
| Random Forest | 0.8782 | 0.9751 | Solid baseline |
| Logistic Regression | 0.7217 | 0.9731 | Interpretable |

---

## Features Engineered

| Feature | Source | Reason |
|---------|--------|--------|
| `log_amount` | Amount | Reduce skewness 17 → 2.1 |
| `hour_of_day` | Time | Capture temporal patterns |
| `is_night` | hour_of_day | Fraud peaks 1-4 AM |
| `amount_zscore` | Amount | How unusual is this amount? |
| `is_small_transaction` | Amount | Card testing pattern (<$10) |

**EDA Finding:** Fraud median $9 vs normal $22. Fraudsters test stolen cards with small amounts.

---

## SHAP Feature Importance

Top 10 predictive features (5K test sample):

1. V14 — 4.30
2. V4 — 1.40
3. V10 — 1.34
4. V12 — 0.99
5. V11 — 0.45
6. V17 — 0.45
7. V8 — 0.32
8. V1 — 0.29
9. V3 — 0.26
10. V18 — 0.24

---

## Classification Report (threshold=0.20)

```
              precision    recall  f1-score   support
      Normal       1.00      1.00      1.00     56864
       Fraud       0.76      0.90      0.82        98
    accuracy                           1.00     56962
   macro avg       0.88      0.95      0.91     56962
weighted avg       1.00      1.00      1.00     56962
```

---

## Decision Rules

```
Probability < 0.20  →  APPROVE (automatic)
Probability 0.20-0.70 →  REVIEW (SMS verification)
Probability > 0.70  →  BLOCK
```

Most false positives land in REVIEW tier (customer verifies with SMS), not automatic block.

---

## Project Structure

```
fraud-detection/
├── fraud_detection_optimized.py    # Main pipeline
├── README.md                        # This file
├── .gitignore                       # Git ignore
├── requirements.txt                 # Dependencies
└── models/
    ├── fraud_model.pkl             # Trained model (auto-generated)
    ├── scaler.pkl                  # Fitted scaler (auto-generated)
    └── model_metadata.json         # Metadata (auto-generated)
```

---

## Tech Stack

- **ML Framework:** LightGBM, XGBoost, scikit-learn
- **Data:** pandas, numpy
- **Hyperparameter Tuning:** Optuna
- **Feature Importance:** SHAP
- **Imbalance Handling:** imbalanced-learn (SMOTE)
- **Serialization:** joblib

---

## Key Insights

1. **Temporal validation matters** — Walk-forward (0.7597) vs random split (0.8832) gap shows 12% leakage
2. **V14 is the strongest predictor** — SHAP analysis shows 4.3x importance vs other features
3. **Small transactions are suspicious** — Card testing attack pattern captured in feature engineering
4. **Threshold 0.20 is business-optimal** — 61:1 cost ratio drives recall > precision
5. **90% fraud recall is achievable** — With only 0.02% false positives

---

## Author

**Shivansh Tripathi**  
Data Scientist | Risk Analytics + ML  
2.5+ years in BFSI and CAT modeling

---

## Dataset Source

[Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## License

MIT License — Free to use and modify
