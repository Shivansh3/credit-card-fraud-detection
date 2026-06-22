# Credit Card Fraud Detection

Production ML pipeline for credit card fraud detection using real European bank transaction data.

**Dataset:** 284,807 transactions | 492 fraud cases (0.172% fraud rate)

---

## Results

| Metric                  | Score      |
| ----------------------- | ---------- |
| **Walk-Forward PR-AUC** | **0.7597** |
| Test Set PR-AUC         | 0.8832     |
| ROC-AUC                 | 0.9851     |
| F2 Score                | 0.8661     |
| Fraud Recall            | 90%        |
| False Positive Rate     | 0.02%      |

**Walk-Forward PR-AUC (0.7597) is the honest metric.** Random split (0.8832) is inflated by temporal data leakage.

---

## Business Impact

At $1M daily transaction volume:

* **Fraud prevented:** $1,610,400/month
* **Investigation costs:** $8,400/month
* **NET BENEFIT:** $1,602,000/month

---

## How to Run

```bash
pip install -r requirements.txt

# Download creditcard.csv from Kaggle
# Place in project folder

python fraud_detection.py
```

---

## Key Technical Decisions

### Metric Choice: PR-AUC

With 578:1 class imbalance, accuracy is misleading. PR-AUC focuses on fraud detection performance and is not inflated by true negatives.

### Walk-Forward Validation

Transactions are time-ordered. Random train-test splits can leak future information. Walk-forward validation provides a more realistic estimate of production performance.

### Cost-Based Threshold

Missed fraud is significantly more expensive than investigating a legitimate transaction. Thresholds were selected using cost-benefit analysis.

---

## Model Comparison

| Model               | PR-AUC | ROC-AUC |
| ------------------- | ------ | ------- |
| LightGBM            | 0.8832 | 0.9851  |
| XGBoost             | 0.8810 | 0.9825  |
| Random Forest       | 0.8782 | 0.9751  |
| Logistic Regression | 0.7217 | 0.9731  |

---

## Features Engineered

* log_amount
* hour_of_day
* is_night
* amount_zscore
* is_small_transaction

---

## SHAP Explainability

Used SHAP TreeExplainer to provide:

* Global feature importance
* Local transaction-level explanations
* Model transparency for fraud decisions

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* LightGBM
* XGBoost
* SHAP
* Imbalanced-Learn
* Joblib

---

## Project Structure

```text
fraud-detection/
├── fraud_detection.py
├── README.md
├── requirements.txt
├── .gitignore
└── models/
```

---

## Author

Shivansh Tripathi

Data Science | Machine Learning | Risk Analytics
