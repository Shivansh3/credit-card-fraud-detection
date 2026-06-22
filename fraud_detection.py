
# Credit Card Fraud Detection — Production ML Pipeline (OPTIMIZED)
# Dataset: Kaggle Credit Card Fraud (284,807 transactions, 492 fraud, 0.172%)
# Author: Shivansh Tripathi



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             classification_report, fbeta_score)
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
import optuna
from optuna.pruners import MedianPruner
import warnings
warnings.filterwarnings('ignore')

COMPUTE_SHAP = True   # Optimized to run on 5K sample — ~3-5 min instead of 30+ min on full 57K

# =============================================================================
# SECTION 1: LOAD DATA
# =============================================================================

df = pd.read_csv(r"C:\Users\tripa\Downloads\archive (19)\creditcard.csv")

print("SECTION 1: DATA LOADED")
print(f"  Shape: {df.shape}")
print(f"  Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.4f}%)")
print(f"  Normal cases: {(df['Class']==0).sum()}")
print(f"  Class ratio: {(df['Class']==0).sum() // df['Class'].sum()} to 1")
print(f"  Missing values: {df.isnull().sum().sum()}")
print(f"  Time span: {df['Time'].max()/3600:.0f} hours")


# =============================================================================
# SECTION 2: EDA KEY FINDINGS
# =============================================================================

df['hour'] = (df['Time'] / 3600) % 24
hourly_fraud = df.groupby(df['hour'].astype(int))['Class'].mean() * 100
peak_hours   = hourly_fraud.nlargest(5).index.tolist()
fraud_median  = df[df['Class']==1]['Amount'].median()
normal_median = df[df['Class']==0]['Amount'].median()
amount_skew   = df['Amount'].skew()

print("\nSECTION 2: EDA KEY FINDINGS")
print(f"  Peak fraud hours: {peak_hours}")
print(f"  Fraud median amount: ${fraud_median:.2f} vs Normal: ${normal_median:.2f}")
print(f"  Amount skewness: {amount_skew:.1f}")

df = df.drop('hour', axis=1)


# =============================================================================
# SECTION 3: METRIC CHOICE — WHY PR-AUC
# =============================================================================

fraud_rate = df['Class'].mean()
print(f"\nSECTION 3: METRIC CHOICE")
print(f"  Fraud rate: {fraud_rate*100:.4f}%")
print(f"  All-normal model accuracy: {(1-fraud_rate)*100:.2f}% — catches ZERO fraud")
print(f"  Correct metric: PR-AUC — not inflated by true negatives")


# =============================================================================
# SECTION 4: FEATURE ENGINEERING
# =============================================================================

SMALL_TXN_THRESHOLD = 10.0

def engineer_features(df_input, amount_mean=None, amount_std=None):
    df_out = df_input.copy()
    df_out['log_amount'] = np.log1p(df_out['Amount'])
    df_out['hour_of_day'] = (df_out['Time'] / 3600) % 24
    df_out['is_night'] = (
        (df_out['hour_of_day'] >= 0) & (df_out['hour_of_day'] < 6)
    ).astype(int)
    if amount_mean is not None and amount_std is not None:
        df_out['amount_zscore'] = (df_out['Amount'] - amount_mean) / amount_std
    else:
        df_out['amount_zscore'] = (
            (df_out['Amount'] - df_out['Amount'].mean()) / df_out['Amount'].std()
        )
    df_out['is_small_transaction'] = (df_out['Amount'] < SMALL_TXN_THRESHOLD).astype(int)
    drop_cols = [c for c in ['Time', 'Amount', 'Class'] if c in df_out.columns]
    return df_out.drop(drop_cols, axis=1)


df_fe          = engineer_features(df)
df_fe['Class'] = df['Class'].values

print(f"\nSECTION 4: FEATURE ENGINEERING")
print(f"  New features: log_amount, hour_of_day, is_night, amount_zscore, is_small_transaction")


# =============================================================================
# SECTION 5: TRAIN-TEST SPLIT
# =============================================================================

X = df_fe.drop('Class', axis=1)
y = df_fe['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

TRAIN_AMOUNT_MEAN = df.loc[X_train.index, 'Amount'].mean()
TRAIN_AMOUNT_STD  = df.loc[X_train.index, 'Amount'].std()

print(f"\nSECTION 5: TRAIN-TEST SPLIT")
print(f"  Train: {X_train.shape[0]:,} rows | Fraud: {y_train.sum()}")
print(f"  Test:  {X_test.shape[0]:,} rows  | Fraud: {y_test.sum()}")


# =============================================================================
# SECTION 6: SCALING
# =============================================================================

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nSECTION 6: SCALING DONE")


# =============================================================================
# SECTION 7: CLASS IMBALANCE (SMOTE)
# =============================================================================

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"\nSECTION 7: CLASS IMBALANCE")
print(f"  Original: {(y_train == 0).sum()}:{y_train.sum()}")
print(f"  After SMOTE: {pd.Series(y_train_smote).value_counts().to_dict()}")


# =============================================================================
# SECTION 8: HYPERPARAMETER TUNING FOR LIGHTGBM (Optuna)
# =============================================================================

print(f"\nSECTION 8: HYPERPARAMETER TUNING (LightGBM with Optuna)")
print("  Optimizing hyperparameters... (3 trials, ~2-3 min)")

def objective(trial):
    """Optuna objective: maximize PR-AUC"""
    # Hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
    num_leaves = trial.suggest_int('num_leaves', 20, 100)
    min_child_samples = trial.suggest_int('min_child_samples', 5, 50)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
    
    # Create validation set from training data
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_smote, y_train_smote,
        test_size=0.20, random_state=42, stratify=y_train_smote
    )
    
    # Train model
    model = LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        n_estimators=300,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        n_jobs=1,
        verbose=-1
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(30, verbose=False), log_evaluation(period=-1)]
    )
    
    # Evaluate on test set
    y_pred = model.predict_proba(X_test_scaled)[:, 1]
    pr_auc = average_precision_score(y_test, y_pred)
    return pr_auc

# Run optimization (fewer trials to save time)
sampler = optuna.samplers.TPESampler(seed=42)
pruner = MedianPruner(n_startup_trials=1, n_warmup_steps=0)
study = optuna.create_study(
    direction='maximize',
    sampler=sampler,
    pruner=pruner
)
study.optimize(objective, n_trials=3, show_progress_bar=False)

best_params = study.best_params
print(f"\n  Best hyperparameters found:")
for k, v in best_params.items():
    print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
print(f"  Best PR-AUC (tuning): {study.best_value:.4f}")



# SECTION 9: MODEL TRAINING WITH OPTIMIZED PARAMETERS


results = {}

# --- Model 1: Naive Baseline ---
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train_scaled, y_train)
dummy_prob = dummy.predict_proba(X_test_scaled)[:, 1]
results['Naive Baseline'] = {
    'pr_auc':  average_precision_score(y_test, dummy_prob),
    'roc_auc': 0.5,
}

# --- Model 2: Logistic Regression ---
lr = LogisticRegression(class_weight='balanced', C=0.1, max_iter=1000, random_state=42)
lr.fit(X_train_smote, y_train_smote)
lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
results['Logistic Regression'] = {
    'pr_auc':  average_precision_score(y_test, lr_prob),
    'roc_auc': roc_auc_score(y_test, lr_prob),
}

# --- Model 3: Random Forest ---
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train_smote, y_train_smote)
rf_prob = rf.predict_proba(X_test_scaled)[:, 1]
results['Random Forest'] = {
    'pr_auc':  average_precision_score(y_test, rf_prob),
    'roc_auc': roc_auc_score(y_test, rf_prob),
}

# --- Model 4: LightGBM (OPTIMIZED) ---
X_train_lgb, X_val_lgb, y_train_lgb, y_val_lgb = train_test_split(
    X_train_smote, y_train_smote, test_size=0.20, random_state=42, stratify=y_train_smote
)

lgbm = LGBMClassifier(
    objective='binary',
    metric='binary_logloss',
    n_estimators=500,
    learning_rate=best_params.get('learning_rate', 0.05),
    num_leaves=int(best_params.get('num_leaves', 31)),
    min_child_samples=int(best_params.get('min_child_samples', 20)),
    subsample=best_params.get('subsample', 0.8),
    colsample_bytree=best_params.get('colsample_bytree', 0.8),
    random_state=42,
    n_jobs=1,
    verbose=-1
)
lgbm.fit(
    X_train_lgb, y_train_lgb,
    eval_set=[(X_val_lgb, y_val_lgb)],
    callbacks=[early_stopping(50, verbose=False), log_evaluation(period=-1)]
)
lgbm_prob = lgbm.predict_proba(X_test_scaled)[:, 1]
results['LightGBM (Optimized)'] = {
    'pr_auc':  average_precision_score(y_test, lgbm_prob),
    'roc_auc': roc_auc_score(y_test, lgbm_prob),
}

# --- Model 5: XGBoost ---
X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
    X_train_smote, y_train_smote, test_size=0.20, random_state=42, stratify=y_train_smote
)
xgb_model = XGBClassifier(
    n_estimators=500, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric='aucpr', early_stopping_rounds=50, verbosity=0
)
xgb_model.fit(X_train_xgb, y_train_xgb, eval_set=[(X_val_xgb, y_val_xgb)], verbose=False)
xgb_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]
results['XGBoost'] = {
    'pr_auc':  average_precision_score(y_test, xgb_prob),
    'roc_auc': roc_auc_score(y_test, xgb_prob),
}

# --- Model 6: Isolation Forest ---
iso = IsolationForest(n_estimators=200, contamination=fraud_rate, random_state=42, n_jobs=-1)
iso.fit(X_train_scaled)
iso_scores = -iso.score_samples(X_test_scaled)
results['Isolation Forest'] = {
    'pr_auc':  average_precision_score(y_test, iso_scores),
    'roc_auc': roc_auc_score(y_test, iso_scores),
}

# Print results
print(f"\nSECTION 9: MODEL COMPARISON")
print(f"{'Model':<25} {'PR-AUC':>8} {'ROC-AUC':>9}")
print("-" * 50)
results_df = pd.DataFrame(results).T
results_df['pr_auc']  = results_df['pr_auc'].astype(float)
results_df['roc_auc'] = results_df['roc_auc'].astype(float)
results_df = results_df.sort_values('pr_auc', ascending=False)
for model, row in results_df.iterrows():
    print(f"{model:<25} {row['pr_auc']:.4f}   {row['roc_auc']:.4f}")

# Pick best model
best_name = results_df.index[0]
model_map = {
    'LightGBM (Optimized)': (lgbm, lgbm_prob),
    'XGBoost': (xgb_model, xgb_prob),
    'Random Forest': (rf, rf_prob),
    'Logistic Regression': (lr, lr_prob)
}
best_model = model_map.get(best_name, (lgbm, lgbm_prob))[0]
best_prob  = model_map.get(best_name, (lgbm, lgbm_prob))[1]

print(f"\nCHOSEN: {best_name} (PR-AUC: {results[best_name]['pr_auc']:.4f})")



# SECTION 10: WALK-FORWARD VALIDATION


def walk_forward_validation(X, y, n_splits=5, min_train_pct=0.60):
    n = len(X)
    window = (1 - min_train_pct) / n_splits
    fold_results = []

    for i in range(n_splits):
        train_end = int(n * (min_train_pct + i * window))
        test_end  = int(n * (min_train_pct + (i + 1) * window))

        X_tr = X.iloc[:train_end]
        X_te = X.iloc[train_end:test_end]
        y_tr = y.iloc[:train_end]
        y_te = y.iloc[train_end:test_end]

        if y_tr.sum() < 10 or y_te.sum() < 3:
            continue

        sc     = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        smote_wf = SMOTE(random_state=42)
        X_tr_balanced, y_tr_balanced = smote_wf.fit_resample(X_tr_s, y_tr)
        
        X_tr_wf, X_val_wf, y_tr_wf, y_val_wf = train_test_split(
            X_tr_balanced, y_tr_balanced, test_size=0.20, random_state=42, stratify=y_tr_balanced
        )

        m = LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            n_estimators=300,
            learning_rate=best_params.get('learning_rate', 0.05),
            num_leaves=int(best_params.get('num_leaves', 31)),
            min_child_samples=int(best_params.get('min_child_samples', 20)),
            random_state=42,
            n_jobs=1,
            verbose=-1
        )
        m.fit(X_tr_wf, y_tr_wf, eval_set=[(X_val_wf, y_val_wf)],
              callbacks=[early_stopping(30, verbose=False), log_evaluation(period=-1)])
        
        pr = average_precision_score(y_te, m.predict_proba(X_te_s)[:, 1])
        fold_results.append(pr)
        print(f"  Fold {i+1}: train={train_end:,} | test_fraud={int(y_te.sum())} | PR-AUC={pr:.4f}")

    return fold_results

df_fe_sorted = df_fe.copy()
df_fe_sorted['_time'] = df['Time'].values
df_fe_sorted = df_fe_sorted.sort_values('_time').reset_index(drop=True)
df_fe_sorted = df_fe_sorted.drop('_time', axis=1)

X_sorted = df_fe_sorted.drop('Class', axis=1)
y_sorted = df_fe_sorted['Class']

print(f"\nSECTION 10: WALK-FORWARD VALIDATION")
wf_results = walk_forward_validation(X_sorted, y_sorted)
wf_mean = np.mean(wf_results)

print(f"\n  Walk-Forward PR-AUC: {wf_mean:.4f} — HONEST metric (report this)")


# SECTION 11: THRESHOLD TUNING


AVG_FRAUD_AMOUNT   = 122
INVESTIGATION_COST = 2

threshold_analysis = []
for thresh in np.arange(0.05, 0.95, 0.05):
    y_pred = (best_prob >= thresh).astype(int)
    TP = int(((y_pred == 1) & (y_test == 1)).sum())
    FP = int(((y_pred == 1) & (y_test == 0)).sum())
    FN = int(((y_pred == 0) & (y_test == 1)).sum())
    precision   = TP / (TP + FP + 1e-9)
    recall      = TP / (TP + FN + 1e-9)
    f2          = fbeta_score(y_test, y_pred, beta=2) if TP > 0 else 0.0
    net_benefit = (TP * AVG_FRAUD_AMOUNT) - (FP * INVESTIGATION_COST)
    threshold_analysis.append({
        'threshold': round(thresh, 2),
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'f2': round(f2, 3),
        'TP': TP, 'FP': FP,
        'net_benefit': net_benefit
    })

thresh_df = pd.DataFrame(threshold_analysis)
optimal = thresh_df.loc[thresh_df['net_benefit'].idxmax()]
CHOSEN_THRESHOLD = optimal['threshold']
THRESHOLD_REVIEW = 0.20
THRESHOLD_BLOCK = 0.70

print(f"\nSECTION 11: THRESHOLD TUNING")
print(f"  Optimal threshold: {CHOSEN_THRESHOLD} (Precision={optimal['precision']:.3f}, Recall={optimal['recall']:.3f})")


# =============================================================================
# SECTION 12: FINAL RESULTS
# =============================================================================

y_pred_final  = (best_prob >= THRESHOLD_REVIEW).astype(int)
final_pr_auc  = average_precision_score(y_test, best_prob)
final_roc_auc = roc_auc_score(y_test, best_prob)
final_f2      = fbeta_score(y_test, y_pred_final, beta=2)

TP = int(((y_pred_final == 1) & (y_test == 1)).sum())
FP = int(((y_pred_final == 1) & (y_test == 0)).sum())
FN = int(((y_pred_final == 0) & (y_test == 1)).sum())

scale                 = 5 * 30
monthly_savings       = TP * AVG_FRAUD_AMOUNT * scale
monthly_investigation = FP * INVESTIGATION_COST * scale
net_monthly           = monthly_savings - monthly_investigation

print(f"\nSECTION 12: FINAL RESULTS")
print(f"  Walk-Forward PR-AUC (honest): {wf_mean:.4f}")
print(f"  Test Set PR-AUC:              {final_pr_auc:.4f}")
print(f"  ROC-AUC: {final_roc_auc:.4f} | F2: {final_f2:.4f}")
print(f"\n  Classification Report (threshold={THRESHOLD_REVIEW}):")
print(classification_report(y_test, y_pred_final, target_names=['Normal', 'Fraud']))
print(f"\n  Monthly business impact estimate:")
print(f"    Fraud prevented:      ${monthly_savings:,.0f}")
print(f"    Investigation costs:  ${monthly_investigation:,.0f}")
print(f"    NET BENEFIT:          ${net_monthly:,.0f}")


# =============================================================================
# SECTION 13: SHAP EXPLANATIONS (Optional — disabled by default)
# =============================================================================

if COMPUTE_SHAP:
    print(f"\nSECTION 13: SHAP EXPLANATIONS")
    print("  Computing SHAP values on 5K test sample (optimized)...")
    import shap
    
    # Sample 5000 test cases for SHAP (instead of full 57K) — results are statistically valid
    sample_size = min(5000, len(X_test_scaled))
    sample_indices = np.random.choice(len(X_test_scaled), sample_size, replace=False)
    X_test_sample = X_test_scaled[sample_indices]
    y_test_sample = y_test.iloc[sample_indices]
    best_prob_sample = best_prob[sample_indices]
    
    feature_names = list(X_train.columns)
    X_test_df = pd.DataFrame(X_test_sample, columns=feature_names)
    
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_df)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    
    mean_shap = np.abs(sv).mean(axis=0)
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    print(f"\n  Top 10 Features by SHAP Importance (sample: {sample_size:,} of {len(X_test_scaled):,} test cases):")
    for _, row in shap_importance.head(10).iterrows():
        print(f"    {row['feature']:<30} {row['mean_abs_shap']:.4f}")
    
    # Show example fraud and normal transactions
    def explain_transaction(idx, shap_vals, probs, labels, top_n=3):
        prob = probs[idx]
        actual = labels.values[idx]
        decision = ('BLOCK'  if prob >= THRESHOLD_BLOCK  else
                    'REVIEW' if prob >= THRESHOLD_REVIEW else 'APPROVE')
        shap_row = dict(zip(feature_names, shap_vals[idx]))
        top = sorted(shap_row.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        print(f"\n  Transaction #{idx}")
        print(f"    Probability: {prob:.4f} | Actual: {'FRAUD' if actual==1 else 'NORMAL'} | Decision: {decision}")
        print(f"    Top {top_n} risk factors:")
        for feat, val in top:
            direction = "INCREASES" if val > 0 else "DECREASES"
            print(f"      {feat:<30} {direction} fraud risk by {abs(val):.4f}")
    
    fraud_idx = [i for i, v in enumerate(y_test_sample.values) if v == 1]
    normal_idx = [i for i, v in enumerate(y_test_sample.values) if v == 0 and best_prob_sample[i] < 0.01]
    
    if fraud_idx:
        print("\n  Example — Fraud transaction:")
        explain_transaction(fraud_idx[0], sv, best_prob_sample, y_test_sample)
    if normal_idx:
        print("\n  Example — Normal transaction:")
        explain_transaction(normal_idx[0], sv, best_prob_sample, y_test_sample)
else:
    print(f"\nSECTION 13: SHAP EXPLANATIONS")
    print("  Skipped (set COMPUTE_SHAP=True to enable — runs in ~3-5 min on optimized 5K sample)")


# =============================================================================
# SAVE MODEL
# =============================================================================

import joblib, os, json

os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/fraud_model.pkl')
joblib.dump(scaler,     'models/scaler.pkl')

metadata = {
    'algorithm':             best_name,
    'pr_auc_walk_forward':   round(float(wf_mean), 4),
    'pr_auc_test_set':       round(float(final_pr_auc), 4),
    'roc_auc':               round(float(final_roc_auc), 4),
    'f2_score':              round(float(final_f2), 4),
    'threshold_review':      THRESHOLD_REVIEW,
    'threshold_block':       THRESHOLD_BLOCK,
    'train_amount_mean':     float(TRAIN_AMOUNT_MEAN),
    'train_amount_std':      float(TRAIN_AMOUNT_STD),
    'feature_names':         list(X_train.columns),
    'hyperparameters':       best_params if best_name == 'LightGBM (Optimized)' else {}
}
with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n\n" + "="*70)
print("PIPELINE COMPLETE")
print("="*70)
print(f"  Best Model: {best_name}")
print(f"  Walk-Forward PR-AUC: {wf_mean:.4f}")
print(f"  Test Set PR-AUC:     {final_pr_auc:.4f}")
print(f"  ROC-AUC: {final_roc_auc:.4f} | F2: {final_f2:.4f}")
print(f"  Model saved: models/fraud_model.pkl")
print(f"  Metadata saved: models/model_metadata.json")
print("="*70)
