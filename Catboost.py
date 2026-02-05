#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nested CV + Optuna + SMOTE + ROC-AUC & AUPR
CatBoost Classifier  Leakage_Free
"""

import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import fbeta_score
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    f1_score
)
from imblearn.over_sampling import SMOTE

# ================== CONFIG ==================
RANDOM_SEED = 92
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
N_TRIALS = 150

# ================== LOAD DATA ==================
df = pd.read_csv(Data_Path)
X = df.drop("label", axis=1).values
y = df["label"].values

outer_cv = StratifiedKFold(
    n_splits=N_OUTER_FOLDS,
    shuffle=True,
    random_state=RANDOM_SEED
)

# ================== STORAGE ==================
outer_auc, outer_aupr = [], []
all_true, all_pred, all_proba = [], [], []

mean_fpr = np.linspace(0, 1, 200)
mean_recall = np.linspace(0, 1, 200)

roc_tprs = []
pr_precisions = []

# ================== OUTER CV ==================
for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):

    print(f"\n===== OUTER FOLD {fold} =====")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ================== OPTUNA OBJECTIVE ==================
# ================== OPTUNA OBJECTIVE ==================
    def objective(trial):
    
        params = {
            "iterations": trial.suggest_int("iterations", 500, 2000, step=250),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "verbose": 0,
            "random_seed": RANDOM_SEED,
        }
    
        inner_cv = StratifiedKFold(
            n_splits=N_INNER_FOLDS,
            shuffle=True,
            random_state=RANDOM_SEED
        )
    
        roc_scores, pr_scores = [], []
    
        for tr_idx, val_idx in inner_cv.split(X_train, y_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    
            # class weight from ORIGINAL data
            pos_weight = (len(y_tr) - sum(y_tr)) / sum(y_tr)
    
            model = CatBoostClassifier(
                **params,
                scale_pos_weight=pos_weight
            )
    
            model.fit(X_tr, y_tr)
    
            y_val_proba = model.predict_proba(X_val)[:, 1]
    
            roc_scores.append(roc_auc_score(y_val, y_val_proba))
            pr_scores.append(average_precision_score(y_val, y_val_proba))
    
        return 0.6 * np.mean(roc_scores) + 0.4 * np.mean(pr_scores)


    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_params = study.best_params

    # ================== TRAIN FINAL MODEL ==================
    X_train_res, y_train_res = SMOTE(random_state=RANDOM_SEED).fit_resample(X_train, y_train)
    pos_weight = (len(y_train_res) - sum(y_train_res)) / sum(y_train_res)

    model = CatBoostClassifier(
        **best_params,
        scale_pos_weight=pos_weight,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=0,
        random_seed=RANDOM_SEED
    )
    model.fit(X_train_res, y_train_res)

    # ================== THRESHOLD TUNING ==================
    train_probs = model.predict_proba(X_train)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    
    best_thr, best_f2 = 0.5, 0
    for t in thresholds:
        f2 = fbeta_score(y_train, (train_probs >= t).astype(int), beta=2)
        if f2 > best_f2:
            best_f2, best_thr = f2, t
    # ================== TEST ==================
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= best_thr).astype(int)

    auc = roc_auc_score(y_test, y_test_proba)
    aupr = average_precision_score(y_test, y_test_proba)

    outer_auc.append(auc)
    outer_aupr.append(aupr)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_tprs.append(np.interp(mean_fpr, fpr, tpr))

    # PR
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    pr_precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1]))

    print(f"ROC-AUC : {auc:.4f}")
    print(f"AUPR    : {aupr:.4f}")
    print(f"Thresh  : {best_thr:.2f}")
    print(classification_report(y_test, y_test_pred, digits=4))

    all_true.extend(y_test)
    all_pred.extend(y_test_pred)
    all_proba.extend(y_test_proba)

# ================== ROC CURVE ==================
plt.figure(figsize=(7, 6))
mean_tpr = np.mean(roc_tprs, axis=0)
std_tpr = np.std(roc_tprs, axis=0)

plt.plot(mean_fpr, mean_tpr, color="black",
         label=f"Mean ROC (AUC={np.mean(outer_auc):.3f})")
plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.25)
plt.plot([0, 1], [0, 1], "--", color="gray")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("CatBoost ROC Curve (Nested 5-fold CV)")
plt.legend()
plt.tight_layout()
plt.savefig("CatBoost_ROC.pdf", dpi=300)
plt.show()

# ================== PR CURVE ==================
plt.figure(figsize=(7, 6))
mean_prec = np.mean(pr_precisions, axis=0)
std_prec = np.std(pr_precisions, axis=0)

plt.plot(mean_recall, mean_prec, color="black",
         label=f"Mean PR (AUPR={np.mean(outer_aupr):.3f})")
plt.fill_between(mean_recall, mean_prec - std_prec, mean_prec + std_prec, alpha=0.25)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("CatBoost PrecisionRecall Curve (Nested 5-fold CV)")
plt.legend()
plt.tight_layout()
plt.savefig("CatBoost_PR.pdf", dpi=300)
plt.show()

# ================== FINAL SUMMARY ==================
print("\n===== FINAL PERFORMANCE =====")
print(f"ROC-AUC : {np.mean(outer_auc):.4f} ± {np.std(outer_auc):.4f}")
print(f"AUPR    : {np.mean(outer_aupr):.4f} ± {np.std(outer_aupr):.4f}")

print("\nAggregated Classification Report:")
print(classification_report(all_true, all_pred, digits=4))

# ================== SAVE MODEL ==================
joblib.dump(model, "/final_catboost_model.pkl")
joblib.dump(best_params, "/final_catboost_params.pkl")

print("✅ CatBoost model and parameters saved.")
