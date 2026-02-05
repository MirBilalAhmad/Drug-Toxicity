#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nested CV + Optuna + SMOTE
XGBoost Classifier (XGB)
Leakage-free,  publication-grade
ROC-AUC & AUPR curves + final model saving
"""

import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score
)
from imblearn.over_sampling import SMOTE

# ================== CONFIG ==================
RANDOM_SEED = 92
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
N_TRIALS = 150

# ================== LOAD DATA ==================
df = pd.read_csv(data_path)
X = df.drop("label", axis=1).values
y = df["label"].values

outer_cv = StratifiedKFold(
    n_splits=N_OUTER_FOLDS,
    shuffle=True,
    random_state=RANDOM_SEED
)

# ================== STORAGE ==================
outer_auc, outer_aupr = [], []

mean_fpr = np.linspace(0, 1, 300)
mean_recall = np.linspace(0, 1, 300)

roc_tprs = []
pr_precisions = []

best_params_outer = None   # store final hyperparameters

# ================== OUTER CV ==================
for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):

    print(f"\n===== OUTER FOLD {fold} =====")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ================== OPTUNA OBJECTIVE ==================
    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
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

            X_tr_res, y_tr_res = SMOTE(
                random_state=RANDOM_SEED
            ).fit_resample(X_tr, y_tr)

            model = XGBClassifier(
                **params,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_SEED,
                n_jobs=-1,
                tree_method="hist"
            )

            model.fit(X_tr_res, y_tr_res)

            y_val_proba = model.predict_proba(X_val)[:, 1]

            roc_scores.append(roc_auc_score(y_val, y_val_proba))
            pr_scores.append(average_precision_score(y_val, y_val_proba))

        # PR-weighted objective (imbalanced data)
        return 0.4 * np.mean(roc_scores) + 0.6 * np.mean(pr_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_params = study.best_params
    best_params_outer = best_params

    # ================== TRAIN MODEL ON OUTER TRAIN ==================
    X_train_res, y_train_res = SMOTE(
        random_state=RANDOM_SEED
    ).fit_resample(X_train, y_train)

    model = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist"
    )

    model.fit(X_train_res, y_train_res)

    # ================== THRESHOLD TUNING (INNER CV) ==================
    inner_probs, inner_labels = [], []

    inner_cv = StratifiedKFold(
        n_splits=N_INNER_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    for tr_idx, val_idx in inner_cv.split(X_train, y_train):

        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        X_tr_res, y_tr_res = SMOTE(
            random_state=RANDOM_SEED
        ).fit_resample(X_tr, y_tr)

        model.fit(X_tr_res, y_tr_res)
        inner_probs.extend(model.predict_proba(X_val)[:, 1])
        inner_labels.extend(y_val)

    thresholds = np.linspace(0.1, 0.9, 81)
    best_thr, best_f1 = 0.5, 0

    for t in thresholds:
        f1 = f1_score(inner_labels, (np.array(inner_probs) >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_thr = f1, t

    # ================== TEST ==================
    y_test_proba = model.predict_proba(X_test)[:, 1]

    outer_auc.append(roc_auc_score(y_test, y_test_proba))
    outer_aupr.append(average_precision_score(y_test, y_test_proba))

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_tprs.append(np.interp(mean_fpr, fpr, tpr))

    # PR
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    order = np.argsort(recall)
    pr_precisions.append(np.interp(mean_recall, recall[order], precision[order]))

# ================== ROC CURVE ==================
plt.figure(figsize=(7, 6))
mean_tpr = np.mean(roc_tprs, axis=0)
std_tpr = np.std(roc_tprs, axis=0)

plt.plot(mean_fpr, mean_tpr, lw=2, color="black",
         label=f"Mean ROC (AUC={np.mean(outer_auc):.3f})")
plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.25)
plt.plot([0, 1], [0, 1], "--", color="gray")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("XGBoost ROC Curve (5-fold Nested CV)")
plt.legend()
plt.tight_layout()
plt.savefig("XGB_ROC.pdf", dpi=300)
plt.show()

# ================== PR CURVE ==================
plt.figure(figsize=(7, 6))
mean_prec = np.mean(pr_precisions, axis=0)
std_prec = np.std(pr_precisions, axis=0)

plt.plot(mean_recall, mean_prec, lw=2, color="black",
         label=f"Mean PR (AUPR={np.mean(outer_aupr):.3f})")
plt.fill_between(mean_recall, mean_prec - std_prec, mean_prec + std_prec, alpha=0.25)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("XGBoost PrecisionRecall Curve (5-fold Nested CV)")
plt.legend()
plt.tight_layout()
plt.savefig("XGB_PR.pdf", dpi=300)
plt.show()

# ================== TRAIN FINAL MODEL ON FULL DATA ==================
X_res, y_res = SMOTE(random_state=RANDOM_SEED).fit_resample(X, y)

final_xgb = XGBClassifier(
    **best_params_outer,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_SEED,
    n_jobs=-1,
    tree_method="hist"
)

final_xgb.fit(X_res, y_res)

# ================== SAVE MODEL & PARAMS ==================
joblib.dump(final_xgb, "final_xgb_model.pkl")
joblib.dump(best_params_outer, "final_xgb_params.pkl")

print("\n===== FINAL PERFORMANCE =====")
print(f"ROC-AUC : {np.mean(outer_auc):.4f} ± {np.std(outer_auc):.4f}")
print(f"AUPR    : {np.mean(outer_aupr):.4f} ± {np.std(outer_aupr):.4f}")

print("✅ XGBoost final model and parameters saved successfully.")
