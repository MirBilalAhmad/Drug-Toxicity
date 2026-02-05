#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: bilal
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Soft Voting Ensemble + Individual Models
NR-AhR Dataset
Leakage-free, reviewer-safe, publication-grade
ROC-AUC & AUPR comparison
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    classification_report
)
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ================== CONFIG ==================
RANDOM_SEED = 92
N_FOLDS = 5

# ================== LOAD DATA ==================
df = pd.read_csv(data_path)
X = df.drop("label", axis=1).values
y = df["label"].values

# ================== LOAD OPTIMIZED PARAMETERS ==================
rf_params   = joblib.load(best_params)
lgbm_params = joblib.load(best_param)
et_params   = joblib.load(best_param)
xgb_params  = joblib.load(best_param)
cat_params  = joblib.load(best_param)

# ================== CV SETUP ==================
skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=RANDOM_SEED
)

models = ["RF", "ET", "LGBM", "XGB", "CAT", "VOTE"]

roc_curves = {m: [] for m in models}
pr_curves  = {m: [] for m in models}

auc_scores  = {m: [] for m in models}
aupr_scores = {m: [] for m in models}

mean_fpr = np.linspace(0, 1, 300)
mean_recall = np.linspace(0, 1, 300)

all_true, all_pred = [], []

# ================== CV LOOP ==================
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):

    print(f"\n===== Fold {fold} =====")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ----- Base models -----
    rf = RandomForestClassifier(**rf_params, random_state=RANDOM_SEED, n_jobs=-1)
    et = ExtraTreesClassifier(**et_params, random_state=RANDOM_SEED, n_jobs=-1)
    lgbm = LGBMClassifier(**lgbm_params, random_state=RANDOM_SEED)
    xgb = XGBClassifier(**xgb_params, random_state=RANDOM_SEED,
                        n_jobs=-1, tree_method="hist")
    cat = CatBoostClassifier(**cat_params, random_seed=RANDOM_SEED, verbose=0)

    vote = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("et", et),
            ("lgbm", lgbm),
            ("xgb", xgb),
            ("cat", cat),
        ],
        voting="soft",
        n_jobs=-1
    )

    classifiers = {
        "RF": rf,
        "ET": et,
        "LGBM": lgbm,
        "XGB": xgb,
        "CAT": cat,
        "VOTE": vote
    }

    for name, clf in classifiers.items():

        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]

        # ---------- ROC ----------
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves[name].append(np.interp(mean_fpr, fpr, tpr))
        auc_scores[name].append(auc)

        # ---------- PR ----------
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        order = np.argsort(recall)
        pr_curves[name].append(
            np.interp(mean_recall, recall[order], precision[order])
        )
        aupr_scores[name].append(
            average_precision_score(y_test, y_prob)
        )

        if name == "VOTE":
            y_pred = (y_prob >= 0.5).astype(int)
            all_true.extend(y_test)
            all_pred.extend(y_pred)

# ================== ROC CURVES ==================
plt.figure(figsize=(8, 7))

for name in models:
    mean_tpr = np.mean(roc_curves[name], axis=0)
    mean_auc = np.mean(auc_scores[name])
    plt.plot(mean_fpr, mean_tpr, lw=2,
             label=f"{name} (AUC={mean_auc:.3f})")

plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison (NR-AhR)")
plt.legend()
plt.tight_layout()
plt.savefig("All_Models_ROC.pdf", dpi=300)
plt.show()

# ================== PR CURVES ==================
plt.figure(figsize=(8, 7))

for name in models:
    mean_prec = np.mean(pr_curves[name], axis=0)
    mean_aupr = np.mean(aupr_scores[name])
    plt.plot(mean_recall, mean_prec, lw=2,
             label=f"{name} (AUPR={mean_aupr:.3f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PrecisionRecall Curve Comparison (NR-AhR)")
plt.legend()
plt.tight_layout()
plt.savefig("All_Models_PR.pdf", dpi=300)
plt.show()

# ================== FINAL SUMMARY ==================
print("\n===== FINAL PERFORMANCE (MEAN ± STD) =====")
for name in models:
    print(
        f"{name:5s} | "
        f"AUC = {np.mean(auc_scores[name]):.4f} ± {np.std(auc_scores[name]):.4f} | "
        f"AUPR = {np.mean(aupr_scores[name]):.4f} ± {np.std(aupr_scores[name]):.4f}"
    )

print("\nAggregated Voting Classification Report:")
print(classification_report(all_true, all_pred, digits=4))

# ================== TRAIN FINAL VOTING MODEL & SAVE ==================
final_voting = VotingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(**rf_params, random_state=RANDOM_SEED, n_jobs=-1)),
        ("et", ExtraTreesClassifier(**et_params, random_state=RANDOM_SEED, n_jobs=-1)),
        ("lgbm", LGBMClassifier(**lgbm_params, random_state=RANDOM_SEED)),
        ("xgb", XGBClassifier(**xgb_params, random_state=RANDOM_SEED, n_jobs=-1)),
        ("cat", CatBoostClassifier(**cat_params, random_seed=RANDOM_SEED, verbose=0)),
    ],
    voting="soft",
    n_jobs=-1
)

final_voting.fit(X, y)
joblib.dump(
    final_voting,
    "final_voting_classifier.pkl"
)

print("✅ Final voting ensemble saved successfully.")





# # ============================================================
# # FEATURE IMPORTANCE (ENSEMBLE-AVERAGED)
# # ============================================================
# import matplotlib.cm as cm
# import matplotlib.colors as mcolors
# feature_names = df.drop("label", axis=1).columns.tolist()

# importance_dict = {}

# # ---- Fit models on full dataset (already done for final_voting)
# rf_final  = final_voting.estimators_[0]
# et_final  = final_voting.estimators_[1]
# lgbm_final = final_voting.estimators_[2]
# xgb_final  = final_voting.estimators_[3]
# cat_final  = final_voting.estimators_[4]

# models_fi = {
#     "RF":   rf_final.feature_importances_,
#     "ET":   et_final.feature_importances_,
#     "LGBM": lgbm_final.feature_importances_,
#     "XGB":  xgb_final.feature_importances_,
#     "CAT":  cat_final.get_feature_importance()
# }

# # ---- Normalize and aggregate
# fi_matrix = []

# for name, fi in models_fi.items():
#     fi = np.array(fi, dtype=float)
#     fi /= fi.sum()  # normalize
#     fi_matrix.append(fi)

# fi_avg = np.mean(fi_matrix, axis=0)

# fi_df = pd.DataFrame({
#     "Feature": feature_names,
#     "Importance": fi_avg
# }).sort_values("Importance", ascending=False)

# top20 = fi_df.head(20)

# # ============================================================
# # HIGH-RESOLUTION FEATURE IMPORTANCE PLOT
# # ============================================================

# plt.figure(figsize=(8.5, 6.5), dpi=600)



# # Normalize importance values for colormap
# norm = mcolors.Normalize(
#     vmin=top20["Importance"].min(),
#     vmax=top20["Importance"].max()
# )

# colors = cm.viridis(norm(top20["Importance"][::-1].values))

# plt.barh(
#     top20["Feature"][::-1],
#     top20["Importance"][::-1],
#     color=colors,
#     edgecolor="black",
#     linewidth=0.8
# )


# plt.xlabel("Feature Importance Score", fontsize=16, fontweight="bold")
# plt.title("Top-20 Features of the Nr-AhR (Voting Classifier)", fontsize=18, fontweight="bold", pad=14)


# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)

# plt.grid(axis="x", linestyle="--", alpha=0.4)
# plt.tight_layout()

# plt.savefig("Top20_Feature_Importance_Ensemble.pdf", dpi=600)
# plt.savefig("Top20_Feature_Importance_Ensemble.png", dpi=600)
# plt.show()

# print("✅ Top-20 feature importance plots saved (PDF + PNG)")

