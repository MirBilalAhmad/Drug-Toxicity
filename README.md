# Drug-Toxicity
Interpretable Ensemble Learning Framework for Accurate and Scalable Drug Toxicity Prediction
## Overview
This repository presents an AI-driven framework for computational toxicity prediction using classical machine learning and ensemble models. The pipeline integrates physicochemical molecular descriptors and extended-connectivity fingerprints (ECFP) to model toxicological endpoints across benchmark datasets, including  ClinTox, and Tox21.
Multiple tree-based learners (Random Forest, Extra Trees, LightGBM, XGBoost, and CatBoost) are combined using a soft voting ensemble. Model development follows leakage-free, stratified cross-validation with imbalance-aware training strategies. Comprehensive evaluation is performed using ROC-AUC and precision–recall metrics, alongside ensemble-averaged feature importance analysis for interpretability.
The framework is designed to be reproducible, publication-ready, and applicable to drug discovery and safety assessment tasks.
<img width="4413" height="1203" alt="drug_tox_model" src="https://github.com/user-attachments/assets/17db245b-b8dd-4166-8eff-1b85d6cdf05a" />
