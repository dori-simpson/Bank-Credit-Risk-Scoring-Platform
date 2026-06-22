# Bank Credit Risk Scoring Platform: Explainable AI Pipeline ⬡

This repository houses the core data science pipeline and analytical modeling framework designed to quantify retail credit risk. The system processes complex consumer credit portfolios to model individual default liabilities, converting mathematical probability values into actionable, auditable risk tiers and underwriting decisions.

---

## 📊 Business & Core Objective
The primary operational goal is to predict whether a prospective borrower will **default** (`1`) or remain **non-default** (`0`), allowing financial institutions to make automated, risk-adjusted lending decisions.

---

## ⚙️ Data Architecture & Pipeline Engineering
The framework handles end-to-end processing using an isolated, data-leakage-safe **Scikit-Learn ColumnTransformer Pipeline**:
* **Feature Engineering:** Extracts advanced signals like `credit_utilization_ratio`, `installment_to_income`, and ordinal grade parameters.
* **Preprocessing Suite:** Implements outlier-resistant **Median Imputation** and standardization for numerical arrays alongside strategic **One-Hot Encoding** for categorical markers.

---

## 🤖 Model Zoo & Benchmarks
The pipeline trains and audits three distinct algorithmic archetypes to compare performance distributions:

| Algorithmic Model | Benchmark ROC-AUC | Target Metrics Focus |
| :--- | :--- | :--- |
| **Logistic Regression** | ~0.77 | Linear Regularized Baseline |
| **Random Forest** | ~0.82 | Multi-variable Splitting |
| **🏆 XGBoost Classifier** | **~0.85+** | Optimized Gradient Tree Ensemble |

---

## 🔍 Regulated Credit Scoring & Decision Engine
Individual default probabilities are mapped to an intuitive **0 to 1000 credit score scale** ($Score = 1000 \times (1 - P(\text{Default}))$). Higher scores denote minimized default risks:
* **Score: 800 – 1000** ──► `APPROVE` | Low Risk Tier
* **Score: 650 – 799** ──► `APPROVE WITH CONDITIONS` | Moderate Risk Tier
* **Score: 500 – 649** ──► `MANUAL REVIEW` | Elevated Risk Tier
* **Score: 000 – 499** ──► `DECLINE` | High Risk Tier

---

## 🧠 Model Explainability (XAI & SHAP Compliance)
To maintain alignment with institutional model risk audit standards, model predictions are passed through a structural **SHAP TreeExplainer**. This provides global variable rankings alongside localized mathematical force plots detailing the primary financial drivers pushing an applicant toward approval or rejection.
