# telco-customer-churn-prediction
An end-to-end Machine Learning pipeline and Streamlit web application predicting Telco customer churn using preprocessed data, feature engineering, and tuned Random Forest models.
# 📊 End-to-End Telco Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end Machine Learning project developed as part of the **Devixo Solutions AI/ML Internship (Week 3 Task)**. This project encompasses data cleaning, exploratory feature engineering, hyperparameter tuning, model persistence, and an interactive prediction interface built with Streamlit.

---

## 📌 Project Overview
Customer retention is crucial for telecommunications companies. This project leverages the **Telco Customer Churn** dataset (~7,043 rows) to build a predictive binary classification model that flags potential churners based on tenure, service usage, contracts, and payment metrics.

---

## 🚀 Key Features & Workflow

### 1. Data Preparation & Feature Engineering
- **Missing Value Imputation:** Coerced whitespace strings in `TotalCharges` into numeric representations and imputed missing values using median statistics.
- **Feature Engineering:** Developed custom features such as `AverageMonthlySpend` (`TotalCharges` / `tenure + 1`) to capture financial velocity over account lifespan.
- **Categorical Encoding:** Applied binary mapping to standard boolean indicators and One-Hot Encoding (`get_dummies`) to multi-class variables.
- **Scaling:** Normalization using `StandardScaler` to preserve feature variance across distance-sensitive algorithms.

### 2. Multi-Model Benchmarking
Evaluated candidate models across Accuracy, Precision, Recall, F1-Score, and ROC-AUC:
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **XGBoost Classifier**

### 3. Optimization & Persistence
- **Hyperparameter Tuning:** Conducted 5-Fold `GridSearchCV` on the top-performing Random Forest architecture to optimize `n_estimators`, `max_depth`, and split metrics.
- **Cross-Validation:** Verified cross-fold stability using 5-fold evaluation.
- **Feature Importance:** Analyzed key decision drivers using Random Forest feature importance scoring.
- **Model Serialization:** Persisted the optimized classifier and standard scaler pipelines via `joblib` and `pickle`.

---

## 🛠️ Technologies Used
- **Language:** Python 3.8+
- **Data Manipulation & Viz:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **Model Serialization:** Joblib, Pickle
- **Web App:** Streamlit

---

## 📁 Repository Structure
├── app.py                          # Streamlit interactive application script
├── best_churn_model.joblib         # Saved tuned Random Forest model
├── scaler.joblib                   # Saved StandardScaler object
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset file
└── README.md                       # Project documentation

## Name: Arfa Momna
## Program: Devixo Solutions Summer Internship (AI/ML Track)
