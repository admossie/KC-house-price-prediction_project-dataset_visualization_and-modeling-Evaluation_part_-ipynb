# 🏠 KC House Price Prediction — Dataset Visualization and Modeling Evaluation

> Predicting home sale prices in King County using exploratory data analysis (EDA), feature engineering, and supervised regression models.

![status](https://img.shields.io/badge/status-complete-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 🧭 Project Overview
The goal of this project is to **predict house prices** in King County (Seattle area) based on property and location features.  
Using **supervised learning**, the project explores relationships between features, builds regression models, and evaluates their accuracy.

**Objective:** Train models to predict the `price` column.  
**Dataset:** `kc_house_data.csv`  
**Target variable:** `price` (continuous).  
**Final Model:** Gradient Boosting Regressor — achieved **R² ≈ 0.89** on the test set.

---

## 📂 Repository Structure

---

## 📊 Data Validation & Preprocessing
- Inspected missing values and handled nulls via mean/median imputation.
- Converted date to datetime and extracted year/month features.
- Removed outliers and unrealistic property values.
- Scaled numeric features with `StandardScaler`.
- Encoded categorical variables using `OneHotEncoder`.
- Split data into **80% train / 20% test**.

---

## 🔎 Exploratory Data Analysis (EDA)
- **Distribution of price:** Right-skewed; log-transform applied to normalize.  
- **Correlation matrix:** Strong positive correlation between `sqft_living`, `grade`, and `price`.  
- **Geospatial insights:** Houses near Seattle’s urban core show higher prices.  
- **Feature visualization:** Pairplots, scatter plots, boxplots, and heatmaps used for interpretation.

---

## 🧠 Modeling and Evaluation

### Models Tested
| Model | R² Score | RMSE | Notes |
|--------|-----------|------|-------|
| Linear Regression | 0.73 | 134,000 | Simple, interpretable baseline |
| Decision Tree Regressor | 0.85 | 95,000 | Overfits slightly |
| **Gradient Boosting Regressor (final)** | **0.89** | **82,000** | Best generalization |
| Random Forest Regressor | 0.88 | 85,000 | Strong alternative |

### Model Tuning
Used `GridSearchCV` for parameter optimization:
```python
param_grid = {
    'n_estimators': [100, 250, 500],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}
