# 💼 Job Market Salary Prediction using Machine Learning

## 📌 Project Overview

This project focuses on building an end-to-end Machine Learning pipeline to predict whether a job posting is **High Paying** or **Low Paying** based on various job-related attributes.

The project covers the complete Data Science workflow including data preprocessing, exploratory data analysis (EDA), feature engineering, Weight of Evidence (WoE), Information Value (IV), multicollinearity analysis (VIF), model building, hyperparameter tuning, and model comparison.

---

## 🎯 Objective

Develop a binary classification model capable of predicting whether a job belongs to the **High Paying** or **Low Paying** category.

Target Variable:

* **1 → High Paying Job**
* **0 → Low Paying Job**

The target was created using the **median salary (Salary_LPA)** as the threshold.

---

# 📂 Dataset

* Total Records: **20,000**
* Original Features: **18**
* Domain: **Job Market / Recruitment**

Features include:

* Job Title
* Company Name
* Company Size
* Industry
* Remote Mode
* Experience Required
* Education
* Salary (LPA)
* Company Rating
* Applicants
* Open Positions
* Posting Date
* Required Tools
* Skills

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Statsmodels
* ScorecardPy

---

# 📊 Project Workflow

```
Data Collection
        ↓
Exploratory Data Analysis (EDA)
        ↓
Missing Value Treatment
        ↓
Duplicate Check
        ↓
Outlier Analysis
        ↓
Feature Engineering
        ↓
Target Creation
        ↓
Variance Inflation Factor (VIF)
        ↓
Weight of Evidence (WoE)
        ↓
Information Value (IV)
        ↓
Train-Test Split
        ↓
Logistic Regression
        ↓
Decision Tree
        ↓
GridSearchCV Hyperparameter Tuning
        ↓
Model Comparison
```

---

# 📈 Exploratory Data Analysis

Performed:

* Dataset Overview
* Missing Value Analysis
* Missing Value Treatment
* Duplicate Detection
* Statistical Summary
* Histogram Analysis
* Box Plot Analysis

Visualizations Included:

* Salary Distribution
* Salary Boxplot
* Company Rating Boxplot
* Applicants Boxplot
* Open Positions Boxplot
* Bins
* Decision Tree

---

# 🧹 Data Cleaning

Performed the following preprocessing steps:

* Median Imputation for Salary
* Mean Imputation for Company Rating
* Duplicate Check
* Outlier Analysis
* Removed unnecessary columns
* Converted Posting Date into DateTime format

---

# ⚙️ Feature Engineering

Created additional features from Posting Date:

* Posting Year
* Posting Month
* Posting Month Name

Removed:

* Skills
* Required Tools

(These columns contained thousands of unique combinations and were excluded from modeling.)

---

# 🎯 Target Engineering

Created a binary target variable:

```
Salary ≥ Median Salary  → High Paying (1)

Salary < Median Salary → Low Paying (0)
```

Target Distribution:

* High Paying : 10,733
* Low Paying : 9,267

---

# 📉 Multicollinearity Analysis

Variance Inflation Factor (VIF) was calculated to identify highly correlated independent variables before model training.

---

# 📊 WoE & Information Value

Performed:

* Weight of Evidence (WoE)
* Information Value (IV)

These techniques were used to understand the predictive strength of each feature before model building.

---

# 🤖 Machine Learning Models

## 1. Logistic Regression

Performed:

* Train-Test Split
* Model Training
* Prediction
* Evaluation

Metrics Used:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 2. Decision Tree Classifier

Performed:

* Decision Tree Training
* Hyperparameter Tuning using GridSearchCV
* Tree Visualization
* Model Evaluation

Hyperparameters Tuned:

* Max Depth
* Min Samples Split
* Criterion (Gini / Entropy)

Best Parameters:

```
Criterion : Entropy
Max Depth : 3
Min Samples Split : 2
```

---

# 📊 Model Performance

| Model               | Accuracy | F1 Score |
| ------------------- | -------- | -------- |
| Logistic Regression | 64%      | 0.72     |
| Decision Tree       | 64%      | 0.72     |

Both models produced similar performance.

Considering interpretability and simplicity, **Logistic Regression** can be selected as the final model.

---

# 📁 Project Structure

```
Job_Market_Analysis/

│
├── Job_dataset.csv
├── Job_dataset_cleaned.csv
├── Job_Market_Analysis.py
├── README.md
│
├── Images/
│   ├── Salary Distribution
│   ├── Salary Boxplot
│   ├── Company Rating Boxplot
│   ├── Applicants Boxplot
│   ├── Open Positions Boxplot
│   ├── Decision Tree
│   └── WoE Plots

```



# 📚 Key Concepts Used

* Exploratory Data Analysis (EDA)
* Missing Value Treatment
* Feature Engineering
* Target Engineering
* One-Hot Encoding
* Variance Inflation Factor (VIF)
* Weight of Evidence (WoE)
* Information Value (IV)
* Logistic Regression
* Decision Tree
* GridSearchCV
* Model Evaluation
* Hyperparameter Tuning

---

## ⭐ Conclusion

This project demonstrates a complete end-to-end Machine Learning workflow for salary classification using job market data. It covers every major stage of the ML lifecycle—from data preprocessing and feature engineering to model training, evaluation, and hyperparameter tuning—making it a practical example of applying classification techniques to a real-world business problem.
