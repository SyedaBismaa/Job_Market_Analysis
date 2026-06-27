# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 21:52:14 2026

@author: syeda
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scorecardpy as sc

#model imports 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
#matrix
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
#Loadt  Data 
df=pd.read_csv(r"C:\Users\syeda\OneDrive\Desktop\Job_Market_Analyses\Job_dataset.csv")


#Data overview
print(df.head(4))
print(df.columns)
print(df.shape)

print(df.describe())
df['City'].unique
df['State'].unique()   #nothing or maybe label 

#missing vals and null vals
print(df.isna().sum())
df["Salary_LPA"].describe()  
#right skewde data , so will using mideian imputation
df["Salary_LPA"].fillna(df["Salary_LPA"].median(), inplace=True)

df['Company_Rating'].describe()
#using mean imputation
df["Company_Rating"].fillna(df["Company_Rating"].mean(), inplace=True)

# Company Size
df = pd.get_dummies(
    df,
    columns=["Company_Size"],
    dtype=int,
    drop_first=True
)

# Remote Mode
df = pd.get_dummies(
    df,
    columns=["Remote_Mode"],
    dtype=int,
    drop_first=True
)

# Employment Type
df = pd.get_dummies(
    df,
    columns=["Employment_Type"],
    dtype=int,
    drop_first=True
)


#Duplicates and outliers
df.duplicated().sum()
df['Salary_LPA'].describe()
df['Company_Rating'].describe()
df['Applicants'].describe()
df['Open_Positions'].describe()

#Charts Anlysis to check outliers 
plt.figure(figsize=(8,5))
plt.hist(df["Salary_LPA"], bins=30)
plt.title("Salary Distribution")
plt.xlabel("Salary (LPA)")
plt.ylabel("Frequency")
plt.show()   #Mostly Jobs are in Low paying side , less high paying jobs 

plt.figure(figsize=(8,5))
plt.boxplot(df["Salary_LPA"])
plt.title("Salary_LPA Boxplot")
plt.show()  #many high paying salaries 

plt.figure(figsize=(8,5))
plt.boxplot(df["Company_Rating"])
plt.title("Company Rating Boxplot")
plt.show() # mormal rating no outliers 

plt.figure(figsize=(8,5))
plt.boxplot(df["Applicants"])
plt.title("Applicants Boxplot")
plt.show() #some jobs have very high num of apllicants 

plt.figure(figsize=(8,5))
plt.boxplot(df["Open_Positions"])
plt.title("Open Positions Boxplot")
plt.show() # some companies have many openings 


df.to_csv(
    r"C:\Users\syeda\OneDrive\Desktop\Job_Market_Analyses\Job_dataset_cleaned.csv",
    index=False
)

print("Cleaned dataset saved successfully!")

#drop
df = df.drop(columns=["Skills", "Required_Tools"])

#Feature Engineer
#Posting_date
df["Posting_Date"] = pd.to_datetime(
    df["Posting_Date"],
    dayfirst=True
)
df["Posting_Year"] = df["Posting_Date"].dt.year
df["Posting_Month"] = df["Posting_Date"].dt.month
df["Posting_Month_Name"] = df["Posting_Date"].dt.month_name()

#target Creation
df["Salary_LPA"].median()

df["Target"] = np.where(
    df["Salary_LPA"] >= df["Salary_LPA"].median(),
    1,
    0
)

print(df["Target"].value_counts())
 


X = df.drop(columns=[
    "Target",
    "Salary_LPA",          # Target created from Salary
    "Job_ID",
    "Job_Title",
    "Company_Name",
    "City",
    "State",
    "Industry",
    "Education",
    "Experience_Required",
    "Posting_Date",
    "Posting_Month_Name"
])

vif = pd.DataFrame()

vif["Feature"] = X.columns

vif["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

print(vif)

# WoE & IV

df_woe = df.drop(columns=[
    "Job_ID",
    "Job_Title",
    "Company_Name",
    "City",
    "Posting_Date",
    "Salary_LPA"          # Prevent target leakage
])

# Generate WoE bins
bins = sc.woebin(df_woe, y="Target")

# Print WoE table for one feature
print(bins["Experience_Required"])

# Plot WoE
sc.woebin_plot(bins)

# Information Value
iv = sc.iv(df_woe, y="Target")

print(iv)

# Model
X = df.drop(columns=[
    "Job_ID",
    "Job_Title",
    "Company_Name",
    "City",
    "Posting_Date",
    "Salary_LPA",
    "Industry","Experience_Required","Education", "Posting_Month_Name","State","Target"
    ])
y = df["Target"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Logistic Regression Model  65%accuracy ,61f1 
model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

# Train Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print(X.select_dtypes(include="object").columns.tolist())
# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#Desicion Tree Model (keep)

model=DecisionTreeClassifier()

param_grid={
    "max_depth":[3,5,7,10,15],
    "min_samples_split":[2,5,10],
    "criterion":['gini','entropy']
    }
grid_search=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='f1'
    )

grid_search.fit(X_train,y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)  #72 

best_tree = grid_search.best_estimator_

plt.figure(figsize=(20,10))

plot_tree(
    best_tree,
    feature_names=X.columns,
    class_names=["Low Paying", "High Paying"],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.show()

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
