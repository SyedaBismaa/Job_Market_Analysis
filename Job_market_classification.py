# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 21:52:14 2026

@author: syeda
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

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

#One-Hot_encoded 
df['Industry'].unique()
df=pd.get_dummies(df,columns=['Industry'],dtype=int)

df['Company_Size'].unique()
df=pd.get_dummies(df,columns=['Company_Size'], dtype=int)

df['Remote_Mode'].unique()
df=pd.get_dummies(df,columns=['Remote_Mode'],dtype=int)


df['Education'].unique()
df=pd.get_dummies(df,columns=['Education'],dtype=int)

df['Employment_Type'].unique() #onehot 
df=pd.get_dummies(df,columns=['Employment_Type'],dtype=int)




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