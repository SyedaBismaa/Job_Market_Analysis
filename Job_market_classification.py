# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 21:52:14 2026

@author: syeda
"""

import pandas as pd
import numpy as np 


#Loadt  Data 
df=pd.read_csv(r"C:\Users\syeda\OneDrive\Desktop\Job_Market_Analyses\Job_dataset.csv")


#Data overview
print(df.head(4))
print(df.columns)
print(df.shape)

print(df.describe())
df['City'].unique()
df['State'].unique()   #nothing or maybe label 

df['Company_Size'].unique()
df['Remote_Mode'].unique()
df['Education'].unique()
df['Employment_Type'].unique() #onehot 


