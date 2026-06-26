import pandas as pd
import numpy as np


df = pd.read_csv("customer_churn_dataset.csv")

print(f"Shape of dataset: {df.shape}\n")



print(f"Missing values per column:\n{df.isnull().sum()}")
print(f"Duplicate rows found: {df.duplicated().sum()}\n")



print(df["Churn_Status"].value_counts(normalize=True))  # Shows the % of churn vs retained

print(df.groupby("Subscription_Type")[["Monthly_Spending", "Tenure", "Churn_Status"]].mean())


ml_df = df.drop(columns=["Customer_ID", "Last_Activity_Date"])

# Convert Gender and Subscription_Type into numbers using One-Hot Encoding
ml_df = pd.get_dummies(ml_df, columns=["Gender", "City", "Subscription_Type"], drop_first=True)


for col in ml_df.columns:
    if ml_df[col].dtype == 'bool':
        ml_df[col] = ml_df[col].astype(int)


ml_df.to_csv("prepared_customer_data.csv", index=False)

print(f"New feature : {ml_df.shape[1]} columns.")