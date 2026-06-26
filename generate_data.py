import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set seed for reproducibility so your results stay consistent
np.random.seed(42)
num_records = 1550

# 1. Basic Demographics & IDs
customer_ids = [f"CUST-{i:04d}" for i in range(1, num_records + 1)]
ages = np.random.randint(18, 70, size=num_records)
genders = np.random.choice(["Male", "Female"], size=num_records, p=[0.49, 0.51])
cities = np.random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Miami"], size=num_records)

# 2. Subscription Information
subscription_types = np.random.choice(["Basic", "Standard", "Premium"], size=num_records, p=[0.5, 0.3, 0.2])
tenures = np.random.randint(1, 60, size=num_records)  # Months active

# Adjust monthly spending based on subscription type to make it realistic
spending_map = {"Basic": 15, "Standard": 35, "Premium": 60}
monthly_spending = [spending_map[sub] + np.random.uniform(-5, 10) for sub in subscription_types]

# 3. Engagement & Behavior
num_purchases = np.random.randint(0, 15, size=num_records)
login_frequency = np.random.randint(1, 31, size=num_records)  # Logins per month
support_requests = np.random.randint(0, 10, size=num_records)
satisfaction_scores = np.random.randint(1, 6, size=num_records)  # 1 to 5 scale

# Generate realistic dates within the last year
base_date = datetime(2026, 6, 20)
last_activity_dates = [
    (base_date - timedelta(days=int(np.random.randint(0, 180)))).strftime('%Y-%m-%d')
    for _ in range(num_records)
]

# 4. Define Churn Logic (Creating a realistic target variable based on behavior)
# High support requests, low satisfaction, and low tenure increase the risk of leaving
churn_probs = []
for i in range(num_records):
    prob = 0.1  # base probability
    if satisfaction_scores[i] <= 2: prob += 0.4
    if support_requests[i] >= 5: prob += 0.3
    if tenures[i] < 6: prob += 0.2
    if login_frequency[i] < 5: prob += 0.15
    
    churn_probs.append(min(max(prob, 0.0), 1.0))

# 1 = Churned, 0 = Stayed
churn_status = [np.random.choice([1, 0], p=[p, 1-p]) for p in churn_probs]

# 5. Create DataFrame and Save
df = pd.DataFrame({
    "Customer_ID": customer_ids,
    "Age": ages,
    "Gender": genders,
    "City": cities,
    "Subscription_Type": subscription_types,
    "Monthly_Spending": np.round(monthly_spending, 2),
    "Tenure": tenures,
    "Number_of_Purchases": num_purchases,
    "Customer_Support_Requests": support_requests,
    "Login_Frequency": login_frequency,
    "Last_Activity_Date": last_activity_dates,
    "Satisfaction_Score": satisfaction_scores,
    "Churn_Status": churn_status
})

df.to_csv("customer_churn_dataset.csv", index=False)
print(f" Success! Dataset created with {df.shape[0]} rows and saved as 'customer_churn_dataset.csv'!")