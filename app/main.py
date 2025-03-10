import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

# Load dataset
df = pd.read_csv("../data/transactions.csv")

# Selecting only 'Amount' column for anomaly detection
X = df[['Amount']]

# Train Isolation Forest model
model = IsolationForest(contamination=0.1, random_state=42)
df['Anomaly'] = model.fit_predict(X)

# Mark anomalies where Anomaly == -1
df['Anomaly'] = df['Anomaly'].apply(lambda x: "Bill Shock" if x == -1 else "Normal")

# Save the model
joblib.dump(model, "../models/anomaly_model.pkl")

# Save the results
df.to_csv("../data/transactions_with_anomalies.csv", index=False)

# Print anomalies
print("🔹 Anomalies Detected:")
print(df[df['Anomaly'] == "Bill Shock"])