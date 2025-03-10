import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest

def train_anomaly_model(file_path="data/transactions.csv", model_path="models/anomaly_model.pkl"):
    """Trains and saves an Isolation Forest model for anomaly detection."""
    
    # Check if CSV file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found!")
        return
    
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Check if 'Amount' column exists
    if "Amount" not in df.columns:
        print("❌ Error: 'Amount' column not found in CSV!")
        return
    
    # Train Isolation Forest model
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(df[["Amount"]])  

    # Save the trained model
    os.makedirs("models", exist_ok=True)  # Ensure 'models' folder exists
    joblib.dump(model, model_path)
    
    print("✅ Model trained & saved successfully!")

def detect_anomalies(df, model_path="models/anomaly_model.pkl"):
    """Loads the trained model and detects anomalies."""
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found! Train the model first.")
        return df
    
    # Load trained model
    model = joblib.load(model_path)  
    
    # Predict anomalies
    df["Anomaly"] = model.predict(df[["Amount"]])  
    df["Anomaly"] = df["Anomaly"].map({1: "Normal", -1: "Bill Shock"})  # Convert labels
    
    return df[df["Anomaly"] == "Bill Shock"]  # Return only anomalies

# Train model when script is run directly
if __name__ == "__main__":
    train_anomaly_model()