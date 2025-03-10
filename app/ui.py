import streamlit as st
import pandas as pd

# Load processed data with anomalies
df = pd.read_csv("../data/transactions_with_anomalies.csv")

# Title
st.set_page_config(page_title="💰 Bill Shock Detector", layout="wide")
st.title("⚡ Bill Shock Detector Dashboard")

# Summary
total_shocks = df[df['Anomaly'] == "Bill Shock"].shape[0]
st.metric("⚠️ Total Bill Shocks Detected", total_shocks)

# Display transactions
st.subheader("🔍 Identified Bill Shock Transactions")
st.dataframe(df[df['Anomaly'] == "Bill Shock"], use_container_width=True)

# Run the app
if __name__ == "__main__":
    st.write("Your Streamlit App is Running!")