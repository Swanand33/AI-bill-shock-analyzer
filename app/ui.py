import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add parent directory to path to import from utils
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.anomaly_detection import detect_anomalies, MODELS_DIR

# Page config
st.set_page_config(
    page_title="💰 Bill Shock Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("⚡ Bill Shock Detector Dashboard")
st.markdown("Upload your transaction data to detect unusual spending patterns using AI-powered anomaly detection.")

# Sidebar
st.sidebar.header("📂 Data Upload")
st.sidebar.markdown("Upload a CSV file with an **Amount** column to analyze transactions.")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Choose CSV file",
    type=['csv'],
    help="CSV must contain an 'Amount' column"
)

# Model status check
model_path = MODELS_DIR / "anomaly_model.pkl"
if not model_path.exists():
    st.error("⚠️ Model not found! Please train the model first by running `python app/main.py`")
    st.stop()

# Main content
if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)

        # Validate data
        if "Amount" not in df.columns:
            st.error(f"❌ Error: 'Amount' column not found! Available columns: {', '.join(df.columns)}")
            st.stop()

        # Show basic stats
        st.subheader("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Transactions", len(df))
        with col2:
            st.metric("Total Amount", f"${df['Amount'].sum():,.2f}")
        with col3:
            st.metric("Average Amount", f"${df['Amount'].mean():,.2f}")
        with col4:
            st.metric("Max Amount", f"${df['Amount'].max():,.2f}")

        # Detect anomalies
        st.subheader("🔍 Anomaly Detection")

        with st.spinner("Analyzing transactions..."):
            anomalies = detect_anomalies(df)

        if anomalies is not None and not anomalies.empty:
            # Metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "⚠️ Bill Shocks Detected",
                    len(anomalies),
                    delta=f"{(len(anomalies)/len(df)*100):.1f}% of total"
                )
            with col2:
                st.metric("Total Shock Amount", f"${anomalies['Amount'].sum():,.2f}")
            with col3:
                st.metric("Avg Shock Amount", f"${anomalies['Amount'].mean():,.2f}")

            # Visualizations
            st.subheader("📈 Visualizations")

            # Create two columns for charts
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                # Distribution chart
                st.markdown("**Amount Distribution**")
                fig_hist = px.histogram(
                    df,
                    x="Amount",
                    nbins=30,
                    title="Transaction Amount Distribution",
                    labels={"Amount": "Amount ($)", "count": "Frequency"},
                    color_discrete_sequence=["#1f77b4"]
                )
                fig_hist.add_vline(
                    x=df['Amount'].mean(),
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Mean"
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with chart_col2:
                # Box plot
                st.markdown("**Anomaly Detection Box Plot**")

                # Add anomaly labels to full dataframe
                df_full = df.copy()
                df_full["Category"] = "Normal"
                df_full.loc[df_full.index.isin(anomalies.index), "Category"] = "Bill Shock"

                fig_box = px.box(
                    df_full,
                    y="Amount",
                    x="Category",
                    color="Category",
                    title="Normal vs Bill Shock Amounts",
                    labels={"Amount": "Amount ($)"},
                    color_discrete_map={"Normal": "#2ecc71", "Bill Shock": "#e74c3c"}
                )
                st.plotly_chart(fig_box, use_container_width=True)

            # Scatter plot (if there are other columns)
            if len(df.columns) > 1:
                st.markdown("**Transaction Scatter Plot**")

                # Prepare data with anomaly labels
                df_scatter = df.copy()
                df_scatter["Is_Anomaly"] = df_scatter.index.isin(anomalies.index)
                df_scatter["Label"] = df_scatter["Is_Anomaly"].map({True: "Bill Shock", False: "Normal"})

                fig_scatter = px.scatter(
                    df_scatter,
                    x=df_scatter.index,
                    y="Amount",
                    color="Label",
                    title="Transactions Timeline",
                    labels={"x": "Transaction ID", "Amount": "Amount ($)"},
                    color_discrete_map={"Normal": "#3498db", "Bill Shock": "#e74c3c"},
                    hover_data={"Label": True, "Amount": ":$.2f"}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            # Anomaly table
            st.subheader("⚠️ Detected Bill Shock Transactions")
            st.dataframe(
                anomalies.style.highlight_max(subset=['Amount'], color='#ffcccb'),
                use_container_width=True,
                height=300
            )

            # Download button
            csv = anomalies.to_csv(index=False)
            st.download_button(
                label="📥 Download Anomaly Report (CSV)",
                data=csv,
                file_name="bill_shock_report.csv",
                mime="text/csv"
            )

        else:
            st.success("✅ No bill shocks detected! All transactions appear normal.")

            # Still show distribution chart
            st.subheader("📈 Transaction Distribution")
            fig_hist = px.histogram(
                df,
                x="Amount",
                nbins=30,
                title="Transaction Amount Distribution",
                labels={"Amount": "Amount ($)", "count": "Frequency"}
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")

else:
    # Show instructions when no file uploaded
    st.info("👈 Please upload a CSV file from the sidebar to begin analysis.")

    st.markdown("""
    ### 📋 How to Use:
    1. **Upload CSV**: Use the sidebar to upload a CSV file with transaction data
    2. **Required Format**: Your CSV must have an **Amount** column
    3. **Optional Columns**: Include Date, ID, or other fields for better insights
    4. **View Results**: See detected anomalies with visualizations
    5. **Download Report**: Export anomaly report as CSV

    ### 📊 Sample CSV Format:
    ```
    ID,Amount,Date
    1,150.50,2024-01-01
    2,200.00,2024-01-02
    3,8500.00,2024-01-03
    ```

    ### 🤖 About the Model:
    This tool uses **Isolation Forest** algorithm to detect unusual spending patterns.
    Transactions that deviate significantly from normal behavior are flagged as "Bill Shocks".
    """)
