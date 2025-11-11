#  Bill Shock Detector - AI-Powered Anomaly Detection  

An AI-driven **Bill Shock Detector** that identifies unusual spending patterns using **Isolation Forests**. This project is designed for **financial analysts, consultants, and businesses** to detect unexpected billing anomalies.  

---

##  Features  
✅ **Anomaly Detection**: Identifies bill shocks based on transaction history.  
✅ **Machine Learning**: Uses `IsolationForest` for anomaly detection.  
✅ **Customizable**: Works with any transaction dataset.  
✅ **Efficient & Lightweight**: Trains in seconds, detects anomalies in real time.  

---

##  Tech Stack  
- **Python** (Data Processing & Model)  
- **scikit-learn** (`IsolationForest` for anomaly detection)  
- **pandas** (Data Handling)  
- **joblib** (Model Saving & Loading)  

---

### 1. Clone the Repository
```sh
git clone https://github.com/Swanand33/AI-bill-shock-analyzer.git
cd AI-bill-shock-analyzer
```

### 2. Install Dependencies  
Ensure you have Python 3.8+ installed, then run:  
```sh
pip install -r requirements.txt  
```

### 3. Prepare Your Data  
Add your transaction data to:  
```sh
data/transactions.csv  
```
 **Ensure your CSV has an "Amount" column.**  

### 4. Train the Model  
Run the following command:  
```sh
python utils/anomaly_detection.py  
```
✅ This will generate a trained model at:  
```sh
models/anomaly_model.pkl  
```

### 5. Detect Anomalies  
Use the function in your script:  
```python
import pandas as pd  
from utils.anomaly_detection import detect_anomalies  

df = pd.read_csv("data/transactions.csv")  
anomalies = detect_anomalies(df)  
print(anomalies)  
```

📊 **Example Output:**  
```
    ID   Amount  Anomaly  
3  104  7500.0  Bill Shock  
7  202  9900.0  Bill Shock  
```

---

## 📂 Project Structure
```
📦 AI-bill-shock-analyzer
 ┣ 📂 app
 ┃ ┣ 📜 main.py  # Training script
 ┃ ┗ 📜 ui.py  # Streamlit dashboard
 ┣ 📂 data
 ┃ ┣ 📜 transactions.csv  # Sample transaction data
 ┃ ┗ 📜 transactions_with_anomalies.csv  # Results
 ┣ 📂 models
 ┃ ┗ 📜 anomaly_model.pkl  # Saved ML model
 ┣ 📂 utils
 ┃ ┣ 📜 anomaly_detection.py  # Core ML logic
 ┃ ┗ 📜 data_processing.py  # Data loading utilities
 ┣ 📜 README.md  # Project documentation
 ┣ 📜 requirements.txt  # Dependencies
 ┗ 📜 LICENSE
```

---

---

## 🚀 Running the Dashboard

To launch the interactive Streamlit dashboard:

```sh
streamlit run app/ui.py
```

Then upload your CSV file with transaction data to analyze!

---

## ⚙️ Advanced Configuration

### Adjusting Sensitivity

You can adjust the anomaly detection sensitivity by changing the `contamination` parameter when training:

```python
from utils.anomaly_detection import train_anomaly_model

# More sensitive (detects more anomalies)
train_anomaly_model(contamination=0.10)  # 10%

# Less sensitive (detects fewer anomalies)
train_anomaly_model(contamination=0.02)  # 2%
```

**Default:** 0.05 (5% of transactions expected to be anomalies)

---

## 🐛 Troubleshooting

### Issue 1: "Model not found" error
**Problem:** Dashboard shows "Model not found! Please train the model first"

**Solution:**
```sh
python app/main.py
```
This will train and save the model to `models/anomaly_model.pkl`

---

### Issue 2: "'Amount' column not found"
**Problem:** Uploaded CSV doesn't have an 'Amount' column

**Solution:** Ensure your CSV has a column named exactly `Amount` (case-sensitive). Example:
```csv
ID,Amount,Date
1,150.50,2024-01-01
2,200.00,2024-01-02
```

---

### Issue 3: No anomalies detected
**Problem:** All transactions show as "Normal"

**Possible causes:**
- Transaction amounts are very similar (no real anomalies)
- Contamination parameter is too low

**Solution:** Retrain with higher contamination:
```sh
# Edit app/main.py to use higher contamination
python app/main.py
```

---

### Issue 4: Too many false positives
**Problem:** Normal transactions flagged as "Bill Shock"

**Solution:** Retrain with lower contamination (e.g., 0.02 instead of 0.05)

---

### Issue 5: Import errors
**Problem:** `ModuleNotFoundError` when running scripts

**Solution:** Make sure you're running from the project root directory:
```sh
cd AI-bill-shock-analyzer
python app/main.py
```

---

## 📊 How It Works

1. **Training Phase:**
   - Reads transaction data from CSV
   - Trains Isolation Forest ML model on Amount values
   - Saves trained model to `models/` folder

2. **Detection Phase:**
   - Loads trained model
   - Analyzes new transactions
   - Identifies outliers as "Bill Shocks"

3. **Visualization:**
   - Interactive Streamlit dashboard
   - Charts: Histogram, Box Plot, Scatter Plot
   - Export anomaly report as CSV

---

## 🤝 Contributing
Feel free to fork, contribute, and submit PRs. For major changes, open an issue first.

---

## 🏆 Why This Project?
This AI-powered financial tool can be used by consulting firms, financial analysts, and businesses to:
- 💰 Detect unusual billing patterns
- 🔍 Identify potential fraud
- 📈 Optimize cost management
- ⚠️ Alert on spending anomalies

**Perfect for job interviews showcasing:**
- Machine Learning (Isolation Forest)
- Data Visualization (Plotly)
- Web Applications (Streamlit)
- Python Best Practices (Logging, Validation, Testing)

---

## 📄 License
MIT License - see LICENSE file for details

---

**Let's make finance smarter with AI!**
📩 Reach out if you have any questions!