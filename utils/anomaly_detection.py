import pandas as pd
import joblib
import os
from pathlib import Path
from sklearn.ensemble import IsolationForest
from .logger import setup_logger
from .validation import validate_csv_file, validate_dataframe, validate_contamination, validate_model_file

# Get project root directory (parent of 'utils' folder)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Setup logger
logger = setup_logger('anomaly_detection')

def train_anomaly_model(file_path=None, model_path=None, contamination=0.05):
    """
    Trains and saves an Isolation Forest model for anomaly detection.

    Args:
        file_path: Path to CSV file with transaction data
        model_path: Path to save trained model
        contamination: Expected proportion of anomalies (0.01-0.5, default 0.05 = 5%)

    Returns:
        bool: True if training successful, False otherwise
    """

    try:
        # Set default paths if not provided
        if file_path is None:
            file_path = DATA_DIR / "transactions.csv"
        if model_path is None:
            model_path = MODELS_DIR / "anomaly_model.pkl"

        # Validate contamination parameter
        is_valid, error_msg = validate_contamination(contamination)
        if not is_valid:
            logger.warning(f"{error_msg}, using default 0.05")
            contamination = 0.05

        # Validate CSV file
        is_valid, error_msg = validate_csv_file(file_path)
        if not is_valid:
            logger.error(error_msg)
            return False

        # Load dataset
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)

        # Validate DataFrame
        is_valid, error_msg = validate_dataframe(df, "Amount")
        if not is_valid:
            logger.error(error_msg)
            return False

        # Remove NaN values
        df_clean = df[["Amount"]].dropna()
        if df_clean.empty:
            logger.error("No valid data after removing NaN values!")
            return False

        logger.info(f"Training on {len(df_clean)} transactions (contamination={contamination})...")

        # Train Isolation Forest model
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(df_clean)

        # Save the trained model
        MODELS_DIR.mkdir(exist_ok=True)  # Ensure 'models' folder exists
        joblib.dump(model, model_path)

        logger.info(f"Model trained & saved successfully to {model_path}!")
        return True

    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty or corrupted!")
        return False
    except pd.errors.ParserError:
        logger.error("Could not parse CSV file. Check file format!")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during training: {str(e)}")
        return False

def detect_anomalies(df, model_path=None):
    """Loads the trained model and detects anomalies."""

    try:
        # Set default model path if not provided
        if model_path is None:
            model_path = MODELS_DIR / "anomaly_model.pkl"

        # Validate model file
        is_valid, error_msg = validate_model_file(model_path)
        if not is_valid:
            logger.error(error_msg)
            return None

        # Validate DataFrame
        is_valid, error_msg = validate_dataframe(df, "Amount")
        if not is_valid:
            logger.error(error_msg)
            return None

        # Load trained model
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)

        # Create copy to avoid modifying original
        df_result = df.copy()

        # Predict anomalies
        logger.info(f"Detecting anomalies in {len(df)} transactions...")
        df_result["Anomaly"] = model.predict(df_result[["Amount"]])
        df_result["Anomaly"] = df_result["Anomaly"].map({1: "Normal", -1: "Bill Shock"})

        anomalies = df_result[df_result["Anomaly"] == "Bill Shock"]
        logger.info(f"Found {len(anomalies)} anomalies ({len(anomalies)/len(df)*100:.1f}%)")

        return anomalies

    except Exception as e:
        logger.error(f"Error during anomaly detection: {str(e)}")
        return None

# Train model when script is run directly
if __name__ == "__main__":
    train_anomaly_model()