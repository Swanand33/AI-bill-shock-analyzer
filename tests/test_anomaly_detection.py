import pytest
import pandas as pd
from pathlib import Path
import sys
import joblib
from sklearn.ensemble import IsolationForest

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.anomaly_detection import train_anomaly_model, detect_anomalies


class TestTrainAnomalyModel:
    """Tests for model training"""

    def test_train_with_valid_data(self, tmp_path):
        """Test training succeeds with valid data"""
        # Create test CSV
        csv_file = tmp_path / "transactions.csv"
        df = pd.DataFrame({
            "Amount": [100, 150, 200, 250, 300, 350, 400, 9000]  # 9000 is anomaly
        })
        df.to_csv(csv_file, index=False)

        # Train model
        model_file = tmp_path / "model.pkl"
        result = train_anomaly_model(
            file_path=csv_file,
            model_path=model_file,
            contamination=0.1
        )

        assert result is True
        assert model_file.exists()

    def test_train_with_missing_file(self, tmp_path):
        """Test training fails with missing file"""
        csv_file = tmp_path / "nonexistent.csv"
        model_file = tmp_path / "model.pkl"

        result = train_anomaly_model(
            file_path=csv_file,
            model_path=model_file
        )

        assert result is False
        assert not model_file.exists()

    def test_train_with_empty_csv(self, tmp_path):
        """Test training fails with empty CSV"""
        csv_file = tmp_path / "empty.csv"
        df = pd.DataFrame()
        df.to_csv(csv_file, index=False)

        model_file = tmp_path / "model.pkl"
        result = train_anomaly_model(
            file_path=csv_file,
            model_path=model_file
        )

        assert result is False

    def test_train_with_missing_amount_column(self, tmp_path):
        """Test training fails when Amount column is missing"""
        csv_file = tmp_path / "no_amount.csv"
        df = pd.DataFrame({"Price": [100, 200, 300]})
        df.to_csv(csv_file, index=False)

        model_file = tmp_path / "model.pkl"
        result = train_anomaly_model(
            file_path=csv_file,
            model_path=model_file
        )

        assert result is False

    def test_train_with_invalid_contamination(self, tmp_path):
        """Test training uses default contamination for invalid values"""
        csv_file = tmp_path / "transactions.csv"
        df = pd.DataFrame({"Amount": [100, 200, 300, 400, 500]})
        df.to_csv(csv_file, index=False)

        model_file = tmp_path / "model.pkl"
        result = train_anomaly_model(
            file_path=csv_file,
            model_path=model_file,
            contamination=0.9  # Invalid, should use default 0.05
        )

        # Should still succeed with default value
        assert result is True


class TestDetectAnomalies:
    """Tests for anomaly detection"""

    def test_detect_with_valid_data(self, tmp_path):
        """Test detection works with valid data and model"""
        # Create and train model
        model = IsolationForest(contamination=0.2, random_state=42)
        df_train = pd.DataFrame({"Amount": [100, 150, 200, 250, 300]})
        model.fit(df_train)

        model_file = tmp_path / "model.pkl"
        joblib.dump(model, model_file)

        # Test detection
        df_test = pd.DataFrame({"Amount": [120, 180, 9000]})  # 9000 should be anomaly
        anomalies = detect_anomalies(df_test, model_path=model_file)

        assert anomalies is not None
        assert len(anomalies) > 0
        assert 9000 in anomalies["Amount"].values

    def test_detect_with_missing_model(self, tmp_path):
        """Test detection fails when model doesn't exist"""
        df = pd.DataFrame({"Amount": [100, 200, 300]})
        model_file = tmp_path / "nonexistent.pkl"

        anomalies = detect_anomalies(df, model_path=model_file)

        assert anomalies is None

    def test_detect_with_empty_dataframe(self, tmp_path):
        """Test detection fails with empty DataFrame"""
        # Create dummy model
        model = IsolationForest(contamination=0.1, random_state=42)
        df_train = pd.DataFrame({"Amount": [100, 200, 300]})
        model.fit(df_train)

        model_file = tmp_path / "model.pkl"
        joblib.dump(model, model_file)

        # Test with empty DataFrame
        df_empty = pd.DataFrame()
        anomalies = detect_anomalies(df_empty, model_path=model_file)

        assert anomalies is None

    def test_detect_with_missing_amount_column(self, tmp_path):
        """Test detection fails when Amount column is missing"""
        # Create dummy model
        model = IsolationForest(contamination=0.1, random_state=42)
        df_train = pd.DataFrame({"Amount": [100, 200, 300]})
        model.fit(df_train)

        model_file = tmp_path / "model.pkl"
        joblib.dump(model, model_file)

        # Test with wrong column
        df_test = pd.DataFrame({"Price": [100, 200, 300]})
        anomalies = detect_anomalies(df_test, model_path=model_file)

        assert anomalies is None

    def test_detect_returns_only_anomalies(self, tmp_path):
        """Test that only anomalies are returned, not normal transactions"""
        # Create and train model
        model = IsolationForest(contamination=0.1, random_state=42)
        df_train = pd.DataFrame({"Amount": [100, 150, 200, 250, 300]})
        model.fit(df_train)

        model_file = tmp_path / "model.pkl"
        joblib.dump(model, model_file)

        # Test detection
        df_test = pd.DataFrame({"Amount": [120, 150, 180]})
        anomalies = detect_anomalies(df_test, model_path=model_file)

        if anomalies is not None:
            # All returned rows should be labeled as "Bill Shock"
            assert all(anomalies["Anomaly"] == "Bill Shock")
