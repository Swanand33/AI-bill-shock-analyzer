import pytest
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.validation import (
    validate_csv_file,
    validate_dataframe,
    validate_contamination,
    validate_model_file
)


class TestValidateCsvFile:
    """Tests for CSV file validation"""

    def test_nonexistent_file(self):
        """Test validation fails for non-existent file"""
        is_valid, error_msg = validate_csv_file(Path("nonexistent.csv"))
        assert is_valid is False
        assert "does not exist" in error_msg

    def test_wrong_extension(self, tmp_path):
        """Test validation fails for non-CSV file"""
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("test")
        is_valid, error_msg = validate_csv_file(txt_file)
        assert is_valid is False
        assert "CSV format" in error_msg


class TestValidateDataFrame:
    """Tests for DataFrame validation"""

    def test_empty_dataframe(self):
        """Test validation fails for empty DataFrame"""
        df = pd.DataFrame()
        is_valid, error_msg = validate_dataframe(df)
        assert is_valid is False
        assert "empty" in error_msg

    def test_missing_required_column(self):
        """Test validation fails when required column missing"""
        df = pd.DataFrame({"Price": [100, 200, 300]})
        is_valid, error_msg = validate_dataframe(df, "Amount")
        assert is_valid is False
        assert "Amount" in error_msg
        assert "not found" in error_msg

    def test_non_numeric_column(self):
        """Test validation fails for non-numeric Amount column"""
        df = pd.DataFrame({"Amount": ["abc", "def", "ghi"]})
        is_valid, error_msg = validate_dataframe(df)
        assert is_valid is False
        assert "numeric" in error_msg

    def test_all_nan_values(self):
        """Test validation fails when all values are NaN"""
        df = pd.DataFrame({"Amount": [None, None, None]})
        is_valid, error_msg = validate_dataframe(df)
        assert is_valid is False
        assert "NaN" in error_msg

    def test_valid_dataframe(self):
        """Test validation passes for valid DataFrame"""
        df = pd.DataFrame({"Amount": [100, 200, 300]})
        is_valid, error_msg = validate_dataframe(df)
        assert is_valid is True
        assert error_msg is None


class TestValidateContamination:
    """Tests for contamination parameter validation"""

    def test_below_range(self):
        """Test validation fails for value below range"""
        is_valid, error_msg = validate_contamination(0.001)
        assert is_valid is False
        assert "0.01 and 0.5" in error_msg

    def test_above_range(self):
        """Test validation fails for value above range"""
        is_valid, error_msg = validate_contamination(0.6)
        assert is_valid is False
        assert "0.01 and 0.5" in error_msg

    def test_non_numeric(self):
        """Test validation fails for non-numeric value"""
        is_valid, error_msg = validate_contamination("abc")
        assert is_valid is False
        assert "must be a number" in error_msg

    def test_valid_contamination(self):
        """Test validation passes for valid value"""
        is_valid, error_msg = validate_contamination(0.05)
        assert is_valid is True
        assert error_msg is None


class TestValidateModelFile:
    """Tests for model file validation"""

    def test_nonexistent_model(self):
        """Test validation fails for non-existent model"""
        is_valid, error_msg = validate_model_file(Path("nonexistent.pkl"))
        assert is_valid is False
        assert "not found" in error_msg

    def test_wrong_extension(self, tmp_path):
        """Test validation fails for non-PKL file"""
        txt_file = tmp_path / "model.txt"
        txt_file.write_text("test")
        is_valid, error_msg = validate_model_file(txt_file)
        assert is_valid is False
        assert ".pkl" in error_msg
