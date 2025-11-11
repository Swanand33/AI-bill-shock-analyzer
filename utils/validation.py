import pandas as pd
from pathlib import Path
from typing import Optional, Tuple


def validate_csv_file(file_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate if CSV file exists and is readable.

    Args:
        file_path: Path to CSV file

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    if not file_path.exists():
        return False, f"File '{file_path}' does not exist"

    if not file_path.is_file():
        return False, f"'{file_path}' is not a file"

    if file_path.suffix.lower() not in ['.csv']:
        return False, f"File must be CSV format, got {file_path.suffix}"

    return True, None


def validate_dataframe(df: pd.DataFrame, required_column: str = "Amount") -> Tuple[bool, Optional[str]]:
    """
    Validate if DataFrame has required structure.

    Args:
        df: DataFrame to validate
        required_column: Name of required column (default: "Amount")

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"

    if df.empty:
        return False, "DataFrame is empty"

    if required_column not in df.columns:
        available = ", ".join(df.columns)
        return False, f"Required column '{required_column}' not found. Available: {available}"

    if not pd.api.types.is_numeric_dtype(df[required_column]):
        return False, f"Column '{required_column}' must contain numeric values"

    # Check for all NaN
    if df[required_column].isna().all():
        return False, f"Column '{required_column}' contains only NaN values"

    return True, None


def validate_contamination(contamination: float) -> Tuple[bool, Optional[str]]:
    """
    Validate contamination parameter.

    Args:
        contamination: Expected proportion of anomalies (should be between 0.01 and 0.5)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(contamination, (int, float)):
        return False, f"Contamination must be a number, got {type(contamination)}"

    if not 0.01 <= contamination <= 0.5:
        return False, f"Contamination must be between 0.01 and 0.5, got {contamination}"

    return True, None


def validate_model_file(model_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate if model file exists and is readable.

    Args:
        model_path: Path to model file

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(model_path, Path):
        model_path = Path(model_path)

    if not model_path.exists():
        return False, f"Model file '{model_path}' not found. Train the model first."

    if not model_path.is_file():
        return False, f"'{model_path}' is not a file"

    if model_path.suffix.lower() not in ['.pkl', '.pickle']:
        return False, f"Model file must be .pkl format, got {model_path.suffix}"

    return True, None
