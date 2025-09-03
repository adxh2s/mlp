from __future__ import annotations

from pathlib import Path

# Standard library
from typing import Any, Union

import numpy as np

# Third-party
import pandas as pd

# Local
from src.instrumentation.logger_mixin import LoggerMixin


class DataManager:
    """Manage pure data transformations and ML preparation.

    Handles data loading, cleaning, validation, type inference,
    and X/y splitting for machine learning workflows.
    """

    # Class constants
    NUMERIC_TYPES = {"int64", "float64", "int32", "float32"}
    CATEGORICAL_THRESHOLD = 0.1  # If <10% unique values -> categorical
    TARGET_CANDIDATES = {"target", "label", "class", "y", "Target", "Label"}
    MIN_SAMPLES_THRESHOLD = 10

    def __init__(self, config=None) -> None:
        """Initialize DataManager with configuration."""
        self.config = config or {}

    def load_from_raw(self, raw_data: Any) -> pd.DataFrame:
        """Convert raw data (dict, DataFrame, etc.) into pandas DataFrame."""
        if isinstance(raw_data, pd.DataFrame):
            return raw_data.copy()
        elif isinstance(raw_data, dict):
            return pd.DataFrame([raw_data])  # Single record
        elif isinstance(raw_data, list):
            return pd.DataFrame(raw_data)
        else:
            raise ValueError(f"Unsupported raw data type: {type(raw_data)}")

    def infer_target_column(self, df: pd.DataFrame) -> str | None:
        """Auto-detect the target column based on naming conventions."""
        target_col = self.config.get("target_column")
        if target_col and target_col in df.columns:
            return target_col

        # Auto-detection
        for col in df.columns:
            if col in self.TARGET_CANDIDATES:
                return col
        return None

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data cleaning: duplicates, missing values, outliers."""
        df_clean = df.copy()

        # Remove duplicates
        initial_shape = df_clean.shape
        df_clean = df_clean.drop_duplicates()

        # Handle missing values based on strategy
        missing_strategy = self.config.get("missing_strategy", "auto")
        if missing_strategy == "drop":
            df_clean = df_clean.dropna()
        elif missing_strategy == "fill":
            # Simple fill strategy
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns

            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
            df_clean[categorical_cols] = df_clean[categorical_cols].fillna("Unknown")

        # Drop columns if requested
        drop_cols = self.config.get("drop_columns", [])
        df_clean = df_clean.drop(columns=[col for col in drop_cols if col in df_clean.columns])

        return df_clean

    def infer_column_types(self, df: pd.DataFrame) -> dict[str, str]:
        """Infer optimal data types for each column."""
        type_map = {}
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if df[col].dtype in self.NUMERIC_TYPES:
                if unique_ratio < self.CATEGORICAL_THRESHOLD:
                    type_map[col] = "categorical"
                else:
                    type_map[col] = "numeric"
            else:
                type_map[col] = "categorical"
        return type_map

    def split_features_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
        """Split DataFrame into features (X) and target (y)."""
        target_col = self.infer_target_column(df)

        if target_col:
            X = df.drop(columns=[target_col])
            y = df[target_col]
            return X, y
        else:
            return df, None

    def validate_data(self, X: pd.DataFrame, y: pd.Series | None = None) -> bool:
        """Validate prepared data for ML workflows."""
        if len(X) < self.MIN_SAMPLES_THRESHOLD:
            raise ValueError(f"Insufficient samples: {len(X)} < {self.MIN_SAMPLES_THRESHOLD}")

        if y is not None and len(X) != len(y):
            raise ValueError(f"Feature/target length mismatch: {len(X)} != {len(y)}")

        return True

    def prepare_for_ml(self, raw_data: Any) -> tuple[pd.DataFrame, pd.Series | None]:
        """Full preparation pipeline: load → clean → split → validate."""
        # 1. Load data
        df = self.load_from_raw(raw_data)

        # 2. Clean data
        df_clean = self.clean_data(df)

        # 3. Split X/y
        X, y = self.split_features_target(df_clean)

        # 4. Validate
        self.validate_data(X, y)

        return X, y
