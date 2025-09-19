from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class DataManager:
    """Manage pure data transformations and ML preparation."""

    # Class constants
    NUMERIC_TYPES = {"int64", "float64", "int32", "float32"}
    CATEGORICAL_THRESHOLD = 0.1  # If <10% unique values -> categorical
    TARGET_CANDIDATES = {"target", "label", "class", "y", "Target", "Label"}
    MIN_SAMPLES_THRESHOLD = 10

    def __init__(self, config=None) -> None:
        """Initialize DataManager with configuration."""
        # Accepter dict ou objet type Pydantic ayant model_dump fait en amont
        self.config = config or {}

    # ---------- IO helpers ----------

    @staticmethod
    def load_csv(path: Path, encoding: str | None = None, sep: str | None = None, **kwargs) -> pd.DataFrame:
        """Charger un CSV avec encodage/séparateur optionnels."""
        if encoding is not None:
            kwargs["encoding"] = encoding
        if sep is not None:
            kwargs["sep"] = sep
        return pd.read_csv(path, **kwargs)

    def load_from_raw(self, raw_data: Any) -> pd.DataFrame:
        """Convert raw data (dict, DataFrame, list records) into pandas DataFrame."""
        if isinstance(raw_data, pd.DataFrame):
            return raw_data.copy()
        if isinstance(raw_data, dict):
            return pd.DataFrame([raw_data])
        if isinstance(raw_data, list):
            return pd.DataFrame(raw_data)
        raise ValueError(f"Unsupported raw data type: {type(raw_data)}")

    # ---------- Inference / cleaning ----------

    def infer_target_column(self, df: pd.DataFrame) -> str | None:
        """Return explicit target if configured; else optionally auto-detect."""
        cfg = self.config or {}
        target_col = cfg.get("target_column")
        auto_detect = cfg.get("auto_detect_target", True)

        # Priorité à la colonne explicitement fournie
        if target_col:
            return target_col if target_col in df.columns else None

        # Si auto-détection désactivée, ne rien inférer
        if not auto_detect:
            return None

        # Auto-detec par conventions basiques
        for col in df.columns:
            if col in self.TARGET_CANDIDATES:
                return col
        return None

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data cleaning: duplicates, missing values, optional drops."""
        df_clean = df.copy()

        # Remove duplicates
        df_clean = df_clean.drop_duplicates()

        # Handle missing values based on strategy
        missing_strategy = (self.config or {}).get("missing_strategy", "auto")
        if missing_strategy == "drop":
            df_clean = df_clean.dropna()
        elif missing_strategy == "fill":
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
            df_clean[categorical_cols] = df_clean[categorical_cols].fillna("Unknown")

        # Drop columns if requested (filtrer vides/espaces)
        drop_cols = (self.config or {}).get("drop_columns", []) or []
        drop_cols = [c for c in (d.strip() if isinstance(d, str) else d for d in drop_cols) if c and c in df_clean.columns]
        if drop_cols:
            df_clean = df_clean.drop(columns=drop_cols)

        return df_clean

    def infer_column_types(self, df: pd.DataFrame) -> dict[str, str]:
        """Infer optimal data types for each column."""
        type_map: dict[str, str] = {}
        n = len(df)
        for col in df.columns:
            # Si la colonne est entièrement NA, la considérer catégorielle pour éviter divisions
            if n == 0 or df[col].isna().all():
                type_map[col] = "categorical"
                continue
            unique_ratio = df[col].nunique(dropna=True) / max(n, 1)
            if str(df[col].dtype) in self.NUMERIC_TYPES:
                type_map[col] = "categorical" if unique_ratio < self.CATEGORICAL_THRESHOLD else "numeric"
            else:
                type_map[col] = "categorical"
        return type_map

    # ---------- Split / validation ----------

    def split_features_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
        """Split DataFrame into features (X) and target (y) honoring configuration."""
        target_col = self.infer_target_column(df)
        if target_col:
            y = df[target_col]
            X = df.drop(columns=[target_col])
            return X, y
        return df, None

    def validate_data(self, X: pd.DataFrame, y: pd.Series | None = None) -> bool:
        """Basic validations for ML workflows."""
        if len(X) < self.MIN_SAMPLES_THRESHOLD:
            raise ValueError(f"Insufficient samples: {len(X)} < {self.MIN_SAMPLES_THRESHOLD}")
        if y is not None and len(X) != len(y):
            raise ValueError(f"Feature/target length mismatch: {len(X)} != {len(y)}")
        return True

    # ---------- Main entry ----------

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
