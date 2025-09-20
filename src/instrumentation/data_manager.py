# src/instrumentation/data_manager.py
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
        elif strategy == "fill":
            num = df_clean.select_dtypes(include=[np.number]).columns
            cat = df_clean.select_dtypes(exclude=[np.number]).columns
            df_clean[num] = df_clean[num].fillna(df_clean[num].median())
            df_clean[cat] = df_clean[cat].fillna("Unknown")

        # Filtrer les colonnes vides et n’enlever que celles présentes
        drops = (self.config or {}).get("drop_columns", []) or []
        drops = [c.strip() if isinstance(c, str) else c for c in drops]
        drops = [c for c in drops if c and c in df_clean.columns]
        if drops:
            df_clean = df_clean.drop(columns=drops)

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
        # Forcer la cible si configurée
        cfg_target = (self.config or {}).get("target_column")
        if cfg_target and cfg_target in df.columns:
            return df.drop(columns=[cfg_target]), df[cfg_target]

        # Sinon fallback auto-détection
        tcol = self.infer_target_column(df)
        if tcol:
            return df.drop(columns=[tcol]), df[tcol]
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
        df = self.clean_data(df)
        X, y = self.split_features_target(df)
        self.validate_data(X, y)
        return X, y
