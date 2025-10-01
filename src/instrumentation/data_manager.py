# src/instrumentation/data_manager.py
from __future__ import annotations

"""DataManager: transformations de données et préparation ML avec télémétrie structurée."""

# Décorateurs: import robuste avec fallback no-op
try:
    from decorators import log_call
except Exception:  # pragma: no cover
    from typing import Callable, TypeVar, ParamSpec

    T = TypeVar("T")
    P = ParamSpec("P")

    def log_call(name: str | None = None) -> Callable[[Callable[P, T]], Callable[P, T]]:  # type: ignore[override]
        def deco(fn: Callable[P, T]) -> Callable[P, T]:
            return fn
        return deco

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd


class DataManager:
    """Manage pure data transformations and ML preparation."""

    # Class constants
    NUMERIC_TYPES = {"int64", "float64", "int32", "float32"}
    CATEGORICAL_THRESHOLD = 0.1  # If <10% unique values -> categorical
    TARGET_CANDIDATES = {"target", "label", "class", "y", "Target", "Label"}
    MIN_SAMPLES_THRESHOLD = 10

    @log_call("data_manager.__init__")
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize DataManager with configuration."""
        # Accepter dict ou objet type Pydantic ayant model_dump fait en amont
        self.config = config or {}

    # ---------- IO helpers ----------
    @staticmethod
    @log_call("data_manager.load_csv")
    def load_csv(path: Path, encoding: str | None = None, sep: str | None = None, **kwargs) -> pd.DataFrame:
        """Charger un CSV avec encodage/séparateur optionnels."""
        if encoding is not None:
            kwargs["encoding"] = encoding
        if sep is not None:
            kwargs["sep"] = sep
        return pd.read_csv(path, **kwargs)

    @staticmethod
    @log_call("data_manager.load_xlsx")
    def load_xlsx(path: Path, **kwargs) -> pd.DataFrame:
        """Charger un fichier Excel (xlsx/xls)."""
        return pd.read_excel(path, **kwargs)

    @staticmethod
    @log_call("data_manager.load_json")
    def load_json(path: Path, **kwargs) -> pd.DataFrame:
        """Charger un JSON tabulaire (records/lines selon le contenu)."""
        return pd.read_json(path, **kwargs)

    @staticmethod
    @log_call("data_manager.load_from_path")
    def load_from_path(path: Path, encoding: str | None = None, sep: str | None = None, **kwargs) -> pd.DataFrame:
        """Heuristique de chargement par extension."""
        suf = path.suffix.lower()
        if suf == ".csv":
            return DataManager.load_csv(path, encoding=encoding, sep=sep, **kwargs)
        if suf in (".xlsx", ".xls"):
            return DataManager.load_xlsx(path, **kwargs)
        if suf == ".json":
            return DataManager.load_json(path, **kwargs)
        # Fallback CSV
        return DataManager.load_csv(path, encoding=encoding, sep=sep, **kwargs)

    @log_call("data_manager.load_from_raw")
    def load_from_raw(self, raw_data: Any, encoding: str | None = None, sep: str | None = None, **kwargs) -> pd.DataFrame:
        """Convertit une donnée brute en DataFrame: DataFrame|dict|list|str|Path|{'path':...}."""
        if isinstance(raw_data, pd.DataFrame):
            return raw_data.copy()
        if isinstance(raw_data, dict) and "path" not in raw_data:
            return pd.DataFrame([raw_data])
        if isinstance(raw_data, list):
            return pd.DataFrame(raw_data)
        # chemins
        if isinstance(raw_data, (str, Path)):
            return self.load_from_path(Path(raw_data), encoding=encoding, sep=sep, **kwargs)
        if isinstance(raw_data, dict) and "path" in raw_data:
            return self.load_from_path(Path(raw_data["path"]), encoding=encoding, sep=sep, **kwargs)
        raise ValueError(f"Unsupported raw data type: {type(raw_data)}")

    # ---------- Inference / cleaning ----------
    @log_call("data_manager.infer_target_column")
    def infer_target_column(self, df: pd.DataFrame) -> str | None:
        """Retourne la cible explicite si configurée; sinon auto-détection optionnelle."""
        cfg = self.config or {}
        target_col = cfg.get("target_column")
        auto_detect = cfg.get("auto_detect_target", True)

        # Priorité à la colonne explicitement fournie
        if target_col:
            return target_col if target_col in df.columns else None

        # Si auto-détection désactivée, ne rien inférer
        if not auto_detect:
            return None

        # Auto-détection par conventions basiques
        for col in df.columns:
            if col in self.TARGET_CANDIDATES:
                return col
        return None

    @log_call("data_manager.clean_data")
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique la politique de nettoyage: doublons, valeurs manquantes, colonnes à exclure."""
        df_clean = df.copy()

        # Doublons
        try:
            df_clean = df_clean.drop_duplicates()
        except Exception:
            pass

        # Politique valeurs manquantes
        missing_strategy = (self.config or {}).get("missing_strategy", "auto")
        if missing_strategy == "drop":
            df_clean = df_clean.dropna()
        elif missing_strategy == "fill":
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns
            try:
                df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
                    df_clean[numeric_cols].median(numeric_only=True)
                )
            except Exception:
                df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
            try:
                df_clean[categorical_cols] = df_clean[categorical_cols].fillna("Unknown")
            except Exception:
                pass
        elif missing_strategy == "auto":
            # Heuristique par défaut: ne rien faire ici (pipeline aval gère si besoin)
            pass
        else:
            # Valeur inconnue: fallback sûr (ne rien faire)
            pass

        # Colonnes à supprimer (si listées)
        drop_cols = (self.config or {}).get("drop_columns", [])
        if drop_cols:
            keep = [c for c in df_clean.columns if c not in drop_cols]
            df_clean = df_clean[keep]

        return df_clean

    @log_call("data_manager.infer_column_types")
    def infer_column_types(self, df: pd.DataFrame) -> dict[str, str]:
        """Infère un type 'numeric'/'categorical' par colonne."""
        type_map: dict[str, str] = {}
        n = len(df)
        for col in df.columns:
            if n == 0 or df[col].isna().all():
                type_map[col] = "categorical"
                continue
            unique_ratio = df[col].nunique(dropna=True) / max(n, 1)
            if str(df[col].dtype) in self.NUMERIC_TYPES:
                type_map[col] = "categorical" if unique_ratio < self.CATEGORICAL_THRESHOLD else "numeric"
            else:
                type_map[col] = "categororical" if unique_ratio < 1.0 else "categorical"  # keep categorical
                type_map[col] = "categorical"
        return type_map

    # ---------- Split / validation ----------
    @log_call("data_manager.split_features_target")
    def split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series | None]:
        """Sépare X et y selon la config (prioritaire) puis auto-détection."""
        cfg_target = (self.config or {}).get("target_column")
        if cfg_target and cfg_target in df.columns:
            return df.drop(columns=[cfg_target]), df[cfg_target]
        tcol = self.infer_target_column(df)
        if tcol:
            return df.drop(columns=[tcol]), df[tcol]
        return df, None

    @log_call("data_manager.validate_data")
    def validate_data(self, X: pd.DataFrame, y: pd.Series | None = None) -> bool:
        """Validations de base pour les workflows ML."""
        if len(X) < self.MIN_SAMPLES_THRESHOLD:
            raise ValueError(f"Insufficient samples: {len(X)} < {self.MIN_SAMPLES_THRESHOLD}")
        if y is not None and len(X) != len(y):
            raise ValueError(f"Feature/target length mismatch: {len(X)} != {len(y)}")
        return True

    # ---------- Main entry ----------
    @log_call("data_manager.prepare_for_ml")
    def prepare_for_ml(self, raw_or_df: Any) -> Tuple[pd.DataFrame, pd.Series | None]:
        """Pipeline de préparation: load → clean → split → validate."""
        # 1) Charger, en acceptant DataFrame ou données brutes
        if isinstance(raw_or_df, pd.DataFrame):
            df = raw_or_df
        else:
            df = self.load_from_raw(
                raw_or_df,
                encoding=(self.config or {}).get("encoding"),
                sep=(self.config or {}).get("sep"),
            )

        # 2) Nettoyage selon la policy
        df = self.clean_data(df)

        # 3) Split
        X, y = self.split_features_target(df)

        # 4) Validation
        self.validate_data(X, y)

        return X, y
