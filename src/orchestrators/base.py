from __future__ import annotations

from typing import Any

import pandas as pd


class IOrchestrator:
    """Interface minimale pour les orchestrateurs."""

    def run(self, X: pd.DataFrame, y: pd.Series | None = None) -> dict[str, Any]:
        """Exécuter l'orchestrateur et retourner un dict de résultats."""
        raise NotImplementedError
