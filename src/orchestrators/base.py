from __future__ import annotations

from typing import Any

import pandas as pd


class IOrchestrator:
    def run(self, X: pd.DataFrame, y: pd.Series | None = None) -> dict[str, Any]:
        raise NotImplementedError
