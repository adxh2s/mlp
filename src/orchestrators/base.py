from __future__ import annotations
from typing import Any, Dict, Optional
import pandas as pd

class IOrchestrator:
    def run(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        raise NotImplementedError
