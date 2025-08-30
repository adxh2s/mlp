from __future__ import annotations
from typing import Any, Dict, Optional

class SamplersFactory:
    TYPE_SMOTE = "smote"
    TYPE_UNDER = "under"
    TYPE_OVER = "over"

    @staticmethod
    def make_sampler(cfg: Optional[Dict[str, Any]]):
        return None
