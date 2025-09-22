from __future__ import annotations

from typing import Final

# Export/defaults
DEF_MODEL_PATH: Final = "outputs/dl/final_model.keras"
DEF_CKPT_PATH: Final = "outputs/dl/best_model.keras"
DEF_MONITOR: Final = "val_loss"

# Tâches et types de modèle
TASK_BINARY: Final = "binary"
TASK_MULTICLASS: Final = "multiclass"
MODEL_SEQUENTIAL: Final = "sequential"
MODEL_FUNCTIONAL: Final = "functional"

# Noms de couches supportées
LAYER_NAMES: Final = frozenset({"Dense", "Dropout", "BatchNormalization", "Activation", "Input"})

# Noms standardisés de modèles Keras construits
MODEL_NAME_SEQ: Final = "dl_sequential"
MODEL_NAME_FUNC: Final = "dl_functional"
