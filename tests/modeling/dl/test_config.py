from __future__ import annotations

from src.modeling.dl.config import DLConfig, LayerSpec


def test_layers_default_is_typed_list() -> None:
    cfg = DLConfig()
    assert isinstance(cfg.model.layers, list)
    assert cfg.model.layers == []

    cfg2 = DLConfig(model={"layers": [LayerSpec(type="Dense", params={"units": 8}).model_dump()]})
    assert isinstance(cfg2.model.layers, list)
    assert isinstance(cfg2.model.layers[0], LayerSpec)
