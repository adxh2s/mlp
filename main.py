from __future__ import annotations

import hydra
from pathlib import Path
from omegaconf import DictConfig

from src.instrumentation.config_manager import ConfigManager
from src.instrumentation.logger_factory import build_logger_manager
from src.orchestrators.general import GeneralOrchestrator

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg_mgr = ConfigManager(cfg)
    app_cfg = cfg_mgr.load()

    # 1) Construire des LoggerSettings enrichis depuis la conf (file_path absolu -> <racine>/logs/app.log si absent)
    logger_settings = cfg_mgr.build_logger_settings()

    # 2) S’assurer que le dossier parent existe
    if logger_settings.file_path:
        Path(logger_settings.file_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    # 3) Créer et configurer le manager; console + fichier seront actifs si file_path est renseigné
    lm = build_logger_manager(logger_settings)
    lm.configure()
    log = lm.get_logger(__name__)
    log.info("app_start", extra={"extra_fields": {"entry": "main", "log_file": logger_settings.file_path}})

    # demo dataset and run...
    from sklearn.datasets import load_breast_cancer
    ds = load_breast_cancer(as_frame=True)
    X = ds.frame.drop(columns=["target"])
    y = ds.frame["target"]

    go = GeneralOrchestrator(cfg_mgr, logger_manager=lm)
    res = go.run(X, y)
    log.info("app_done", extra={"extra_fields": {"report_artifacts": res.get('report', {}).get('artifacts')}})

if __name__ == "__main__":
    main()
