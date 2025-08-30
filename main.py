from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.instrumentation.config_manager import ConfigManager
from src.instrumentation.logger_factory import build_logger_manager
from src.orchestrators.general import GeneralOrchestrator


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg_mgr = ConfigManager(cfg)
    app_cfg = cfg_mgr.load()

    # Init logger via ConfigurationManager
    lm = build_logger_manager(app_cfg.logger)
    lm.configure()
    log = lm.get_logger(__name__)
    log.info("app_start", extra={"extra_fields": {"entry": "main"}})

    # demo dataset and run...
    from sklearn.datasets import load_breast_cancer

    ds = load_breast_cancer(as_frame=True)
    X = ds.frame.drop(columns=["target"])
    y = ds.frame["target"]

    go = GeneralOrchestrator(cfg_mgr)
    res = go.run(X, y)
    log.info("app_done", extra={"extra_fields": {"report_artifacts": res.get('report', {}).get('artifacts')}})


if __name__ == "__main__":
    main()
