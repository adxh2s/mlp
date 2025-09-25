from __future__ import annotations

from pathlib import Path

from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from src.instrumentation.config_manager import ConfigManager
from src.instrumentation.logger_manager import LoggerManager
from src.orchestrators.config import ConfigOrchestrator
from src.orchestrators.logger import LoggerOrchestrator
from src.orchestrators.message import MessageOrchestrator

"""
App orchestrator: bootstrap logging and configuration, and build a Hydra-safe ctx.
"""


class AppOrchestrator:
    """Boot logger + config, expose logger_manager, config_manager and ctx."""

    def __init__(self, hydra_cfg: DictConfig) -> None:
        # 1) Config manager (Hydra -> Pydantic)
        self.config_manager = ConfigManager(hydra_cfg)

        # 2) Logger bootstrap first
        self.logger_orchestrator = LoggerOrchestrator(hydra_cfg)
        self.logger_manager: LoggerManager = self.logger_orchestrator.run(self.config_manager)

        # 3) Config orchestrator: validate and expose AppConfig + message
        self.config_orchestrator = ConfigOrchestrator(self.config_manager, logger_manager=self.logger_manager)
        app_cfg = self.config_orchestrator.get_app_config()

        # 4) Message (shared) for downstream orchestrators
        self.message_orchestrator = MessageOrchestrator(self.config_manager, logger_manager=self.logger_manager)

        # 5) Build ctx (Hydra-safe absolute paths)
        root = Path(get_original_cwd())
        outputs_root = (root / app_cfg.project.output_dir).resolve()
        project_dir = (outputs_root / app_cfg.project.name).resolve()

        file_cfg = getattr(app_cfg.orchestrators, "file", None)
        if file_cfg is None:
            self.message_orchestrator.emit(
                "config",
                "config_section_missing",
                section="orchestrators.file",
                used_defaults={"data_dir": "data", "in_dir": "in", "out_dir": "out"},
            )
            data_dir, in_dir, out_dir = "data", "in", "out"
        else:
            data_dir, in_dir, out_dir = file_cfg.data_dir, file_cfg.in_dir, file_cfg.out_dir

        data_root = (root / data_dir).resolve()
        data_in = (data_root / in_dir).resolve()
        data_out = (data_root / out_dir).resolve()
        eda_dir = (project_dir / "eda").resolve()
        report_dir = (project_dir / "report").resolve()

        for d in (outputs_root, project_dir, data_in, data_out, eda_dir, report_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.ctx: dict[str, str] = {
            "root_dir": str(root),
            "outputs_root": str(outputs_root),
            "project_dir": str(project_dir),
            "data_root": str(data_root),
            "data_in": str(data_in),
            "data_out": str(data_out),
            "eda_dir": str(eda_dir),
            "report_dir": str(report_dir),
        }
