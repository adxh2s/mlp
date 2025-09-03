from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import streamlit as st
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from src.instrumentation.config_manager import ConfigManager
from src.instrumentation.messages_manager import MessageManager
from src.instrumentation.messages_taxonomy import STREAMLIT_INIT
from src.orchestrators.config import ConfigOrchestrator
from src.orchestrators.messages import MessageOrchestrator
from src.ui.constants import (
    DEFAULT_OUTPUTS,
    DEFAULT_PROJECT,
    DEFAULT_TEXTS,
    KEY_APP_TITLE,
    KEY_CFG_MISSING,
    KEY_CFG_READ_ERROR,
    KEY_OUTPUTS_DIR_LABEL,
    KEY_PAGE_EDA,
    KEY_PAGE_HOME,
    KEY_PAGE_PIPES,
    KEY_PAGE_REPORTS,
    KEY_PROJECT_LABEL,
    KEY_ROOT_PREFIX,
    KEY_SIDEBAR_SECTION,
    UI_DOMAIN,
)

"""
Class-based Streamlit app using ConfigOrchestrator + MessageOrchestrator
and a MessageManager for UI i18n under the "streamlit_app" domain.

- Loads Hydra config safely with explicit typing to satisfy Pylance.
- Emits a localized init event via MessageOrchestrator.
- Resolves UI labels via MessageManager with robust fallbacks.
- Uses modern multipage API: st.Page + st.navigation.
"""


CONF_PATH = Path("conf/config.yaml")


def _load_cfg_safe(conf_path: Path) -> tuple[dict[str, Any], str]:
    """Load Hydra config safely, returning (dict, message_info)."""
    if not conf_path.exists():
        return {}, f"Configuration absente: {conf_path}"
    try:
        cfg = OmegaConf.load(conf_path)
        raw_any = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(raw_any, dict):
            return {}, "La configuration résolue n'est pas un dictionnaire racine."
        raw: dict[str, Any] = cast(dict[str, Any], raw_any)
        return raw, ""
    except OmegaConfBaseException as exc:
        return {}, f"Erreur de lecture configuration: {exc}"


class MLPStreamlitApp:
    """Streamlit app wrapper managing config, messages, and UI i18n."""

    def __init__(self) -> None:
        """Initialize placeholders for orchestrators and UI MessageManager."""
        self.cfg_orch: ConfigOrchestrator | None = None
        self.msg_orch: MessageOrchestrator | None = None
        self.ui_mm: MessageManager | None = None

    def _ui(self, key: str, **params: Any) -> str:
        """
        Resolve a UI string by key using MessageManager on UI_DOMAIN.

        Falls back to DEFAULT_TEXTS if translation or key is missing.
        """
        default = DEFAULT_TEXTS.get(key, key)
        if self.ui_mm is None:
            return default.format(**params) if params else default
        text = self.ui_mm.msg(UI_DOMAIN, key, None, **params)
        if text == key:
            return default.format(**params) if params else default
        return text

    def get_ui_text(self, key: str, **params: Any) -> str:
        """
        Public UI text resolver for pages and tests.

        Wraps the internal _ui() method.
        """
        return self._ui(key, **params)

    def bootstrap(self) -> None:
        """Initialize Config and Message orchestrators, and UI MessageManager."""
        if (
            "cfg_orch" in st.session_state
            and "msg_orch" in st.session_state
            and "ui_mm" in st.session_state
        ):
            self.cfg_orch = cast(ConfigOrchestrator, st.session_state["cfg_orch"])
            self.msg_orch = cast(MessageOrchestrator, st.session_state["msg_orch"])
            self.ui_mm = cast(MessageManager, st.session_state["ui_mm"])
            return

        raw_any, info = _load_cfg_safe(CONF_PATH)
        if info:
            # Localized warnings via UI keys
            if "absente" in info:
                st.warning(self._ui(KEY_CFG_MISSING, path=str(CONF_PATH)))
            else:
                st.warning(self._ui(KEY_CFG_READ_ERROR, error=info))
            # Typed fallback config
            raw: dict[str, Any] = cast(
                dict[str, Any],
                {
                    "project": {
                        "name": DEFAULT_PROJECT,
                        "random_state": 42,
                        "output_dir": DEFAULT_OUTPUTS,
                    },
                    "orchestrators": {
                        "config": {"enabled": True},
                        "messages": {
                            "enabled": True,
                            "locale": "fr",
                            "locales_dir": "i18n/locales",
                            "domains": [
                                "general",
                                "config",
                                "file",
                                "data",
                                "eda",
                                "pipelines",
                                "report",
                                "streamlit_app",
                            ],
                        },
                    },
                    "logger": {
                        "backend": "stdlib",
                        "app_name": "mlp",
                        "level": "INFO",
                        "json_mode": False,
                        "file_path": None,
                        "default_fields": {"app": "mlp", "env": "dev"},
                    },
                },
            )
        else:
            raw = raw_any

        cfg_mgr = ConfigManager(OmegaConf.create(raw))
        self.cfg_orch = ConfigOrchestrator(cfg_mgr)
        self.msg_orch = MessageOrchestrator(
            cfg_mgr,
            logger_manager=self.cfg_orch.get_logger_manager(),
        )

        # UI MessageManager uses messages orchestrator locale/dir
        locales_dir = (
            Path(self.cfg_orch.get_config_manager().project_root)
            / cast(
                str,
                cast(dict[str, Any], cfg_mgr.raw.get("orchestrators", {})).get(
                    "messages", {}
                ).get("locales_dir", "i18n/locales"),
            )
        )
        default_locale = cast(
            str,
            cast(dict[str, Any], cfg_mgr.raw.get("orchestrators", {})).get(
                "messages", {}
            ).get("locale", "fr"),
        )
        self.ui_mm = MessageManager(locales_dir, default_locale=default_locale)  # type: ignore[arg-type]

        st.session_state["cfg_orch"] = self.cfg_orch
        st.session_state["msg_orch"] = self.msg_orch
        st.session_state["ui_mm"] = self.ui_mm

        self.msg_orch.emit("general", STREAMLIT_INIT, title=self._ui(KEY_APP_TITLE))

    def sidebar(self) -> None:
        """Render sidebar and persist chosen parameters."""
        st.sidebar.header(self._ui(KEY_SIDEBAR_SECTION))
        outputs_dir = st.sidebar.text_input(
            self._ui(KEY_OUTPUTS_DIR_LABEL),
            value=DEFAULT_OUTPUTS,
        )
        project_name = st.sidebar.text_input(
            self._ui(KEY_PROJECT_LABEL),
            value=DEFAULT_PROJECT,
        )
        root = Path(outputs_dir) / project_name
        st.sidebar.caption(f"{self._ui(KEY_ROOT_PREFIX)}{root}")
        st.session_state["outputs_dir"] = outputs_dir
        st.session_state["project_name"] = project_name

    def navigation(self) -> None:
        """Define and run programmatic navigation."""
        import streamlit_pages.pipelines as pipes  # noqa: PLC0415
        from streamlit_pages import eda, home, reports  # noqa: PLC0415

        pg_home = st.Page(home.run, title=self._ui(KEY_PAGE_HOME), icon="🏠", default=True)
        pg_eda = st.Page(eda.run, title=self._ui(KEY_PAGE_EDA), icon="🧭")
        pg_pipes = st.Page(pipes.run, title=self._ui(KEY_PAGE_PIPES), icon="🧪")
        pg_reports = st.Page(reports.run, title=self._ui(KEY_PAGE_REPORTS), icon="📄")
        st.navigation([pg_home, pg_eda, pg_pipes, pg_reports]).run()

    def run(self) -> None:
        """Entrypoint to render and run the app."""
        st.set_page_config(page_title=self._ui(KEY_APP_TITLE), page_icon="📊", layout="wide")
        st.title(self._ui(KEY_APP_TITLE))
        self.bootstrap()
        self.sidebar()
        self.navigation()
