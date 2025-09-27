# MLP Project Session Context

## 📊 Project Overview
- **Architecture**: Automated analysis of modular ML pipeline
- **Last Updated**: 2025-09-26 13:33
- **Files Analyzed**: 74 Python files, 56 config files
- **Root Path**: `/home/adxh2s/Projects/mlp`

## 🎯 Automated Analysis Results

### 📁 Python Files by Directory
**./:**
- `create_project_from_yaml.py`
- `main.py`
- `session_context_md.py` - Générateur de contexte de session pour projets ML/DS.

Ce module parcourt la base de code afin de pr...
- `streamlit_app.py`

**list/bin/:**
- `activate_this.py` - Activate virtualenv for current interpreter:

import runpy
runpy.run_path(this_file)

This can be us...

**list/lib/python3.11/site-packages/:**
- `_virtualenv.py` - Patches that are applied at runtime to the virtual environment....

**src/:**
- `__init__.py`

**src/config/:**
- `schemas.py`

**src/datanalysis/:**
- `eda_profile.py`
- `eda_summary.py`

**src/datavisualization/:**
- `report_renderer.py`

**src/evaluation/:**
- `metrics.py`
- `splitting.py`

**src/instrumentation/:**
- `bootstrap_logging.py`
- `config_manager.py`
- `data_manager.py`
- `file_manager.py`
- `logger_factory.py`
- `logger_manager.py`
- `logger_manager_structlog.py`
- `logger_mixin.py`
- `message_manager.py`
- `message_taxonomy.py`

**src/modeling/dl/:**
- `config.py`
- `consts.py`
- `factory.py`
- `trainer.py`

**src/modeling/pipeline/:**
- `consts.py`
- `evaluator.py`
- `factory.py`

**src/orchestrators/:**
- `app.py`
- `base.py`
- `config.py`
- `data.py`
- `eda.py`
- `file.py`
- `general.py`
- `logger.py`
- `message.py`
- `pipeline.py`
- `report.py`

**src/preprocessing/:**
- `reducers.py`
- `samplers.py`
- `selectors.py`

**src/ui/:**
- `app.py`
- `constants.py`

**streamlit_pages/:**
- `__init__.py`
- `demo.py`
- `eda.py`
- `home.py`
- `notebook.py`
- `pipeline.py`
- `report.py`
- `utils_runs.py`

**tests/:**
- `conftest.py`
- `test_create_project_from_yaml.py` - Create project directories and files from a YAML specification.

- Reads project_structure.yaml from...

**tests/config/:**
- `test_schemas.py`

**tests/datanalysis/:**
- `test_eda_summary.py`

**tests/datavisualization/:**
- `test_report_renderer.py`

**tests/instrumentation/:**
- `test_file_manager.py`

**tests/integration/:**
- `test_demo_flow.py`

**tests/modeling/dl/:**
- `test_config.py`
- `test_factory.py`
- `test_trainer.py`

**tests/modeling/pipelines/:**
- `test_evaluator.py`
- `test_evaluator_dl.py`
- `test_factory.py`

**tests/orchestrators/:**
- `test_config_orchestrator.py`
- `test_general.py`
- `test_messages_orchestrator.py`
- `test_pipelines_orchestrator.py`

**tests/ui/:**
- `test_app_bootstrap.py`
- `test_app_ui_fallback.py`
- `test_ui_messages.py`


### 📦 Production Dependencies
- `pyyaml>=6.0.1`
- `hydra-core>=1.3.2`
- `omegaconf>=2.3.0`
- `pydantic>=2.5,<3`
- `pandas>=2.2,<2.3`
- `numpy==1.25.2`
- `scipy==1.11.4`
- `scikit-learn==1.3.2`
- `imbalanced-learn>=0.12,<0.13`
- `ydata-profiling>=4.6,<4.7`
- `umap-learn>=0.5.5,<0.6`
- `tpot>=0.12,<0.13`
- `matplotlib>=3.8,<3.9`
- `jinja2>=3.1,<4`
- `streamlit>=1.36,<1.38`
- `lazypredict>=0.2.16,<0.3`
- `plotly>=5.22,<6`
- `structlog>=24.1,<25`
- `mlflow>=2.12,<3`
- `jupyterlab>=4,<5`
- `ipykernel>=6,<7`
- `nbconvert>=7,<8`
- `nbformat>=5,<6`
- `protobuf==4.25.3`
- `tensorflow==2.16.1`
- `setuptools>=80.0`
- `voila>=0.5,<0.6`
- `ipywidgets>=8,<9`
- `jupyter-server>=2,<3`

### 🔧 Development Dependencies
**dev:**
- `pytest>=8.0`
- `pytest-cov>=4.1`
- `mypy>=1.17`
- `ruff>=0.12`
- `pre-commit>=3.7`
- `types-PyYAML`

### ⚙️ Configured Tools
`ruff`, `mypy`, `pytest`, `uv`, `setuptools`


### 🔄 Import Dependencies Flow
**Main Entry Points:**
- `main.py`
 └── __future__.annotations
 └── sys
 └── hydra
 └── omegaconf.DictConfig
 └── omegaconf.OmegaConf

**Orchestrators Chain:**
- `tests/orchestrators/test_messages_orchestrator.py`
- `tests/orchestrators/test_pipelines_orchestrator.py`
- `tests/orchestrators/test_config_orchestrator.py`
- `tests/orchestrators/test_general.py`
- `src/orchestrators/eda.py`
- `src/orchestrators/data.py`
- `src/orchestrators/general.py`
- `src/orchestrators/logger.py`
- `src/orchestrators/file.py`
- `src/orchestrators/report.py`
- `src/orchestrators/pipeline.py`
- `src/orchestrators/base.py`
- `src/orchestrators/app.py`
- `src/orchestrators/config.py`
- `src/orchestrators/message.py`

### 📚 Key Components Documentation
**session_context_md.py:**
Générateur de contexte de session pour projets ML/DS.

Ce module parcourt la base de code afin de produire un fichier Markdown
(session-context.md) décrivant:
- l’arborescence des fichiers Python par dossier,
- les dépendances extraites de pyproject.toml,
- un flux des dépendances d’import,
- un résumé des docstrings,
- une section Configuration (index des YAML sous conf/),
- une section finale d’API complète (signatures + docstrings) pour les modules sous src/.

Conformité:
- PEP 8 (style), Ruf...

**tests/test_create_project_from_yaml.py:**
Create project directories and files from a YAML specification.

- Reads project_structure.yaml from the current working directory.
- Creates missing directories and files without overwriting existing content.
- Applies permission overrides based on glob-like patterns.

**list/bin/activate_this.py:**
Activate virtualenv for current interpreter:

import runpy
runpy.run_path(this_file)

This can be used when you must use an existing Python interpreter, not the virtualenv bin/python.

**list/lib/python3.11/site-packages/_virtualenv.py:**
Patches that are applied at runtime to the virtual environment.


### ⚙️ Configuration
**conf:**
- `conf/config.yaml` — clés: defaults, project, logger, mlflow

**conf/orchestrators/config:**
- `conf/orchestrators/config/config.yaml` — clés: package, enabled

**conf/orchestrators/data:**
- `conf/orchestrators/data/data.yaml` — clés: package, enabled, target_column, auto_detect_target, drop_columns, missing_strategy, categorical_threshold, min_samples_threshold, outlier_detection, encoding, sep

**conf/orchestrators/eda:**
- `conf/orchestrators/eda/eda.yaml` — clés: package, enabled

**conf/orchestrators/file:**
- `conf/orchestrators/file/file.yaml` — clés: package, enabled, data_dir, in_dir, out_dir, extensions, save_input_file, save_input_file_compression

**conf/orchestrators/message:**
- `conf/orchestrators/message/message.yaml` — clés: package, enabled, locale, locales_dir, domains

**conf/orchestrators/pipeline:**
- `conf/orchestrators/pipeline/pipeline.yaml` — clés: package, enabled, out_dir, active, cv, policy, pipeline

**conf/orchestrators/report:**
- `conf/orchestrators/report/report.yaml` — clés: package, enabled, formats

_Cette configuration sera détaillée par les orchestrateurs correspondants._


## 🔧 Coding Standards (Auto-Applied)
- **Style**: PEP8 + Ruff strict compliance
- **Docstrings**: Mandatory (summary + detailed description)
- **Imports**: Ordered (stdlib → third-party → local)
- **Constants**: UPPERCASE at class top
- **Types**: PEP 604 unions (X | None), builtin generics (dict/list/tuple)
- **Format**: Production-ready, copy-pastable code blocks


## 📁 Project Structure Analysis
MLP/
├── Python files: 74
├── Config files: 56
├── Total modules with docs: 4
└── Dependencies analyzed: 29


## 🚀 Current State (Auto-Detected)
- ✅ Files discovered and analyzed automatically
- ✅ Dependencies extracted from configuration
- ✅ Docstrings catalogued and summarized
- ✅ Import relationships mapped
- ✅ Architecture flow documented


## 📋 Session Usage
1. Start new chat with: "Context: MLP project from session-context.md"
2. Attach this file to provide immediate context
3. Mention specific files from the analysis above
4. Standards auto-applied - no need to re-specify


## 🎯 Analysis Insights
- **Most documented module**: src/modeling/pipeline/evaluator.py
- **Main entry points**: 1 detected
- **Orchestrator pattern**: ✅ Detected


### 🧭 API complète (src)
**src/__init__.py**

**src/config/schemas.py**
Classes:
- AppConfig: 
- DataConfig: Configuration for data processing orchestrator.
- EDAConfig: 
- EstimatorConfig: 
- FileConfig: Config section for the file orchestrator.
- LoggerSettings: 
- OrchestratorsConfig: 
- PipelineConfig: 
- PipelineSpec: 
- ProjectConfig: 
- ReductionConfig: 
- ReportConfig: 
- StepsConfig: 

**src/datanalysis/eda_profile.py**
Classes:
- EDAProfile: 
  - _ts() -> str: 
  - generate_profile(df: pd.DataFrame, out_dir: str, minimal: bool = False, title: str = DEFAULT_TITLE) -> str: 

**src/datanalysis/eda_summary.py**
Classes:
- EDASummary: 
  - _ts() -> str: 
  - summarize(X: pd.DataFrame, y: pd.Series | None, out_dir: str) -> tuple[str, dict[str, Any], dict[str, bool]]: 

**src/datavisualization/report_renderer.py**
Classes:
- ReportRenderer: Render HTML/Markdown report from templates and a data context.

Uses Jinja2 templates loaded from a filesystem directory to separate
presentation from logic and to support reusable layouts.
  - __init__(self, templates_dir: str) -> None: Initialize the renderer with a filesystem templates directory.

Args:
    templates_dir: Directory containing Jinja2 templates.
  - _build_context(self, project_name: str, eda_payload: dict[str, Any], pipe_payload: dict[str, Any]) -> dict[str, Any]: Build a normalized context dict consumed by templates.

Args:
    project_name: Project display name.
    eda_payload: EDA orchestrator output payload.
    pipe...
  - render(self, out_dir: str, project_name: str, formats: list[str], eda_payload: dict[str, Any], pipe_payload: dict[str, Any]) -> dict[str, Any]: Render report in the requested formats and return artifact paths.

Args:
    out_dir: Output directory for rendered report.
    project_name: Project display na...

**src/evaluation/metrics.py**

**src/evaluation/splitting.py**

**src/instrumentation/bootstrap_logging.py**
Fonctions:
- init_logging_from_config(cfg_mgr: ConfigManager): Construit et configure le logger manager (stdlib/structlog) à partir de ConfigManager.
Retourne un logger (racine de l'app) prêt à l'emploi.

**src/instrumentation/config_manager.py**
Classes:
- ConfigManager: Load and validate OmegaConf (Hydra) config into Pydantic models.
  - __init__(self, hydra_cfg: DictConfig) -> None: Initialize with a Hydra/OmegaConf DictConfig.
  - build_logger_settings(self) -> LoggerSettings: 
  - load(self) -> AppConfig: Resolve and validate the configuration, returning AppConfig.
  - make_logs_file_path(self, filename: str = 'app.log') -> str: 
  - model(self) -> AppConfig: Return the validated Pydantic model, loading if needed.
  - project_root(self) -> Path: Best-effort pour déduire la racine du projet.
  - raw(self) -> dict[str, Any]: Return the raw resolved dictionary form.

**src/instrumentation/data_manager.py**
Classes:
- DataManager: Manage pure data transformations and ML preparation.
  - __init__(self, config = None) -> None: Initialize DataManager with configuration.
  - clean_data(self, df: pd.DataFrame) -> pd.DataFrame: Apply data cleaning: duplicates, missing values, outliers.
  - infer_column_types(self, df: pd.DataFrame) -> dict[str, str]: Infer optimal data types for each column.
  - infer_target_column(self, df: pd.DataFrame) -> str | None: Return explicit target if configured; else optionally auto-detect.
  - load_csv(path: Path, encoding: str | None = None, sep: str | None = None, **kwargs) -> pd.DataFrame: Charger un CSV avec encodage/séparateur optionnels.
  - load_from_raw(self, raw_data: Any) -> pd.DataFrame: Convert raw data (dict, DataFrame, list records) into pandas DataFrame.
  - prepare_for_ml(self, raw_data: Any) -> tuple[pd.DataFrame, pd.Series | None]: Full preparation pipeline: load → clean → split → validate.
  - split_features_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]: 
  - validate_data(self, X: pd.DataFrame, y: pd.Series | None = None) -> bool: Basic validations for ML workflows.

**src/instrumentation/file_manager.py**
Classes:
- FileManager: File and directory utilities with pragmatic defaults.

Provides helpers for:
- Existence and type checks.
- Permission checks (read/write/execute).
- Directory creation (idempotent).
- Listing files b...
  - check_path_exists(self, path: str | Path) -> bool: Return True if path exists (file or directory), else False.
  - compress_file_gz(self, path: str | Path, delete_original: bool = False) -> Path: Compress single file to gzip (path.ext.gz); optionally delete original.
  - copy_file(self, src: str | Path, dst_dir: str | Path, rename: str | None = None) -> Path: Copy file to dst_dir, optionally renaming the target (preserves metadata).
  - ensure_dir(self, path: str | Path) -> Path: Create directory path with parents, no error if exists.
  - has_perm_exec(self, path: str | Path) -> bool: Return True if current process can execute path.
  - has_perm_read(self, path: str | Path) -> bool: Return True if current process can read path.
  - has_perm_write(self, path: str | Path) -> bool: Return True if current process can write to path.
  - is_dir(self, path: str | Path) -> bool: Return True if path points to a directory.
  - is_file(self, path: str | Path) -> bool: Return True if path points to a regular file.
  - list_files_by_ext(self, dir_path: str | Path, exts: Iterable[str]) -> list[Path]: List files in dir_path whose suffix is in exts (case-insensitive, non-recursive).
  - make_timestamp_name(self, src: str | Path) -> str: Return a timestamped filename YYYYMMDD_HHMMSS_<basename>.
  - read_file(self, path: str | Path) -> Any: Read csv/xlsx/json into a pandas object.
  - write_file(self, data: Any, path: str | Path) -> None: Write pandas/tabular data to csv/xlsx/json by extension; fallback to text/bytes.

**src/instrumentation/logger_factory.py**
Fonctions:
- build_logger_manager(settings: LoggerSettings) -> LoggerManager: Build a logger manager (stdlib or structlog) from settings.
- build_logger_manager_from_config(cfg_mgr: ConfigManager) -> LoggerManager: Shortcut: construit LoggerManager à partir de ConfigManager.

**src/instrumentation/logger_manager.py**
Classes:
- JsonFormatter: 
  - format(self, record: logging.LogRecord) -> str: 
- LoggerBaseConfig: Common logger config loaded from ConfigManager.
- LoggerManager: Base logger manager using Python stdlib logging + dictConfig.
  - __init__(self, cfg: LoggerBaseConfig) -> None: Initialize the stdlib logger manager.
  - _build_dict_config(self) -> dict[str, Any]: Build a dictConfig mapping for logging.config.dictConfig.
  - _default_fields_filter(common: dict[str, Any]) -> logging.Filter: Attach default extra fields to all log records.
  - configure(self) -> None: Apply stdlib logging configuration once.
  - get_logger(self, name: str | None = None) -> logging.Logger: Return a stdlib logger, configuring on first use.
Fonctions:
- _json_default(o: Any) -> Any: 

**src/instrumentation/logger_manager_structlog.py**
Classes:
- StructlogLoggerManager: structlog-based manager inheriting the LoggerManager interface.

Falls back to stdlib if structlog is not installed.
  - __init__(self, cfg: LoggerBaseConfig) -> None: Initialize the structlog logger manager.
  - configure(self) -> None: Configure structlog and stdlib integration (once).
  - get_logger(self, name: str | None = None): Return a structlog logger if available, stdlib otherwise.

**src/instrumentation/logger_mixin.py**
Classes:
- LoggerMixin: Provide a self.log attribute using a provided LoggerManager-like object.
  - _init_logger(self, lm: SupportsGetLogger) -> None: Initialize self.log with a named logger from the logger manager.

Args:
    lm: Object exposing get_logger(name) (LoggerManager/StructlogLoggerManager).
- SupportsGetLogger: Protocol for objects exposing get_logger(name)->logger.
  - get_logger(self, name: str | None = None): Return a logger instance (stdlib or structlog-compatible).

**src/instrumentation/message_manager.py**
Classes:
- MessageManager: Gestionnaire de traduction pour multiples domaines.
  - __init__(self, locales_dir: Path | str, default_locale: str = 'fr') -> None: 
  - _build_fallback(self) -> Callable[[str], str]: 
  - msg(self, domain: str, key: str, locale: str | None = None, **params: Any) -> str: Résout key dans domain, applique format(**params) si fourni.
  - translator(self, domain: str, locale: str | None = None) -> Callable[[str], str]: Retourne une fonction gettext pour un domaine/locale.

**src/instrumentation/message_taxonomy.py**

**src/modeling/dl/config.py**
Classes:
- CallbacksConfig: Agrégat des callbacks usuels.
- CheckpointCfg: Callback ModelCheckpoint.
- CompileConfig: Compilation: optimizer (string ou dict avec lr), loss (auto), metrics.
- DLConfig: Config DL racine.
- EarlyStoppingCfg: Callback EarlyStopping.
- ExportConfig: Options d’export: modèle et historique.
- FitConfig: Paramètres d’entraînement fit().
- LayerSpec: Couche déclarative: type + paramètres (ex: Dense, Dropout).
- ModelConfig: Description du modèle Keras et de la tâche.
- ReduceLRCfg: Callback ReduceLROnPlateau.
Fonctions:
- _empty_layers() -> list[LayerSpec]: Fabrique typée pour éviter list[Unknown] côté Pylance.

**src/modeling/dl/consts.py**

**src/modeling/dl/factory.py**
Fonctions:
- _auto_loss(cfg: DLConfig) -> str: Détermine automatiquement la loss:
  - binary -> 'binary_crossentropy'
  - multiclass -> 'sparse_categorical_crossentropy' (cible y attendue 1D entiers)

Args:
...
- _build_layers(layer_specs: list[LayerSpec], input_shape: list[int] | None) -> list[Any]: Construit les instances Keras pour chaque LayerSpec, en injectant input_shape
sur la première couche si absent dans ses paramètres.

Args:
    layer_specs: Spéc...
- _get_keras() -> Any: Charge tensorflow.keras dynamiquement.

Retour:
    Module tensorflow.keras (Any) si disponible, sinon None.
- _get_optimizer_factory(optim_mod: Any, name: str) -> Any | None: Retourne le constructeur d'optimizer (classe/fabrique) si disponible
dans keras.optimizers, en testant le nom en minuscules puis capitalisé.

Args:
    optim_mo...
- _last_dense_signature(layer: Any) -> tuple[int | None, str | None]: Extrait (units, activation_name) pour une couche Dense.

Args:
    layer: Instance Keras de couche.

Retour:
    (units, activation_name) ou (None, None) si ce ...
- _make_optimizer(optimizer_cfg: str | dict[str, Any]) -> Any: Crée un optimizer Keras depuis:
  - une chaîne (e.g., "adam"),
  - un dict (e.g., {"name": "adam", "lr": 0.001}), en mappant lr -> learning_rate.

Args:
    opt...
- _maybe_append_output(cfg: DLConfig, layers: list[Any]) -> list[Any]: Ajoute une couche de sortie adaptée à la tâche si auto_output est actif
et si la dernière couche n'est pas déjà une sortie conforme.

Règles:
  - Binary: Dense(...
- _resolve_layer(name: str) -> Any: Retourne la classe de couche Keras (e.g., Dense, Dropout) par son nom.

Args:
    name: Nom de la classe de couche Keras (e.g., "Dense").

Retour:
    La classe...
- build_model(cfg: DLConfig) -> Any: Construit le modèle Keras (Sequential ou Functional) à partir de cfg.

- Applique l'injection input_shape sur la première couche si nécessaire.
- Ajoute automat...
- compile_model(model: Any, cfg: DLConfig) -> Any: Compile le modèle avec optimizer / loss / metrics.

Args:
    model: Modèle Keras à compiler.
    cfg: DLConfig contenant la config de compilation.

Retour:
   ...

**src/modeling/dl/trainer.py**
Fonctions:
- _make_callbacks(cfg: DLConfig) -> list[Any]: 
- _summary_to_string(model: Any) -> str: 
- _to_array(y: npt.ArrayLike) -> Float32Array: 
- _to_float32_array(x: npt.ArrayLike) -> Float32Array: 
- get_keras() -> Any: 
- train_dense(x_train: npt.ArrayLike, y_train: npt.ArrayLike, x_val: npt.ArrayLike | None, y_val: npt.ArrayLike | None, cfg: DLConfig) -> dict[str, Any]: 

**src/modeling/pipeline/consts.py**
Classes:
- AutoLib: 

**src/modeling/pipeline/evaluator.py**
Classes:
- PipelineEvaluator: Évalue un pipeline ML ou DL selon la configuration déclarative fournie.
- ML: pipeline sklearn + CV (Grid/Random/Halving), export des cv_results_.csv.
- DL: modèles Keras séquentiels/fonctionnels via ...
  - __init__(self, out_dir: str, random_state: int = 42, mlflow_enabled: bool = False, experiment: str = 'mlp-experiments', logger_manager: SupportsGetLogger | None = None) -> None: 
  - _cv(cv_cfg: dict[str, Any]) -> sk_ms.StratifiedKFold: Construit un StratifiedKFold robuste depuis la config CV.
  - _instantiate_tpot(self, trials: list[dict[str, Any]]) -> tuple[Any | None, Exception | None]: Instancie TPOTClassifier en essayant plusieurs combinaisons de paramètres tolérantes.
  - _make_dask_client(self, tcfg: dict[str, Any]) -> tuple[Any | None, Any | None]: Crée un cluster local Dask si demandé; sinon retourne (None, None).
  - _maybe_run_dl(self, spec: dict[str, Any], x: pd.DataFrame, y: pd.Series) -> dict[str, Any] | None: Branche DL: applique d'abord le ColumnTransformer si configuré, puis entraîne un MLP dense.
- Construit le preprocess depuis la spec + policy (via la fabrique)....
  - _maybe_run_lazy(self, spec: dict[str, Any], x: pd.DataFrame, y: pd.Series) -> dict[str, Any] | None: Exécute LazyPredict si configuré; sinon None.
  - _maybe_run_tpot(self, spec: dict[str, Any], x: pd.DataFrame, y: pd.Series) -> dict[str, Any] | None: Exécute TPOT si configuré; sinon None.
  - _resolve_class(path_or_name: Any) -> type | None: Résout une classe sklearn depuis un nom court ou chemin complet module.Classe.
  - _resolve_classifiers(items: list[Any] | None) -> list[type] | None: Résout une liste de classes sklearn depuis une liste d'items hétérogènes.
  - _resolve_metric(metric_spec: Any) -> Callable[..., float] | None: Résout un metric custom pour LazyPredict, sinon None.
  - _resolve_sklearn_class(short_name: str) -> type | None: Résout une classe sklearn via recherche dans modules fréquents.
  - _sanitize_space(pipe: Pipeline, space: dict[str, Any] | None) -> dict[str, Any]: Filtre l’espace de recherche pour ne garder que les clés valides pour le pipeline.
  - _scoring_and_refit(cv_cfg: dict[str, Any]) -> tuple[ScoringType, RefitType]: Retourne le couple scoring/refit compatible sklearn, avec défauts sûrs.
  - _search(self, cv_type: str, estimator: Any, scoring: ScoringType, refit: RefitType, cv: Any, grid: dict[str, list[Any]], dists: dict[str, Any], cv_cfg: dict[str, Any]) -> Any: Construit l’objet de recherche hyperparamétrique selon le type souhaité.
  - _tpot_trials(self, tcfg: dict[str, Any], client: Any | None, safe_cv: int) -> list[dict[str, Any]]: Génère une liste d’essais de paramètres compatibles TPOT1/TPOT2.
  - _warn(self, msg: str) -> None: Émet un warning via le logger mixin si présent, sinon logger Python.
  - evaluate(self, x: pd.DataFrame, y: pd.Series, spec: dict[str, Any], cv_cfg: dict[str, Any], global_policy: dict[str, Any]) -> dict[str, Any]: Évalue le pipeline: tente DL, puis TPOT, puis Lazy, sinon sklearn+CV avec export CSV.
- Retourne un dict de résultats harmonisé incluant artefacts et meilleurs ...
- _F1Weighted: 
  - __call__(self, y_true: Any, y_pred: Any, *, average: Literal['weighted']) -> float: 
Fonctions:
- _as_mapping(obj: Any) -> dict[str, Any]: Copie défensive d’un mapping arbitraire en dict[str, Any].
- _f1_weighted() -> Callable[[Any, Any], float]: Scorer F1 weighted simple (pour LazyPredict custom_metric).
- _get_scorer_safe(name: str | None) -> Callable[[Any, Any, Any], float] | None: Wrapper typé sur sklearn.metrics.get_scorer, retourne (estimator, X, y) -> float ou None.
- _safe_close(obj: Any, warn: Callable[[str], None], what: str) -> None: Ferme proprement un objet (Client/Cluster) sans déclencher d’alertes de typage.
- _tts_df(x: pd.DataFrame, y: pd.Series, *, test_size: float, random_state: int, stratify: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Train/test split DataFrame/Series avec stratification robuste.
- _tts_np(x: npt.NDArray[np.float64], y: npt.NDArray[np.int64], *, test_size: float, random_state: int, stratify: npt.NDArray[np.int64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]: Train/test split NDArray avec typage explicite, via cast sur l’API sklearn.

**src/modeling/pipeline/factory.py**
Classes:
- PipelineFactory: Construit un sklearn.Pipeline + grilles d’hyperparamètres à partir d’un spec.
- ColumnTransformer auto/manuelle via selectors (noms/regex/dtypes), avec imputation intégrée.
- Grilles/distributions com...
  - _build_categorical_encoder(cfg: Mapping[str, Any]) -> Any: Retourne l’encodeur catégoriel selon la config locale.
  - _build_column_transformer(cls, spec: Mapping[str, Any], global_policy: Mapping[str, Any]) -> ColumnTransformer | None: Construit un ColumnTransformer en première étape, avec imputation intégrée.
- policy: "manual" (règles de colonnes) ou "auto" (sélection par dtype).
- numeric: ...
  - _build_numeric_scaler(scaler_name: str | None) -> Any: Retourne un scaler numérique depuis une chaîne.
  - _flatten_distributions(cls, prefix: str, dists: Mapping[str, Any]) -> dict[str, Any]: Aplati des distributions -> param_distributions sklearn (step__param: dist).
  - _flatten_grid(prefix: str, params: Mapping[str, Any]) -> dict[str, list[Any]]: Aplati un dictionnaire de paramètres -> param_grid sklearn (step__param).
  - _instantiate_estimator(est_cfg: Mapping[str, Any]) -> Any: Crée l'estimateur à partir de est_cfg['type'] (chemin fully-qualified ou alias).
Garantit la présence de predict/decision_function pour la compatibilité GridSea...
  - _make_cat_pipe(cls, local_cat: Mapping[str, Any] | None, fallback_cat: Mapping[str, Any] | None) -> Any: Construit le sous-pipeline catégoriel (imputer + encodeur) selon overrides locaux/politiques globales.
  - _make_num_pipe(cls, local_num: Mapping[str, Any] | None, fallback_num: Mapping[str, Any] | None) -> Any: Construit le sous-pipeline numérique (imputer + scaler) selon overrides locaux/politiques globales.
  - _scipy_dist(name: str, low: float, high: float) -> Any: Construit une distribution scipy.stats depuis un spec déclaratif.
  - _selector_from_rule(rule: Mapping[str, Any]) -> Callable[[pd.DataFrame], list[str]]: Construit un sélecteur de colonnes depuis une règle, toujours un callable ColumnSelector.
  - build(cls, spec: Mapping[str, Any], global_policy: Mapping[str, Any]) -> tuple[Pipeline, dict[str, list[Any]], dict[str, Any]]: Construit (pipeline, param_grid, param_distributions) depuis un spec.
- Ordre garanti: ColumnTransformer (avec imputation) -> feature_selection
  -> pre_pca_imp...
Fonctions:
- as_mapping(obj: Mapping[str, Any] | None) -> dict[str, Any]: Retourne un dict[str, Any] garanti à partir d'un Mapping optionnel, sans dict(obj).
- as_str_list(items: Any) -> list[str]: Convertit en list[str] en couvrant None/str/bytes/Iterable et en évitant les Unknown.

**src/orchestrators/app.py**
Classes:
- AppOrchestrator: Boot logger + config, expose logger_manager, config_manager and ctx.
  - __init__(self, hydra_cfg: DictConfig) -> None: 

**src/orchestrators/base.py**
Classes:
- IOrchestrator: Interface minimale pour les orchestrateurs.
  - run(self, X: pd.DataFrame, y: pd.Series | None = None) -> dict[str, Any]: Exécuter l'orchestrateur et retourner un dict de résultats.

**src/orchestrators/config.py**
Classes:
- ConfigOrchestrator: Load config, build logger/message, and compute project context.
  - __init__(self, config_manager: ConfigManager, logger_manager: LoggerManager | None = None) -> None: 
  - _resolve_root(self) -> Path: Retourne une racine de projet stable selon le contexte (Hydra ou non).
  - get_app_config(self) -> Any: 
  - get_config_manager(self) -> ConfigManager: 
  - get_context(self) -> dict[str, str]: 
  - get_logger_manager(self) -> LoggerManager: 
  - run(self) -> dict[str, str]: 

**src/orchestrators/data.py**
Classes:
- DataOrchestrator: Orchestrate data preparation workflows using DataManager.
  - __init__(self, cfg: DataConfig, logger_manager = None) -> None: 
  - _load_df_from_payload(raw_data: Any, encoding: str | None, sep: str | None) -> pd.DataFrame: 
  - analyze_df(self, df: pd.DataFrame) -> dict[str, Any]: 
  - attach_message(self, msg: MessageOrchestratorApp) -> None: 
  - process_data(self, raw_data: Any) -> tuple[pd.DataFrame, pd.Series | None]: 
  - run(self, raw_data: Any) -> dict[str, Any]: 

**src/orchestrators/eda.py**
Classes:
- EDAOrchestrator: Run EDA: profile and summary, and emit localized events.
  - __init__(self, cfg: EDAConfig, project_dir: str, logger_manager: LoggerManager) -> None: 
  - attach_message(self, msg: MessageOrchestratorApp) -> None: 
  - run(self, x: pd.DataFrame, y: pd.Series | None = None) -> dict[str, Any]: 
Fonctions:
- _as_target(y: pd.Series | None) -> pd.Series | None: 

**src/orchestrators/file.py**
Classes:
- FileConfig: Configuration interne normalisée pour l'orchestrateur de fichiers.

Notes:
- La config effective vient typiquement de Pydantic (FileConfigModel) ou d'un dict Hydra;
  ce dataclass sert de réceptacle a...
- FileConfigDict: Type dict minimal pour la normalisation de configuration en entrée.
- FileOrchestrator: Orchestrateur de fichiers.

Responsabilités:
- Découvrir un fichier d'entrée dans data_in selon les extensions autorisées.
- Copier/compresser (optionnel) vers data_out pour traçabilité.
- Charger les...
  - __init__(self, cfg: FileConfigModel | FileConfig | Mapping[str, Any], logger_manager: LoggerManager | None = None, ctx: dict[str, str] | None = None) -> None: 
  - attach_message(self, msg: MessageOrchestratorApp) -> None: Attache l’orchestrateur de messages pour émettre les événements localisés.
  - from_cfg_mgr(cls, cfg_mgr: Any, logger_manager: LoggerManager | None = None, ctx: dict[str, str] | None = None) -> FileOrchestrator: Fabrique un FileOrchestrator à partir d’un ConfigManager-like (.model.orchestrators.file attendu).
  - from_config_manager(cls, config_manager: Any, logger_manager: LoggerManager | None = None, ctx: dict[str, str] | None = None) -> FileOrchestrator: Alias de from_cfg_mgr pour compatibilité d’API explicite.
  - pick_input_file(self) -> Path | None: Sélectionne le fichier d’entrée:
- Priorité à preferred_filename s’il existe.
- Sinon, premier fichier correspondant aux extensions autorisées.
  - process_input(self) -> dict[str, Any]: Traite l’entrée fichier:
- Émet FILE_INIT et file_paths_resolved.
- Cherche un fichier; si absent, NO_INPUT_FILE et payload found=False.
- Copie/compresse si co...
Fonctions:
- _coerce_cfg_dict(raw: Mapping[str, Any]) -> FileConfigDict: Normalise un mapping arbitraire en FileConfigDict.

- Cast des types simples.
- Filtrage des séquences pour extensions.
- _evt(e: object) -> str: Uniformise un identifiant d’événement issu d’un Enum/tuple/str en str.
- _to_dict_cfg(cfg: object) -> FileConfigDict: Convertit une config Pydantic/dataclass/dict en FileConfigDict.

- Utilise model_dump() si disponible (Pydantic v2).
- asdict() pour dataclass.
- Mapping direct...

**src/orchestrators/general.py**
Classes:
- GeneralOrchestrator: Coordinate the full workflow with consistent logging and message.
  - __init__(self, config_manager: ConfigManager, logger_manager: Any | None = None, message_orchestrator: MessageOrchestratorApp | None = None, ctx: dict[str, str] | None = None) -> None: 
  - _attach_message(self, *children: Any) -> None: 
  - _fallback_logger(self): 
  - _run_ml_orchestrators(self, x: pd.DataFrame, y: pd.Series | None, results: dict[str, Any]) -> dict[str, Any]: 
  - load_example_data(self) -> tuple[pd.DataFrame, pd.Series]: 
  - run(self, x: pd.DataFrame | None = None, y: pd.Series | None = None) -> dict[str, Any]: 
  - run_from_data(self, x: pd.DataFrame, y: pd.Series | None = None) -> dict[str, Any]: 
  - run_from_files(self) -> dict[str, Any]: 
Fonctions:
- _example_data() -> tuple[pd.DataFrame, pd.Series]: 

**src/orchestrators/logger.py**
Classes:
- LoggerOrchestrator: Thin wrapper around LoggerManager to standardize logger bootstrap.
  - __init__(self, hydra_cfg: DictConfig) -> None: 
  - run(self, config_manager: ConfigManager) -> LoggerManager: 

**src/orchestrators/message.py**
Classes:
- MessageOrchestrator: Noyau i18n minimal basé sur gettext:
- Paramétré par localedir / domain / default_lang
- Expose get(key, params) et load(lang)
- Fournit translate(domain, key, **params) en secours pour compatibilité
  - __init__(self, localedir: str | Path = 'i18n/locales', domain: str = 'streamlit_app', default_lang: str = 'fr') -> None: 
  - _load_fallback(self) -> None: Charge l’anglais comme secours s’il existe, sinon identité.
  - get(self, key: str, params: dict[str, Any] | None = None) -> str: Traduit key dans le domaine courant, puis applique format(**params) si fourni.
  - load(self, lang: str) -> None: Charge/active la langue courante pour le domaine par défaut.
  - set_lang(self, lang: str) -> None: 
  - translate(self, domain: str, key: str, **params: Any) -> str: Traduit une clé dans un domaine donné, avec fallback si absent.
- MessageOrchestratorApp: Orchestrateur i18n applicatif:
- Signature homogène avec les autres orchestrateurs (config_manager, logger_manager)
- Délègue la traduction au core (MessageOrchestrator)
- Expose get(...), translate(....
  - __init__(self, config_manager: Any, logger_manager: Optional[Any] = None, localedir: Optional[str | Path] = None, domain: Optional[str] = None, default_lang: Optional[str] = None) -> None: 
  - emit(self, domain: str, key: str, level: str = 'info', **fields: Any) -> None: Formate un message traduit et l’émet au logger si disponible.
- level: "debug" | "info" | "warning" | "error" | "critical"
- fields: champs structurés additionn...
  - get(self, key: str, params: dict[str, Any] | None = None) -> str: 
  - set_lang(self, lang: str) -> None: 
  - translate(self, domain: str, key: str, **params: Any) -> str: 

**src/orchestrators/pipeline.py**
Classes:
- PipelineOrchestrator: 
  - __init__(self, cfg: PipelineConfig, project_dir: str, random_state: int, logger_manager: SupportsGetLogger | None = None, out_dir: str | None = None, ctx: dict[str, str] | None = None) -> None: 
  - _filter_active_specs(self) -> list[dict[str, Any]]: 
  - attach_message(self, msg: MessageOrchestratorApp) -> None: 
  - run(self, x: pd.DataFrame, y: pd.Series) -> dict[str, Any]: 

**src/orchestrators/report.py**
Classes:
- ReportOrchestrator: Render consolidated report from EDA and pipeline outputs.
  - __init__(self, cfg: ReportConfig, project_dir: str, app_cfg: AppConfig, logger_manager: LoggerManager, ctx: dict[str, str] | None = None) -> None: 
  - attach_message(self, msg: MessageOrchestratorApp) -> None: 
  - run(self, eda_payload: dict[str, Any], pipe_payload: dict[str, Any]) -> dict[str, Any]: 

**src/preprocessing/reducers.py**
Classes:
- ReducersFactory: 
  - _get_umap(): Import paresseux de umap-learn uniquement lorsque requis.
Évite d'importer umap au niveau module car umap expose ParametricUMAP
dans son __init__, ce qui peut t...
  - from_spec(cfg: dict[str, Any] | None, random_state: int = 42): Alias rétro-compatible.
  - instantiate_estimator(cfg: dict[str, Any] | None, random_state: int = 42): Alias rétro-compatible.
  - make_reducer(cfg: dict[str, Any] | None, random_state: int = 42): Construit un réducteur dimensionnel depuis une spec:
- {"type": "pca", "params": {...}}
- {"type": "umap", "params": {...}}
- {"type": "parametric_umap", "param...

**src/preprocessing/samplers.py**
Classes:
- SamplersFactory: 
  - from_spec(cfg: dict[str, Any] | None): Alias for API symmetry.
  - make_sampler(cfg: dict[str, Any] | None): Build a resampler from a spec (imbalanced-learn):
- {"type": "smote", "params": {...}}
- {"type": "over", "params": {...}}   # RandomOverSampler
- {"type": "und...

**src/preprocessing/selectors.py**
Classes:
- SelectorsFactory: 
  - from_spec(cfg: dict[str, Any] | None): Alias kept for symmetry with other factories.
  - instantiate_estimator(cfg: dict[str, Any] | None): 
  - make_selector(cfg: dict[str, Any] | None): Build a feature selector from a spec:
- {"variance_threshold": 0.0}
- {"select_k_best": 100}
- {"select_percentile": 50}
Returns "passthrough" when no selector ...

**src/ui/app.py**
Classes:
- MLPStreamlitApp: Streamlit app wrapper managing config, message, and UI i18n.
  - __init__(self) -> None: Initialize placeholders for orchestrators and UI MessageManager.
  - _ui(self, key: str, **params: Any) -> str: Resolve a UI string by key using MessageManager on UI_DOMAIN.

Falls back to DEFAULT_TEXTS if translation or key is missing.
  - bootstrap(self) -> None: Initialize Config and Message orchestrators, and UI MessageManager.
  - get_ui_text(self, key: str, **params: Any) -> str: Public UI text resolver for pages and tests.

Wraps the internal _ui() method.
  - navigation(self) -> None: Define and run programmatic navigation.
  - run(self) -> None: Entrypoint to render and run the app.
  - sidebar(self) -> None: Render sidebar and persist chosen parameters.
Fonctions:
- _load_cfg_safe(conf_path: Path) -> tuple[dict[str, Any], str]: Load Hydra config safely, returning (dict, message_info).

**src/ui/constants.py**



---
*Generated by AdvancedSessionContextGenerator v2.1*
*Automated analysis of 74 files completed at 2025-09-26 13:33*
