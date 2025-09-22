from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ReductionConfig(BaseModel):
    type: str | None = None
    params: dict[str, Any] | None = None


class EstimatorConfig(BaseModel):
    type: str
    params: dict[str, Any] | None = None


class StepsConfig(BaseModel):
    preprocess: dict[str, Any] | None = None
    feature_selection: dict[str, Any] | None = None
    reduction: ReductionConfig | None = None
    sampler: dict[str, Any] | None = None
    estimator: EstimatorConfig | None = None


class PipelineSpec(BaseModel):
    name: str
    steps: StepsConfig | None = None
    automl: dict[str, Any] | None = None


class PipelinesConfig(BaseModel):
    enabled: bool = True
    out_dir: str | None = None  # Nouveau champ
    active: list[str] = Field(default_factory=list)
    cv: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)
    pipelines: list[PipelineSpec] = Field(default_factory=list)


class EDAConfig(BaseModel):
    enabled: bool = True
    profile: dict[str, Any] = Field(default_factory=dict)


class ReportConfig(BaseModel):
    enabled: bool = True
    formats: list[str] = Field(default_factory=lambda: ["html"])


class FileConfig(BaseModel):
    """Config section for the file orchestrator."""

    enabled: bool = True
    data_dir: str = "data"
    in_dir: str = "in"
    out_dir: str = "out"
    extensions: list[str] = Field(default_factory=lambda: [".csv", ".xlsx", ".json"])
    save_input_file: bool = True
    save_input_file_compression: bool = False


class DataConfig(BaseModel):
    """Configuration for data processing orchestrator."""

    enabled: bool = True
    target_column: str | None = None
    auto_detect_target: bool = True
    drop_columns: list[str] = Field(default_factory=list)
    missing_strategy: str = "auto"  # auto, drop, fill
    categorical_threshold: float = 0.1
    min_samples_threshold: int = 10
    outlier_detection: bool = False
    encoding: str | None = None
    sep: str | None = None


class OrchestratorsConfig(BaseModel):
    file: FileConfig = Field(default_factory=FileConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    eda: EDAConfig = Field(default_factory=EDAConfig)
    pipelines: PipelinesConfig = Field(default_factory=PipelinesConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)


class ProjectConfig(BaseModel):
    name: str
    random_state: int = 42
    output_dir: str = "outputs"
    allow_example_fallback: bool = False


class LoggerSettings(BaseModel):
    backend: str = Field(default="stdlib")  # "stdlib" | "structlog"
    app_name: str = Field(default="mlp")
    level: str = Field(default="INFO")
    json_mode: bool = Field(default=False)
    file_path: str | None = None
    file_max_bytes: int = Field(default=5 * 1024 * 1024)
    file_backup_count: int = Field(default=3)
    uvicorn_noise_filter: bool = Field(default=True)
    default_fields: dict[str, Any] = Field(default_factory=dict)


class AppConfig(BaseModel):
    project: ProjectConfig
    orchestrators: OrchestratorsConfig
    logger: LoggerSettings = Field(default_factory=LoggerSettings)
