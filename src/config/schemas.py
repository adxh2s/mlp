from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ReductionConfig(BaseModel):
    type: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


class EstimatorConfig(BaseModel):
    type: str
    params: Optional[Dict[str, Any]] = None


class StepsConfig(BaseModel):
    preprocess: Optional[Dict[str, Any]] = None
    feature_selection: Optional[Dict[str, Any]] = None
    reduction: Optional[ReductionConfig] = None
    sampler: Optional[Dict[str, Any]] = None
    estimator: Optional[EstimatorConfig] = None


class PipelineSpec(BaseModel):
    name: str
    steps: Optional[StepsConfig] = None
    automl: Optional[Dict[str, Any]] = None


class PipelinesConfig(BaseModel):
    enabled: bool = True
    cv: Dict[str, Any] = Field(default_factory=dict)
    pipelines: List[PipelineSpec] = Field(default_factory=list)


class EDAConfig(BaseModel):
    enabled: bool = True
    profile: Dict[str, Any] = Field(default_factory=dict)


class ReportConfig(BaseModel):
    enabled: bool = True
    formats: List[str] = Field(default_factory=lambda: ["html"])


class OrchestratorsConfig(BaseModel):
    eda: EDAConfig = Field(default_factory=EDAConfig)
    pipelines: PipelinesConfig = Field(default_factory=PipelinesConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)


class ProjectConfig(BaseModel):
    name: str
    random_state: int = 42
    output_dir: str = "outputs"


class LoggerSettings(BaseModel):
    backend: str = Field(default="stdlib")  # "stdlib" | "structlog"
    app_name: str = Field(default="mlp")
    level: str = Field(default="INFO")
    json_mode: bool = Field(default=False)
    file_path: Optional[str] = None
    file_max_bytes: int = Field(default=5 * 1024 * 1024)
    file_backup_count: int = Field(default=3)
    uvicorn_noise_filter: bool = Field(default=True)
    default_fields: Dict[str, Any] = Field(default_factory=dict)


class AppConfig(BaseModel):
    project: ProjectConfig
    orchestrators: OrchestratorsConfig
    logger: LoggerSettings = Field(default_factory=LoggerSettings)
