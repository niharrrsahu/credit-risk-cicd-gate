"""Configuration Schema and Validator using Pydantic V2."""

from typing import Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import yaml
import os

class DataConfig(BaseModel):
    version: str = Field(..., description="Dataset version identifier, e.g., 'v1' or 'v2'")
    raw_path: str = Field(..., description="Relative or absolute path to raw CSV")
    processed_dir: str = Field(default="data/processed")
    test_size: float = Field(default=0.25, ge=0.05, le=0.5, description="Fraction of dataset for test evaluation")
    stratify: bool = Field(default=True)
    manifest_path: str = Field(default="data/data_manifest.json")

    @field_validator("raw_path")
    @classmethod
    def check_file_path(cls, v: str) -> str:
        if not v.endswith(".csv"):
            raise ValueError(f"raw_path must point to a .csv file, got '{v}'")
        return v

class FeatureConfig(BaseModel):
    id_column: str = Field(default="Customer_ID")
    target_column: str = Field(default="High_Risk_Customer")
    utilisation_risk_cutoff: float = Field(default=0.80, ge=0.0, le=2.0)
    delinquency_days_threshold: int = Field(default=30, ge=0, le=365)
    high_spend_ratio_threshold: float = Field(default=0.50, ge=0.0, le=2.0)
    categorical_impute_strategy: Literal["most_frequent", "constant"] = "most_frequent"
    numerical_impute_strategy: Literal["mean", "median", "zero"] = "median"

class ModelConfig(BaseModel):
    family: Literal["logistic_regression", "random_forest", "xgboost", "lightgbm"] = "xgboost"
    params: Dict[str, Any] = Field(default_factory=dict)

class TrainingConfig(BaseModel):
    cv_folds: int = Field(default=5, ge=2, le=10)
    scoring_metric: str = Field(default="roc_auc")
    search_type: Literal["fixed", "grid", "random"] = "fixed"

class TrackingConfig(BaseModel):
    experiment_name: str = Field(..., min_length=1)
    tracking_uri: str = Field(default="file:./mlruns")
    log_models: bool = Field(default=True)
    register_model: bool = Field(default=False)
    registered_model_name: str = Field(default="CreditRiskGateModel")

class QualityGateConfig(BaseModel):
    baseline_metrics_path: str = Field(default="baseline_metrics.json")
    primary_metric: str = Field(default="pr_auc")
    secondary_metric: str = Field(default="roc_auc")
    min_pr_auc_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    max_pr_auc_drop_allowed: float = Field(default=0.02, ge=0.0, le=0.5)
    max_roc_auc_drop_allowed: float = Field(default=0.02, ge=0.0, le=0.5)
    min_recall_threshold: float = Field(default=0.65, ge=0.0, le=1.0)

class AppConfig(BaseModel):
    environment: Literal["development", "ci", "production", "test"]
    version: str = Field(default="1.0.0")
    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    training: TrainingConfig
    tracking: TrackingConfig
    quality_gate: QualityGateConfig

    @model_validator(mode="after")
    def validate_environment_consistency(self):
        # Additional cross-field checks
        if self.environment == "ci" and not self.tracking.register_model:
            # CI should register candidate models for gate checks
            pass
        return self


def load_config(config_path: str) -> AppConfig:
    """Loads and validates configuration from YAML path.
    Raises ValueError with descriptive diagnostics if invalid.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            raw_dict = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"YAML parsing error in {config_path}: {e}")
            
    if not isinstance(raw_dict, dict):
        raise ValueError(f"Configuration in {config_path} must be a valid key-value mapping")

    try:
        config = AppConfig(**raw_dict)
        return config
    except Exception as e:
        raise ValueError(f"Configuration validation failed for '{config_path}':\n{e}")
