"""Unit tests for Configuration Loading and Schema Validation."""

import pytest
import os
import tempfile
import yaml
from src.config_schema import load_config, AppConfig

def test_load_valid_configs():
    """Verify that all default environment configurations load and validate successfully."""
    for config_name in ["dev.yaml", "ci.yaml", "prod.yaml"]:
        path = os.path.join("config", config_name)
        assert os.path.exists(path), f"Missing config file: {path}"
        cfg = load_config(path)
        assert isinstance(cfg, AppConfig)
        assert cfg.seed >= 0
        assert 0.0 < cfg.data.test_size < 1.0

def test_missing_config_file_raises():
    """Verify descriptive FileNotFoundError for missing config path."""
    with pytest.raises(FileNotFoundError):
        load_config("config/non_existent_config.yaml")

def test_invalid_yaml_content_raises():
    """Verify bad YAML syntax causes immediate failure."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("invalid_yaml: [unclosed list")
        temp_path = f.name
    try:
        with pytest.raises(ValueError, match="YAML parsing error"):
            load_config(temp_path)
    finally:
        os.remove(temp_path)

def test_deliberately_broken_schema_raises_informative_error():
    """Verify that illegal schema parameters fail immediately with clear messages."""
    broken_payload = {
        "environment": "development",
        "version": "1.0.0",
        "seed": -10,  # ILLEGAL: seed must be >= 0
        "data": {
            "version": "v1",
            "raw_path": "data/raw/credit_card_customers_v1.csv",
            "test_size": 0.95  # ILLEGAL: test_size > 0.50
        },
        "features": {
            "id_column": "Customer_ID",
            "target_column": "High_Risk_Customer",
            "utilisation_risk_cutoff": -0.5  # ILLEGAL: must be >= 0
        },
        "model": {
            "family": "quantum_unsupported_model"  # ILLEGAL: unsupported family
        },
        "training": {"cv_folds": 15},
        "tracking": {"experiment_name": "test"},
        "quality_gate": {"min_pr_auc_threshold": 1.5}
    }
    
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(broken_payload, f)
        temp_path = f.name
        
    try:
        with pytest.raises(ValueError) as exc_info:
            load_config(temp_path)
        err_msg = str(exc_info.value)
        assert "Configuration validation failed" in err_msg
        assert "seed" in err_msg or "test_size" in err_msg or "family" in err_msg
    finally:
        os.remove(temp_path)
