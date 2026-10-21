"""Model Training, Hyperparameter Execution, and MLflow Tracking Module."""

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import json
import tempfile
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from src.config_schema import AppConfig, load_config
from src.data_loader import get_train_test_split
from src.feature_engineering import build_preprocessor, prepare_features_and_target
from src.evaluate import compute_all_metrics

def instantiate_model(model_family: str, params: Dict[str, Any], seed: int):
    """Instantiates model according to family and hyperparameters."""
    if model_family == "logistic_regression":
        return LogisticRegression(
            random_state=seed,
            **params
        )
    elif model_family == "random_forest":
        return RandomForestClassifier(
            random_state=seed,
            **params
        )
    elif model_family == "xgboost":
        return XGBClassifier(
            random_state=seed,
            **params
        )
    elif model_family == "lightgbm":
        return LGBMClassifier(
            random_state=seed,
            verbose=-1,
            **params
        )
    else:
        raise ValueError(f"Unsupported model family: '{model_family}'")


def train_and_track(config: AppConfig, run_name: str = None) -> Dict[str, Any]:
    """Trains model according to AppConfig and logs full artifacts to MLflow."""
    mlflow.set_tracking_uri(config.tracking.tracking_uri)
    mlflow.set_experiment(config.tracking.experiment_name)

    # 1. Load Data
    train_df, test_df = get_train_test_split(config)
    X_train_raw, y_train = prepare_features_and_target(train_df, config.features)
    X_test_raw, y_test = prepare_features_and_target(test_df, config.features)

    # 2. Build Pipeline
    preprocessor, feat_names = build_preprocessor(config.features)
    classifier = instantiate_model(config.model.family, config.model.params, config.seed)

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    if run_name is None:
        run_name = f"{config.model.family}_{config.data.version}_seed{config.seed}"

    with mlflow.start_run(run_name=run_name) as run:
        # Fit model on training split
        full_pipeline.fit(X_train_raw, y_train)

        # Predict probabilities
        y_test_probs = full_pipeline.predict_proba(X_test_raw)[:, 1]
        test_metrics = compute_all_metrics(y_test.values, y_test_probs)

        # Log parameters
        mlflow.log_param("environment", config.environment)
        mlflow.log_param("dataset_version", config.data.version)
        mlflow.log_param("dataset_raw_path", config.data.raw_path)
        mlflow.log_param("seed", config.seed)
        mlflow.log_param("model_family", config.model.family)
        for k, v in config.model.params.items():
            mlflow.log_param(f"param_{k}", v)
        for k, v in config.features.model_dump().items():
            mlflow.log_param(f"feat_{k}", v)

        # Log metrics
        for metric_name, metric_val in test_metrics.items():
            mlflow.log_metric(metric_name, metric_val)

        # Log config as artifact
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp_f:
            import yaml
            yaml.dump(config.model_dump(), tmp_f)
            tmp_path = tmp_f.name
        mlflow.log_artifact(tmp_path, artifact_path="config")
        os.remove(tmp_path)

        # Log metrics json artifact
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp_f:
            json.dump(test_metrics, tmp_f, indent=2)
            tmp_metric_path = tmp_f.name
        mlflow.log_artifact(tmp_metric_path, artifact_path="metrics")
        os.remove(tmp_metric_path)

        # Log Model artifact
        if config.tracking.log_models:
            try:
                mlflow.sklearn.log_model(
                    sk_model=full_pipeline,
                    artifact_path="model",
                    serialization_format="cloudpickle",
                    registered_model_name=config.tracking.registered_model_name if config.tracking.register_model else None
                )
            except Exception as e:
                # Fallback directly to pickle/cloudpickle artifact logging if skops strict security is triggered
                import cloudpickle
                model_pkl_path = "model.pkl"
                with open(model_pkl_path, "wb") as f_out:
                    cloudpickle.dump(full_pipeline, f_out)
                mlflow.log_artifact(model_pkl_path, artifact_path="model")
                if os.path.exists(model_pkl_path):
                    os.remove(model_pkl_path)

        run_id = run.info.run_id

    return {
        "run_id": run_id,
        "run_name": run_name,
        "metrics": test_metrics,
        "pipeline": full_pipeline,
        "train_df": train_df,
        "test_df": test_df,
        "y_test_probs": y_test_probs
    }
