"""Unit tests for Data Pipeline integrity, leakage prevention, and reproducibility."""

import pytest
import os
import pandas as pd
import numpy as np
from src.config_schema import load_config
from src.data_loader import load_data, get_train_test_split, verify_dataset_integrity
from src.feature_engineering import prepare_features_and_target, build_preprocessor

def test_data_integrity_sha256_v1():
    """Verify v1 dataset sha256 checksum matches manifest."""
    config = load_config("config/dev.yaml")
    info = verify_dataset_integrity(config.data.raw_path, config.data.manifest_path, config.data.version)
    assert info["version"] == "v1"
    assert len(info["sha256"]) == 64

def test_no_target_leakage_and_id_exclusion():
    """Verify that Customer_ID is excluded and target column is not in feature matrix."""
    config = load_config("config/dev.yaml")
    df = load_data(config)
    X, y = prepare_features_and_target(df, config.features)

    assert "Customer_ID" not in X.columns, "Customer_ID leaked into feature matrix!"
    assert config.features.target_column not in X.columns, "Target column leaked into features!"
    assert len(X) == len(y)
    assert set(y.unique()) == {0, 1}

def test_stratified_train_test_split_ratio():
    """Verify train/test split maintains target proportion and seed reproducibility."""
    config = load_config("config/dev.yaml")
    train_df1, test_df1 = get_train_test_split(config)
    train_df2, test_df2 = get_train_test_split(config)

    # Determinism check
    pd.testing.assert_frame_equal(train_df1, train_df2)
    pd.testing.assert_frame_equal(test_df1, test_df2)

    # Class balance check
    train_pos_rate = train_df1["High_Risk_Customer"].mean()
    test_pos_rate = test_df1["High_Risk_Customer"].mean()
    assert abs(train_pos_rate - test_pos_rate) < 0.03

def test_unseen_data_transformation_consistency():
    """Verify that a fitted preprocessor transforms new unseen data without error."""
    config = load_config("config/dev.yaml")
    df_v1 = pd.read_csv("data/raw/credit_card_customers_v1.csv")
    df_v2 = pd.read_csv("data/raw/credit_card_customers_v2.csv")

    X_train, _ = prepare_features_and_target(df_v1.iloc[:600], config.features)
    X_new, _ = prepare_features_and_target(df_v2.iloc[800:], config.features) # 300 new customers

    preprocessor, _ = build_preprocessor(config.features)
    preprocessor.fit(X_train)
    
    transformed_new = preprocessor.transform(X_new)
    assert transformed_new.shape[0] == 300
    assert not np.isnan(transformed_new).any()
