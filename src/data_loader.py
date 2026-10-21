"""Data Loader and Integrity Verifier with SHA-256 Checksum Validation."""

import os
import json
import hashlib
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from src.config_schema import AppConfig

REQUIRED_COLUMNS = [
    "Customer_ID",
    "Age",
    "Income_Bracket",
    "Credit_Limit",
    "Current_Balance",
    "Monthly_Spend",
    "Num_Transactions_Monthly",
    "Payment_Delay_Days",
    "Tenure_Months",
    "High_Risk_Customer"
]

def compute_sha256(file_path: str) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def verify_dataset_integrity(file_path: str, manifest_path: str, expected_version: str) -> Dict[str, Any]:
    """Verifies that the CSV matches the recorded manifest checksum."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file missing: {file_path}")
    
    current_hash = compute_sha256(file_path)
    
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        if expected_version in manifest:
            expected_hash = manifest[expected_version]["sha256"]
            if current_hash != expected_hash:
                raise ValueError(
                    f"Data integrity violation for version '{expected_version}'!\n"
                    f"Expected SHA-256: {expected_hash}\n"
                    f"Actual SHA-256:   {current_hash}"
                )
    return {"sha256": current_hash, "version": expected_version}

def load_data(config: AppConfig) -> pd.DataFrame:
    """Loads CSV and asserts column presence."""
    verify_dataset_integrity(config.data.raw_path, config.data.manifest_path, config.data.version)
    df = pd.read_csv(config.data.raw_path)
    
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
    
    return df

def get_train_test_split(config: AppConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Performs stratified train/test split without data leakage."""
    df = load_data(config)
    target_col = config.features.target_column
    
    stratify_target = df[target_col] if config.data.stratify else None
    train_df, test_df = train_test_split(
        df,
        test_size=config.data.test_size,
        random_state=config.seed,
        stratify=stratify_target
    )
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
