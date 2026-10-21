"""Unit tests for Feature Engineering logic and edge cases."""

import pytest
import numpy as np
import pandas as pd
from src.config_schema import FeatureConfig
from src.feature_engineering import CreditRiskFeatureEngineer, build_preprocessor

@pytest.fixture
def sample_feature_df():
    return pd.DataFrame({
        "Customer_ID": ["CC-TEST1", "CC-TEST2", "CC-TEST3", "CC-TEST4"],
        "Age": [25, 45, 60, 35],
        "Income_Bracket": ["Low", "Medium", "High", "Low"],
        "Credit_Limit": [5000.0, 10000.0, 0.0, 2000.0],  # TEST3 has 0.0 credit limit edge case!
        "Current_Balance": [4500.0, 2000.0, 500.0, 0.0],
        "Monthly_Spend": [1200.0, 800.0, 100.0, 100.0],
        "Num_Transactions_Monthly": [15, 10, 2, 5],
        "Payment_Delay_Days": [35.0, 5.0, 45.0, 0.0],    # TEST1 and TEST3 have > 30 days
        "Tenure_Months": [6, 24, 60, 12]                 # Covers: <=12 (new), 13-36 (established), >36 (mature)
    })

def test_feature_creation_columns(sample_feature_df):
    """Verify that all 4 required engineered features are created."""
    cfg = FeatureConfig()
    engineer = CreditRiskFeatureEngineer(cfg)
    transformed_df = engineer.transform(sample_feature_df)

    expected_cols = [
        "credit_utilisation_ratio",
        "delinquency_high_risk_flag",
        "tenure_maturity_stage",
        "financial_stress_index"
    ]
    for col in expected_cols:
        assert col in transformed_df.columns

def test_zero_credit_limit_edge_case(sample_feature_df):
    """Verify that Credit_Limit == 0 does NOT raise ZeroDivisionError and handles risk gracefully."""
    cfg = FeatureConfig()
    engineer = CreditRiskFeatureEngineer(cfg)
    transformed_df = engineer.transform(sample_feature_df)

    # Customer 3 has Credit_Limit = 0 and Current_Balance = 500 -> should be capped at max risk (2.5)
    row3 = transformed_df.iloc[2]
    assert not np.isnan(row3["credit_utilisation_ratio"])
    assert not np.isinf(row3["credit_utilisation_ratio"])
    assert row3["credit_utilisation_ratio"] == 2.5
    assert not np.isnan(row3["financial_stress_index"])
    assert not np.isinf(row3["financial_stress_index"])

def test_ratio_feature_values(sample_feature_df):
    """Verify calculation for standard ratio feature."""
    cfg = FeatureConfig()
    engineer = CreditRiskFeatureEngineer(cfg)
    transformed_df = engineer.transform(sample_feature_df)

    # Customer 1: 4500 / 5000 = 0.90
    assert pytest.approx(transformed_df.iloc[0]["credit_utilisation_ratio"], 0.01) == 0.90
    # Customer 2: 2000 / 10000 = 0.20
    assert pytest.approx(transformed_df.iloc[1]["credit_utilisation_ratio"], 0.01) == 0.20

def test_binary_flag_delinquency(sample_feature_df):
    """Verify binary delinquency flag triggers on either delay > 30 OR util > 0.8."""
    cfg = FeatureConfig(utilisation_risk_cutoff=0.8, delinquency_days_threshold=30)
    engineer = CreditRiskFeatureEngineer(cfg)
    transformed_df = engineer.transform(sample_feature_df)

    # Customer 1: delay 35 (>30) and util 0.9 (>0.8) -> Flag = 1
    assert transformed_df.iloc[0]["delinquency_high_risk_flag"] == 1
    # Customer 2: delay 5, util 0.2 -> Flag = 0
    assert transformed_df.iloc[1]["delinquency_high_risk_flag"] == 0
    # Customer 3: delay 45 (>30) -> Flag = 1
    assert transformed_df.iloc[2]["delinquency_high_risk_flag"] == 1
    # Customer 4: delay 0, util 0.0 -> Flag = 0
    assert transformed_df.iloc[3]["delinquency_high_risk_flag"] == 0

def test_binned_feature_categories(sample_feature_df):
    """Verify tenure categorisation into new, established, mature."""
    cfg = FeatureConfig()
    engineer = CreditRiskFeatureEngineer(cfg)
    transformed_df = engineer.transform(sample_feature_df)

    assert transformed_df.iloc[0]["tenure_maturity_stage"] == "new"          # 6 months
    assert transformed_df.iloc[1]["tenure_maturity_stage"] == "established"  # 24 months
    assert transformed_df.iloc[2]["tenure_maturity_stage"] == "mature"       # 60 months
    assert transformed_df.iloc[3]["tenure_maturity_stage"] == "new"          # 12 months

def test_pipeline_transform_consistency(sample_feature_df):
    """Verify full ColumnTransformer fits and transforms without error."""
    cfg = FeatureConfig()
    preprocessor, feature_names = build_preprocessor(cfg)
    
    # Fit and transform
    X_matrix = preprocessor.fit_transform(sample_feature_df)
    assert X_matrix.shape[0] == len(sample_feature_df)
    assert not np.isnan(X_matrix).any(), "Processed feature matrix contains NaNs"
