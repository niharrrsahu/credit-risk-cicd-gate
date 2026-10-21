"""Feature Engineering and Preprocessing Pipeline.
Implements 4 engineered features with zero data leakage and scikit-learn compatibility:
1. Ratio: Credit Utilisation (Current_Balance / Credit_Limit)
2. Binary Flag: Delinquency High Risk Flag (Payment_Delay_Days > 30 or Utilisation > 0.8)
3. Binned: Tenure Maturity Stage (new <= 12m, established 13-36m, mature > 36m)
4. Domain Custom: Financial Stress Index ((Monthly_Spend / Credit_Limit) * (1 + Delay / 30))
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from src.config_schema import FeatureConfig

class CreditRiskFeatureEngineer(BaseEstimator, TransformerMixin):
    """Generates the 4 required features and handles edge cases safely."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = FeatureConfig()
        self.config = config
        self.utilisation_risk_cutoff = config.utilisation_risk_cutoff
        self.delinquency_days_threshold = config.delinquency_days_threshold
        self.high_spend_ratio_threshold = config.high_spend_ratio_threshold

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        
        # 1. Ratio Feature: Credit Utilisation Ratio (safe division for zero credit limit)
        credit_limit_safe = df["Credit_Limit"].replace(0, np.nan)
        # If Credit_Limit is 0 or NaN, fill with fallback 1.0 to avoid division by zero
        credit_limit_safe = credit_limit_safe.fillna(1.0)
        
        utilisation = df["Current_Balance"] / credit_limit_safe
        # Clip outliers between 0.0 and 2.5
        df["credit_utilisation_ratio"] = utilisation.clip(lower=0.0, upper=2.5)
        # If original limit was 0 and balance > 0, utilisation is maximum risk (2.5)
        df.loc[(df["Credit_Limit"] == 0) & (df["Current_Balance"] > 0), "credit_utilisation_ratio"] = 2.5
        df.loc[(df["Credit_Limit"] == 0) & (df["Current_Balance"] <= 0), "credit_utilisation_ratio"] = 0.0

        # 2. Binary Flag: Delinquency / High Utilization Alert
        delay_condition = df["Payment_Delay_Days"] > self.delinquency_days_threshold
        util_condition = df["credit_utilisation_ratio"] > self.utilisation_risk_cutoff
        df["delinquency_high_risk_flag"] = (delay_condition | util_condition).astype(int)

        # 3. Binned Feature: Tenure Maturity Stage
        # <= 12 months: new, 13-36 months: established, > 36 months: mature
        df["tenure_maturity_stage"] = pd.cut(
            df["Tenure_Months"],
            bins=[-np.inf, 12, 36, np.inf],
            labels=["new", "established", "mature"]
        ).astype(str)

        # 4. Domain Feature: Financial Stress Index
        # Spend burn rate relative to approved limit, compounded by delay in days
        spend_ratio = (df["Monthly_Spend"] / credit_limit_safe).clip(lower=0.0, upper=2.0)
        delay_factor = 1.0 + (df["Payment_Delay_Days"].clip(lower=0) / 30.0)
        df["financial_stress_index"] = spend_ratio * delay_factor

        return df


def build_preprocessor(feature_config: FeatureConfig) -> Tuple[Pipeline, List[str]]:
    """Builds an end-to-end ColumnTransformer and Feature Pipeline."""
    
    engineered_numerical = [
        "Age",
        "Credit_Limit",
        "Current_Balance",
        "Monthly_Spend",
        "Num_Transactions_Monthly",
        "Payment_Delay_Days",
        "Tenure_Months",
        "credit_utilisation_ratio",
        "delinquency_high_risk_flag",
        "financial_stress_index"
    ]
    
    engineered_categorical = [
        "Income_Bracket",
        "tenure_maturity_stage"
    ]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=feature_config.numerical_impute_strategy)),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=feature_config.categorical_impute_strategy, fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    col_transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, engineered_numerical),
            ("cat", cat_pipeline, engineered_categorical)
        ],
        remainder="drop"
    )

    full_pipeline = Pipeline([
        ("feature_creator", CreditRiskFeatureEngineer(feature_config)),
        ("preprocessor", col_transformer)
    ])

    return full_pipeline, engineered_numerical + engineered_categorical


def prepare_features_and_target(
    df: pd.DataFrame, 
    feature_config: FeatureConfig
) -> Tuple[pd.DataFrame, pd.Series]:
    """Drops identifier (Customer_ID) and isolates target without leakage."""
    id_col = feature_config.id_column
    target_col = feature_config.target_column
    
    # Strictly ensure Customer_ID is excluded from features
    feature_cols = [c for c in df.columns if c not in [id_col, target_col]]
    X = df[feature_cols].copy()
    y = df[target_col].copy() if target_col in df.columns else None
    
    return X, y
