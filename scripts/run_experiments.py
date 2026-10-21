"""Experiment Tracking Execution Script (Task 5).
Executes at least 10 distinct experiments across multiple model families and hyperparameters.
Logs all parameters, metrics, run configurations, and models to MLflow SQLite backend.
Registers the winning model in the MLflow Model Registry.
"""

import sys
import os

# Ensure project root is in python path before importing src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

import json
import pandas as pd
import mlflow
from src.config_schema import load_config
from src.train import train_and_track

EXPERIMENTS = [
    # 1-3: Logistic Regression Family
    {
        "name": "logreg_baseline_c1.0",
        "family": "logistic_regression",
        "params": {"C": 1.0, "max_iter": 300, "class_weight": "balanced", "solver": "lbfgs"}
    },
    {
        "name": "logreg_l2_regularized_c0.1",
        "family": "logistic_regression",
        "params": {"C": 0.1, "max_iter": 300, "class_weight": "balanced", "solver": "lbfgs"}
    },
    {
        "name": "logreg_high_c10.0",
        "family": "logistic_regression",
        "params": {"C": 10.0, "max_iter": 300, "class_weight": "balanced", "solver": "lbfgs"}
    },
    # 4-6: Random Forest Family
    {
        "name": "rf_trees50_depth3",
        "family": "random_forest",
        "params": {"n_estimators": 50, "max_depth": 3, "class_weight": "balanced"}
    },
    {
        "name": "rf_trees100_depth5_balanced",
        "family": "random_forest",
        "params": {"n_estimators": 100, "max_depth": 5, "class_weight": "balanced_subsample"}
    },
    {
        "name": "rf_trees200_depth7",
        "family": "random_forest",
        "params": {"n_estimators": 200, "max_depth": 7, "class_weight": "balanced"}
    },
    # 7-9: XGBoost Family
    {
        "name": "xgb_default_depth3_lr0.1",
        "family": "xgboost",
        "params": {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.1, "eval_metric": "logloss"}
    },
    {
        "name": "xgb_tuned_depth4_lr0.05_posweight",
        "family": "xgboost",
        "params": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.85, "scale_pos_weight": 4.25, "eval_metric": "logloss"}
    },
    {
        "name": "xgb_deep_depth5_lr0.01",
        "family": "xgboost",
        "params": {"n_estimators": 120, "max_depth": 5, "learning_rate": 0.01, "subsample": 0.8, "eval_metric": "logloss"}
    },
    # 10-12: LightGBM Family
    {
        "name": "lgbm_default_leaves15_lr0.1",
        "family": "lightgbm",
        "params": {"n_estimators": 80, "num_leaves": 15, "learning_rate": 0.1}
    },
    {
        "name": "lgbm_unbalanced_leaves31_lr0.05",
        "family": "lightgbm",
        "params": {"n_estimators": 100, "num_leaves": 31, "learning_rate": 0.05, "is_unbalance": True}
    },
    {
        "name": "lgbm_tuned_leaves20_lr0.03",
        "family": "lightgbm",
        "params": {"n_estimators": 120, "num_leaves": 20, "learning_rate": 0.03, "is_unbalance": True}
    }
]

def main():
    base_cfg = load_config("config/ci.yaml")
    print(f"==================================================")
    print(f"Starting Task 5: Running {len(EXPERIMENTS)} Tracked Experiments")
    print(f"Tracking Backend: {base_cfg.tracking.tracking_uri}")
    print(f"Experiment Name:  {base_cfg.tracking.experiment_name}")
    print(f"==================================================")

    results = []
    best_pr_auc = -1.0
    best_run = None

    for idx, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n[{idx}/{len(EXPERIMENTS)}] Running: {exp['name']} ({exp['family']})...")
        cfg = base_cfg.model_copy(deep=True)
        cfg.model.family = exp["family"]
        cfg.model.params = exp["params"]
        
        # We will register only the highest scoring candidate at the end
        cfg.tracking.register_model = False

        res = train_and_track(cfg, run_name=exp["name"])
        metrics = res["metrics"]
        print(f"   -> PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | Recall: {metrics['recall']:.4f} | Accuracy: {metrics['accuracy']:.4f}")

        row = {
            "Run_Name": exp["name"],
            "Model_Family": exp["family"],
            "Run_ID": res["run_id"],
            "PR_AUC": metrics["pr_auc"],
            "ROC_AUC": metrics["roc_auc"],
            "Recall": metrics["recall"],
            "Precision": metrics["precision"],
            "F1_Score": metrics["f1"],
            "Accuracy": metrics["accuracy"],
            "Brier_Score": metrics["brier_score"]
        }
        results.append(row)

        if metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = metrics["pr_auc"]
            best_run = (exp, cfg, res)

    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    os.makedirs("reports", exist_ok=True)
    df_results.to_csv("reports/experiment_runs_comparison.csv", index=False)
    print("\n==================================================")
    print("All Experiments Completed Successfully!")
    print(f"Best Model: {best_run[0]['name']} with PR-AUC = {best_pr_auc:.4f}")
    
    # Register the Best Winning Model in MLflow Model Registry
    print(f"Registering winning model into MLflow Model Registry as '{base_cfg.tracking.registered_model_name}'...")
    winning_cfg = best_run[1].model_copy(deep=True)
    winning_cfg.tracking.register_model = True
    winning_res = train_and_track(winning_cfg, run_name=f"FINAL_REGISTERED_{best_run[0]['name']}")
    print(f"Winning Model successfully registered! MLflow Run ID: {winning_res['run_id']}")
    
    # Save markdown summary table
    md_table = df_results.to_markdown(index=False)
    with open("reports/experiment_runs_summary.md", "w", encoding="utf-8") as f:
        f.write("# Task 5: Experiment Tracking & Model Selection Summary\n\n")
        f.write(f"Total Runs Logged: {len(results)}\n\n")
        f.write(md_table)
        f.write(f"\n\n**Selected & Registered Model:** `{best_run[0]['name']}` (PR-AUC: `{best_pr_auc:.4f}`)\n")
    print("Summary saved to reports/experiment_runs_summary.md")

if __name__ == "__main__":
    main()
