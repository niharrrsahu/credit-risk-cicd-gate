"""Model Evaluation, Metric Calculations, and Fairness / Subgroup Breakdown."""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    brier_score_loss,
    confusion_matrix
)

def compute_all_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Computes comprehensive metrics with emphasis on imbalanced classification."""
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    # In case of single class in batch edge case
    try:
        roc_auc = float(roc_auc_score(y_true, y_pred_prob))
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = float(average_precision_score(y_true, y_pred_prob))
    except Exception:
        pr_auc = float(np.mean(y_true))
        
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_pred_prob))
    
    # Baseline comparison: Majority class trivial predictor accuracy
    majority_acc = float(max(np.mean(y_true == 0), np.mean(y_true == 1)))

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "f1": round(f1, 4),
        "recall": round(rec, 4),
        "precision": round(prec, 4),
        "accuracy": round(acc, 4),
        "majority_class_accuracy": round(majority_acc, 4),
        "brier_score": round(brier, 4)
    }


def evaluate_subgroup_fairness(
    df_raw: pd.DataFrame,
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    subgroup_col: str = "Income_Bracket",
    threshold: float = 0.5
) -> pd.DataFrame:
    """Computes subgroup error distributions across sensitive attributes (Income_Bracket).
    Fulfills Task 8 requirements for Fairness and Error Breakdown.
    """
    df_eval = df_raw.copy()
    df_eval["y_true"] = y_true
    df_eval["y_prob"] = y_pred_prob
    df_eval["y_pred"] = (y_pred_prob >= threshold).astype(int)

    records = []
    for group_name, group_df in df_eval.groupby(subgroup_col):
        g_y_true = group_df["y_true"].values
        g_y_pred = group_df["y_pred"].values
        g_y_prob = group_df["y_prob"].values
        
        n_samples = len(group_df)
        n_positives = int(np.sum(g_y_true))
        base_rate = float(np.mean(g_y_true))
        selection_rate = float(np.mean(g_y_pred))
        
        tn, fp, fn, tp = confusion_matrix(g_y_true, g_y_pred, labels=[0, 1]).ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # False Positive Rate
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0 # False Negative Rate (missed defaults)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (prec * tpr) / (prec + tpr) if (prec + tpr) > 0 else 0.0
        
        try:
            subgroup_pr_auc = average_precision_score(g_y_true, g_y_prob)
        except Exception:
            subgroup_pr_auc = base_rate

        records.append({
            subgroup_col: group_name,
            "Total_Count": n_samples,
            "Actual_High_Risk_Rate": round(base_rate, 4),
            "Predicted_High_Risk_Rate": round(selection_rate, 4),
            "Recall_TPR": round(tpr, 4),
            "FPR": round(fpr, 4),
            "FNR_Missed_Defaults": round(fnr, 4),
            "Precision": round(prec, 4),
            "F1": round(f1, 4),
            "PR_AUC": round(subgroup_pr_auc, 4)
        })

    subgroup_df = pd.DataFrame(records)
    return subgroup_df
