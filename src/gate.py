"""Automated CI/CD Quality Gate Comparator.
Evaluates candidate model performance against the stored repository baseline.
Fails with non-zero exit status if performance regresses beyond accepted thresholds.
"""

import os
import json
import sys
from typing import Dict, Any, Tuple
from src.config_schema import AppConfig

def load_baseline_metrics(baseline_path: str) -> Dict[str, Any]:
    """Loads repository baseline metrics."""
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline metrics file not found: {baseline_path}")
    with open(baseline_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_quality_gate(
    candidate_metrics: Dict[str, float],
    baseline_metrics: Dict[str, Any],
    config: AppConfig
) -> Tuple[bool, str, Dict[str, Any]]:
    """Evaluates candidate metrics against baseline and gate thresholds."""
    gate_cfg = config.quality_gate
    
    cand_pr_auc = candidate_metrics["pr_auc"]
    cand_roc_auc = candidate_metrics["roc_auc"]
    cand_recall = candidate_metrics["recall"]
    
    base_pr_auc = baseline_metrics["metrics"]["pr_auc"]
    base_roc_auc = baseline_metrics["metrics"]["roc_auc"]
    base_recall = baseline_metrics["metrics"]["recall"]
    
    pr_auc_delta = cand_pr_auc - base_pr_auc
    roc_auc_delta = cand_roc_auc - base_roc_auc
    recall_delta = cand_recall - base_recall
    
    reasons = []
    passed = True

    # Check 1: Primary metric drop (PR-AUC)
    if pr_auc_delta < -gate_cfg.max_pr_auc_drop_allowed:
        passed = False
        reasons.append(
            f"PR-AUC degraded by {abs(pr_auc_delta):.4f} (Allowed drop: {gate_cfg.max_pr_auc_drop_allowed:.4f})"
        )

    # Check 2: Secondary metric drop (ROC-AUC)
    if roc_auc_delta < -gate_cfg.max_roc_auc_drop_allowed:
        passed = False
        reasons.append(
            f"ROC-AUC degraded by {abs(roc_auc_delta):.4f} (Allowed drop: {gate_cfg.max_roc_auc_drop_allowed:.4f})"
        )

    # Check 3: Absolute PR-AUC minimum
    if cand_pr_auc < gate_cfg.min_pr_auc_threshold:
        passed = False
        reasons.append(
            f"Candidate PR-AUC ({cand_pr_auc:.4f}) is below minimum floor ({gate_cfg.min_pr_auc_threshold:.4f})"
        )

    # Check 4: Sensitivity / Recall floor
    if cand_recall < gate_cfg.min_recall_threshold:
        passed = False
        reasons.append(
            f"Candidate Recall ({cand_recall:.4f}) is below credit safety threshold ({gate_cfg.min_recall_threshold:.4f})"
        )

    status_str = "PASSED" if passed else "FAILED"
    
    # Markdown PR Summary
    summary_md = f"""### 🛡️ CI/CD Credit Risk Quality Gate: **{status_str}**

**Model Family:** `{config.model.family}` | **Dataset Version:** `{config.data.version}` | **Environment:** `{config.environment}`

| Metric | Candidate | Baseline (`{baseline_metrics.get('model_family', 'prod')}`) | Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **PR-AUC (Primary)** | `{cand_pr_auc:.4f}` | `{base_pr_auc:.4f}` | `{pr_auc_delta:+.4f}` | {"✅ OK" if pr_auc_delta >= -gate_cfg.max_pr_auc_drop_allowed else "❌ REGRESSION"} |
| **ROC-AUC** | `{cand_roc_auc:.4f}` | `{base_roc_auc:.4f}` | `{roc_auc_delta:+.4f}` | {"✅ OK" if roc_auc_delta >= -gate_cfg.max_roc_auc_drop_allowed else "❌ REGRESSION"} |
| **Recall (Default Capture)** | `{cand_recall:.4f}` | `{base_recall:.4f}` | `{recall_delta:+.4f}` | {"✅ OK" if cand_recall >= gate_cfg.min_recall_threshold else "❌ BELOW FLOOR"} |
| **F1-Score** | `{candidate_metrics['f1']:.4f}` | `{baseline_metrics['metrics'].get('f1', 0):.4f}` | `{candidate_metrics['f1'] - baseline_metrics['metrics'].get('f1', 0):+.4f}` | ℹ️ Tracked |
| **Accuracy** | `{candidate_metrics['accuracy']:.4f}` | `{baseline_metrics['metrics'].get('accuracy', 0):.4f}` | `{candidate_metrics['accuracy'] - baseline_metrics['metrics'].get('accuracy', 0):+.4f}` | ℹ️ Reference |

**Gate Decision:**
{"🟢 **MERGE ALLOWED**: The candidate model meets all performance criteria without unacceptable regressions." if passed else f"🔴 **MERGE BLOCKED**: Quality gate failed for the following reasons:\n- " + "\n- ".join(reasons)}
"""
    results_detail = {
        "passed": passed,
        "reasons": reasons,
        "deltas": {
            "pr_auc": pr_auc_delta,
            "roc_auc": roc_auc_delta,
            "recall": recall_delta
        }
    }
    return passed, summary_md, results_detail


def run_gate_cli(candidate_metrics_path: str, config_path: str, summary_output_path: str = None) -> int:
    """CLI runner for Quality Gate."""
    from src.config_schema import load_config
    config = load_config(config_path)
    
    with open(candidate_metrics_path, "r", encoding="utf-8") as f:
        candidate_metrics = json.load(f)
        
    baseline_metrics = load_baseline_metrics(config.quality_gate.baseline_metrics_path)
    
    passed, summary_md, detail = evaluate_quality_gate(candidate_metrics, baseline_metrics, config)
    
    print(summary_md)
    
    if summary_output_path:
        with open(summary_output_path, "w", encoding="utf-8") as f:
            f.write(summary_md)
            
    return 0 if passed else 1

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m src.gate <candidate_metrics.json> <config.yaml> [summary_output.md]")
        sys.exit(2)
    cand_path = sys.argv[1]
    cfg_path = sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None
    sys.exit(run_gate_cli(cand_path, cfg_path, out_path))
