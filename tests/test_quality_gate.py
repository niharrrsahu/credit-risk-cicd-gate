"""Unit tests for CI/CD Quality Gate decisions across the 3 PR scenarios."""

import pytest
import os
import json
from src.config_schema import load_config
from src.gate import evaluate_quality_gate, load_baseline_metrics

@pytest.fixture
def baseline_data():
    return load_baseline_metrics("baseline_metrics.json")

def test_gate_scenario_1_passing_pr(baseline_data):
    """PR Scenario 1: Model improves or matches baseline metrics -> GATE PASSES."""
    config = load_config("config/ci.yaml")
    candidate_metrics = {
        "pr_auc": baseline_data["metrics"]["pr_auc"] + 0.02, # Improved PR-AUC
        "roc_auc": baseline_data["metrics"]["roc_auc"] + 0.01,
        "f1": 0.72,
        "recall": 0.88,
        "precision": 0.62,
        "accuracy": 0.86,
        "brier_score": 0.10
    }

    passed, summary, detail = evaluate_quality_gate(candidate_metrics, baseline_data, config)
    assert passed is True
    assert "PASSED" in summary
    assert len(detail["reasons"]) == 0
    assert "MERGE ALLOWED" in summary

def test_gate_scenario_2_quality_drop_fails_pr(baseline_data):
    """PR Scenario 2: Model quality regresses (untrained or degraded) -> GATE BLOCKS MERGE."""
    config = load_config("config/ci.yaml")
    # Severe degradation in PR-AUC and Recall
    candidate_metrics = {
        "pr_auc": 0.3500,  # Below minimum and large drop
        "roc_auc": 0.6200, # Large regression
        "f1": 0.30,
        "recall": 0.40,    # Fails recall safety floor (0.70)
        "precision": 0.25,
        "accuracy": 0.75,
        "brier_score": 0.28
    }

    passed, summary, detail = evaluate_quality_gate(candidate_metrics, baseline_data, config)
    assert passed is False
    assert "FAILED" in summary
    assert len(detail["reasons"]) > 0
    assert "MERGE BLOCKED" in summary
    # Verify exact reason triggers
    reason_text = " ".join(detail["reasons"])
    assert "PR-AUC" in reason_text or "Recall" in reason_text

def test_gate_scenario_3_boundary_drop_rejection(baseline_data):
    """Verify that even a drop of 0.03 (exceeding 0.015 allowance) triggers rejection."""
    config = load_config("config/ci.yaml")
    candidate_metrics = {
        "pr_auc": baseline_data["metrics"]["pr_auc"] - 0.03, # 3% drop exceeds 1.5% tolerance
        "roc_auc": baseline_data["metrics"]["roc_auc"],
        "f1": 0.65,
        "recall": 0.85,
        "precision": 0.52,
        "accuracy": 0.82,
        "brier_score": 0.12
    }

    passed, summary, detail = evaluate_quality_gate(candidate_metrics, baseline_data, config)
    assert passed is False
    assert "PR-AUC degraded" in " ".join(detail["reasons"])
