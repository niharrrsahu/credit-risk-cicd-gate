# Credit Risk Model with a CI/CD Quality Gate

[![CI/CD Quality Gate](https://github.com/niharsahu/credit-risk-cicd-gate/actions/workflows/ci.yml/badge.svg)](https://github.com/niharsahu/credit-risk-cicd-gate/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![MLflow Tracking](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-brightgreen.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Elective II:** Machine Learning Operations (MLOps)  
**Assigned Project:** Project 1: Credit Risk Model with a CI/CD Quality Gate  
**Group 13 (Sheet 2):**  
* **Nihar Sahu** (Roll No: **B010**)  
* **Shaun Shaju** (Roll No: **B012**)  
**Instructor:** Dr. Santosh Kumar  

---

## 📌 Project Overview & Business Scenario
A commercial retail bank retrains its credit-card default prediction model on a monthly cadence. Recently, an untested code and feature modification lowered predictive quality unnoticed for two weeks, exposing the loan portfolio to unexpected charge-offs. Post-incident analysis revealed that engineering teams could not trace which commit, hyperparameter configuration, or data partition produced the degraded model.

This project implements an end-to-end, automated **CI/CD Quality Gate**:
1. **Decoupled Data & Configuration:** Code, data hashes (`data_manifest.json`), and YAML environments (`dev`, `ci`, `prod`) are versioned and validated via Pydantic schemas.
2. **Leakage-Free Feature Engineering:** Implements 4 custom domain features (Credit Utilisation Ratio, 30-Day Delinquency Flag, Tenure Maturity Stages, and Financial Stress Index) with zero data leakage.
3. **Statistical Quality Gate:** Automatically evaluates pull requests against a locked baseline (`baseline_metrics.json`) using **PR-AUC (Precision-Recall AUC)** and a minimum **Recall floor (≥ 0.70)** to prevent defaulting borrower leakage.
4. **Data-Change Retraining Drill:** Rigorously assesses model retraining on Version 2 data (1,100 records) across 3 random seeds with two-sample t-test significance testing.
5. **Demographic Fairness Defense:** Audits model error parity across `Income_Bracket` groups to guarantee compliance with Fair Lending regulations.

---

## 📂 Repository Directory Layout

```
credit_risk_cicd_gate/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Automated GitHub Actions CI workflow with quality gate
├── config/
│   ├── dev.yaml                   # Fast development environment configuration
│   ├── ci.yaml                    # Full CI Quality Gate testing & training configuration
│   └── prod.yaml                  # Production serving & monitoring configuration
├── data/
│   ├── data_manifest.json         # Cryptographic SHA-256 checksums and version metadata
│   ├── raw/                       # Raw datasets (excluded from git history)
│   │   ├── credit_card_customers_v1.csv  # 800 rows (initial dataset)
│   │   └── credit_card_customers_v2.csv  # 1,100 rows (v1 + 300 new customers)
│   └── processed/                 # Processed matrices (cache)
├── src/
│   ├── __init__.py
│   ├── config_schema.py           # Pydantic V2 schema validation for YAML configurations
│   ├── data_loader.py             # Data loading and SHA-256 integrity check
│   ├── feature_engineering.py     # 4 domain features, imputation, scikit-learn Pipeline
│   ├── train.py                   # Model training, hyperparameter execution, MLflow logging
│   ├── evaluate.py                # Imbalanced metrics & subgroup fairness evaluations
│   ├── gate.py                    # Quality Gate comparator CLI against baseline
│   └── utils.py                   # Global seed control and reproducibility utilities
├── tests/
│   ├── __init__.py
│   ├── test_config.py             # Config validation & intentional break unit tests
│   ├── test_features.py           # Unit tests for 4 features & edge cases (zero credit limit)
│   ├── test_pipeline.py           # Leakage prevention, ID exclusion, & determinism tests
│   └── test_quality_gate.py       # Gate logic verification across the 3 PR scenarios
├── scripts/
│   ├── run_experiments.py         # Executes 12 tracked MLflow runs and registers champion
│   ├── run_v1_v2_drill.py         # Runs Task 7 3-seed v1 vs v2 retraining drill & t-test
│   ├── run_fairness_defense.py    # Runs Task 8 3-rule benchmark & Income_Bracket fairness audit
│   ├── run_gate_scenarios.py      # Task 6 demonstration of PR1 Pass, PR2 Quality Fail, PR3 Test Fail
│   └── generate_report_pdf.py     # Generates the official 4-page submittable PDF report
├── reports/
│   ├── FINAL_PROJECT_REPORT.pdf   # Exactly 4-page official submission PDF
│   ├── LIFECYCLE_MAP.md           # Task 1: MLOps vs DevOps lifecycle mapping table & diagram
│   ├── FEATURE_DOCUMENTATION.md   # Task 4: Complete feature dictionary, formulas & justifications
│   ├── METRIC_JUSTIFICATION.md    # Task 5: Mathematical defense of PR-AUC vs Accuracy
│   ├── DATA_DRILL_V1_V2.md        # Task 7: Statistical comparison & gate decision on v2 data
│   ├── FAIRNESS_AND_GATE_DEFENSE_MEMO.md # Task 8: 1-page formal defense memo for risk committee
│   ├── PRESENTATION_AND_VIVA_GUIDE.md    # 5-minute presentation script & viva Q&A preparation
│   ├── experiment_runs_summary.md        # Formatted summary of all 12 experimental runs
│   └── candidate_rules_comparison.csv    # Rule A, Rule B, Rule C acceptance comparison
├── baseline_metrics.json          # Locked production baseline performance metrics
├── requirements.txt               # Locked project dependencies
├── .gitignore                     # Git tracking exclusions
└── README.md                      # Comprehensive project guide & documentation
```

---

## ⚡ Quickstart & Reproducibility Guide

Follow these steps to reproduce all experiments, unit tests, and quality gate evaluations from a fresh environment:

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/niharsahu/credit-risk-cicd-gate.git
cd credit-risk-cicd-gate

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute Automated Unit Tests
Run the 17 unit tests verifying configuration schemas, zero-leakage pipelines, and feature edge cases (e.g. division by zero credit limit):
```bash
python -m pytest -v tests/
```

### 3. Run the MLflow Experiment Tracking Suite (Task 5)
Trains 12 distinct models across Logistic Regression, Random Forests, LightGBM, and XGBoost, logs all artifacts into the SQLite MLflow backend (`sqlite:///mlflow.db`), and registers the champion model into the Model Registry:
```bash
python scripts/run_experiments.py
```
*To inspect the interactive MLflow UI:*
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://127.0.0.1:5000 in your browser
```

### 4. Execute the CI/CD Quality Gate Demonstration (Task 6)
Demonstrates the three mandatory Pull Request conditions:
* **PR #101 (PASS):** Tuned model improves PR-AUC (+0.0180) $\rightarrow$ **Merge Approved**.
* **PR #102 (FAIL - Regression):** Untested feature change degrades PR-AUC to 0.4120 and Recall to 0.45 $\rightarrow$ **Merge Blocked**.
* **PR #103 (FAIL - Unit Test):** Breaking syntax/logic error triggers fast failure in pytest $\rightarrow$ **Merge Blocked**.
```bash
python scripts/run_gate_scenarios.py
```

### 5. Run the Version 1 vs Version 2 Retraining Drill (Task 7)
Retrains the champion architecture across 3 random seeds (42, 101, 777) on v1 vs v2 datasets and computes statistical significance:
```bash
python scripts/run_v1_v2_drill.py
```

### 6. Run the Quality Gate Defense & Fairness Audit (Task 8)
Benchmarks Candidate Rules A, B, and C, and executes the `Income_Bracket` demographic parity audit:
```bash
python scripts/run_fairness_defense.py
```

### 7. Compile the Official 4-Page Submission PDF Report
```bash
python scripts/generate_report_pdf.py
```

---

## 📊 Summary of Key Results

| Milestone / Evaluation | Primary Metric (PR-AUC) | Secondary (ROC-AUC) | Recall (Defaults) | F1-Score | Gate Decision / Action |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Locked Baseline Model (LogReg)** | `0.7978` | `0.9215` | `0.8947` | `0.6733` | Production Reference |
| **Registered Champion (XGBoost Tuned)** | `0.8091` | `0.9414` | `0.7895` | `0.7407` | **PROMOTED** to Model Registry |
| **Version 2 Drill (Seed 42)** | `0.8459` | `0.9398` | `0.8654` | `0.7586` | **ACCEPTED**: 10× Variance Drop |
| **Version 2 Drill (Seed 101)** | `0.8505` | `0.9558` | `0.7885` | `0.7130` | **ACCEPTED**: High-spend density |
| **Version 2 Drill (Seed 777)** | `0.8402` | `0.9446` | `0.7500` | `0.6818` | **ACCEPTED**: Stable bounds |

---

## ⚖️ Academic Integrity & AI Tool Usage Disclosure

In compliance with **Section 8 of the Course Project Brief**:
* **Tools Consulted:** Anthropic Claude / Google Gemini / OpenAI assistants were utilized during the project lifecycle strictly as technical syntax accelerators (looking up Pydantic V2 model validator syntax, ReportLab Flowable table layout constraints, and SQLite tracking URI flags).
* **Independent Contributions:** All architectural decisions, mathematical justifications (PR-AUC vs Accuracy proof), baseline gate threshold formulations, feature formulas (`financial_stress_index`), data drill statistical analyses, and fairness defense trade-off memos were independently designed, implemented, and defended by Group 13 members (**Nihar Sahu** and **Shaun Shaju**).
* Both partners are fully prepared to explain, modify live, and defend every component of this repository during the scheduled demonstration on **26th/27th October 2026**.
