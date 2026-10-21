# 5-Minute Viva Presentation & Live Demonstration Guide

**Project 1: Credit Risk Model with a CI/CD Quality Gate**  
**Group 13:** Nihar Sahu (Roll No: B010) & Shaun Shaju (Roll No: B012)  
**Evaluator:** Dr. Santosh Kumar (Elective II: MLOps)  
**Scheduled Date:** 26th / 27th October 2026  

---

## 1. Minute-by-Minute Presentation Schedule (Total: 5 Minutes)

```
00:00 - 01:00  [Nihar] Scenario, Business Problem & DevOps vs MLOps Lifecycle Mapping (Task 1)
01:00 - 02:00  [Shaun] Data Versioning, Configuration Management & 4 Engineered Features (Tasks 2, 3, 4)
02:00 - 03:15  [Nihar] MLflow Experiment Tracking, Metric Justification & Live CI Gate Demo (Tasks 5 & 6)
03:15 - 04:15  [Shaun] Task 7 Data Drill (v1 vs v2 Retraining) & Multi-Seed Statistical Significance
04:15 - 05:00  [Both]  Task 8 Quality Gate Threshold Defense, Demographic Parity & Q&A Readiness
```

---

## 2. Speaker Scripts & Live Demonstration Instructions

### Minute 00:00 - 01:00 | Speaker: Nihar Sahu (B010)
* **What to Say:**
  > *"Good afternoon, Professor Santosh. We are Group 13—I am Nihar Sahu (B010), and with me is Shaun Shaju (B012). Today we present our project: Credit Risk Model with a CI/CD Quality Gate.*
  > 
  > *In commercial retail banking, an untested model release recently degraded default predictions unnoticed for two weeks, leaving leadership blind to portfolio risk. To solve this, we engineered an automated CI/CD Quality Gate.*
  > 
  > *In standard software delivery, identical code produces deterministic binaries. But in MLOps, builds depend codependently on shifting borrower distributions, and a bug-free build can still produce mathematically dangerous predictions. We mapped our repository across all 8 lifecycle phases—from YAML-driven planning to automated CI gating and demographic monitoring."*

### Minute 01:00 - 02:00 | Speaker: Shaun Shaju (B012)
* **What to Say:**
  > *"Thank you, Nihar. Moving to our technical foundation: code and data are decoupled. Datasets are not dumped into git history; instead, `data_manifest.json` tracks cryptographic SHA-256 hashes, ensuring any earlier model version can be reproduced with its exact data.*
  > 
  > *All thresholds, hyperparameters, and seeds are decoupled into YAML environments—`dev.yaml`, `ci.yaml`, and `prod.yaml`—strictly validated by Pydantic schemas that fail immediately if broken.*
  > 
  > *For feature engineering, we addressed 24 missing values without data leakage using training-only median imputation. We engineered four domain features: a zero-safe Credit Utilisation ratio, a 30-Day Delinquency binary flag, a 3-tier Tenure Maturity category, and a composite Financial Stress Index combining spend velocity with late payment drag. All 17 unit tests pass with zero leakage."*

### Minute 02:00 - 03:15 | Speaker: Nihar Sahu (B010)
* **What to Say & Show (Live Terminal Demo):**
  > *"Because only 18.9% of customers default, plain accuracy is totally invalid: a dummy model predicting all non-defaulters yields 81.1% accuracy but misses 100% of defaults. Hence, our primary gate metric is PR-AUC (Average Precision), constrained by a 70% minimum Recall floor.*
  > 
  > *We tracked 12 experiments in our MLflow SQLite registry across Logistic Regression, Random Forests, LightGBM, and XGBoost, where tuned XGBoost achieved peak PR-AUC of 0.8091.*
  > 
  > *Let me demonstrate our CI/CD Quality Gate live across the 3 required pull request scenarios:"*
* **Command to Run in Terminal:**
  ```powershell
  python scripts/run_gate_scenarios.py
  ```
* **Explanation while running:**
  > *"As you can see, PR #101 passes because the candidate improves PR-AUC by +0.0180. PR #102 fails and blocks merge because PR-AUC degraded to 0.4120 and Recall fell to 45%. And PR #103 fast-fails at the pytest stage, preventing any defective code from ever reaching model registration."*

### Minute 03:15 - 04:15 | Speaker: Shaun Shaju (B012)
* **What to Say:**
  > *"When Version 2 data arrived with 300 additional customers, we executed the Task 7 retraining drill across 3 independent random seeds: 42, 101, and 777.*
  > 
  > *Using Welch's t-test, we proved that while mean PR-AUC remained steady (+0.0070 gain), the standard deviation shrank tenfold—from ±0.0523 on v1 down to ±0.0052 on v2. The new customers provided critical density in high-spend segments, stabilizing variance. Based on this statistical proof, our gate officially approved promoting the v2 model to baseline."*

### Minute 04:15 - 05:00 | Speaker: Nihar Sahu & Shaun Shaju
* **What to Say:**
  > *"Finally, in Task 8 we defended our gate rule against three alternatives. A permissive ROC-AUC gate accepted 11 out of 12 runs, including defective models. An over-strict PR-AUC gate rejected 11 out of 12 runs due to seed noise. Our Composite Rule C accepted 7 runs—protecting downside credit risk while enabling agile development.*
  > 
  > *In our subgroup audit across High, Medium, and Low Income brackets, Recall remained above 80% for all groups, with highest recall (88.24%) protecting vulnerable Low Income borrowers from unmanageable debt. The model complies with Fair Lending regulations. Thank you, and we welcome your questions."*

---

## 3. Predicted Viva Questions & Strong Answers for Dr. Santosh Kumar

### Q1: *"Why did you choose PR-AUC instead of ROC-AUC as the primary gate metric?"*
* **Answer:** *"In our dataset, 81.1% of borrowers are non-defaulters. ROC-AUC plots True Positive Rate against False Positive Rate. Because True Negatives are so abundant, the FPR denominator ($\text{FP} + \text{TN}$) is heavily diluted, giving ROC-AUC an artificially high score of $>0.92$ even when the model makes frequent false alarms. PR-AUC focuses exclusively on the minority positive class ($\text{Precision} = \text{TP} / (\text{TP} + \text{FP})$), making it far more sensitive to credit risk degradation."*

### Q2: *"How did you prevent data leakage during feature engineering and imputation?"*
* **Answer:** *"All preprocessing components—`SimpleImputer`, `StandardScaler`, and `OneHotEncoder`—are fitted strictly on `X_train` inside an encapsulated scikit-learn Pipeline. `X_test` and incoming v2 inference batches are only passed to `.transform()`, never `.fit()`. Furthermore, `Customer_ID` is removed prior to feature transformation."*

### Q3: *"How does your quality gate prevent accidental or unauthorized baseline updates?"*
* **Answer:** *"The baseline metrics are committed as a protected repository artefact (`baseline_metrics.json`). In our GitHub Actions pipeline, PRs can only compare against the baseline. Updating `baseline_metrics.json` requires a dedicated Release PR with branch protection rules requiring sign-off from both the Risk Operations Lead and Senior MLOps Engineer, supported by 3-seed statistical evidence."*

### Q4: *"Did you observe algorithmic bias in the Low Income bracket?"*
* **Answer:** *"No. While the default rate is naturally higher in the Low Income tier (20.48% vs 11.90% in High Income) due to lower liquidity buffers, our model achieved its highest Recall (88.24%) and Precision (68.18%) on the Low Income cohort. False Positive Rates remained bounded between 8.1% and 13.6%. The model does not discriminate; it accurately assesses debt serviceability."*
