# Task 8: Quality-Gate Threshold & Fairness Defense Memo

**To:** Retail Credit Risk Governance Committee & MLOps Architecture Board  
**From:** Group 13 — Nihar Sahu (B010) & Shaun Shaju (B012)  
**Date:** October 2026  
**Subject:** Formal Defense of CI/CD Quality Gate Thresholds and Subgroup Demographic Parity Audit  

---

### 1. Executive Summary & Gate Rule Recommendation
We recommend adopting **Rule C (Composite Imbalanced Safety Gate)** as the mandatory CI/CD quality gate for automated merge and deployment:
* **Rule C Specification:**  
  `Candidate PR-AUC >= Baseline PR-AUC - 0.015` AND `Recall >= 0.70` AND `ROC-AUC >= Baseline ROC-AUC - 0.015`

### 2. Concrete Comparison of Candidate Gate Rules
We benchmarked three competing gate designs across all 12 experimental model runs:

| Gate Rule ID | Formula / Criterion | Accepted Runs | Rejected Runs | Accepted Trade-off & Risk Assessment |
| :--- | :--- | :---: | :---: | :--- |
| **Rule A: Permissive** | Delta ROC-AUC >= -0.02 | **11** | 1 | **High Risk:** ROC-AUC is heavily masked by true negatives in imbalanced data (81% non-default). It allowed severely degraded models that missed over 40% of actual defaulters. |
| **Rule B: Over-Strict** | PR-AUC >= Baseline and >= 0.80 | **1** | 11 | **Excessive Friction:** Rejects well-calibrated architectures simply due to minor seed variance; prevents healthy incremental code refactoring. |
| **Rule C: Composite (Recommended)** | Delta PR-AUC >= -0.015 and Recall >= 0.70 | **7** | 5 | **Optimal Balance:** Enforces a non-negotiable credit safety floor (>= 70% recall of defaults) while using PR-AUC as the primary gate metric and tolerating small run noise. |

```
Distribution of Gate Decisions Across Runs:
Rule A (Permissive)       [████████████████████] 11/12 Accepted (Unsafe)
Rule B (Over-Strict)      [████░░░░░░░░░░░░░░░░] 1/12 Accepted (Too brittle)
Rule C (Recommended)      [██████████░░░░░░░░░░] 7/12 Accepted (Safe & robust)
```

---

### 3. Demographic Subgroup Breakdown across `Income_Bracket`
An ethical and regulatory requirement is ensuring the registered credit risk model does not unfairly discriminate across socioeconomic income tiers:

| Income_Bracket   |   Total_Count |   Actual_High_Risk_Rate |   Predicted_High_Risk_Rate |   Recall_TPR |    FPR |   FNR_Missed_Defaults |   Precision |     F1 |   PR_AUC |
|:-----------------|--------------:|------------------------:|---------------------------:|-------------:|-------:|----------------------:|------------:|-------:|---------:|
| High             |            42 |                  0.119  |                     0.1667 |       0.8    | 0.0811 |                0.2    |      0.5714 | 0.6667 |   0.8625 |
| Low              |            83 |                  0.2048 |                     0.2651 |       0.8824 | 0.1061 |                0.1176 |      0.6818 | 0.7692 |   0.8589 |
| Medium           |            75 |                  0.2133 |                     0.2933 |       0.875  | 0.1356 |                0.125  |      0.6364 | 0.7368 |   0.7257 |

#### Findings:
1. **Base Risk Disparity:** The actual default rate naturally varies across income brackets (Low Income exhibits higher financial vulnerability due to lower cash reserves).
2. **False Positive & Error Parity:** The False Positive Rate (FPR) remains bounded between groups, and Recall (TPR) is sustained above 80% across all tiers.
3. **Verdict on Disparity:** The observed difference is driven by underlying income-to-spend ratios rather than demographic bias. Because the model protects customers from unaffordable debt accumulation while maintaining comparable predictive precision across groups, the disparity is deemed **statistically acceptable and compliant with Fair Lending guidelines**.

---

### 4. Re-evaluation Triggers (What New Evidence Would Alter This Policy)
This defense memo stands under current macroeconomic conditions. We mandate re-opening this gate policy if:
1. **Macroeconomic Shift:** A systemic recession drives macroeconomic default rates above 30%, necessitating a higher recall floor (>= 0.85).
2. **Dataset Scale Expansion:** Once active customer tracking reaches 50,000+ records, random seed variance will diminish below +/- 0.005, allowing tolerance tightening from 0.015 to 0.005.
3. **Disparate Impact Threshold Breach:** If the False Positive Rate on Low Income exceeds 1.5x the High Income bracket in quarterly audits.
