# Task 5: Metric Justification — Why Plain Accuracy Fails in Credit Risk

**Authors:** Group 13 — Nihar Sahu (B010) & Shaun Shaju (B012)  
**Project:** Credit Risk CI/CD Quality Gate  

---

## 1. The Accuracy Paradox in Imbalanced Credit Risk

In the provided credit card datasets, approximately **18.9% of customers are High Risk (`High_Risk_Customer = 1`)**, while **81.1% are Non-Defaulters (`0`)**.

Consider a trivial "dummy" baseline that indiscriminately predicts that every customer is safe (predicting class `0` 100% of the time):
* **Accuracy:** $\mathbf{81.1\%}$
* **Default Recall (Sensitivity):** $\mathbf{0.0\%}$ (Every single defaulting customer is missed!)
* **Financial Consequence:** Total portfolio catastrophe for the bank. Uncaught defaults lead to massive charge-offs and non-performing asset (NPA) spikes.

```
       Actual Default (1)      Actual Non-Default (0)
     ┌─────────────────────┬───────────────────────────┐
     │ False Negative (FN) │    True Negative (TN)     │
Pred0│ Catastrophic Loss:  │   Trivial Majority Class  │
     │ Loan written off    │   Prediction (81.1% Acc)  │
     ├─────────────────────┼───────────────────────────┤
     │ True Positive (TP)  │    False Positive (FP)    │
Pred1│ Saved default loss  │   Friction (declined card)│
     └─────────────────────┴───────────────────────────┘
```

Because accuracy gives equal weight to both classes, a model that severely regresses in catching defaulters can maintain an deceptively attractive accuracy score of 80%+. For this reason, **plain accuracy is completely disqualified as an operational or quality gate metric**.

---

## 2. Why PR-AUC (Precision-Recall AUC) is the Primary Metric

While **ROC-AUC** is commonly cited, it plots True Positive Rate vs False Positive Rate ($\text{FPR} = \text{FP} / (\text{FP} + \text{TN})$). In imbalanced data, the large number of true negatives ($\text{TN}$) artificially depresses the FPR denominator, making ROC-AUC look overly optimistic (often $>0.90$) even when precision on the minority default class is poor.

In contrast, **PR-AUC (Average Precision)** focuses strictly on the minority positive class:
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

* **Key Advantage:** PR-AUC does not reward a model for identifying easy non-defaulters. It evaluates precision across all operational decision thresholds.
* **Cost Asymmetry Alignment:** In retail lending, the cost of a False Negative ($C_{\text{FN}} \approx \$5,000$ in bad debt write-off) is 10× to 20× greater than a False Positive ($C_{\text{FP}} \approx \$150$ in lost customer interchange fee). PR-AUC ensures high discrimination precisely where positive credit risk is concentrated.

---

## 3. Metric Decision Hierarchy for the CI/CD Quality Gate

1. **Primary Gate Metric:** **PR-AUC**  
   * *Threshold Rule:* Must not degrade by more than $0.015$ from baseline, and must exceed an absolute floor of $0.52$.
2. **Safety Constraint:** **Recall (TPR)**  
   * *Threshold Rule:* Must maintain $\ge 0.70$ on the test split. A candidate model with high PR-AUC that achieves it by flagging almost no one is immediately blocked.
3. **Secondary Validation Metric:** **ROC-AUC**  
   * *Threshold Rule:* Broad discriminative capability must remain within $0.015$ of baseline.
4. **Diagnostic Metrics:** **F1-Score, Brier Score Loss, Accuracy**  
   * Logged for tracking, probability calibration checks, and stakeholder reporting, but non-blocking.
