# Task 4: Feature Documentation Table & Engineering Architecture

**Authors:** Group 13 — Nihar Sahu (B010) & Shaun Shaju (B012)  
**Project:** Credit Risk CI/CD Quality Gate  

---

## 1. Engineered Features Table

The credit risk pipeline engineers four specialized features covering the required modalities (Ratio, Binary Flag, Binned/Categorical, Domain Custom):

| Feature Name | Category | Mathematical / Logical Formula | Edge Case Handling | Domain Reasoning & Business Justification |
| :--- | :--- | :--- | :--- | :--- |
| `credit_utilisation_ratio` | **Ratio** | $\\frac{\\text{Current\\_Balance}}{\\max(\\text{Credit\\_Limit}, 1.0)}$ | If $\\text{Credit\\_Limit} = 0$: set to $2.5$ if $\\text{Balance} > 0$, else $0.0$. Clipped between $[0.0, 2.5]$. | Credit utilisation is the foremost metric in retail credit risk scoring. Borrowers exhausting $>80\\%$ of limit suffer liquidity strain and exhibit dramatically higher default probabilities. |
| `delinquency_high_risk_flag` | **Binary Flag** | $\\mathbb{I}(\\text{Delay} > 30 \\lor \\text{Util} > 0.80)$ | Handled via boolean integer casting `(0 or 1)`. NaN delays imputed via median before flag evaluation. | Flags immediate credit impairment. Late payment exceeding 30 days is the standard delinquency threshold across commercial banking frameworks (e.g. 30 DPD). |
| `tenure_maturity_stage` | **Binned / Categorical** | `new` if $\\text{Tenure} \\le 12$, `established` if $12 < \\text{Tenure} \\le 36$, `mature` if $\\text{Tenure} > 36$. | Open-ended boundary $[-\\infty, 12, 36, \\infty]$; zero or negative tenure mapped to `new`. | Captures credit "seasoning". Accounts in their first year exhibit heightened default volatility compared to long-standing cardholders with established credit history. |
| `financial_stress_index` | **Domain Custom** | $\\left(\\frac{\\text{Monthly\\_Spend}}{\\max(\\text{Credit\\_Limit}, 1.0)}\\right) \\times \\left(1 + \\frac{\\max(\\text{Delay}, 0)}{30}\\right)$ | Division by zero protected; spend ratio clipped to $2.0$; delay capped non-negative. | Domain composite feature: measures spend velocity relative to credit capacity compounded by delinquent payment drag. A borrower spending heavily while missing payment dates is under severe cash flow stress. |

---

## 2. Leakage Prevention & Transformation Consistency
1. **Target Leakage Safeguard:**
   - The target `High_Risk_Customer` is strictly isolated prior to any preprocessing.
   - Preprocessors (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) are fitted **exclusively on the training split** and subsequently applied to test and incoming inference data.
2. **Identifier Exclusion:**
   - `Customer_ID` is removed from the predictive matrix during initial ingestion (`prepare_features_and_target`), preventing spurious overfitting on customer hash identifiers.
3. **Serving Consistency:**
   - Feature engineering logic is packaged inside a custom scikit-learn `BaseEstimator` / `TransformerMixin` class (`CreditRiskFeatureEngineer`), ensuring that identical transformations are applied at train time, CI test time, and batch prediction time.
