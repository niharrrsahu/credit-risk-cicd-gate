# Task 1: Lifecycle Mapping — DevOps vs MLOps

**Authors:** Group 13 — Nihar Sahu (B010) & Shaun Shaju (B012)  
**Project:** Credit Risk CI/CD Quality Gate  
**Course:** Elective II: Machine Learning Operations (MLOps)  
**Instructor:** Dr. Santosh Kumar  

---

## 1. Lifecycle Mapping Table

The table below maps each classic DevOps lifecycle phase to the corresponding MLOps phase and links each to concrete artefacts and automated workflow steps in this repository:

| Classic DevOps Phase | MLOps Equivalent Phase | Concrete Repository Artefact / Workflow Step | Purpose & Operational Function |
| :--- | :--- | :--- | :--- |
| **Plan** | Business Problem Formulation & Metric Definition | [`config/dev.yaml`](file:///config/dev.yaml), [`baseline_metrics.json`](file:///baseline_metrics.json) | Defining the imbalanced classification objective, cost asymmetry of credit defaults, PR-AUC baseline, and gate thresholds. |
| **Code** | Modular Feature & Model Engineering | [`src/feature_engineering.py`](file:///src/feature_engineering.py), [`src/train.py`](file:///src/train.py) | Writing deterministic data cleaning, leakage-free feature transformers, and model pipelines. |
| **Build** | Pipeline Construction & Schema Validation | [`src/config_schema.py`](file:///src/config_schema.py), [`src/data_loader.py`](file:///src/data_loader.py) | Validating YAML configurations with Pydantic; verifying raw data integrity using SHA-256 hashes against [`data_manifest.json`](file:///data/data_manifest.json). |
| **Test** | Automated Pre-merge Testing & Quality Gate | [`tests/`](file:///tests/), [`src/gate.py`](file:///src/gate.py), [`.github/workflows/ci.yml`](file:///.github/workflows/ci.yml) | Executing pytest unit tests across edge cases (e.g., zero credit limit), training CI candidate models, and gating merges against baseline. |
| **Release** | Model Registry & Version Tagging | `mlflow.db` (Model Registry: `CreditRiskGateModel`) | Formal versioning and tagging of champion models with Git commit SHA and data version lineage. |
| **Deploy** | Containerized / Local Serving Preparation | [`config/prod.yaml`](file:///config/prod.yaml), Serialized Model Artefacts | Packaging fitted pipelines and deployment configurations for downstream credit decisioning. |
| **Operate** | Quality Gate Enforcement & Retraining Trigger | [`scripts/run_v1_v2_drill.py`](file:///scripts/run_v1_v2_drill.py) | Retraining drills when new batches (v2 data) arrive; multi-seed statistical significance verification. |
| **Monitor** | Fairness & Degradation Auditing | [`scripts/run_fairness_defense.py`](file:///scripts/run_fairness_defense.py), Subgroup Reports | Auditing model error across `Income_Bracket` to protect against demographic disparate impact. |

---

## 2. Key Differences: Machine Learning Delivery vs Traditional Software Delivery

Traditional software engineering focuses on code correctness, whereas MLOps manages a triad of **Code, Data, and Model**. We identify three critical operational differences demonstrated in this project:

```
Traditional DevOps:
Code  ──[ Compile / Unit Test ]──>  Deterministic Binary / Build

MLOps:
Code + Data (Evolving) + Hyperparams  ──[ Train / Validate ]──>  Non-Deterministic Probabilistic Model
                                                                           │
                                                                           ▼
                                                               Quality Gate Validation
                                                               (PR-AUC, ROC-AUC, Recall)
```

### 1. Dual-Dependency Drift and Non-Deterministic Builds
* **Software:** Code alone dictates build behaviour. Given the same compiler and dependencies, identical code consistently yields the exact same executable binary.
* **ML:** Model behaviour is codependent on data distributions. Even if code is completely frozen, an incoming shift in macroeconomic customer balance (covariate shift) can severely degrade model predictions. Furthermore, stochastic optimization requires multi-seed evaluations to distinguish genuine capability from random noise.

### 2. Testing Logic: Code Correctness vs Statistical Quality
* **Software:** Unit tests assert binary deterministic outcomes (e.g., $f(x) = y$). If inputs match expectations and exceptions are caught, the test passes.
* **ML:** Code may execute with zero runtime exceptions, zero syntax errors, and 100% test coverage, yet produce an utterly defective model (e.g. predicting all zeros yielding 81% accuracy but 0% recall of defaults). Hence, MLOps requires a **Statistical Quality Gate** to block regression in statistical performance (PR-AUC / Recall).

### 3. Continuous Training (CT) and Model Versioning vs Binary Release
* **Software:** Once a software binary is released to production, it functions indefinitely until a new feature is authored or a bug is discovered.
* **ML:** Models naturally decay due to concept drift. MLOps introduces Continuous Training (CT) workflows where newly acquired data (e.g., moving from v1 800 customers to v2 1,100 customers) triggers automated retraining, baseline evaluation, and governance approval.

---

## 3. Stages Intentionally Excluded & Domain Justification

1. **Live Production Infrastructure Deployment (Kubernetes / ECS Endpoint):**
   * *Reason:* The project scope deliberately targets the **Pre-Merge CI/CD Quality Gate**. In retail banking regulations (e.g. Basel II, Federal Reserve SR 11-7), credit risk models cannot be automatically pushed to live customer underwriting without offline Model Risk Management (MRM) committee review. The CI gate acts as the automated technical barrier before MRM submission.
2. **Real-Time Online Latency Monitoring (Prometheus/Grafana):**
   * *Reason:* Credit card delinquency evaluation in this retail banking scenario is run on monthly credit reporting cycles, not sub-second fraud scoring endpoints. Real-time sub-millisecond serving metrics are replaced with batch calibration and demographic fairness monitoring.
