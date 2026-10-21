"""Professional 4-Page Submittable PDF Report Generator.
Uses ReportLab to generate 'reports/FINAL_PROJECT_REPORT.pdf'
conforming exactly to the 4-page constraint from Dr. Santosh Kumar's brief.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

def create_pdf(output_path="reports/FINAL_PROJECT_REPORT.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 0.5 inch margins to comfortably fit within 4 dense, professional pages
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#1A365D"),
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#2B6CB0"),
        alignment=TA_CENTER,
        fontName="Helvetica",
        spaceAfter=8
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#2C5282"),
        fontName="Helvetica-Bold",
        spaceBefore=6,
        spaceAfter=4
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#2D3748"),
        fontName="Helvetica-Bold",
        spaceBefore=4,
        spaceAfter=3
    )

    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#2D3748"),
        alignment=TA_JUSTIFY,
        spaceAfter=4
    )

    body_bold = ParagraphStyle(
        'BodyBoldCustom',
        parent=body_style,
        fontName="Helvetica-Bold"
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#1A202C")
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        fontName="Helvetica-Bold"
    )

    story = []

    # =========================================================================
    # PAGE 1: Title, Executive Summary, Lifecycle Mapping, Repo Architecture
    # =========================================================================
    story.append(Paragraph("PROJECT 1: CREDIT RISK MODEL WITH A CI/CD QUALITY GATE", title_style))
    story.append(Paragraph(
        "<b>Group 13:</b> Nihar Sahu (B010) &amp; Shaun Shaju (B012) &nbsp;|&nbsp; "
        "<b>Course:</b> Elective II - MLOps &nbsp;|&nbsp; "
        "<b>Instructor:</b> Dr. Santosh Kumar",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E0"), spaceAfter=6))

    story.append(Paragraph("1. Executive Summary &amp; Problem Context", h1_style))
    story.append(Paragraph(
        "In retail banking, credit-card default prediction models are periodically retrained. Untested changes can introduce "
        "silent model degradation, remaining undetected for weeks and causing non-performing loan spikes without auditability. "
        "This project engineers an end-to-end, production-grade MLOps architecture featuring a <b>CI/CD Quality Gate</b>. "
        "Every modification to code, YAML configuration, engineered features, or data distributions is automatically tested, "
        "experiment-tracked in MLflow, and accepted only if candidate models meet statistical quality thresholds without regressing "
        "subgroup fairness across customer demographic brackets.",
        body_style
    ))

    story.append(Paragraph("2. DevOps vs MLOps Lifecycle Mapping (Task 1)", h1_style))
    story.append(Paragraph(
        "ML delivery fundamentally departs from standard software engineering across three operational dimensions: "
        "<b>(1) Dual-Dependency Drift:</b> Model performance depends codependently on code logic and changing borrower data distributions. "
        "<b>(2) Statistical Quality Verification:</b> Syntax-clean code with 100% test coverage can produce mathematically defective models. "
        "<b>(3) Continuous Training (CT):</b> Continuous learning loops retrain and version models against newly arriving data batches.",
        body_style
    ))

    lifecycle_data = [
        [Paragraph("DevOps Stage", table_header), Paragraph("MLOps Equivalent Stage", table_header), Paragraph("Repository Artefact / Step", table_header), Paragraph("Operational Function", table_header)],
        [Paragraph("Plan", table_cell), Paragraph("Problem &amp; Metric Definition", table_cell), Paragraph("config/dev.yaml, baseline_metrics.json", table_cell), Paragraph("Defines PR-AUC &gt;= 0.79 baseline &amp; credit cost asymmetry.", table_cell)],
        [Paragraph("Code", table_cell), Paragraph("Feature &amp; Model Engineering", table_cell), Paragraph("src/feature_engineering.py, train.py", table_cell), Paragraph("Leakage-free transformer pipeline &amp; model algorithms.", table_cell)],
        [Paragraph("Build", table_cell), Paragraph("Schema &amp; Data Verification", table_cell), Paragraph("src/config_schema.py, data_manifest.json", table_cell), Paragraph("Pydantic strict schema validation &amp; SHA-256 data hashing.", table_cell)],
        [Paragraph("Test", table_cell), Paragraph("Automated CI Quality Gate", table_cell), Paragraph("tests/, src/gate.py, .github/workflows/ci.yml", table_cell), Paragraph("Executes pytest unit suite &amp; evaluates candidate against baseline.", table_cell)],
        [Paragraph("Release", table_cell), Paragraph("Model Registry Tagging", table_cell), Paragraph("MLflow Registry (CreditRiskGateModel)", table_cell), Paragraph("Immutable model lineage tied to git commit and data version.", table_cell)],
        [Paragraph("Operate", table_cell), Paragraph("Data Retraining Drill", table_cell), Paragraph("scripts/run_v1_v2_drill.py", table_cell), Paragraph("Retraining on v2 data across 3 seeds with t-test significance.", table_cell)],
        [Paragraph("Monitor", table_cell), Paragraph("Fairness &amp; Error Auditing", table_cell), Paragraph("scripts/run_fairness_defense.py", table_cell), Paragraph("Audits error across Income_Bracket to prevent disparate impact.", table_cell)]
    ]
    t_lifecycle = Table(lifecycle_data, colWidths=[65, 115, 160, 200])
    t_lifecycle.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_lifecycle)
    story.append(Spacer(1, 6))

    story.append(Paragraph("3. Versioning &amp; Configuration Architecture (Tasks 2 &amp; 3)", h1_style))
    story.append(Paragraph(
        "<b>Data Versioning:</b> Raw CSVs are decoupled from git history via <code>data_manifest.json</code> storing cryptographic "
        "SHA-256 hashes (v1: <code>91c12c5c...</code>, v2: <code>e9514b4f...</code>). Preprocessing pipelines guarantee deterministic replication.<br/>"
        "<b>Configuration Management:</b> All file paths, random seeds, hyperparameter spaces, and quality gate cut-offs are managed in "
        "modular YAML configs (<code>config/dev.yaml</code>, <code>config/ci.yaml</code>, <code>config/prod.yaml</code>). Pydantic schemas enforce "
        "strict type-checking and fail immediately on illegal or corrupted configurations.",
        body_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: Feature Engineering, Imbalanced Metric Justification, Experiments
    # =========================================================================
    story.append(Paragraph("4. Feature Engineering Architecture (Task 4)", h1_style))
    story.append(Paragraph(
        "Four specialized domain features were engineered. Preprocessing is encapsulated in a custom scikit-learn transformer "
        "(<code>CreditRiskFeatureEngineer</code>). Imputers and scalers fit exclusively on training folds to prevent target leakage, "
        "and <code>Customer_ID</code> is strictly dropped from predictive matrices.",
        body_style
    ))

    feat_data = [
        [Paragraph("Feature Name", table_header), Paragraph("Category", table_header), Paragraph("Mathematical Formula", table_header), Paragraph("Edge-Case Handling &amp; Business Domain Justification", table_header)],
        [Paragraph("credit_utilisation_ratio", table_cell), Paragraph("Ratio", table_cell), Paragraph("Current_Balance / max(Credit_Limit, 1.0)", table_cell), Paragraph("If Limit=0, set to 2.5 (max risk) if Bal&gt;0 else 0. Clipped [0.0, 2.5]. Basel II liquidity strain indicator.", table_cell)],
        [Paragraph("delinquency_high_risk_flag", table_cell), Paragraph("Binary Flag", table_cell), Paragraph("I(Delay &gt; 30 or Util &gt; 0.80)", table_cell), Paragraph("Captures 30+ Days Past Due (DPD) commercial delinquency threshold and acute over-utilisation.", table_cell)],
        [Paragraph("tenure_maturity_stage", table_cell), Paragraph("Binned", table_cell), Paragraph("new (&lt;=12m), est (13-36m), mature (&gt;36m)", table_cell), Paragraph("Accounts in first year exhibit elevated seasoning default volatility compared to established accounts.", table_cell)],
        [Paragraph("financial_stress_index", table_cell), Paragraph("Domain Custom", table_cell), Paragraph("(Monthly_Spend / Limit) * (1 + Delay/30)", table_cell), Paragraph("Spend burn-rate multiplied by delinquent payment drag. Identifies severe cash-flow distress.", table_cell)]
    ]
    t_feat = Table(feat_data, colWidths=[110, 60, 150, 220])
    t_feat.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_feat)
    story.append(Spacer(1, 6))

    story.append(Paragraph("5. Metric Justification: The Accuracy Paradox (Task 5)", h1_style))
    story.append(Paragraph(
        "The dataset exhibits severe class imbalance: <b>18.9% High Risk defaults vs 81.1% Non-defaults</b>. "
        "A trivial dummy classifier predicting all zeros achieves <b>81.1% accuracy but 0.0% default recall</b>, missing all credit write-offs. "
        "ROC-AUC is artificially inflated by the high volume of true negatives. "
        "Consequently, <b>PR-AUC (Precision-Recall AUC)</b> is established as the primary gate metric because it penalizes false positives and "
        "rewards accurate identification of the minority default class, paired with a mandatory <b>Recall floor (&gt;= 0.70)</b>.",
        body_style
    ))

    story.append(Paragraph("6. Experiment Tracking &amp; Model Selection (Task 5)", h1_style))
    story.append(Paragraph(
        "12 tracked experiments were logged into the MLflow SQLite backend across 4 model families. "
        "The tuned XGBoost ensemble achieved peak PR-AUC (0.8091) and ROC-AUC (0.9414), outperforming baseline Logistic Regression.",
        body_style
    ))

    exp_data = [
        [Paragraph("Run Name", table_header), Paragraph("Family", table_header), Paragraph("PR-AUC", table_header), Paragraph("ROC-AUC", table_header), Paragraph("Recall", table_header), Paragraph("Precision", table_header), Paragraph("F1", table_header), Paragraph("Acc", table_header), Paragraph("Brier", table_header)],
        [Paragraph("logreg_baseline_c1.0", table_cell), Paragraph("LogReg", table_cell), Paragraph("0.7978", table_cell), Paragraph("0.9215", table_cell), Paragraph("0.8947", table_cell), Paragraph("0.5397", table_cell), Paragraph("0.6733", table_cell), Paragraph("0.835", table_cell), Paragraph("0.1189", table_cell)],
        [Paragraph("logreg_l2_c0.1", table_cell), Paragraph("LogReg", table_cell), Paragraph("0.7925", table_cell), Paragraph("0.9185", table_cell), Paragraph("0.8947", table_cell), Paragraph("0.5397", table_cell), Paragraph("0.6733", table_cell), Paragraph("0.835", table_cell), Paragraph("0.1215", table_cell)],
        [Paragraph("rf_trees100_depth5", table_cell), Paragraph("RF", table_cell), Paragraph("0.7498", table_cell), Paragraph("0.9193", table_cell), Paragraph("0.8158", table_cell), Paragraph("0.5849", table_cell), Paragraph("0.6813", table_cell), Paragraph("0.855", table_cell), Paragraph("0.0974", table_cell)],
        [Paragraph("xgb_default_depth3_lr0.1", table_cell), Paragraph("XGBoost", table_cell), Paragraph("0.8091", table_cell), Paragraph("0.9414", table_cell), Paragraph("0.7895", table_cell), Paragraph("0.6977", table_cell), Paragraph("0.7407", table_cell), Paragraph("0.895", table_cell), Paragraph("0.0777", table_cell)],
        [Paragraph("xgb_tuned_depth4_posweight", table_cell), Paragraph("XGBoost", table_cell), Paragraph("0.7945", table_cell), Paragraph("0.9397", table_cell), Paragraph("0.8684", table_cell), Paragraph("0.6111", table_cell), Paragraph("0.7174", table_cell), Paragraph("0.870", table_cell), Paragraph("0.0992", table_cell)],
        [Paragraph("lgbm_unbalanced_leaves31", table_cell), Paragraph("LightGBM", table_cell), Paragraph("0.7949", table_cell), Paragraph("0.9404", table_cell), Paragraph("0.7895", table_cell), Paragraph("0.6122", table_cell), Paragraph("0.6897", table_cell), Paragraph("0.865", table_cell), Paragraph("0.0955", table_cell)]
    ]
    t_exp = Table(exp_data, colWidths=[130, 60, 45, 45, 45, 45, 40, 40, 50])
    t_exp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_exp)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: CI/CD Quality Gate (Task 6) & Data Retraining Drill (Task 7)
    # =========================================================================
    story.append(Paragraph("7. Automated CI/CD Quality Gate &amp; Pull Request Drill (Task 6)", h1_style))
    story.append(Paragraph(
        "A GitHub Actions CI workflow (<code>.github/workflows/ci.yml</code>) triggers on pull requests to <code>main</code>. "
        "The workflow runs pytest unit tests, executes model training with <code>config/ci.yaml</code>, and compares candidate metrics "
        "against <code>baseline_metrics.json</code>. If statistical regression exceeds allowed tolerances (PR-AUC drop &gt; 0.015 or Recall &lt; 0.70), "
        "the gate exits with code 1, automatically blocking the merge. Three distinct PR scenarios were simulated and validated:",
        body_style
    ))

    pr_data = [
        [Paragraph("PR Scenario", table_header), Paragraph("Trigger / Modification", table_header), Paragraph("Candidate vs Baseline", table_header), Paragraph("Gate Verdict &amp; Merging Action", table_header)],
        [Paragraph("PR #101 (Pass)", table_cell), Paragraph("Tuned XGBoost feature optimization", table_cell), Paragraph("PR-AUC: 0.8158 (+0.0180)<br/>Recall: 0.8950 (+0.0003)", table_cell), Paragraph("🟢 <b>PASSED</b>: All gate criteria satisfied. Merge approved.", table_cell)],
        [Paragraph("PR #102 (Regression)", table_cell), Paragraph("Untested feature deletion / bug", table_cell), Paragraph("PR-AUC: 0.4120 (-0.3858)<br/>Recall: 0.4500 (Below 0.70)", table_cell), Paragraph("🔴 <b>FAILED</b>: Severe regression detected. Merge blocked.", table_cell)],
        [Paragraph("PR #103 (Broken Test)", table_cell), Paragraph("Breaking schema code syntax error", table_cell), Paragraph("Pytest fails on edge-case zero credit limit unit test", table_cell), Paragraph("🔴 <b>FAILED</b>: Build failure in Step 1. Fast fail before training.", table_cell)]
    ]
    t_pr = Table(pr_data, colWidths=[90, 130, 140, 180])
    t_pr.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_pr)
    story.append(Spacer(1, 6))

    story.append(Paragraph("8. Data-Change Drill: Version 1 vs Version 2 Retraining (Task 7)", h1_style))
    story.append(Paragraph(
        "When Version 2 data (1,100 records: v1 + 300 new customers) was released, the champion model was retrained through the identical "
        "pipeline across <b>3 independent random seeds (42, 101, 777)</b> to determine whether observed gains represent genuine improvement "
        "or stochastic noise. Welch's two-sample independent t-test was computed:",
        body_style
    ))

    drill_data = [
        [Paragraph("Metric", table_header), Paragraph("v1 Mean", table_header), Paragraph("v1 Std", table_header), Paragraph("v2 Mean", table_header), Paragraph("v2 Std", table_header), Paragraph("Delta Mean", table_header), Paragraph("P-Value", table_header), Paragraph("Significant?", table_header)],
        [Paragraph("PR-AUC", table_cell), Paragraph("0.8385", table_cell), Paragraph("0.0523", table_cell), Paragraph("0.8455", table_cell), Paragraph("0.0052", table_cell), Paragraph("+0.0070", table_cell), Paragraph("0.8378", table_cell), Paragraph("Variance Reduction", table_cell)],
        [Paragraph("ROC-AUC", table_cell), Paragraph("0.9484", table_cell), Paragraph("0.0141", table_cell), Paragraph("0.9467", table_cell), Paragraph("0.0082", table_cell), Paragraph("-0.0017", table_cell), Paragraph("0.8679", table_cell), Paragraph("Parity Maintained", table_cell)],
        [Paragraph("Recall", table_cell), Paragraph("0.7982", table_cell), Paragraph("0.0662", table_cell), Paragraph("0.8013", table_cell), Paragraph("0.0588", table_cell), Paragraph("+0.0031", table_cell), Paragraph("0.9551", table_cell), Paragraph("Parity Maintained", table_cell)],
        [Paragraph("F1-Score", table_cell), Paragraph("0.7337", table_cell), Paragraph("0.0305", table_cell), Paragraph("0.7177", table_cell), Paragraph("0.0235", table_cell), Paragraph("-0.0160", table_cell), Paragraph("0.5151", table_cell), Paragraph("Stable", table_cell)]
    ]
    t_drill = Table(drill_data, colWidths=[65, 65, 65, 65, 65, 65, 65, 85])
    t_drill.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_drill)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Finding &amp; Recommendation:</b> Retraining on v2 data stabilizes the model: the standard deviation of PR-AUC shrank 10× "
        "(from ±0.0523 down to ±0.0052). The additional 300 borrowers improve calibration in high-spend brackets. "
        "<b>Recommendation: ACCEPT V2 MODEL FOR PROMOTION</b> and update the protected baseline.",
        body_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: Quality Gate Threshold & Fairness Defense Memo (Task 8)
    # =========================================================================
    story.append(Paragraph("9. Quality-Gate Threshold &amp; Fairness Defense Memo (Task 8)", h1_style))
    story.append(Paragraph(
        "<b>To:</b> Retail Credit Risk Governance Committee &nbsp;|&nbsp; "
        "<b>From:</b> Group 13 (Nihar Sahu B010 &amp; Shaun Shaju B012) &nbsp;|&nbsp; "
        "<b>Subject:</b> Quality Gate Rule Benchmarking and Demographic Parity Audit",
        subtitle_style
    ))

    story.append(Paragraph("A. Comparison of Candidate Gate Rules", h2_style))
    story.append(Paragraph(
        "Three competing gate rules were benchmarked across the 12 tracked experimental runs:",
        body_style
    ))

    rules_data = [
        [Paragraph("Gate Rule ID", table_header), Paragraph("Mathematical Rule", table_header), Paragraph("Accepted", table_header), Paragraph("Rejected", table_header), Paragraph("Operational Trade-off &amp; Risk Justification", table_header)],
        [Paragraph("Rule A (Permissive)", table_cell), Paragraph("Delta ROC-AUC &gt;= -0.02", table_cell), Paragraph("11 / 12", table_cell), Paragraph("1 / 12", table_cell), Paragraph("<b>Unsafe:</b> ROC-AUC is diluted by majority non-defaulters; accepted models missing 40% of actual defaulters.", table_cell)],
        [Paragraph("Rule B (Over-Strict)", table_cell), Paragraph("PR-AUC &gt;= Base &amp; &gt;= 0.80", table_cell), Paragraph("1 / 12", table_cell), Paragraph("11 / 12", table_cell), Paragraph("<b>Too Brittle:</b> Rejects robust architectures due to minor seed variance; halts code updates.", table_cell)],
        [Paragraph("Rule C (Composite)", table_cell), Paragraph("Delta PR-AUC &gt;= -0.015 &amp; Recall &gt;= 0.70", table_cell), Paragraph("7 / 12", table_cell), Paragraph("5 / 12", table_cell), Paragraph("<b>Recommended:</b> Enforces non-negotiable credit safety floor while allowing statistical run tolerance.", table_cell)]
    ]
    t_rules = Table(rules_data, colWidths=[90, 120, 50, 50, 230])
    t_rules.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_rules)
    story.append(Spacer(1, 5))

    story.append(Paragraph("B. Subgroup Demographic Breakdown across Income_Bracket", h2_style))
    story.append(Paragraph(
        "To comply with Fair Lending guidelines (ECOA / FCRA), error distributions were audited across socioeconomic tiers:",
        body_style
    ))

    subgroup_data = [
        [Paragraph("Income Tier", table_header), Paragraph("Count", table_header), Paragraph("Actual Default Rate", table_header), Paragraph("Predicted Default Rate", table_header), Paragraph("Recall (TPR)", table_header), Paragraph("FPR", table_header), Paragraph("Missed Defaults (FNR)", table_header), Paragraph("Precision", table_header), Paragraph("PR-AUC", table_header)],
        [Paragraph("High", table_cell), Paragraph("42", table_cell), Paragraph("11.90%", table_cell), Paragraph("16.67%", table_cell), Paragraph("80.00%", table_cell), Paragraph("8.11%", table_cell), Paragraph("20.00%", table_cell), Paragraph("57.14%", table_cell), Paragraph("0.8625", table_cell)],
        [Paragraph("Low", table_cell), Paragraph("83", table_cell), Paragraph("20.48%", table_cell), Paragraph("26.51%", table_cell), Paragraph("88.24%", table_cell), Paragraph("10.61%", table_cell), Paragraph("11.76%", table_cell), Paragraph("68.18%", table_cell), Paragraph("0.8589", table_cell)],
        [Paragraph("Medium", table_cell), Paragraph("75", table_cell), Paragraph("21.33%", table_cell), Paragraph("29.33%", table_cell), Paragraph("87.50%", table_cell), Paragraph("13.56%", table_cell), Paragraph("12.50%", table_cell), Paragraph("63.64%", table_cell), Paragraph("0.7257", table_cell)]
    ]
    t_subgroup = Table(subgroup_data, colWidths=[55, 35, 75, 75, 60, 50, 70, 60, 60])
    t_subgroup.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_subgroup)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Fairness Verdict:</b> Higher default rates in the Low Income tier reflect economic vulnerability (higher spend-to-limit ratio), "
        "not algorithmic bias. Crucially, <b>Recall is highest (88.24%)</b> for Low Income borrowers, ensuring high protection against unserviceable debt. "
        "False positive rates remain tightly bounded between 8.1% and 13.6%. The disparity is deemed <b>statistically acceptable and fair</b>.",
        body_style
    ))

    story.append(Paragraph("C. Policy Re-evaluation Triggers &amp; Submission Sign-off", h2_style))
    story.append(Paragraph(
        "This policy shall be reopened if: <b>(1)</b> Macroeconomic recession lifts portfolio defaults above 30%; "
        "<b>(2)</b> Dataset expansion past 50,000 borrowers reduces seed variance below ±0.005 (allowing gate tightening to 0.005); or "
        "<b>(3)</b> Quarterly audits show Low Income FPR exceeding 1.5× High Income FPR.<br/>"
        "<b>Submitted by:</b> Group 13 — Nihar Sahu (B010) &amp; Shaun Shaju (B012) &nbsp;|&nbsp; <b>Status:</b> Ready for Submission",
        body_style
    ))

    doc.build(story)
    print(f"PDF successfully built at: {output_path}")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "reports/FINAL_PROJECT_REPORT.pdf"
    create_pdf(out)
