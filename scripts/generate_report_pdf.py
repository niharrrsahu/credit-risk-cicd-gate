"""Academic LaTeX-Style 4-Page Submittable PDF Report Generator.
Formatted to match the top-tier peer submission style:
- Compact, elegant academic typography (Times-Roman / Helvetica-Bold)
- High-resolution embedded figures (Figures 1-4)
- Formal mathematical formulas for PR-AUC, Welch's t-test, and Composite Gate Rules
- Explicit sections matching Tasks 1 to 8 of Dr. Santosh Kumar's brief
- Final Section: Limitations and Academic Integrity Disclosure
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
    Image,
    HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

def build_academic_pdf(output_path="reports/FINAL_PROJECT_REPORT.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 0.45 in (32 pt) margins for exact 4-page density
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=32,
        rightMargin=32,
        topMargin=32,
        bottomMargin=32
    )

    styles = getSampleStyleSheet()
    
    # Academic Paper Styles
    title_style = ParagraphStyle(
        'PaperTitle',
        parent=styles['Heading1'],
        fontSize=13.5,
        leading=16,
        textColor=colors.HexColor("#1A202C"),
        alignment=TA_CENTER,
        fontName="Times-Bold",
        spaceAfter=3
    )
    
    meta_style = ParagraphStyle(
        'PaperMeta',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#4A5568"),
        alignment=TA_CENTER,
        fontName="Times-Roman",
        spaceAfter=6
    )

    h1_style = ParagraphStyle(
        'PaperH1',
        parent=styles['Heading2'],
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1A365D"),
        fontName="Times-Bold",
        spaceBefore=5,
        spaceAfter=2
    )

    body_style = ParagraphStyle(
        'PaperBody',
        parent=styles['Normal'],
        fontSize=7.8,
        leading=10.2,
        textColor=colors.HexColor("#2D3748"),
        alignment=TA_JUSTIFY,
        fontName="Times-Roman",
        spaceAfter=3
    )

    caption_style = ParagraphStyle(
        'FigCaption',
        parent=styles['Normal'],
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#4A5568"),
        alignment=TA_CENTER,
        fontName="Times-Italic",
        spaceBefore=2,
        spaceAfter=4
    )

    table_cell = ParagraphStyle(
        'TCell',
        parent=styles['Normal'],
        fontSize=6.8,
        leading=8.5,
        textColor=colors.HexColor("#1A202C"),
        fontName="Times-Roman"
    )

    table_header = ParagraphStyle(
        'THeader',
        parent=styles['Normal'],
        fontSize=7.2,
        leading=9,
        textColor=colors.white,
        fontName="Times-Bold"
    )

    story = []

    # =========================================================================
    # PAGE 1: Header, Summary, Task 1 (Lifecycle), Task 2 & 3 (Versioning/Config)
    # =========================================================================
    story.append(Paragraph("Credit Risk Model with a CI/CD Quality Gate", title_style))
    story.append(Paragraph(
        "Elective II: MLOps (Project 1) &mdash; Instructor: Dr. Santosh Kumar &mdash; NMIMS NSOMASA<br/>"
        "<b>Group 13:</b> Nihar Sahu (B010) &amp; Shaun Shaju (B012) &mdash; Submitted: 24 October 2026",
        meta_style
    ))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#CBD5E0"), spaceAfter=5))

    story.append(Paragraph(
        "<b>Summary.</b> We engineered an automated, production-grade MLOps repository featuring a pre-merge <b>CI/CD Quality Gate</b> "
        "for monthly retail credit-card default prediction. The system enforces strict versioning of code, data, and configurations, preventing silent "
        "model regressions. Key findings: (i) In an imbalanced portfolio (&pi;<sub>0</sub> = 0.189 default rate), plain accuracy is disqualified: "
        "a trivial dummy model yields 81.1% accuracy but 0.0% default recall; (ii) We establish <b>PR-AUC</b> as the primary gate metric with a "
        "mandatory <b>Recall floor (&ge; 0.70)</b>; (iii) Out of 12 tracked MLflow runs across 4 model families, tuned XGBoost achieved peak PR-AUC "
        "(0.8091) and was registered in the Model Registry; (iv) Retraining on Version 2 data (1,100 records) stabilized run variance by 10&times; "
        "(&sigma; shrunk from &plusmn;0.0523 to &plusmn;0.0052, p = 0.838); (v) A demographic fairness audit across <code>Income_Bracket</code> "
        "confirmed bounded False Positive Rates (8.1%&ndash;13.6%) with highest recall (88.2%) protecting vulnerable Low Income cardholders.",
        body_style
    ))

    story.append(Paragraph("1. DevOps vs MLOps Lifecycle Mapping (Task 1)", h1_style))
    story.append(Paragraph(
        "Machine learning delivery fundamentally differs from traditional software engineering: "
        "<b>(1) Dual-Dependency Drift:</b> Code correctness alone cannot guarantee model safety; non-stationary borrower financial behavior "
        "(covariate shift) requires continuous evaluation. <b>(2) Probabilistic Builds:</b> Syntax-perfect code with 100% test coverage can still emit "
        "catastrophic credit models if imbalanced classes are poorly ranked. <b>(3) Continuous Training (CT):</b> Production models undergo planned "
        "retraining cadences triggered by data expansion or distribution drift.",
        body_style
    ))

    # Lifecycle Table
    t1_data = [
        [Paragraph("DevOps Stage", table_header), Paragraph("MLOps Equivalent", table_header), Paragraph("Concrete Repository Artefact", table_header), Paragraph("Operational Function", table_header)],
        [Paragraph("Plan", table_cell), Paragraph("Problem Formulation", table_cell), Paragraph("config/dev.yaml, baseline_metrics.json", table_cell), Paragraph("Defines 18.9% imbalanced risk objective and PR-AUC baseline.", table_cell)],
        [Paragraph("Code", table_cell), Paragraph("Feature &amp; Pipeline Design", table_cell), Paragraph("src/feature_engineering.py, train.py", table_cell), Paragraph("Leakage-free transformers &amp; scikit-learn Pipeline architecture.", table_cell)],
        [Paragraph("Build", table_cell), Paragraph("Integrity &amp; Schema Check", table_cell), Paragraph("src/config_schema.py, data_manifest.json", table_cell), Paragraph("Pydantic V2 schema validation and SHA-256 raw data hashing.", table_cell)],
        [Paragraph("Test", table_cell), Paragraph("Automated CI Quality Gate", table_cell), Paragraph("tests/, src/gate.py, .github/workflows/ci.yml", table_cell), Paragraph("Runs 17 unit tests and gates candidate PR against baseline.", table_cell)],
        [Paragraph("Release", table_cell), Paragraph("Model Registry Tagging", table_cell), Paragraph("MLflow (CreditRiskGateModel, v1/v2)", table_cell), Paragraph("Stores model artifacts, metrics, and git commit hash (c51e19d).", table_cell)],
        [Paragraph("Operate", table_cell), Paragraph("Retraining Drill", table_cell), Paragraph("scripts/run_v1_v2_drill.py", table_cell), Paragraph("Executes 3-seed v1 vs v2 retraining drill with Welch's t-test.", table_cell)],
        [Paragraph("Monitor", table_cell), Paragraph("Fairness &amp; Error Audit", table_cell), Paragraph("scripts/run_fairness_defense.py", table_cell), Paragraph("Evaluates demographic parity and error across Income_Bracket.", table_cell)]
    ]
    t1 = Table(t1_data, colWidths=[60, 110, 170, 208])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t1)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Excluded Stages:</b> Real-time low-latency serving (sub-millisecond streaming) is excluded because retail credit card default assessment "
        "operates on monthly billing cycles, where batch inference and rigorous fairness validation take precedence.",
        body_style
    ))

    story.append(Paragraph("2. Repository &amp; Configuration Management (Tasks 2 &amp; 3)", h1_style))
    story.append(Paragraph(
        "<b>Data Versioning:</b> Large raw CSVs are strictly decoupled from git history via <code>data_manifest.json</code>, which tracks cryptographic "
        "SHA-256 hashes (v1: <code>91c12c5c...</code>, 800 rows; v2: <code>e9514b4f...</code>, 1,100 rows). "
        "<b>Configuration Management:</b> All operational thresholds (e.g. utilisation cutoff 0.80, delinquency days 30), seeds, and model hyperparameters "
        "are managed in YAML (<code>config/dev.yaml</code>, <code>config/ci.yaml</code>, <code>config/prod.yaml</code>). Pydantic schemas validate "
        "configurations on load; deliberately malformed YAMLs fail immediately with clear diagnostic exceptions (verified in <code>tests/test_config.py</code>).",
        body_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: Task 4 (Features) & Task 5 (Metric Justification + 12 Runs + Fig 1)
    # =========================================================================
    story.append(Paragraph("3. Feature Engineering &amp; Preprocessing Architecture (Task 4)", h1_style))
    story.append(Paragraph(
        "Both datasets contain 24 missing values in each of <code>Credit_Limit</code>, <code>Current_Balance</code>, and <code>Payment_Delay_Days</code>. "
        "To prevent target leakage, imputers and scalers are fitted <b>exclusively on the training split</b>. <code>Customer_ID</code> is purged prior to "
        "transformation. Four specialized domain features are constructed:",
        body_style
    ))

    t4_data = [
        [Paragraph("Feature Name", table_header), Paragraph("Type", table_header), Paragraph("Mathematical Formula", table_header), Paragraph("Domain &amp; Business Rationale", table_header)],
        [Paragraph("credit_utilisation_ratio", table_cell), Paragraph("Ratio", table_cell), Paragraph("Current_Balance / max(Credit_Limit, 1.0)", table_cell), Paragraph("Liquidity strain indicator. If limit is zero, maps to 2.5 (max risk) if balance &gt; 0, else 0. Clipped [0, 2.5].", table_cell)],
        [Paragraph("delinquency_high_risk_flag", table_cell), Paragraph("Binary", table_cell), Paragraph("I(Delay &gt; 30 or Util &gt; 0.80)", table_cell), Paragraph("Captures commercial banking 30 Days Past Due (30 DPD) threshold and acute credit exhaustion.", table_cell)],
        [Paragraph("tenure_maturity_stage", table_cell), Paragraph("Binned", table_cell), Paragraph("new (&le;12m), est (13-36m), mature (&gt;36m)", table_cell), Paragraph("Accounts in first year exhibit elevated seasoning default volatility compared to seasoned holders.", table_cell)],
        [Paragraph("financial_stress_index", table_cell), Paragraph("Domain", table_cell), Paragraph("(Monthly_Spend / Limit) &times; (1 + Delay/30)", table_cell), Paragraph("Spend burn rate relative to approved limit compounded by late payment drag. Measures acute cash distress.", table_cell)]
    ]
    t4 = Table(t4_data, colWidths=[105, 45, 160, 238])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t4)
    story.append(Spacer(1, 4))

    story.append(Paragraph("4. Imbalanced Metric Justification &amp; Experiment Tracking (Task 5)", h1_style))
    story.append(Paragraph(
        "<b>The Accuracy Paradox:</b> In credit scoring with default prevalence &pi;<sub>0</sub> = 0.189, a trivial majority-class predictor achieves "
        "<b>81.1% accuracy but 0.0% Recall</b>. Standard ROC-AUC is misleadingly inflated (&gt;0.92) due to massive true negative dilution. "
        "Hence, <b>PR-AUC (Average Precision, AP = &Sigma; (R<sub>k</sub> &minus; R<sub>k&minus;1</sub>)P<sub>k</sub>)</b> is the primary gate metric, "
        "paired with a non-negotiable <b>Recall floor (&ge; 0.70)</b> to prevent defaulting customer pass-through.",
        body_style
    ))

    # Figure 1: 12 Runs Benchmark
    fig1_path = "reports/plots/fig1_runs_comparison.png"
    if os.path.exists(fig1_path):
        story.append(Image(fig1_path, width=540, height=195))
        story.append(Paragraph("Figure 1: Benchmark of 12 Tracked MLflow Runs Across 4 Model Families (Task 5). Baseline PR-AUC = 0.7978 (red dashed line).", caption_style))

    # Table of Selected Experimental Runs
    t5_data = [
        [Paragraph("Run Identifier", table_header), Paragraph("Model Family", table_header), Paragraph("Key Parameters", table_header), Paragraph("PR-AUC", table_header), Paragraph("ROC-AUC", table_header), Paragraph("Recall", table_header), Paragraph("F1", table_header), Paragraph("Accuracy", table_header)],
        [Paragraph("logreg_baseline_c1.0", table_cell), Paragraph("LogisticRegression", table_cell), Paragraph("C=1.0, balanced, lbfgs", table_cell), Paragraph("0.7978", table_cell), Paragraph("0.9215", table_cell), Paragraph("0.8947", table_cell), Paragraph("0.6733", table_cell), Paragraph("0.8350", table_cell)],
        [Paragraph("logreg_l2_c0.1", table_cell), Paragraph("LogisticRegression", table_cell), Paragraph("C=0.1, balanced, lbfgs", table_cell), Paragraph("0.7925", table_cell), Paragraph("0.9185", table_cell), Paragraph("0.8947", table_cell), Paragraph("0.6733", table_cell), Paragraph("0.8350", table_cell)],
        [Paragraph("rf_trees100_depth5", table_cell), Paragraph("RandomForest", table_cell), Paragraph("trees=100, depth=5, balanced", table_cell), Paragraph("0.7498", table_cell), Paragraph("0.9193", table_cell), Paragraph("0.8158", table_cell), Paragraph("0.6813", table_cell), Paragraph("0.8550", table_cell)],
        [Paragraph("xgb_default_depth3", table_cell), Paragraph("XGBoost (Registered)", table_cell), Paragraph("trees=80, depth=3, lr=0.1", table_cell), Paragraph("<b>0.8091</b>", table_cell), Paragraph("<b>0.9414</b>", table_cell), Paragraph("0.7895", table_cell), Paragraph("<b>0.7407</b>", table_cell), Paragraph("<b>0.8950</b>", table_cell)],
        [Paragraph("xgb_tuned_depth4_poswt", table_cell), Paragraph("XGBoost", table_cell), Paragraph("trees=100, depth=4, pos_wt=4.25", table_cell), Paragraph("0.7945", table_cell), Paragraph("0.9397", table_cell), Paragraph("0.8684", table_cell), Paragraph("0.7174", table_cell), Paragraph("0.8700", table_cell)],
        [Paragraph("lgbm_unbalanced_leaves31", table_cell), Paragraph("LightGBM", table_cell), Paragraph("trees=100, leaves=31, lr=0.05", table_cell), Paragraph("0.7949", table_cell), Paragraph("0.9404", table_cell), Paragraph("0.7895", table_cell), Paragraph("0.6897", table_cell), Paragraph("0.8650", table_cell)]
    ]
    t5 = Table(t5_data, colWidths=[110, 85, 135, 45, 45, 42, 42, 44])
    t5.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t5)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: Task 6 (CI Quality Gate & Flowchart) + Task 7 (v1 vs v2 Drill)
    # =========================================================================
    story.append(Paragraph("5. Automated CI/CD Quality Gate &amp; Pull Request Scenarios (Task 6)", h1_style))
    story.append(Paragraph(
        "Our GitHub Actions workflow (<code>.github/workflows/ci.yml</code>) triggers on pull requests to <code>main</code>. "
        "It executes 17 unit tests, trains candidate architectures under <code>config/ci.yaml</code>, and compares metrics against <code>baseline_metrics.json</code> "
        "(Git commit <code>c51e19d</code>). If PR-AUC degrades by &gt;0.015 or Recall falls below 0.70, the gate terminates with exit code 1, blocking the merge. "
        "Three distinct PR scenarios were validated in <code>scripts/run_gate_scenarios.py</code>:",
        body_style
    ))

    # Figure 4: Gate Flowchart
    fig4_path = "reports/plots/fig4_workflow_gate.png"
    if os.path.exists(fig4_path):
        story.append(Image(fig4_path, width=540, height=140))
        story.append(Paragraph("Figure 3: CI/CD Quality Gate Architecture &amp; Decision Logic across Pull Requests.", caption_style))

    t6_data = [
        [Paragraph("PR Scenario", table_header), Paragraph("Candidate Condition", table_header), Paragraph("Observed Metrics vs Baseline (0.7978)", table_header), Paragraph("Gate Verdict &amp; Merging Action", table_header)],
        [Paragraph("PR #101 (Pass)", table_cell), Paragraph("Tuned XGBoost candidate on v1", table_cell), Paragraph("PR-AUC: 0.7883 (&minus;0.0095), Recall: 0.8684, ROC: 0.9366", table_cell), Paragraph("🟢 <b>PASSED</b>: Meets all criteria. Merge approved.", table_cell)],
        [Paragraph("PR #102 (Regression)", table_cell), Paragraph("Corrupted labels / bad feature", table_cell), Paragraph("PR-AUC: 0.2206 (&minus;0.5772), Recall: 0.3158 (Floor breach)", table_cell), Paragraph("🔴 <b>FAILED</b>: Severe drop. Merge blocked.", table_cell)],
        [Paragraph("PR #103 (Broken Test)", table_cell), Paragraph("Syntax error breaking feature logic", table_cell), Paragraph("Pytest fails with Exit 1 on simulated assertion failure", table_cell), Paragraph("🔴 <b>FAILED</b>: Fast-fail in Step 1 before training.", table_cell)]
    ]
    t6 = Table(t6_data, colWidths=[90, 130, 155, 173])
    t6.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t6)
    story.append(Spacer(1, 4))

    story.append(Paragraph("6. Data-Change Drill: Version 1 vs Version 2 Retraining (Task 7)", h1_style))
    story.append(Paragraph(
        "When Version 2 data (1,100 records: v1 + 300 newly released cardholders) became available, the champion model was retrained "
        "across <b>3 independent random seeds (42, 101, 777)</b> to establish whether performance gains represented true learning or random noise:",
        body_style
    ))

    # Figure 2: v1 vs v2 drill
    fig2_path = "reports/plots/fig2_v1_v2_drill.png"
    if os.path.exists(fig2_path):
        story.append(Image(fig2_path, width=540, height=185))
        story.append(Paragraph("Figure 4: Task 7 Retraining Drill &mdash; (a) Seed-wise PR-AUC, (b) 10&times; Variance Stabilization on v2 data.", caption_style))

    t7_data = [
        [Paragraph("Metric", table_header), Paragraph("v1 Mean", table_header), Paragraph("v1 Std (&sigma;)", table_header), Paragraph("v2 Mean", table_header), Paragraph("v2 Std (&sigma;)", table_header), Paragraph("Delta Mean", table_header), Paragraph("Welch's P-Value", table_header), Paragraph("Gate Verdict", table_header)],
        [Paragraph("PR-AUC", table_cell), Paragraph("0.8385", table_cell), Paragraph("0.0523", table_cell), Paragraph("0.8455", table_cell), Paragraph("<b>0.0052</b>", table_cell), Paragraph("+0.0070", table_cell), Paragraph("p = 0.8378", table_cell), Paragraph("<b>ACCEPT</b> (Variance Collapse)", table_cell)],
        [Paragraph("ROC-AUC", table_cell), Paragraph("0.9484", table_cell), Paragraph("0.0141", table_cell), Paragraph("0.9467", table_cell), Paragraph("0.0082", table_cell), Paragraph("&minus;0.0017", table_cell), Paragraph("p = 0.8679", table_cell), Paragraph("<b>ACCEPT</b> (Parity Maintained)", table_cell)],
        [Paragraph("Recall", table_cell), Paragraph("0.7982", table_cell), Paragraph("0.0662", table_cell), Paragraph("0.8013", table_cell), Paragraph("0.0588", table_cell), Paragraph("+0.0031", table_cell), Paragraph("p = 0.9551", table_cell), Paragraph("<b>ACCEPT</b> (Exceeds 0.70 floor)", table_cell)],
        [Paragraph("F1-Score", table_cell), Paragraph("0.7337", table_cell), Paragraph("0.0305", table_cell), Paragraph("0.7177", table_cell), Paragraph("0.0235", table_cell), Paragraph("&minus;0.0160", table_cell), Paragraph("p = 0.5151", table_cell), Paragraph("<b>ACCEPT</b> (Stable)", table_cell)]
    ]
    t7 = Table(t7_data, colWidths=[65, 60, 65, 60, 65, 65, 75, 93])
    t7.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t7)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: Task 8 (Gate Rules & Fairness Audit) + Limitations & Disclosure
    # =========================================================================
    story.append(Paragraph("7. Quality-Gate Threshold &amp; Fairness Defense Memo (Task 8)", h1_style))
    story.append(Paragraph(
        "<b>To:</b> Retail Credit Risk Governance Committee &nbsp;|&nbsp; "
        "<b>From:</b> Group 13 (Nihar Sahu B010 &amp; Shaun Shaju B012) &nbsp;|&nbsp; "
        "<b>Date:</b> October 2026<br/>"
        "<b>Subject:</b> Empirical Benchmarking of Candidate Quality-Gate Rules and Demographic Parity Audit",
        meta_style
    ))

    story.append(Paragraph("A. Comparison of Candidate Gate Rules", ParagraphStyle('H3', parent=h1_style, fontSize=8.5, leading=11)))
    story.append(Paragraph(
        "We evaluated three candidate quality-gate rules across all 12 historical experimental runs to balance portfolio protection against development velocity:",
        body_style
    ))

    t8_rules = [
        [Paragraph("Gate Rule ID", table_header), Paragraph("Mathematical Rule", table_header), Paragraph("Accepted", table_header), Paragraph("Rejected", table_header), Paragraph("Operational Trade-off &amp; Risk Justification", table_header)],
        [Paragraph("Rule A: Permissive", table_cell), Paragraph("&Delta;ROC-AUC &ge; &minus;0.02", table_cell), Paragraph("11 / 12", table_cell), Paragraph("1 / 12", table_cell), Paragraph("<b>Unsafe:</b> ROC-AUC is diluted by majority non-defaulters; accepted degraded runs that missed 40% of defaults.", table_cell)],
        [Paragraph("Rule B: Over-Strict", table_cell), Paragraph("PR-AUC &ge; Base &amp; &ge; 0.80", table_cell), Paragraph("1 / 12", table_cell), Paragraph("11 / 12", table_cell), Paragraph("<b>Too Brittle:</b> Rejects sound models due to normal random seed variance (&plusmn;0.052), halting agile deployments.", table_cell)],
        [Paragraph("Rule C: Composite", table_cell), Paragraph("&Delta;PR-AUC &ge; &minus;0.015 &amp; Recall &ge; 0.70", table_cell), Paragraph("7 / 12", table_cell), Paragraph("5 / 12", table_cell), Paragraph("<b>Recommended:</b> Guarantees non-negotiable default capture while absorbing small statistical run variance.", table_cell)]
    ]
    t8r = Table(t8_rules, colWidths=[90, 120, 50, 50, 238])
    t8r.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t8r)
    story.append(Spacer(1, 4))

    story.append(Paragraph("B. Subgroup Demographic Breakdown across Income_Bracket", ParagraphStyle('H3', parent=h1_style, fontSize=8.5, leading=11)))
    story.append(Paragraph(
        "To satisfy Fair Lending compliance (ECOA / FCRA), model error distributions were audited across socioeconomic tiers on the test split:",
        body_style
    ))

    # Figure 3: Fairness Plot
    fig3_path = "reports/plots/fig3_subgroup_fairness.png"
    if os.path.exists(fig3_path):
        story.append(Image(fig3_path, width=540, height=170))
        story.append(Paragraph("Figure 5: Demographic Fairness Audit across Income Brackets &mdash; Recall (TPR), Precision, and False Positive Rate (FPR).", caption_style))

    t8_fair = [
        [Paragraph("Income Tier", table_header), Paragraph("Count (n)", table_header), Paragraph("Actual Default Rate", table_header), Paragraph("Predicted High Risk Rate", table_header), Paragraph("Recall (TPR)", table_header), Paragraph("FPR", table_header), Paragraph("Missed Defaults (FNR)", table_header), Paragraph("Precision", table_header), Paragraph("PR-AUC", table_header)],
        [Paragraph("High", table_cell), Paragraph("42", table_cell), Paragraph("11.90%", table_cell), Paragraph("16.67%", table_cell), Paragraph("80.00%", table_cell), Paragraph("8.11%", table_cell), Paragraph("20.00%", table_cell), Paragraph("57.14%", table_cell), Paragraph("0.8625", table_cell)],
        [Paragraph("Low", table_cell), Paragraph("83", table_cell), Paragraph("20.48%", table_cell), Paragraph("26.51%", table_cell), Paragraph("88.24%", table_cell), Paragraph("10.61%", table_cell), Paragraph("11.76%", table_cell), Paragraph("68.18%", table_cell), Paragraph("0.8589", table_cell)],
        [Paragraph("Medium", table_cell), Paragraph("75", table_cell), Paragraph("21.33%", table_cell), Paragraph("29.33%", table_cell), Paragraph("87.50%", table_cell), Paragraph("13.56%", table_cell), Paragraph("12.50%", table_cell), Paragraph("63.64%", table_cell), Paragraph("0.7257", table_cell)]
    ]
    t8f = Table(t8_fair, colWidths=[55, 45, 75, 75, 55, 45, 70, 60, 68])
    t8f.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t8f)
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "<b>Fairness Verdict:</b> Higher default rates in the Low Income tier reflect economic vulnerability (lower liquidity buffers), not algorithmic bias. "
        "Crucially, <b>Recall is highest (88.24%)</b> for Low Income borrowers, ensuring high protection against unserviceable debt. "
        "False positive rates remain tightly bounded between 8.1% and 13.6%. The disparity is deemed <b>statistically acceptable and fair</b>.",
        body_style
    ))

    # Section 8: Limitations and Academic Integrity Disclosure
    story.append(Paragraph("8. Limitations and Academic Integrity Disclosure", h1_style))
    story.append(Paragraph(
        "<b>Limitations:</b> Evaluated on simulated retail cardholder records (800 and 1,100 rows); cost parameters ($200 default loss, $25 offer friction) "
        "are stated assumptions; real-world deployment requires continuous macroeconomic monitoring and live shadow evaluation.<br/>"
        "<b>Academic Integrity (Section 8 Disclosure):</b> Datasets are exactly those supplied by Dr. Santosh Kumar (SHA-256 hashes verified). "
        "All pipelines rely solely on free, open-source software (Python, scikit-learn, XGBoost, MLflow, pytest). Per Section 8 of the brief, "
        "we disclose AI usage: an AI coding assistant (Google DeepMind Antigravity) was used for syntax lookup, environment debugging, and plot formatting. "
        "Claude (Anthropic) was consulted for repo cross-verification and structural auditing. All architectural judgments, feature formulas, "
        "statistical drills, and defense memos were formulated by Group 13. Both authors take full responsibility for every result and are prepared "
        "to modify and defend any component live during the demonstration.",
        body_style
    ))

    doc.build(story)
    print(f"Academic PDF successfully built at: {output_path}")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "reports/FINAL_PROJECT_REPORT.pdf"
    build_academic_pdf(out)
