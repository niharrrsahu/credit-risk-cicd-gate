"""Generate High-Resolution Figures for the Academic Project Report."""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs("reports/plots", exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 8.5

# ==============================================================================
# Figure 1: Model Family PR-AUC & ROC-AUC Comparison across 12 Runs
# ==============================================================================
df_runs = pd.read_csv("reports/experiment_runs_comparison.csv")
fig, ax1 = plt.subplots(figsize=(6.8, 2.5), dpi=300)

x = np.arange(len(df_runs))
width = 0.38

bars1 = ax1.bar(x - width/2, df_runs['PR_AUC'], width, label='PR-AUC (Primary)', color='#2B6CB0', alpha=0.9)
bars2 = ax1.bar(x + width/2, df_runs['ROC_AUC'], width, label='ROC-AUC', color='#48BB78', alpha=0.85)

ax1.axhline(0.7978, color='#E53E3E', linestyle='--', linewidth=1.2, label='Baseline PR-AUC (0.7978)')
ax1.set_ylabel('Score Metric', fontweight='bold')
ax1.set_xticks(x)
labels = [r.replace('_', '\n') for r in df_runs['Run_Name']]
ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=6.8)
ax1.set_ylim(0.65, 1.0)
ax1.legend(loc='lower right', frameon=True, fontsize=7)
ax1.set_title("Figure 1: Benchmark of 12 Tracked MLflow Runs Across 4 Model Families (Task 5)", fontsize=9, fontweight='bold', pad=6)
plt.tight_layout()
plt.savefig("reports/plots/fig1_runs_comparison.png", dpi=300)
plt.close()
print("Saved reports/plots/fig1_runs_comparison.png")

# ==============================================================================
# Figure 2: Task 7 Version 1 vs Version 2 Retraining Drill & Variance Reduction
# ==============================================================================
df_v1_v2 = pd.read_csv("reports/v1_vs_v2_drill_raw_runs.csv")
v1_prauc = df_v1_v2[df_v1_v2['Dataset_Version'] == 'v1']['PR_AUC'].values
v2_prauc = df_v1_v2[df_v1_v2['Dataset_Version'] == 'v2']['PR_AUC'].values
seeds = [42, 101, 777]

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.8, 2.4), dpi=300)

# Panel A: Seed-by-seed comparison
x_seeds = np.arange(len(seeds))
ax_a.plot(x_seeds, v1_prauc, marker='o', linewidth=1.5, markersize=5, color='#DD6B20', label='v1 (800 rows)')
ax_a.plot(x_seeds, v2_prauc, marker='s', linewidth=1.5, markersize=5, color='#3182CE', label='v2 (1,100 rows)')
ax_a.set_xticks(x_seeds)
ax_a.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=7.5)
ax_a.set_ylabel("PR-AUC", fontweight='bold')
ax_a.set_title("(a) Seed-wise PR-AUC Trajectory", fontsize=8.5, fontweight='bold')
ax_a.legend(loc='lower right', fontsize=7.5)
ax_a.set_ylim(0.75, 0.92)

# Panel B: Mean and Variance (Std Dev errorbar)
means = [np.mean(v1_prauc), np.mean(v2_prauc)]
stds = [np.std(v1_prauc, ddof=1), np.std(v2_prauc, ddof=1)]
bars = ax_b.bar(['Version 1\n(800 rows)', 'Version 2\n(1,100 rows)'], means, yerr=stds, capsize=5, 
         color=['#ED8936', '#4299E1'], width=0.45, alpha=0.9, error_kw={'elinewidth': 1.5, 'ecolor': '#2D3748'})
ax_b.set_ylabel("Mean PR-AUC (± 1 Std Dev)", fontweight='bold')
ax_b.set_title("(b) 10x Variance Stabilization", fontsize=8.5, fontweight='bold')
ax_b.set_ylim(0.70, 0.95)

# Text labels on bars
for bar, m, s in zip(bars, means, stds):
    ax_b.text(bar.get_x() + bar.get_width()/2., 0.72, f"μ={m:.4f}\nσ=±{s:.4f}", ha='center', va='bottom', fontsize=7, color='white', fontweight='bold')

plt.suptitle("Figure 2: Data-Change Drill — Version 1 vs Version 2 Retraining (Task 7)", fontsize=9, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("reports/plots/fig2_v1_v2_drill.png", dpi=300)
plt.close()
print("Saved reports/plots/fig2_v1_v2_drill.png")

# ==============================================================================
# Figure 3: Task 8 Demographic Fairness Audit Across Income_Bracket
# ==============================================================================
df_fair = pd.read_csv("reports/income_bracket_subgroup_fairness.csv")
fig, ax = plt.subplots(figsize=(6.8, 2.3), dpi=300)

groups = df_fair['Income_Bracket'].values
x_grp = np.arange(len(groups))
w = 0.25

ax.bar(x_grp - w, df_fair['Recall_TPR'], w, label='Recall (Default Capture, TPR)', color='#2B6CB0')
ax.bar(x_grp, df_fair['Precision'], w, label='Precision', color='#4FD1C5')
ax.bar(x_grp + w, df_fair['FPR'], w, label='False Positive Rate (FPR)', color='#F56565')

ax.axhline(0.70, color='#C53030', linestyle=':', label='Recall Safety Floor (0.70)')
ax.set_xticks(x_grp)
ax.set_xticklabels([f"{g} Income\n(n={cnt})" for g, cnt in zip(groups, df_fair['Total_Count'])], fontsize=8)
ax.set_ylabel("Metric Rate", fontweight='bold')
ax.set_ylim(0.0, 1.05)
ax.legend(loc='upper right', frameon=True, fontsize=7.5)
ax.set_title("Figure 3: Subgroup Error Parity & Demographic Fairness across Income Tiers (Task 8)", fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig("reports/plots/fig3_subgroup_fairness.png", dpi=300)
plt.close()
print("Saved reports/plots/fig3_subgroup_fairness.png")

# ==============================================================================
# Figure 4: CI/CD Quality Gate Workflow Architecture
# ==============================================================================
fig, ax = plt.subplots(figsize=(6.8, 1.8), dpi=300)
ax.axis('off')

# Draw flowchart boxes
boxes = [
    ("PR Opened\n(Code/Data/YAML)", 0.08, 0.5, "#E2E8F0", "#4A5568"),
    ("Automated Tests\n(pytest - 17 tests)", 0.30, 0.5, "#BEE3F8", "#2B6CB0"),
    ("Candidate Train\n(config/ci.yaml)", 0.52, 0.5, "#C6F6D5", "#22543D"),
    ("Quality Gate\n(PR-AUC & Recall)", 0.74, 0.5, "#FEEBC8", "#7B341E"),
    ("Merge &\nRegister", 0.94, 0.72, "#C6F6D5", "#22543D"),
    ("Block PR\n(Exit 1)", 0.94, 0.28, "#FED7D7", "#742A2A")
]

for text, xc, yc, bg, fg in boxes:
    ax.text(xc, yc, text, ha='center', va='center', fontsize=6.8, fontweight='bold', color=fg,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=bg, edgecolor=fg, linewidth=1))

# Arrows
ax.annotate("", xy=(0.20, 0.5), xytext=(0.14, 0.5), arrowprops=dict(arrowstyle="->", lw=1.2, color="#4A5568"))
ax.annotate("", xy=(0.42, 0.5), xytext=(0.38, 0.5), arrowprops=dict(arrowstyle="->", lw=1.2, color="#4A5568"))
ax.annotate("", xy=(0.64, 0.5), xytext=(0.60, 0.5), arrowprops=dict(arrowstyle="->", lw=1.2, color="#4A5568"))

# Branching arrows
ax.annotate("Pass", xy=(0.87, 0.72), xytext=(0.81, 0.58), arrowprops=dict(arrowstyle="->", lw=1.2, color="#22543D"), fontsize=6.8, fontweight='bold', color="#22543D")
ax.annotate("Regression", xy=(0.87, 0.28), xytext=(0.81, 0.42), arrowprops=dict(arrowstyle="->", lw=1.2, color="#742A2A"), fontsize=6.8, fontweight='bold', color="#742A2A")

ax.set_title("Figure 4: Automated CI/CD Quality Gate Pipeline Architecture (.github/workflows/ci.yml)", fontsize=9, fontweight='bold', pad=4)
plt.tight_layout()
plt.savefig("reports/plots/fig4_workflow_gate.png", dpi=300)
plt.close()
print("Saved reports/plots/fig4_workflow_gate.png")
