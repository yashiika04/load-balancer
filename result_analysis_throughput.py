import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

BASE = r"E:\LoadBalancer"   # ← change to your local path

def parse_summary(path, key):
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith(key + ","):
                return float(stripped.split(",")[1])
    return None

files = {
    "Round Robin":          os.path.join(BASE, "results_RoundRobin_default.csv"),
    "Least Connection":     os.path.join(BASE, "results_LeastConenction_default.csv"),
    "Model 1\n(Reward 1)": os.path.join(BASE, "results_RLAgent_model1.csv"),
    "Model 3\n(Reward 2)": os.path.join(BASE, "results_RLAgent_model3.csv"),
    "Model 2\n(Reward 3)": os.path.join(BASE, "results_RLAgent_model2.csv"),
}

summary = {}
for name, path in files.items():
    summary[name] = {
        "Success Rate (%)":   parse_summary(path, "Success Rate Mean (%)"),
        "Failure Rate (%)":   parse_summary(path, "Failure Rate Mean (%)"),
        "Throughput (req/s)": parse_summary(path, "Throughput Mean (req/s)"),
        "Avg Latency (s)":    parse_summary(path, "Avg Latency Mean (s)"),
        "Load Variance":      parse_summary(path, "Load Variance Mean"),
        "Success Rate Std":   parse_summary(path, "Success Rate Std Dev (%)"),
        "Latency Std":        parse_summary(path, "Avg Latency Std Dev (s)"),
        "Throughput Std":     parse_summary(path, "Throughput Std Dev (req/s)"),
        "Failure Rate Std":   parse_summary(path, "Failure Rate Std Dev (%)"),
        "Load Variance Std":  parse_summary(path, "Load Variance Std Dev"),
    }

RL_LABELS  = ["Model 1\n(Reward 1)", "Model 3\n(Reward 2)", "Model 2\n(Reward 3)"]
CLASSICAL_COLOR = "#4471A7"
RL_COLORS = ["#E15F2C", "#49BB5A", "#9553B6"]   # Reward 1, Reward 2, Reward 3
rl_x       = np.arange(len(RL_LABELS))

def style_ax(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

# ── 5 subplots — one per metric ──────────────────────────────────────────────
metrics_info = [
    ("Success Rate (%)",   "Success Rate Std",  "Success Rate (%)",   "Success Rate"),
    ("Failure Rate (%)",   "Failure Rate Std",  "Failure Rate (%)",   "Failure Rate"),
    ("Avg Latency (s)",    "Latency Std",       "Average Latency (s)","Average Latency"),
    ("Throughput (req/s)", "Throughput Std",    "Throughput (req/s)", "Throughput"),
    ("Load Variance",      "Load Variance Std", "Load Variance",      "Load Variance"),
]

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.suptitle("RL Model Comparison (Reward 1 → Reward 2 → Reward 3)",
             fontsize=13, fontweight="bold", y=1.02)

# ↓ FIX: iterate over metrics_info with index — no zip with RL_COLORS
for i, (metric, std_key, ylabel, title) in enumerate(metrics_info):
    ax = axes[i]

    vals = [summary[n][metric]  for n in RL_LABELS]
    stds = [summary[n][std_key] for n in RL_LABELS]

    bars = ax.bar(rl_x, vals, width=0.55, color=RL_COLORS, alpha=0.88,
                  yerr=stds, capsize=4,
                  error_kw=dict(elinewidth=1.2, ecolor="#555"))

    ax.set_title(title, fontsize=11, fontweight="semibold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(rl_x)
    ax.set_xticklabels(["Reward 1", "Reward 2", "Reward 3"], fontsize=8)
    style_ax(ax)

    # value labels
    max_std = max(stds) if max(stds) > 0 else vals[0] * 0.02
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_std * 0.15,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#333")

# shared legend
patches = [mpatches.Patch(color=c, label=l)
           for c, l in zip(RL_COLORS, ["Reward 1", "Reward 2", "Reward 3"])]
fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
           framealpha=0.6, edgecolor="none", bbox_to_anchor=(0.5, -0.06))

plt.tight_layout()
plt.savefig("plot2_throughput.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot2_rl_comparison.png")