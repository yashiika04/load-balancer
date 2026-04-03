import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ─── Load Data ────────────────────────────────────────────────────────────────

BASE = r"E:\LoadBalancer"   # change to "." if running locally

def load_trial_rows(path):
    """Return only the per-trial rows (before the SUMMARY block)."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip().startswith("SUMMARY") or line.strip() == "":
                break
            rows.append(line)
    from io import StringIO
    return pd.read_csv(StringIO("".join(rows)))

def parse_summary(path, key):
    """Pull a single summary value by its label."""
    with open(path) as f:
        for line in f:
            if line.strip().startswith(key):
                return float(line.strip().split(",")[1])
    return None

files = {
    "Round Robin":      os.path.join(BASE, "results_RoundRobin_default.csv"),
    "Least Connection": os.path.join(BASE, "results_LeastConenction_default.csv"),
    # ordered: Reward 1, Reward 2, Reward 3
    "Model 1\n(Reward 1)": os.path.join(BASE, "results_RLAgent_model1.csv"),
    "Model 3\n(Reward 2)": os.path.join(BASE, "results_RLAgent_model3.csv"),
    "Model 2\n(Reward 3)": os.path.join(BASE, "results_RLAgent_model2.csv"),
}

# Short keys for internal use
rl_keys = {
    "Model 1\n(Reward 1)": ("model1", os.path.join(BASE, "results_RLAgent_model1.csv")),
    "Model 3\n(Reward 2)": ("model3", os.path.join(BASE, "results_RLAgent_model3.csv")),
    "Model 2\n(Reward 3)": ("model2", os.path.join(BASE, "results_RLAgent_model2.csv")),
}

trial_data = {name: load_trial_rows(path) for name, path in files.items()}

summary_metrics = ["Success Rate Mean (%)", "Failure Rate Mean (%)",
                   "Throughput Mean (req/s)", "Avg Latency Mean (s)", "Load Variance Mean"]

def parse_summary(path, key):
    """Pull a single summary value by its exact label prefix."""
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith(key + ","):
                return float(stripped.split(",")[1])
    return None

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

# ─── Style helpers ────────────────────────────────────────────────────────────

CLASSICAL_COLOR = "#4471A7"
RL_COLORS = ["#E15F2C", "#49BB5A", "#9553B6"]   # Reward 1, Reward 2, Reward 3

CLASSICAL = ["Round Robin", "Least Connection"]
RL_LABELS = ["Model 1\n(Reward 1)", "Model 3\n(Reward 2)", "Model 2\n(Reward 3)"]
ALL_LABELS = CLASSICAL + RL_LABELS

bar_colors = ([CLASSICAL_COLOR] * 2) + RL_COLORS

def style_ax(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

def reward_legend(ax, label, color, loc="lower right"):
    patch = mpatches.Patch(color=color, label=label, alpha=0.85)
    ax.legend(handles=[patch], fontsize=7, loc=loc,
              framealpha=0.6, edgecolor="none")

# ─── PLOT 1 – Classical vs RL: Success Rate & Avg Latency ────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Classical Algorithms vs. RL Models", fontsize=14, fontweight="bold", y=1.01)

x = np.arange(len(ALL_LABELS))
w = 0.55

for ax, metric, std_key, ylabel, title in zip(
    axes,
    ["Success Rate (%)", "Avg Latency (s)"],
    ["Success Rate Std",  "Latency Std"],
    ["Success Rate (%)",  "Average Latency (s)"],
    ["Success Rate",      "Average Latency"]
):
    vals  = [summary[n][metric]  for n in ALL_LABELS]
    stds  = [summary[n][std_key] for n in ALL_LABELS]
    bars  = ax.bar(x, vals, width=w, color=bar_colors, alpha=0.88,
                   yerr=stds, capsize=4, error_kw=dict(elinewidth=1.2, ecolor="#555"))

    ax.set_title(title, fontsize=12, fontweight="semibold")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_LABELS, fontsize=8.5)
    style_ax(ax)

    # value labels on bars
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) * 0.05,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#333")

    # divider between classical and RL
    ax.axvline(x=1.5, color="#aaa", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(0.5,  ax.get_ylim()[1] * 0.97, "Classical", ha="center", fontsize=8, color="#666")
    ax.text(3.0,  ax.get_ylim()[1] * 0.97, "RL Models",  ha="center", fontsize=8, color="#666")

plt.tight_layout()
plt.savefig("plot1_classical_vs_rl.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot1")

# ─── PLOT 2 – RL Model Comparison (5 sub-metrics) ────────────────────────────

rl_x    = np.arange(len(RL_LABELS))
metrics_info = [
    ("Success Rate (%)",   "Success Rate Std",   "Success Rate (%)",   "Success Rate"),
    ("Failure Rate (%)",   "Failure Rate Std",   "Failure Rate (%)",   "Failure Rate"),
    ("Avg Latency (s)",    "Latency Std",        "Average Latency (s)","Average Latency"),
    ("Throughput (req/s)", "Throughput Std",     "Throughput (req/s)", "Throughput"),
    ("Load Variance",      "Load Variance Std",  "Load Variance",      "Load Variance"),
]

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.suptitle("RL Model Comparison (Reward 1 → Reward 2 → Reward 3)",
             fontsize=13, fontweight="bold", y=1.02)

reward_labels = ["Reward 1", "Reward 2", "Reward 3"]

for ax, (metric, std_key, ylabel, title), color, rlabel in zip(
        axes, metrics_info, RL_COLORS, reward_labels):

    # repeat same color per subplot so legend patch makes sense
    colors_here = RL_COLORS
    vals  = [summary[n][metric]  for n in RL_LABELS]
    stds  = [summary[n][std_key] for n in RL_LABELS]
    bars  = ax.bar(rl_x, vals, width=0.55, color=colors_here, alpha=0.88,
                   yerr=stds, capsize=4, error_kw=dict(elinewidth=1.2, ecolor="#555"))

    ax.set_title(title, fontsize=11, fontweight="semibold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(rl_x)
    ax.set_xticklabels(RL_LABELS, fontsize=8)
    style_ax(ax)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.05 if max(stds) > 0 else bar.get_height() * 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#333")

# shared legend bottom
patches = [mpatches.Patch(color=c, label=l)
           for c, l in zip(RL_COLORS, ["Reward 1", "Reward 2", "Reward 3"])]
fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
           framealpha=0.6, edgecolor="none", bbox_to_anchor=(0.5, -0.06))

plt.tight_layout()
plt.savefig("plot2_rl_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot2")

# ─── PLOT 3 – Per-Trial Success Rate Across All Models ───────────────────────

fig, ax = plt.subplots(figsize=(11, 5))
ax.set_title("Per-Trial Success Rate Across All Models", fontsize=13, fontweight="bold")

line_styles = {
    "Round Robin":          dict(color="#5C7EA8", lw=2,   ls="--",  marker="o",  ms=6),
    "Least Connection":     dict(color="#96B4D4", lw=2,   ls="--",  marker="s",  ms=6),
    "Model 1\n(Reward 1)":  dict(color=RL_COLORS[0], lw=2, ls="-", marker="^",  ms=7),
    "Model 3\n(Reward 2)":  dict(color=RL_COLORS[1], lw=2, ls="-", marker="D",  ms=7),
    "Model 2\n(Reward 3)":  dict(color=RL_COLORS[2], lw=2, ls="-", marker="v",  ms=7),
}

clean_labels = {
    "Round Robin": "Round Robin",
    "Least Connection": "Least Connection",
    "Model 1\n(Reward 1)": "Model 1 (Reward 1)",
    "Model 3\n(Reward 2)": "Model 3 (Reward 2)",
    "Model 2\n(Reward 3)": "Model 2 (Reward 3)",
}

for name, style in line_styles.items():
    df = trial_data[name]
    ax.plot(df["Trial"], df["Success Rate (%)"],
            label=clean_labels[name], **style)

ax.set_xlabel("Trial", fontsize=11)
ax.set_ylabel("Success Rate (%)", fontsize=11)
ax.set_xticks(trial_data["Round Robin"]["Trial"].values)
style_ax(ax)

legend = ax.legend(fontsize=9, loc="lower left", framealpha=0.7, edgecolor="none",
                   title="Algorithm / Model", title_fontsize=9)

# small reward note in corner
note = "RL reward order:\nReward 1 → Model 1\nReward 2 → Model 3\nReward 3 → Model 2"
ax.text(0.99, 0.02, note, transform=ax.transAxes, fontsize=7,
        ha="right", va="bottom", color="#555",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.6, ec="none"))

plt.tight_layout()
plt.savefig("plot3_per_trial_success.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot3")

print("All plots saved")