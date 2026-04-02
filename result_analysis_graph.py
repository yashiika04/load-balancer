import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Data ──────────────────────────────────────────────────────────
algorithms = ['Latency-Only\nRL', 'Threshold-Based\nRL', 'SLA-Threshold']
colors     = ['#e8711a', '#34a853', '#1a73e8']

sr_means = [97.80, 98.24, 99.16]
sr_stds  = [1.24,  1.04,  0.65]

bt_means = [122.69, 46.84, 52.28]
bt_stds  = [159.14, 2.63,  3.33]

trials  = [1, 2, 3, 4, 5]
lat_sr  = [98.00, 98.60, 99.00, 97.60, 95.80]
thr_sr  = [98.80, 98.40, 98.80, 96.40, 98.80]
sla_sr  = [98.60, 98.40, 99.80, 99.80, 99.20]

lat_f = [10, 7,  5, 12, 21]
thr_f = [6,  8,  6, 18,  6]
sla_f = [7,  8,  1,  1,  4]

# ── Figure layout ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.patch.set_facecolor('white')

x  = np.arange(len(algorithms))
xp = np.arange(len(trials))

# ── (a) Mean Success Rate ─────────────────────────────────────────
ax = axes[0, 0]
ax.bar(x, sr_means, yerr=None, capsize=0,
       color=colors, edgecolor='white', linewidth=0.5, width=0.55,
       error_kw={'elinewidth': 1.5, 'ecolor': '#555'})
ax.set_ylim(94, 101.5)
ax.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
ax.set_title('(a) Mean Success Rate', fontsize=11, fontweight='bold', pad=8)
ax.set_xticks(x)
ax.set_xticklabels(algorithms, fontsize=9)
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── (b) Mean Batch Processing Time ───────────────────────────────
ax = axes[0, 1]
bars = ax.bar(x, bt_means, yerr=bt_stds, capsize=5,
              color=colors, edgecolor='white', linewidth=0.5, width=0.55,
              error_kw={'elinewidth': 1.5, 'ecolor': '#555'})
ax.set_ylabel('Batch Processing Time (s)', fontsize=11, fontweight='bold')
ax.set_title('(b) Mean Batch Processing Time', fontsize=11, fontweight='bold', pad=8)
ax.set_xticks(x)
ax.set_xticklabels(algorithms, fontsize=9)
ax.grid(False)
for bar, mean in zip(bars, bt_means):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f'{mean:.1f}s',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── (c) Per-Trial Success Rate Stability ─────────────────────────
ax = axes[1, 0]
ax.plot(trials, lat_sr, 's--', color=colors[0], linewidth=2, markersize=7, label='Latency-Only RL')
ax.plot(trials, thr_sr, '^:',  color=colors[1], linewidth=2, markersize=7, label='Threshold-Based RL')
ax.plot(trials, sla_sr, 'o-',  color=colors[2], linewidth=2, markersize=7, label='SLA-Threshold')
ax.set_xlabel('Trial', fontsize=11, fontweight='bold')
ax.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
ax.set_title('(c) Per-Trial Success Rate Stability', fontsize=11, fontweight='bold', pad=8)
ax.set_xticks(trials)
ax.set_ylim(94, 101)
ax.legend(fontsize=8.5, loc='lower left')
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── (d) Per-Trial Request Failures ───────────────────────────────
ax = axes[1, 1]
w = 0.25
ax.bar(xp - w, lat_f, w, color=colors[0], label='Latency-Only RL',    edgecolor='white')
ax.bar(xp,     thr_f, w, color=colors[1], label='Threshold-Based RL', edgecolor='white')
ax.bar(xp + w, sla_f, w, color=colors[2], label='SLA-Threshold',      edgecolor='white')
ax.set_xlabel('Trial', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Failures', fontsize=11, fontweight='bold')
ax.set_title('(d) Per-Trial Request Failures', fontsize=11, fontweight='bold', pad=8)
ax.set_xticks(xp)
ax.set_xticklabels([f'T{t}' for t in trials])
ax.legend(fontsize=8.5)
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Save ──────────────────────────────────────────────────────────
plt.suptitle('Reinforcement Learning Load Balancer — Reward Function Comparison',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('results_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
print('Saved: results_comparison.png')