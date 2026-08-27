"""
Generates publication-quality 300 DPI IEEE figure for Fig. 2: Sentiment Class Distribution of the Working Dataset.
"""

import matplotlib.pyplot as plt
import os

# Set style for IEEE publication
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

categories = ['Negative', 'Neutral', 'Positive']
counts = [44784, 151129, 104087]
percentages = [14.93, 50.38, 34.70]
colors = ['#dc2626', '#475569', '#16a34a'] # Red for negative, Grey/Slate for neutral, Green for positive

bars = ax.bar(categories, counts, color=colors, width=0.55, edgecolor='black', linewidth=1.2, zorder=3)

# Add title and labels
ax.set_title('Sentiment Class Distribution of the Working Dataset', fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Sentiment Class', fontsize=11, fontweight='bold', labelpad=10)
ax.set_ylabel('Number of Headlines', fontsize=11, fontweight='bold', labelpad=10)

# Set y-axis limits and grid
ax.set_ylim(0, 180000)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

# Annotate bars with counts and percentages
for bar, count, pct in zip(bars, counts, percentages):
    height = bar.get_height()
    label_text = f"{count:,}\n({pct:.2f}%)"
    ax.annotate(
        label_text,
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 6),  # 6 points vertical offset
        textcoords="offset points",
        ha='center', va='bottom',
        fontsize=10, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="#f8fafc", ec="#cbd5e1", lw=1)
    )

# Clean spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save in project root and artifact directories
output_path1 = "fig2_sentiment_distribution.png"
output_path2 = "fig2_loss_trend.png" # Also overwrite old fig2
artifact_path = r"C:\Users\ASUS\.gemini\antigravity\brain\c0c4765b-b3ea-4f02-b833-79370412cf97\fig2_sentiment_distribution.png"

plt.savefig(output_path1, dpi=300)
plt.savefig(output_path2, dpi=300)
plt.savefig(artifact_path, dpi=300)
plt.close()

print(f"Successfully generated Fig. 2 at {output_path1} and {artifact_path}")
