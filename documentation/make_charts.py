"""Generate chart images for the project documentation slide deck."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
OUT = ROOT / "documentation" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
BRAND = "#6366f1"
BG = "#0f172a"

metrics = json.loads((MODEL_DIR / "model_metrics.json").read_text(encoding="utf-8"))
MODEL_NAMES = ["linear_regression", "random_forest", "xgboost"]
DISPLAY = ["Linear Regression", "Random Forest", "XGBoost"]

# 1) Accuracy comparison bar chart
fig, ax = plt.subplots(figsize=(8, 4))
accs = [metrics[m]["accuracy"] * 100 for m in MODEL_NAMES]
bars = ax.bar(DISPLAY, accs, color=[BRAND, "#10b981", "#f59e0b"], width=0.55)
ax.axhline(73.4, color="#ef4444", linestyle="--", linewidth=1.5)
ax.text(2.45, 73.9, "majority baseline 73.4%", color="#ef4444", fontsize=9, ha="right")
for b, v in zip(bars, accs):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_ylim(0, 100)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Model accuracy — hold-out test set", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "accuracy_comparison.png", dpi=150)
plt.close()

# 2) Confusion matrix heatmaps
for name, disp in zip(MODEL_NAMES, DISPLAY):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    cm = metrics[name]["confusion_matrix"]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues" if name == "linear_regression" else "Greens",
        cbar=False, square=True, ax=ax,
        xticklabels=["Pred: No", "Pred: Yes"], yticklabels=["Actual: No", "Actual: Yes"],
        annot_kws={"size": 14},
    )
    ax.set_title(f"{disp}\n(accuracy {metrics[name]['accuracy']*100:.1f}%)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / f"confusion_{name}.png", dpi=150)
    plt.close()

# 3) Churn rate donut
fig, ax = plt.subplots(figsize=(4.5, 4.5))
sizes = [5174, 1869]
ax.pie(
    sizes, labels=["Stayed 73.4%", "Churned 26.6%"], autopct="%1.1f%%",
    colors=[BRAND, "#ef4444"], startangle=90, explode=(0, 0.05),
    textprops={"fontsize": 12, "fontweight": "bold"},
)
ax.set_title("Churn distribution (raw dataset)", fontweight="bold")
centre = plt.Circle((0, 0), 0.62, fc="white")
ax.add_artist(centre)
plt.tight_layout()
plt.savefig(OUT / "churn_distribution.png", dpi=150)
plt.close()

# 4) Random Forest top feature importance (horizontal)
importance = json.loads((MODEL_DIR / "feature_importance.json").read_text(encoding="utf-8"))
top = importance["per_model"]["random_forest"]["top_features"][:10]
features = [t["feature"] for t in top][::-1]
values = [t["value"] for t in top][::-1]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.barh(features, values, color=BRAND)
ax.set_title("Random Forest — top churn drivers", fontweight="bold")
ax.set_xlabel("Feature importance")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=150)
plt.close()

print("charts written to", OUT)
for f in sorted(OUT.iterdir()):
    print(" ", f.name)