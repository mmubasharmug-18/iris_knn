"""
DecodeLabs - Project 2: Data Classification Using AI
KNN Classifier on the Iris Dataset
Pipeline: Load → Scale → Split → Train → Evaluate
"""

# ─────────────────────────────────────────────
# STEP 1: IMPORTS
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# COLOUR PALETTE  (matches the DecodeLabs blueprint theme)
# ─────────────────────────────────────────────
NAVY   = "#0D2B4E"
ORANGE = "#E8490F"
STEEL  = "#1A4A7A"
LIGHT  = "#EAF0F8"
WHITE  = "#FFFFFF"
GRID   = "#C5D4E8"

plt.rcParams.update({
    "font.family":      "monospace",
    "axes.facecolor":   LIGHT,
    "figure.facecolor": WHITE,
    "axes.edgecolor":   STEEL,
    "axes.labelcolor":  NAVY,
    "xtick.color":      NAVY,
    "ytick.color":      NAVY,
    "text.color":       NAVY,
    "grid.color":       GRID,
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
})

# ─────────────────────────────────────────────
# STEP 2: LOAD & EXPLORE DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  DecodeLabs · Project 2 · Data Classification")
print("=" * 55)

iris = load_iris()
X    = iris.data
y    = iris.target
names = iris.target_names          # ['setosa', 'versicolor', 'virginica']
feat  = iris.feature_names

df = pd.DataFrame(X, columns=feat)
df["species"] = [names[i] for i in y]

print(f"\n📊  Dataset Shape  : {X.shape}")
print(f"🌸  Classes        : {list(names)}")
print(f"🔢  Samples/class  : {np.bincount(y).tolist()}")
print(f"\n{df.describe().round(2)}\n")

# ─────────────────────────────────────────────
# STEP 3: FEATURE SCALING  (StandardScaler)
# ─────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────────
# STEP 4: TRAIN / TEST SPLIT  (80 / 20, shuffled)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"✂️   Train samples : {len(X_train)}  |  Test samples : {len(X_test)}")

# ─────────────────────────────────────────────
# STEP 5: FIND OPTIMAL K  (elbow method)
# ─────────────────────────────────────────────
k_range    = range(1, 21)
error_rates = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    error_rates.append(1 - accuracy_score(y_test, preds))

best_k = k_range[np.argmin(error_rates)]
print(f"\n🎯  Best K (lowest error) : K = {best_k}")

# ─────────────────────────────────────────────
# STEP 6: TRAIN FINAL MODEL
# ─────────────────────────────────────────────
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ─────────────────────────────────────────────
# STEP 7: EVALUATE
# ─────────────────────────────────────────────
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average="weighted")
cm  = confusion_matrix(y_test, y_pred)

print(f"\n{'─'*45}")
print(f"  Accuracy : {acc*100:.2f}%")
print(f"  F1 Score : {f1:.4f}")
print(f"{'─'*45}")
print(f"\n{classification_report(y_test, y_pred, target_names=names)}")

# ─────────────────────────────────────────────
# STEP 8: VISUALISE  (4-panel blueprint figure)
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14), facecolor=WHITE)
fig.suptitle(
    "DecodeLabs · Project 2 · KNN Classification on Iris",
    fontsize=16, fontweight="bold", color=NAVY, y=0.97
)

gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

COLORS = [STEEL, ORANGE, NAVY]
markers = ["o", "^", "s"]

# ── Panel A: Scatter – Petal Length vs Petal Width ──────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
for cls, col, mk in zip(range(3), COLORS, markers):
    mask = y == cls
    ax0.scatter(X[mask, 2], X[mask, 3],
                c=col, marker=mk, label=names[cls],
                alpha=0.85, edgecolors="white", linewidths=0.4, s=60)
ax0.set_xlabel("Petal Length (cm)")
ax0.set_ylabel("Petal Width  (cm)")
ax0.set_title("Feature Space · Petal Dimensions", fontweight="bold", color=NAVY)
ax0.legend(framealpha=0.85)
ax0.grid(True)

# ── Panel B: Elbow curve ────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(list(k_range), error_rates, color=NAVY, linewidth=2, marker="o",
         markersize=5, markerfacecolor=WHITE, markeredgecolor=NAVY)
ax1.axvline(best_k, color=ORANGE, linestyle="--", linewidth=1.8,
            label=f"Optimal K = {best_k}")
ax1.scatter([best_k], [error_rates[best_k - 1]],
            color=ORANGE, zorder=5, s=100)
ax1.set_xlabel("K Value")
ax1.set_ylabel("Error Rate")
ax1.set_title("Elbow Curve · Choosing Optimal K", fontweight="bold", color=NAVY)
ax1.legend(framealpha=0.85)
ax1.grid(True)

# ── Panel C: Confusion Matrix ────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
cmap = sns.light_palette(STEEL, as_cmap=True)
sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
            xticklabels=names, yticklabels=names,
            ax=ax2, linewidths=0.5, linecolor=GRID,
            annot_kws={"size": 14, "weight": "bold", "color": WHITE})
ax2.set_xlabel("Predicted Label")
ax2.set_ylabel("True Label")
ax2.set_title("Confusion Matrix", fontweight="bold", color=NAVY)
# highlight off-diagonal cells in orange
for i in range(3):
    for j in range(3):
        if i != j and cm[i, j] > 0:
            ax2.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                        edgecolor=ORANGE, lw=2))

# ── Panel D: Per-class F1 bars ───────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
report = classification_report(y_test, y_pred,
                               target_names=names, output_dict=True)
classes = list(names)
f1_vals = [report[c]["f1-score"] for c in classes]
bars = ax3.bar(classes, f1_vals, color=COLORS, edgecolor=WHITE,
               linewidth=0.8, width=0.5)
ax3.set_ylim(0, 1.15)
ax3.set_ylabel("F1 Score")
ax3.set_title(f"Per-class F1 Score  (weighted avg = {f1:.3f})",
              fontweight="bold", color=NAVY)
ax3.grid(True, axis="y")
for bar, val in zip(bars, f1_vals):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             val + 0.03, f"{val:.2f}",
             ha="center", va="bottom", fontweight="bold",
             color=NAVY, fontsize=11)

# Footer annotation
fig.text(0.5, 0.01,
         f"Accuracy: {acc*100:.1f}%   |   Weighted F1: {f1:.4f}   |   K = {best_k}   |   Train/Test = 80/20",
         ha="center", fontsize=10, color=STEEL, style="italic")

plt.savefig("iris_knn_results.png",
            dpi=150, bbox_inches="tight", facecolor=WHITE)
print("✅  Figure saved → iris_knn_results.png")
plt.close()

# ─────────────────────────────────────────────
# STEP 9: LIVE PREDICTION DEMO
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  LIVE PREDICTION DEMO")
print("=" * 55)
sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # a typical Setosa
sample_scaled = scaler.transform(sample)
pred  = model.predict(sample_scaled)[0]
proba = model.predict_proba(sample_scaled)[0]
print(f"\n  Input features : Sepal 5.1×3.5 cm  |  Petal 1.4×0.2 cm")
print(f"  Prediction     : ✅  {names[pred].upper()}")
print(f"  Confidence     : Setosa {proba[0]*100:.1f}%  |  "
      f"Versicolor {proba[1]*100:.1f}%  |  Virginica {proba[2]*100:.1f}%")
print("\n" + "=" * 55)
print("  Project 2 Complete 🎉")
print("=" * 55)
