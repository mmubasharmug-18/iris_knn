"""
DecodeLabs · Project 2 · Data Classification Using AI
Streamlit Dashboard — KNN on Iris Dataset
Run with: streamlit run app.py
"""

import streamlit as st
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
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DecodeLabs · Project 2",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# THEME COLOURS
# ─────────────────────────────────────────────
NAVY   = "#0D2B4E"
ORANGE = "#E8490F"
STEEL  = "#1A4A7A"
LIGHT  = "#EAF0F8"
WHITE  = "#FFFFFF"
GRID   = "#C5D4E8"
COLORS = [STEEL, ORANGE, NAVY]

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
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #F4F8FF; }

    .hero-banner {
        background: linear-gradient(135deg, #0D2B4E 0%, #1A4A7A 60%, #E8490F 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .hero-banner h1 {
        color: white;
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        margin: 0;
        letter-spacing: -1px;
    }
    .hero-banner p {
        color: #C5D4E8;
        margin: 0.5rem 0 0;
        font-size: 1rem;
    }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 5px solid #E8490F;
        box-shadow: 0 2px 12px rgba(13,43,78,0.08);
        text-align: center;
    }
    .metric-card .val {
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #0D2B4E;
    }
    .metric-card .lbl {
        font-size: 0.8rem;
        color: #1A4A7A;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 1.1rem;
        color: #FFFFFF;
        border-bottom: 2px solid #E8490F;
        padding-bottom: 6px;
        margin: 1.5rem 0 1rem;
        letter-spacing: 1px;
    }

    .pred-box {
        background: linear-gradient(135deg, #0D2B4E, #1A4A7A);
        border-radius: 14px;
        padding: 1.5rem;
        color: white;
        text-align: center;
    }
    .pred-box .species {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #E8490F;
        text-transform: uppercase;
    }
    .pred-box .conf { font-size: 0.9rem; color: #C5D4E8; margin-top: 0.3rem; }

    .stSlider > label { font-weight: 600; color: #0D2B4E; }

    section[data-testid="stSidebar"] {
        background-color: #0D2B4E;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] small {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    iris  = load_iris()
    X     = iris.data
    y     = iris.target
    names = iris.target_names
    feat  = iris.feature_names
    df    = pd.DataFrame(X, columns=feat)
    df["species"] = [names[i] for i in y]
    return iris, X, y, names, feat, df

iris, X, y, names, feat, df = load_data()

# ─────────────────────────────────────────────
# SIDEBAR — CONTROLS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Model Controls")
    st.markdown("---")

    test_size  = st.slider("Test Set Size (%)", 10, 40, 20, 5) / 100
    random_seed = st.slider("Random Seed", 0, 99, 42)
    manual_k   = st.checkbox("Set K manually", value=False)
    if manual_k:
        k_val = st.slider("K (n_neighbors)", 1, 20, 5)
    else:
        k_val = None

    st.markdown("---")
    st.markdown("## 🌸 Live Predictor")
    sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
    sw = st.slider("Sepal Width  (cm)", 2.0, 4.5, 3.5, 0.1)
    pl = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
    pw = st.slider("Petal Width  (cm)", 0.1, 2.5, 0.2, 0.1)

    st.markdown("---")
    st.markdown("**DecodeLabs · Batch 2026**")

# ─────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────
@st.cache_data
def train_model(test_size, random_seed, k_val):
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_seed, shuffle=True
    )

    # Elbow
    k_range     = range(1, 21)
    error_rates = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        error_rates.append(1 - accuracy_score(y_test, preds))

    best_k = k_val if k_val else int(list(k_range)[np.argmin(error_rates)])

    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted")
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=names, output_dict=True)

    return scaler, model, X_train, X_test, y_train, y_test, y_pred, \
           acc, f1, cm, report, list(error_rates), best_k

scaler, model, X_train, X_test, y_train, y_test, y_pred, \
acc, f1, cm, report, error_rates, best_k = train_model(test_size, random_seed, k_val)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>🌸 DecodeLabs · Project 2</h1>
  <p>Data Classification Using AI &nbsp;|&nbsp; KNN on the Iris Benchmark &nbsp;|&nbsp; Batch 2026</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TOP METRICS ROW
# ─────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    (f"{acc*100:.1f}%",        "Accuracy"),
    (f"{f1:.4f}",              "F1 Score"),
    (f"K = {best_k}",          "Optimal K"),
    (f"{len(X_train)}",        "Train Samples"),
    (f"{len(X_test)}",         "Test Samples"),
]
for col, (val, lbl) in zip([c1,c2,c3,c4,c5], metrics):
    col.markdown(f"""
    <div class="metric-card">
      <div class="val">{val}</div>
      <div class="lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SECTION 1 — DATASET OVERVIEW
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📊 01 · DATASET OVERVIEW</div>', unsafe_allow_html=True)

col_a, col_b = st.columns([1.2, 1])
with col_a:
    st.dataframe(df.head(10), use_container_width=True, height=280)
with col_b:
    st.dataframe(df.describe().round(3), use_container_width=True, height=280)

# ─────────────────────────────────────────────
# SECTION 2 — FEATURE DISTRIBUTIONS
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📈 02 · FEATURE DISTRIBUTIONS</div>', unsafe_allow_html=True)

fig1, axes = plt.subplots(1, 4, figsize=(18, 4), facecolor=WHITE)
fig1.suptitle("Feature Distribution by Species", fontweight="bold", color=NAVY, fontsize=13)
for i, feature in enumerate(feat):
    for cls, col in zip(range(3), COLORS):
        vals = X[y == cls, i]
        axes[i].hist(vals, bins=15, color=col, alpha=0.65, label=names[cls], edgecolor="white")
    axes[i].set_title(feature.replace(" (cm)", ""), fontsize=10, color=NAVY, fontweight="bold")
    axes[i].set_xlabel("cm")
    axes[i].grid(True)
    if i == 0:
        axes[i].legend(fontsize=8)
plt.tight_layout()
st.pyplot(fig1, use_container_width=True)
plt.close()

# ─────────────────────────────────────────────
# SECTION 3 — FEATURE SCATTER + CORRELATION
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">🔵 03 · FEATURE SPACE & CORRELATION</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fig2, ax = plt.subplots(figsize=(7, 5), facecolor=WHITE)
    markers = ["o", "^", "s"]
    for cls, col, mk in zip(range(3), COLORS, markers):
        mask = y == cls
        ax.scatter(X[mask, 2], X[mask, 3], c=col, marker=mk,
                   label=names[cls], alpha=0.85,
                   edgecolors="white", linewidths=0.4, s=70)
    ax.set_xlabel("Petal Length (cm)")
    ax.set_ylabel("Petal Width (cm)")
    ax.set_title("Petal Dimensions — Feature Space", fontweight="bold", color=NAVY)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

with col2:
    fig3, ax = plt.subplots(figsize=(7, 5), facecolor=WHITE)
    corr = df.drop("species", axis=1).corr()
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, ax=ax,
                linewidths=0.5, linecolor=GRID,
                annot_kws={"size": 11, "weight": "bold"})
    ax.set_title("Feature Correlation Matrix", fontweight="bold", color=NAVY)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close()

# ─────────────────────────────────────────────
# SECTION 4 — SCALING VISUALISATION
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">⚖️ 04 · GATEKEEPER RULE: FEATURE SCALING</div>', unsafe_allow_html=True)

X_scaled_all = scaler.transform(X)
fig4, (ax_r, ax_s) = plt.subplots(1, 2, figsize=(14, 4.5), facecolor=WHITE)
fig4.suptitle("Before vs After StandardScaler", fontweight="bold", color=NAVY)

ax_r.scatter(X[:, 0], X[:, 2], c=[COLORS[i] for i in y], alpha=0.6, edgecolors="white", s=50)
ax_r.set_title("RAW DATA (Biased)", color=NAVY, fontweight="bold")
ax_r.set_xlabel("Sepal Length (cm)")
ax_r.set_ylabel("Petal Length (cm)")
ax_r.grid(True)

ax_s.scatter(X_scaled_all[:, 0], X_scaled_all[:, 2],
             c=[COLORS[i] for i in y], alpha=0.6, edgecolors="white", s=50)
ax_s.set_title("STANDARD SCALED (Balanced) — Mean=0, Var=1", color=NAVY, fontweight="bold")
ax_s.set_xlabel("Sepal Length (scaled)")
ax_s.set_ylabel("Petal Length (scaled)")
ax_s.grid(True)
plt.tight_layout()
st.pyplot(fig4, use_container_width=True)
plt.close()

# ─────────────────────────────────────────────
# SECTION 5 — ELBOW CURVE
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">🎯 05 · ELBOW CURVE — CHOOSING K</div>', unsafe_allow_html=True)

fig5, ax = plt.subplots(figsize=(12, 4), facecolor=WHITE)
k_range = range(1, 21)
ax.plot(list(k_range), error_rates, color=NAVY, linewidth=2.5,
        marker="o", markersize=6, markerfacecolor=WHITE, markeredgecolor=NAVY)
ax.axvline(best_k, color=ORANGE, linestyle="--", linewidth=2,
           label=f"Optimal K = {best_k}")
ax.scatter([best_k], [error_rates[best_k - 1]], color=ORANGE, zorder=6, s=130)
ax.set_xlabel("K Value")
ax.set_ylabel("Error Rate")
ax.set_title(f"Elbow Method — Optimal K = {best_k}", fontweight="bold", color=NAVY)
ax.legend()
ax.grid(True)
plt.tight_layout()
st.pyplot(fig5, use_container_width=True)
plt.close()

# ─────────────────────────────────────────────
# SECTION 6 — CONFUSION MATRIX + F1 BARS
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">🧪 06 · MODEL EVALUATION</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    fig6, ax = plt.subplots(figsize=(6, 5), facecolor=WHITE)
    cmap2 = sns.light_palette(STEEL, as_cmap=True)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap2,
                xticklabels=names, yticklabels=names, ax=ax,
                linewidths=0.5, linecolor=GRID,
                annot_kws={"size": 16, "weight": "bold", "color": WHITE})
    for i in range(3):
        for j in range(3):
            if i != j and cm[i, j] > 0:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                           edgecolor=ORANGE, lw=2.5))
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix", fontweight="bold", color=NAVY)
    plt.tight_layout()
    st.pyplot(fig6, use_container_width=True)
    plt.close()

with col4:
    fig7, ax = plt.subplots(figsize=(6, 5), facecolor=WHITE)
    classes = list(names)
    f1_vals = [report[c]["f1-score"] for c in classes]
    prec_vals = [report[c]["precision"] for c in classes]
    rec_vals  = [report[c]["recall"]    for c in classes]
    x = np.arange(len(classes))
    w = 0.25
    ax.bar(x - w, prec_vals, w, label="Precision", color=STEEL, edgecolor="white")
    ax.bar(x,     f1_vals,   w, label="F1 Score",  color=ORANGE, edgecolor="white")
    ax.bar(x + w, rec_vals,  w, label="Recall",    color=NAVY,  edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score")
    ax.set_title("Precision / F1 / Recall per Class", fontweight="bold", color=NAVY)
    ax.legend()
    ax.grid(True, axis="y")
    for i, (p, f, r) in enumerate(zip(prec_vals, f1_vals, rec_vals)):
        ax.text(i - w, p + 0.03, f"{p:.2f}", ha="center", fontsize=8, color=NAVY)
        ax.text(i,     f + 0.03, f"{f:.2f}", ha="center", fontsize=8, color=NAVY)
        ax.text(i + w, r + 0.03, f"{r:.2f}", ha="center", fontsize=8, color=NAVY)
    plt.tight_layout()
    st.pyplot(fig7, use_container_width=True)
    plt.close()

# ─────────────────────────────────────────────
# SECTION 7 — PCA DECISION BOUNDARY
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">🗺️ 07 · PCA DECISION BOUNDARY VISUALISATION</div>', unsafe_allow_html=True)

@st.cache_data
def compute_pca_boundary(test_size, random_seed, best_k):
    X_sc = StandardScaler().fit_transform(X)
    pca  = PCA(n_components=2)
    X_2d = pca.fit_transform(X_sc)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_2d, y, test_size=test_size, random_state=random_seed
    )
    knn2d = KNeighborsClassifier(n_neighbors=best_k)
    knn2d.fit(X_tr, y_tr)

    h = 0.05
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    return X_2d, xx, yy, Z, y_te, X_te

X_2d, xx, yy, Z, y_te_pca, X_te_pca = compute_pca_boundary(test_size, random_seed, best_k)

fig8, ax = plt.subplots(figsize=(12, 5.5), facecolor=WHITE)
boundary_cmap = plt.cm.colors.ListedColormap([
    "#C5D4E8", "#FAD4C0", "#0D2B4E"
])
ax.contourf(xx, yy, Z, alpha=0.3, cmap=boundary_cmap)
ax.contour(xx, yy, Z, colors=STEEL, linewidths=0.6, alpha=0.5)
for cls, col, mk in zip(range(3), COLORS, ["o", "^", "s"]):
    mask = y == cls
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=col, marker=mk,
               label=names[cls], alpha=0.85,
               edgecolors="white", linewidths=0.4, s=70)
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_title(f"KNN Decision Boundary (K={best_k}) · PCA 2D Projection",
             fontweight="bold", color=NAVY)
ax.legend()
ax.grid(True)
plt.tight_layout()
st.pyplot(fig8, use_container_width=True)
plt.close()

# ─────────────────────────────────────────────
# SECTION 8 — LIVE PREDICTOR
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">🔮 08 · LIVE PREDICTION</div>', unsafe_allow_html=True)

sample        = np.array([[sl, sw, pl, pw]])
sample_scaled = scaler.transform(sample)
pred_idx      = model.predict(sample_scaled)[0]
proba         = model.predict_proba(sample_scaled)[0]
pred_name     = names[pred_idx]

col5, col6 = st.columns([1, 1.5])

with col5:
    emoji_map = {"setosa": "🌺", "versicolor": "🌼", "virginica": "🌸"}
    st.markdown(f"""
    <div class="pred-box">
      <div style="font-size:0.85rem; color:#C5D4E8; margin-bottom:0.5rem;">PREDICTED SPECIES</div>
      <div class="species">{emoji_map.get(pred_name, "🌸")} {pred_name}</div>
      <div class="conf">Sepal {sl}×{sw} cm &nbsp;|&nbsp; Petal {pl}×{pw} cm</div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    fig9, ax = plt.subplots(figsize=(7, 3), facecolor=WHITE)
    bars = ax.barh(list(names), proba * 100,
                   color=COLORS, edgecolor="white", height=0.5)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Prediction Confidence", fontweight="bold", color=NAVY)
    ax.grid(True, axis="x")
    for bar, val in zip(bars, proba * 100):
        ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontweight="bold",
                color=NAVY, fontsize=11)
    plt.tight_layout()
    st.pyplot(fig9, use_container_width=True)
    plt.close()

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='text-align:center; color:#1A4A7A; font-family:monospace; font-size:0.85rem;'>
  DecodeLabs · Project 2 · Data Classification Using AI &nbsp;|&nbsp;
  Accuracy: <b>{acc*100:.1f}%</b> &nbsp;|&nbsp;
  F1: <b>{f1:.4f}</b> &nbsp;|&nbsp;
  K = <b>{best_k}</b> &nbsp;|&nbsp;
  Batch 2026
</div>
""", unsafe_allow_html=True)
