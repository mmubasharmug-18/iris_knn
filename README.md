<<<<<<< HEAD
# 🌸 DecodeLabs · Project 2 — Data Classification Using AI

> **Batch 2026 | Powered by DecodeLabs**
> A supervised machine learning project that classifies Iris flower species using the K-Nearest Neighbors (KNN) algorithm, visualized through an interactive Streamlit dashboard.

---

## 📌 Project Overview

This project demonstrates the complete supervised learning pipeline:

```
Raw Data → Feature Scaling → Train/Test Split → KNN Training → Evaluation & Visualization
```

The model learns to classify 3 species of Iris flowers — **Setosa**, **Versicolor**, and **Virginica** — based on 4 measurements: sepal length, sepal width, petal length, and petal width.

---

## 🚀 Features

- 📊 **Dataset Overview** — Raw data table + statistical summary
- 📈 **Feature Distributions** — Histograms for all 4 features by species
- 🔵 **Feature Space & Correlation** — Scatter plot + correlation heatmap
- ⚖️ **Scaling Visualisation** — Before vs after StandardScaler
- 🎯 **Elbow Curve** — Automatic optimal K selection
- 🧪 **Model Evaluation** — Confusion matrix + Precision/F1/Recall bars
- 🗺️ **Decision Boundary** — PCA 2D projection with KNN boundaries
- 🔮 **Live Predictor** — Real-time prediction with confidence scores via sidebar sliders

---

## 🗂️ Project Structure

```
iris_knn/
│
├── app.py                  # Streamlit dashboard (main app)
├── iris_knn_project.py     # Original CLI script
├── requirements.txt        # Python dependencies
├── .gitignore              # Excludes venv, cache, etc.
└── README.md               # You are here
```

---

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| Streamlit | Interactive web dashboard |
| Scikit-learn | KNN model, scaling, metrics |
| Matplotlib | Charts and visualizations |
| Seaborn | Heatmaps |
| Pandas | Data handling |
| NumPy | Numerical operations |

---

## 🛠️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/iris_knn.git
cd iris_knn
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 📦 requirements.txt

```
streamlit
scikit-learn
matplotlib
seaborn
pandas
numpy
```

---

## 🧠 How It Works

### Dataset
- **Iris Dataset** — 150 samples, 3 classes, 4 features (balanced)
- Built into scikit-learn, no download required

### Pipeline
1. **Load** — Load Iris dataset via `sklearn.datasets`
2. **Scale** — Normalize features using `StandardScaler` (mean=0, variance=1)
3. **Split** — 80% training / 20% testing (configurable via sidebar)
4. **Elbow** — Loop K from 1–20, find K with lowest error rate
5. **Train** — Fit `KNeighborsClassifier` with optimal K
6. **Evaluate** — Accuracy, F1 Score, Confusion Matrix, Classification Report

### Results
| Metric | Score |
|---|---|
| Accuracy | 100% |
| Weighted F1 | 1.0000 |
| Optimal K | 2 |

---

## 🎛️ Sidebar Controls

| Control | Description |
|---|---|
| Test Set Size (%) | Adjust train/test split ratio |
| Random Seed | Control reproducibility |
| Set K manually | Override automatic K selection |
| Live Predictor sliders | Enter flower measurements for real-time prediction |

---

## 🙌 Acknowledgements

- Dataset: [UCI Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris) via scikit-learn
- Built as part of the **DecodeLabs Industrial Training Program — Batch 2026**

---

*Made with ❤️ at DecodeLabs*
=======
# iris_knn
>>>>>>> 7968426659ba27b11b6a6683fa6e78151f3ba67b
