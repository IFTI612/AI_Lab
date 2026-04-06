"""
=============================================================
  Binary Classification – Brain Hemorrhage Dataset
  Models: Decision Tree | Naïve Bayes | K-Nearest Neighbors
=============================================================
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder

# Consistent color palette
PALETTE = {"Decision Tree": "#4C72B0", "Naïve Bayes": "#DD8452", "KNN": "#55A868"}

# =============================================================================
# SECTION 1 – DATA LOADING & EXPLORATION
# =============================================================================
print("=" * 60)
print("  SECTION 1 – DATA LOADING & EXPLORATION")
print("=" * 60)

# Load the dataset
df = pd.read_csv("brain_hemorrhage_dataset.csv")

# ── 1a. Basic inspection ──────────────────────────────────────────────────────
print("\n▸ First 5 rows (head):")
print(df.head())

print(f"\n▸ Shape: {df.shape}  ({df.shape[0]} samples, {df.shape[1]} features)")

print("\n▸ Info:")
df.info()

print("\n▸ Statistical summary (numeric columns):")
print(df.describe())

print("\n▸ Missing values per column:")
print(df.isnull().sum())

# =============================================================================
# SECTION 2 – DATA PREPROCESSING
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 2 – DATA PREPROCESSING")
print("=" * 60)

# ── 2a. Drop irrelevant / non-predictive columns ──────────────────────────────
# 'Patient ID'  -> unique identifier
# 'Date'        -> raw date string
# 'Year'        -> not a clinical predictor 
drop_cols = ["Patient ID", "Date", "Year"]
df.drop(columns=drop_cols, inplace=True)
print(f"\n▸ Dropped columns: {drop_cols}")

# ── 2b. Define the binary target variable ─────────────────────────────────────
# Original 'Outcome' has 4 classes: Fatal, Recovered, Rehabilitation Required, Severe Disability.
# Binary definition: 1 = Adverse outcome (Fatal or Severe Disability)
#                    0 = Better outcome  (Recovered or Rehabilitation Required)
df["Target"] = df["Outcome"].apply(
    lambda x: 1 if x in ["Fatal", "Severe Disability"] else 0
)
print("\n▸ Binary target distribution:")
print(df["Target"].value_counts().rename({0: "Better (0)", 1: "Adverse (1)"}))

# Drop original Outcome column (it's now encoded as Target)
df.drop(columns=["Outcome"], inplace=True)

# ── 2c. Handle categorical features via Label Encoding ───────────────────────
# Keep only numeric features.
# Encode categoricals to integers so they become numeric.
cat_cols = df.select_dtypes(include="object").columns.tolist()
print(f"\n▸ Encoding categorical columns: {cat_cols}")

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
    print(f"   {col}: {df[col].nunique()} unique encoded values")

print("\n▸ All columns after preprocessing:")
print(df.dtypes)

# ── 2d. Confirm no missing values remain ──────────────────────────────────────
assert df.isnull().sum().sum() == 0, "Missing values detected!"
print("\n▸ No missing values – dataset is clean.")

# =============================================================================
# SECTION 3 – TRAIN / TEST SPLIT  (80 / 20)
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 3 – TRAIN / TEST SPLIT")
print("=" * 60)

X = df.drop(columns=["Target"])
y = df["Target"]

# random_state for reproducibility;
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\n▸ Training set : {X_train.shape[0]} samples")
print(f"▸ Testing  set : {X_test.shape[0]} samples")
print(f"▸ Features     : {X_train.shape[1]}")

# =============================================================================
# SECTION 4 – MODEL TRAINING
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 4 – MODEL TRAINING")
print("=" * 60)

# ── 4a. Decision Tree ─────────────────────────────────────────────────────────
# max_depth=5 prevents overfitting; criterion='gini' is standard
dt = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=42)
dt.fit(X_train, y_train)
print("\n▸ Decision Tree trained  (max_depth=5, criterion=gini)")

# ── 4b. Naïve Bayes (Gaussian) ────────────────────────────────────────────────
# GaussianNB assumes continuous features follow a normal distribution
nb = GaussianNB()
nb.fit(X_train, y_train)
print("▸ Naïve Bayes trained    (GaussianNB – default priors)")

# ── 4c. K-Nearest Neighbors ──────────────────────────────────────────────────
# k=9 is odd (avoids ties); weights='distance' gives closer neighbors more say
knn = KNeighborsClassifier(n_neighbors=9, weights="distance", metric="euclidean")
knn.fit(X_train, y_train)
print("▸ KNN trained            (k=9, weights=distance, metric=euclidean)")

# =============================================================================
# SECTION 5 – MODEL EVALUATION
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 5 – MODEL EVALUATION")
print("=" * 60)

models = {
    "Decision Tree": dt,
    "Naïve Bayes":   nb,
    "KNN":           knn,
}

results = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    acc  = accuracy_score (y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score   (y_test, y_pred, zero_division=0)
    f1   = f1_score       (y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    results[name] = dict(Accuracy=acc, Precision=prec, Recall=rec, F1=f1, CM=cm, Pred=y_pred)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred,
          target_names=["Better (0)", "Adverse (1)"]))

# =============================================================================
# SECTION 6 – PERFORMANCE COMPARISON
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 6 – PERFORMANCE COMPARISON")
print("=" * 60)

metrics_df = pd.DataFrame({
    name: {k: v for k, v in res.items() if k not in ("CM", "Pred")}
    for name, res in results.items()
}).T

print("\n▸ Summary table:")
print(metrics_df.round(4).to_string())

best_model = metrics_df["F1"].idxmax()
print(f"\n▸ Best model by F1-Score: {best_model} ({metrics_df.loc[best_model,'F1']:.4f})")

# =============================================================================
# SECTION 7 – VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 7 – GENERATING VISUALIZATIONS")
print("=" * 60)

# ── FIG 1: EDA Dashboard ──────────────────────────────────────────────────────
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle("Brain Hemorrhage Dataset – Exploratory Data Analysis",
              fontsize=15, fontweight="bold", y=1.01)

# 1a. Target distribution
target_counts = y.value_counts().rename({0: "Better\nOutcome", 1: "Adverse\nOutcome"})
axes[0, 0].bar(target_counts.index, target_counts.values,
               color=["#55A868", "#DD8452"], edgecolor="white", width=0.5)
axes[0, 0].set_title("Target Class Distribution", fontweight="bold")
axes[0, 0].set_ylabel("Count")
for i, v in enumerate(target_counts.values):
    axes[0, 0].text(i, v + 10, str(v), ha="center", fontweight="bold")

# 1b. Age distribution by outcome
age_0 = df[df["Target"] == 0]["Age"]
age_1 = df[df["Target"] == 1]["Age"]
axes[0, 1].hist(age_0, bins=20, alpha=0.6, label="Better (0)", color="#55A868")
axes[0, 1].hist(age_1, bins=20, alpha=0.6, label="Adverse (1)", color="#DD8452")
axes[0, 1].set_title("Age Distribution by Outcome", fontweight="bold")
axes[0, 1].set_xlabel("Age")
axes[0, 1].set_ylabel("Count")
axes[0, 1].legend()

# 1c. Hemorrhage Type frequency
ht_counts = df["Hemorrhage Type"].value_counts()
axes[1, 0].barh(range(len(ht_counts)), ht_counts.values,
                color=sns.color_palette("tab10", len(ht_counts)))
axes[1, 0].set_yticks(range(len(ht_counts)))
axes[1, 0].set_yticklabels([f"Type {i}" for i in ht_counts.index])
axes[1, 0].set_title("Hemorrhage Type Frequency (Encoded)", fontweight="bold")
axes[1, 0].set_xlabel("Count")

# 1d. Gender distribution
gender_counts = df["Gender"].value_counts().rename({0: "Female", 1: "Male"})
axes[1, 1].pie(gender_counts.values, labels=gender_counts.index,
               autopct="%1.1f%%", colors=["#4C72B0", "#C44E52"],
               startangle=90, wedgeprops=dict(edgecolor="white", linewidth=1.5))
axes[1, 1].set_title("Gender Distribution", fontweight="bold")

plt.tight_layout()
fig1.savefig("outputs/fig1_eda_dashboard.png",
             dpi=150, bbox_inches="tight")
plt.close(fig1)
print("  Saved: fig1_eda_dashboard.png")

# ── FIG 2: Confusion Matrices ─────────────────────────────────────────────────
fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
fig2.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")

for ax, (name, res) in zip(axes, results.items()):
    sns.heatmap(res["CM"], annot=True, fmt="d", cmap="Blues",
                xticklabels=["Better (0)", "Adverse (1)"],
                yticklabels=["Better (0)", "Adverse (1)"],
                ax=ax, linewidths=0.5, linecolor="white",
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_title(name, fontweight="bold", fontsize=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.tight_layout()
fig2.savefig("outputs/fig2_confusion_matrices.png",
             dpi=150, bbox_inches="tight")
plt.close(fig2)
print("  Saved: fig2_confusion_matrices.png")

# ── FIG 3: Metrics Comparison Bar Chart ──────────────────────────────────────
fig3, ax = plt.subplots(figsize=(11, 6))

metric_names = ["Accuracy", "Precision", "Recall", "F1"]
model_names  = list(results.keys())
x            = np.arange(len(metric_names))
bar_width    = 0.22
offsets      = [-bar_width, 0, bar_width]

for i, (name, res) in enumerate(results.items()):
    vals = [res["Accuracy"], res["Precision"], res["Recall"], res["F1"]]
    bars = ax.bar(x + offsets[i], vals, bar_width,
                  label=name, color=list(PALETTE.values())[i],
                  edgecolor="white", alpha=0.88)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(metric_names, fontsize=12)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig3.savefig("outputs/fig3_model_comparison.png",
             dpi=150, bbox_inches="tight")
plt.close(fig3)
print("  Saved: fig3_model_comparison.png")


# =============================================================================
# SECTION 8 – FINAL ANALYSIS SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("  SECTION 8 – ANALYSIS & CONCLUSION")
print("=" * 60)

print("""
▸ Dataset characteristics:
  - 3000 patient records; 5 features after preprocessing
  - Binary target: Adverse (Fatal/Severe Disability) vs. Better outcome
  - Balanced classes (≈50/50 split)

▸ Model performance summary:
""")
print(metrics_df.round(4).to_string())

print(f"""
▸ Best performing model: {best_model}

▸ Why {best_model} likely leads:
  - Decision Tree: Creates explicit decision boundaries using feature
    thresholds (e.g., Age > 70 → Adverse). Easy to interpret and works
    well with mixed-type encoded features. Controlled depth (7) prevents
    overfitting while capturing real patterns.

  - Naïve Bayes: Assumes feature independence and Gaussian distributions.
    Fast but the independence assumption is too strong for clinical data
    where Age, Hemorrhage Type, and Symptoms are correlated.

  - KNN (k=9): Classifies by majority vote of 9 nearest neighbors.
    Works well when similar patients have similar outcomes. Distance-
    weighted voting reduces noise from outliers.

▸ Trade-off considerations:
  - Interpretability  : Decision Tree > Naïve Bayes > KNN
  - Training speed    : Naïve Bayes > Decision Tree > KNN
  - Inference speed   : Decision Tree ≈ Naïve Bayes >> KNN.
""")

print("=" * 60)
print("  All outputs saved to /outputs/")
print("=" * 60)
