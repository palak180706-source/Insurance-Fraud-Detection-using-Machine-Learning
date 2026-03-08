"""
============================================================
  Insurance Fraud Detection – Complete ML Pipeline
============================================================
Steps:
  1. Import Libraries
  2. Read Dataset
  3. Prepare Dataset  (missing values, outliers)
  4. EDA             (visual, multivariate, encoding, scaling)
  5. Build Models    (DT, RF, KNN, LR, NB, SVM)
  6. Test Models
  7. Compare Models
  8. Save Artefacts  (for Flask deployment)
============================================================
"""

# ─────────────────────────────────────────────
# STEP 1 – IMPORT LIBRARIES
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import os, joblib
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.tree             import DecisionTreeClassifier
from sklearn.ensemble         import RandomForestClassifier
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.naive_bayes      import GaussianNB
from sklearn.svm              import SVC
from sklearn.metrics          import (accuracy_score, classification_report,
                                       confusion_matrix, roc_auc_score,
                                       ConfusionMatrixDisplay, roc_curve)

print("=" * 60)
print("  STEP 1 : Libraries imported successfully ✓")
print("=" * 60)

OUTPUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")


# ─────────────────────────────────────────────
# STEP 2 – READ DATASET
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 2 : Reading Dataset")
print("=" * 60)

df = pd.read_csv("/mnt/user-data/uploads/insurance.csv")

# Drop completely-empty junk column
df.drop(columns=["_c39"], inplace=True, errors="ignore")

print(f"  Shape          : {df.shape}")
print(f"  Columns        : {list(df.columns)}")
print(f"\n  Target Distribution:\n{df['fraud_reported'].value_counts()}")
print(f"\n  Data Types:\n{df.dtypes.value_counts()}")
print(f"\n  First 3 rows:\n{df.head(3).to_string()}")


# ─────────────────────────────────────────────
# STEP 3 – PREPARE DATASET
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 3 : Data Preparation")
print("=" * 60)

# ── 3a. Drop columns that leak or are not predictive ──────
DROP_COLS = [
    "policy_number",       # unique identifier
    "policy_bind_date",    # date string – not in scope
    "incident_date",       # date string – not in scope
    "incident_location",   # free-text address
    "insured_zip",         # high-cardinality numeric zip
]
df.drop(columns=DROP_COLS, inplace=True, errors="ignore")
print(f"  Dropped irrelevant columns: {DROP_COLS}")

# ── 3b. Replace '?' with NaN ──────────────────────────────
df.replace("?", np.nan, inplace=True)
print(f"\n  Missing values BEFORE imputation:\n{df.isnull().sum()[df.isnull().sum()>0]}")

# ── 3c. Impute missing values ─────────────────────────────
#   Categorical → mode imputation
cat_missing = [c for c in df.select_dtypes(include=["object","str"]).columns
               if df[c].isnull().any()]
for col in cat_missing:
    mode_val = df[col].mode()[0]
    df[col].fillna(mode_val, inplace=True)
    print(f"    {col:<30} → filled with mode = '{mode_val}'")

#   Numerical → median imputation
num_missing = [c for c in df.select_dtypes(include="number").columns
               if df[c].isnull().any()]
for col in num_missing:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
    print(f"    {col:<30} → filled with median = {median_val}")

print(f"\n  Missing values AFTER imputation: {df.isnull().sum().sum()}")

# ── 3d. Outlier handling (IQR capping on numeric cols) ────
num_cols = df.select_dtypes(include="number").columns.tolist()
outlier_report = {}
for col in num_cols:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo  = Q1 - 1.5 * IQR
    hi  = Q3 + 1.5 * IQR
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    if n_out:
        df[col] = df[col].clip(lo, hi)
        outlier_report[col] = n_out
print(f"\n  Outliers capped (IQR method):")
for k, v in outlier_report.items():
    print(f"    {k:<35} → {v} outliers capped")

print("\n  Data preparation complete ✓")


# ─────────────────────────────────────────────
# STEP 4 – EDA
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 4 : Exploratory Data Analysis")
print("=" * 60)

# ── 4a. Target distribution ───────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["fraud_reported"].value_counts()
bars   = ax.bar(counts.index, counts.values,
                color=["#4C72B0","#DD8452"], edgecolor="white", width=0.5)
ax.bar_label(bars, padding=3, fontsize=11, fontweight="bold")
ax.set_title("Fraud vs Legitimate Claims", fontsize=13, fontweight="bold")
ax.set_xlabel("Fraud Reported"); ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_target_distribution.png", dpi=150)
plt.close()
print("  [Plot 1] Target distribution saved.")

# ── 4b. Numeric feature distributions ─────────────────────
plot_num = ["age","months_as_customer","total_claim_amount",
            "injury_claim","property_claim","vehicle_claim",
            "policy_annual_premium","incident_hour_of_the_day"]
plot_num = [c for c in plot_num if c in df.columns]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(plot_num):
    axes[i].hist(df[col], bins=25, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[i].set_title(col, fontsize=9, fontweight="bold")
    axes[i].set_ylabel("Frequency")
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Numeric Feature Distributions", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_numeric_distributions.png", dpi=150)
plt.close()
print("  [Plot 2] Numeric distributions saved.")

# ── 4c. Box-plots (fraud vs legitimate) ───────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(plot_num):
    df.boxplot(column=col, by="fraud_reported", ax=axes[i],
               boxprops=dict(color="#4C72B0"),
               medianprops=dict(color="#DD8452", linewidth=2))
    axes[i].set_title(col, fontsize=9, fontweight="bold")
    axes[i].set_xlabel("Fraud Reported")
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Feature Distribution by Fraud Label", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_boxplots_by_fraud.png", dpi=150)
plt.close()
print("  [Plot 3] Box-plots by fraud label saved.")

# ── 4d. Categorical feature counts ────────────────────────
cat_cols = ["incident_type","incident_severity","authorities_contacted",
            "insured_education_level","insured_occupation",
            "collision_type","property_damage","police_report_available"]
cat_cols = [c for c in cat_cols if c in df.columns]

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    order = df[col].value_counts().index
    sns.countplot(data=df, y=col, hue="fraud_reported", order=order,
                  ax=axes[i], palette=["#4C72B0","#DD8452"])
    axes[i].set_title(col, fontsize=9, fontweight="bold")
    axes[i].set_xlabel("Count"); axes[i].set_ylabel("")
    axes[i].legend(title="Fraud", fontsize=7)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Categorical Features vs Fraud", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_categorical_vs_fraud.png", dpi=150)
plt.close()
print("  [Plot 4] Categorical feature plots saved.")

# ── 4e. Correlation heat-map (numeric only) ───────────────
corr_df = df.select_dtypes(include="number")
fig, ax  = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_df.corr(), dtype=bool))
sns.heatmap(corr_df.corr(), mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5,
            annot_kws={"size": 7}, ax=ax)
ax.set_title("Numeric Feature Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_correlation_heatmap.png", dpi=150)
plt.close()
print("  [Plot 5] Correlation heatmap saved.")

# ── 4f. Scatter – total_claim vs injury_claim colored by fraud ──
fig, ax = plt.subplots(figsize=(7, 5))
for label, grp in df.groupby("fraud_reported"):
    ax.scatter(grp["total_claim_amount"], grp["injury_claim"],
               alpha=0.4, label=f"Fraud={label}", s=20)
ax.set_xlabel("Total Claim Amount"); ax.set_ylabel("Injury Claim")
ax.set_title("Total Claim vs Injury Claim (by Fraud)", fontsize=11, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_scatter_claim_vs_injury.png", dpi=150)
plt.close()
print("  [Plot 6] Scatter plot saved.")

# ── 4g. Encode categorical features ───────────────────────
print("\n  Encoding categorical features …")
label_encoders = {}
encode_cols = df.select_dtypes(include=["object","str"]).columns.tolist()
encode_cols = [c for c in encode_cols if c != "fraud_reported"]

for col in encode_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Encode target
df["fraud_reported"] = (df["fraud_reported"] == "Y").astype(int)
print(f"  Encoded {len(encode_cols)} categorical columns.")

# Save encoders
joblib.dump(label_encoders, f"{OUTPUT_DIR}/label_encoders.pkl")
print("  Label encoders saved → label_encoders.pkl")

# ── 4h. Feature / target split ────────────────────────────
X = df.drop(columns=["fraud_reported"])
y = df["fraud_reported"]
feature_names = X.columns.tolist()

# ── 4i. Scale features ────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")
print("  StandardScaler fitted and saved → scaler.pkl")

print("\n  EDA complete ✓")


# ─────────────────────────────────────────────
# STEP 5 – BUILD MODELS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 5 : Building Models")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train size : {X_train.shape}   Test size : {X_test.shape}")

models = {
    "Decision Tree"      : DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN"                : KNeighborsClassifier(n_neighbors=7),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes"        : GaussianNB(),
    "SVM"                : SVC(kernel="rbf", probability=True, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    model.fit(X_train, y_train)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"  {name:<22} → CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("\n  All models trained ✓")


# ─────────────────────────────────────────────
# STEP 6 – TEST MODELS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 6 : Testing Models on Hold-out Test Set")
print("=" * 60)

results = {}
for name, model in models.items():
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    results[name] = {"Accuracy": acc, "ROC-AUC": roc_auc, "y_pred": y_pred, "y_prob": y_prob}

    print(f"\n  ── {name} ──")
    print(f"     Accuracy : {acc:.4f}   ROC-AUC : {roc_auc:.4f}")
    print(classification_report(y_test, y_pred,
                                 target_names=["Legitimate","Fraud"],
                                 digits=3))

# ── Confusion matrices ────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()
for i, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legit","Fraud"])
    disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
    axes[i].set_title(name, fontsize=11, fontweight="bold")
plt.suptitle("Confusion Matrices – All Models", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_confusion_matrices.png", dpi=150)
plt.close()
print("\n  [Plot 7] Confusion matrices saved.")

# ── ROC curves ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
colors  = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860"]
for (name, res), col in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['ROC-AUC']:.3f})", color=col, lw=2)
ax.plot([0,1],[0,1],"k--", lw=1, label="Random Classifier")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves – All Models", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_roc_curves.png", dpi=150)
plt.close()
print("  [Plot 8] ROC curves saved.")


# ─────────────────────────────────────────────
# STEP 7 – COMPARE MODELS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 7 : Model Comparison")
print("=" * 60)

comparison = pd.DataFrame({
    name: {"Accuracy": res["Accuracy"], "ROC-AUC": res["ROC-AUC"]}
    for name, res in results.items()
}).T.sort_values("ROC-AUC", ascending=False)

print(f"\n{comparison.to_string()}")

# Bar chart comparison
fig, ax = plt.subplots(figsize=(11, 6))
x      = np.arange(len(comparison))
width  = 0.38
bars1  = ax.bar(x - width/2, comparison["Accuracy"], width,
                label="Accuracy", color="#4C72B0", edgecolor="white")
bars2  = ax.bar(x + width/2, comparison["ROC-AUC"], width,
                label="ROC-AUC",  color="#DD8452", edgecolor="white")
ax.bar_label(bars1, fmt="%.3f", fontsize=8, padding=2)
ax.bar_label(bars2, fmt="%.3f", fontsize=8, padding=2)
ax.set_xticks(x); ax.set_xticklabels(comparison.index, rotation=20, ha="right", fontsize=10)
ax.set_ylim(0, 1.15); ax.set_ylabel("Score"); ax.legend()
ax.set_title("Model Comparison – Accuracy & ROC-AUC", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_model_comparison.png", dpi=150)
plt.close()
print("\n  [Plot 9] Model comparison chart saved.")

# ── Feature importance (Random Forest) ───────────────────
rf_model  = models["Random Forest"]
feat_imp  = pd.Series(rf_model.feature_importances_, index=feature_names)
feat_imp  = feat_imp.sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(9, 6))
feat_imp.sort_values().plot(kind="barh", color="#4C72B0", edgecolor="white", ax=ax)
ax.set_title("Top 15 Feature Importances – Random Forest", fontsize=12, fontweight="bold")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_feature_importances.png", dpi=150)
plt.close()
print("  [Plot 10] Feature importances saved.")

# ── Pick best model by ROC-AUC ────────────────────────────
best_name  = comparison["ROC-AUC"].idxmax()
best_model = models[best_name]
print(f"\n  ★ Best model : {best_name} (ROC-AUC = {comparison.loc[best_name,'ROC-AUC']:.4f})")

joblib.dump(best_model, f"{OUTPUT_DIR}/best_model.pkl")
joblib.dump(feature_names, f"{OUTPUT_DIR}/feature_names.pkl")
print(f"  Best model saved → best_model.pkl")
print(f"  Feature names saved → feature_names.pkl")

print("\n  Model comparison complete ✓")
print("\n" + "=" * 60)
print("  All steps completed successfully! ✓")
print("=" * 60)
