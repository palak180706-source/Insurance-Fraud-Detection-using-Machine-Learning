"""
retrain_fixed.py  –  Fixed version with class balancing + better fraud detection
Run:  python retrain_fixed.py
"""

import pandas as pd
import numpy as np
import joblib, os
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics         import classification_report, roc_auc_score

# ── UPDATE this path ─────────────────────────────────────
CSV_PATH = r"C:\Users\amitp\Downloads\insurance.csv"
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
# ─────────────────────────────────────────────────────────

print("=" * 55)
print("  Insurance Fraud Detection - Fixed Retraining")
print("=" * 55)

print("\n[1] Loading dataset...")
df = pd.read_csv(CSV_PATH)

# Drop irrelevant columns
DROP_COLS = ["_c39", "policy_number", "policy_bind_date",
             "incident_date", "incident_location", "insured_zip"]
df.drop(columns=DROP_COLS, inplace=True, errors="ignore")

# Replace '?' with NaN
df.replace("?", np.nan, inplace=True)

print("[2] Handling missing values...")
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != "fraud_reported"]
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

num_cols = df.select_dtypes(include="number").columns.tolist()
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# IQR outlier capping
for col in num_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

print("[3] Encoding features...")
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

df["fraud_reported"] = (df["fraud_reported"] == "Y").astype(int)

fraud_count = df["fraud_reported"].sum()
legit_count = len(df) - fraud_count
print(f"    Class distribution → Legitimate: {legit_count}  |  Fraud: {fraud_count}")

X = df.drop(columns=["fraud_reported"])
y = df["fraud_reported"]
feature_names = X.columns.tolist()

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("\n[4] Training models with class_weight='balanced'...")

# ── Model 1: Logistic Regression (balanced) ──────────────
lr = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"   # ← KEY FIX: penalises missing fraud more
)
lr.fit(X_train, y_train)
lr_prob = lr.predict_proba(X_test)[:, 1]
lr_auc  = roc_auc_score(y_test, lr_prob)

# ── Model 2: Random Forest (balanced) ────────────────────
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",  # ← KEY FIX
    max_depth=10,
    min_samples_leaf=2
)
rf.fit(X_train, y_train)
rf_prob = rf.predict_proba(X_test)[:, 1]
rf_auc  = roc_auc_score(y_test, rf_prob)

print(f"\n    Logistic Regression  ROC-AUC: {lr_auc:.4f}")
print(f"    Random Forest        ROC-AUC: {rf_auc:.4f}")

# Pick the better model
if rf_auc >= lr_auc:
    best_model = rf
    best_name  = "Random Forest"
    best_auc   = rf_auc
else:
    best_model = lr
    best_name  = "Logistic Regression"
    best_auc   = lr_auc

print(f"\n    ★ Selected: {best_name}  (AUC = {best_auc:.4f})")

# ── Lower decision threshold for fraud sensitivity ────────
# Default threshold is 0.5 — we lower it to catch more fraud
FRAUD_THRESHOLD = 0.35   # ← if prob > 35% → predict FRAUD

best_prob  = best_model.predict_proba(X_test)[:, 1]
best_pred  = (best_prob >= FRAUD_THRESHOLD).astype(int)

print(f"\n[5] Evaluation at threshold = {FRAUD_THRESHOLD}:")
print(classification_report(y_test, best_pred,
                             target_names=["Legitimate", "Fraud"]))

# ── Save everything ───────────────────────────────────────
print("[6] Saving artefacts...")
joblib.dump(best_model,    os.path.join(OUT_DIR, "best_model.pkl"))
joblib.dump(scaler,        os.path.join(OUT_DIR, "scaler.pkl"))
joblib.dump(label_encoders,os.path.join(OUT_DIR, "label_encoders.pkl"))
joblib.dump(feature_names, os.path.join(OUT_DIR, "feature_names.pkl"))
joblib.dump(FRAUD_THRESHOLD, os.path.join(OUT_DIR, "threshold.pkl"))

print("    ✓ best_model.pkl")
print("    ✓ scaler.pkl")
print("    ✓ label_encoders.pkl")
print("    ✓ feature_names.pkl")
print("    ✓ threshold.pkl")
print("\n✅ Done! Now run:  python app.py")
print("=" * 55)
