"""
app.py  –  Insurance Fraud Detection Flask API (Fixed Version)
Run:  python app.py
Open: http://127.0.0.1:5000
"""

from flask import Flask, request, render_template, jsonify
import joblib, numpy as np, os

app = Flask(__name__)

BASE = os.path.dirname(__file__)

model         = joblib.load(os.path.join(BASE, "best_model.pkl"))
scaler        = joblib.load(os.path.join(BASE, "scaler.pkl"))
le_dict       = joblib.load(os.path.join(BASE, "label_encoders.pkl"))
feature_names = joblib.load(os.path.join(BASE, "feature_names.pkl"))

try:
    THRESHOLD = joblib.load(os.path.join(BASE, "threshold.pkl"))
except:
    THRESHOLD = 0.35

print(f"  Model loaded | Fraud threshold = {THRESHOLD}")


def safe_encode(le, value):
    try:
        return int(le.transform([str(value)])[0])
    except ValueError:
        return 0


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        row = {}
        for feat in feature_names:
            val = data.get(feat, 0)
            if feat in le_dict:
                val = safe_encode(le_dict[feat], val)
            else:
                try:
                    val = float(val) if val != "" else 0.0
                except:
                    val = 0.0
            row[feat] = val

        X_raw    = np.array([[row[f] for f in feature_names]])
        X_scaled = scaler.transform(X_raw)
        prob     = float(model.predict_proba(X_scaled)[0][1])

        # Use lowered threshold for better fraud sensitivity
        pred = 1 if prob >= THRESHOLD else 0

        return jsonify({
            "prediction" : "FRAUD"      if pred == 1 else "LEGITIMATE",
            "probability": round(prob * 100, 2),
            "risk_level" : "HIGH"       if prob >= 0.65
                           else ("MEDIUM" if prob >= 0.35 else "LOW"),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    print("=" * 50)
    print("  Insurance Fraud Detection API")
    print("  Running at http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
