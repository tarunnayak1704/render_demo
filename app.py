# app.py
import os
import pickle
import numpy as np
from flask import Flask, request, render_template, flash

app = Flask(__name__)
app.secret_key = "replace_with_a_random_secret"

# ---------- App config ----------
TITLE = "Package Predictor based on CGPA"
FEATURE_NAMES = ["CGPA (0-10)"]  # single numeric feature
EXPECTED_FEATURES = len(FEATURE_NAMES)

# ---------- Load model (regressor) and optional scaler ----------
MODEL_PATHS_TRY = ["regressor.pkl", "model.pkl"]
SCALER_PATHS_TRY = ["scaler.pkl"]

model = None
for mp in MODEL_PATHS_TRY:
    if os.path.exists(mp):
        with open(mp, "rb") as f:
            model = pickle.load(f)
        break
if model is None:
    raise FileNotFoundError("No model file found. Expected one of: " + ", ".join(MODEL_PATHS_TRY))

scaler = None
for sp in SCALER_PATHS_TRY:
    if os.path.exists(sp):
        with open(sp, "rb") as f:
            scaler = pickle.load(f)
        break

def render_home(prediction_text=None, form_values=None):
    """
    Always render with all variables so Jinja never sees 'Undefined'.
    """
    return render_template(
        "index.html",
        title=TITLE,
        feature_names=FEATURE_NAMES,
        n_features=EXPECTED_FEATURES,
        prediction_text=prediction_text,
        form_values=form_values or {}
    )

@app.route("/", methods=["GET"])
def home():
    return render_home()

@app.route("/predict", methods=["POST"])
def predict():
    # Collect/validate CGPA (0–10 inclusive)
    raw = request.form.get("feature_1", "").strip()

    # Keep what user typed so we can re-fill the form
    form_values = {"feature_1": raw}

    # 1) Numeric check
    try:
        val = float(raw)
    except ValueError:
        flash("Please enter a numeric value for CGPA.")
        return render_home(prediction_text=None, form_values=form_values)

    # 2) Range check
    if not (0.0 <= val <= 10.0):
        flash("CGPA must be between 0 and 10 (inclusive). Please re-enter.")
        return render_home(prediction_text=None, form_values=form_values)

    # Prepare data
    X = np.array([val], dtype=float).reshape(1, -1)

    # If scaler exists, transform
    X_trans = scaler.transform(X) if scaler is not None else X

    # Predict
    try:
        y_pred = model.predict(X_trans)
        pred_val = float(y_pred[0])
        pred_display = f"{pred_val:.4f}"
    except Exception as e:
        flash(f"Error while predicting: {e}")
        return render_home(prediction_text=None, form_values=form_values)

    return render_home(prediction_text=f"Prediction: {pred_display}", form_values=form_values)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render PORT dega
    app.run(host="0.0.0.0", port=port, debug=True)

