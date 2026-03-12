# app.py
import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

import mlflow.pyfunc   # <- uses your MLflow model

# -----------------------------
# CONFIG
# -----------------------------
# Change this to your actual model name & version from MLflow
# Example: "models:/soil_spectra_best/1"
MODEL_URI = os.getenv("SOIL_MODEL_URI", "models:/soil_spectra_best/1")

# If your model expects specific column names, list them here in order.
# For example, if you used wavelengths 400–2499.5 with step 0.5:
# FEATURE_COLUMNS = [str(w) for w in np.arange(400, 2500, 0.5)]
FEATURE_COLUMNS = None  # keep None if the model is a full sklearn/pyfunc pipeline

# -----------------------------
# LOAD MODEL ONCE AT STARTUP
# -----------------------------
print(f"Loading model from MLflow URI: {MODEL_URI}")
model = mlflow.pyfunc.load_model(MODEL_URI)
print("Model loaded.")

app = Flask(__name__)
app.secret_key = "some-secret-key"  # needed for flash messages


# -----------------------------
# HELPERS
# -----------------------------
def prepare_input_from_csv(file_storage):
    """
    Read an uploaded CSV file and return a 2D np.array ready for model.predict().
    We use only the FIRST ROW for demo, but you can easily change it.
    """
    df = pd.read_csv(file_storage)

    # If specific feature columns are required, select/reorder them
    if FEATURE_COLUMNS is not None:
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns: {missing[:5]} ...")
        df = df[FEATURE_COLUMNS]

    # Use only first sample for demo
    X = df.iloc[0:1].values
    return X, df.iloc[0:1]


def prepare_input_from_text(text):
    """
    Convert a comma-separated list of numbers (spectral reflectance values)
    into a 2D np.array suitable for model.predict().
    """
    try:
        values = [float(x.strip()) for x in text.split(",") if x.strip() != ""]
    except ValueError:
        raise ValueError("All values must be numeric (use commas to separate).")

    X = np.array(values, dtype=float).reshape(1, -1)
    return X


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_type = request.form.get("input_type")

        try:
            if input_type == "csv":
                file = request.files.get("csv_file")
                if not file or file.filename == "":
                    flash("Please upload a CSV file.", "error")
                    return redirect(url_for("index"))

                X, first_row_df = prepare_input_from_csv(file)
                raw_preview = first_row_df.to_html(classes="table table-sm table-bordered")

            elif input_type == "manual":
                raw_text = request.form.get("manual_values", "").strip()
                if not raw_text:
                    flash("Please paste comma-separated values.", "error")
                    return redirect(url_for("index"))

                X = prepare_input_from_text(raw_text)
                raw_preview = None

            else:
                flash("Please select an input method.", "error")
                return redirect(url_for("index"))

            # Run prediction with MLflow model
            y_pred = model.predict(X)

            # mlflow.pyfunc may return numpy array, list, or DataFrame
            if isinstance(y_pred, pd.DataFrame):
                pred_dict = y_pred.iloc[0].to_dict()
            elif isinstance(y_pred, (np.ndarray, list)):
                # adjust labels if your model returns multiple outputs
                labels = ["OC", "pH", "EC"]
                values = np.array(y_pred).ravel().tolist()
                pred_dict = {
                    labels[i] if i < len(labels) else f"target_{i}": v
                    for i, v in enumerate(values)
                }
            else:
                pred_dict = {"prediction": float(y_pred)}

            return render_template(
                "result.html",
                predictions=pred_dict,
                raw_preview=raw_preview,
            )

        except Exception as e:
            print("Error during prediction:", e)
            flash(f"Error during prediction: {e}", "error")
            return redirect(url_for("index"))

    # GET request – just render the form
    return render_template("index.html")


@app.route("/health")
def health():
    return {"status": "ok", "model_uri": MODEL_URI}


if __name__ == "__main__":
    # Run locally
    app.run(host="0.0.0.0", port=5000, debug=True)




