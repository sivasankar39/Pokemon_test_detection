from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Paths for artifacts
MODEL_PATH = "model.pkl"
SCALER_PATH = "std_scalar.pkl"
LE_PATH = "label_encoder.pkl"

# Number of distinct type_2 categories used at training
TYPE2_N = 17

model = None
scaler = None
le = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception:
        model = None
if os.path.exists(SCALER_PATH):
    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    except Exception:
        scaler = None
if os.path.exists(LE_PATH):
    try:
        with open(LE_PATH, "rb") as f:
            le = pickle.load(f)
    except Exception:
        le = None


@app.route("/", methods=["GET"]) 
def index():
    # Provide type_2 options as integers 0..16 for demo; replace with real labels if available
    type2_options = list(range(TYPE2_N))
    model_available = model is not None and le is not None
    return render_template("index.html", type2_options=type2_options, model_available=model_available)


@app.route("/predict", methods=["POST"]) 
def predict():
    try:
        height_dm = float(request.form.get("height_dm", 0))
        weight_hg = float(request.form.get("weight_hg", 0))
        type_2 = float(request.form.get("type_2", 0))
        hp = float(request.form.get("hp", 0))
        attack = float(request.form.get("attack", 0))
        defense = float(request.form.get("defense", 0))
    except ValueError:
        return "Invalid input. Please provide numeric values.", 400

    # Build feature vector. We assume training used one-hot encoding for `type_2` with TYPE2_N categories.
    # Feature order used here: [height_dm, weight_hg, hp, attack, defense, type_2_onehot...]
    # Adjust this ordering if your training pipeline used a different order.
    type2_idx = int(type_2)
    if type2_idx < 0 or type2_idx >= TYPE2_N:
        return f"type_2 must be an integer in [0, {TYPE2_N-1}]", 400

    onehot = np.zeros(TYPE2_N, dtype=float)
    onehot[type2_idx] = 1.0

    base_feats = np.array([height_dm, weight_hg, hp, attack, defense], dtype=float)
    features = np.concatenate([base_feats, onehot]).reshape(1, -1)

    if scaler is not None:
        try:
            features = scaler.transform(features)
        except Exception:
            # If scaler fails, continue with raw features
            pass

    if model is None:
        pred = "NO_MODEL"
        probs = None
    else:
        pred_raw = model.predict(features)[0]
        # Map prediction to human label if label encoder is available
        if le is not None:
            try:
                pred = le.inverse_transform([pred_raw])[0]
            except Exception:
                pred = str(pred_raw)
        else:
            # Many sklearn models return numeric class indices or strings in classes_
            if hasattr(model, "classes_"):
                # If model outputs index (0..n-1), map via classes_
                try:
                    pred = model.classes_[int(pred_raw)]
                except Exception:
                    pred = str(pred_raw)
            else:
                pred = str(pred_raw)

        probs = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(features)[0]
                # Determine labels for proba: prefer label encoder, else model.classes_
                if le is not None:
                    labels = le.inverse_transform(np.arange(len(proba)))
                elif hasattr(model, "classes_"):
                    labels = model.classes_
                else:
                    labels = [str(i) for i in range(len(proba))]
                probs = list(zip([str(l) for l in labels], proba.tolist()))
            except Exception:
                probs = None

    return render_template("result.html", prediction=pred, probabilities=probs)


if __name__ == "__main__":
    app.run(debug=True)
