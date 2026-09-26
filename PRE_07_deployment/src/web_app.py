import pickle
from pathlib import Path

import pandas as pd  # type: ignore
from flask import Flask, jsonify, render_template, request  # type: ignore

app = Flask(__name__)
app.config["SECRET_KEY"] = "you-will-never-guess"

FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
]


@app.route("/", methods=["GET", "POST"])
@app.route("/index", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", prediction="")

    data = request.get_json(silent=True)
    if data is None:
        data = request.form

    missing = [key for key in FEATURES if key not in data]
    if missing:
        return f"Faltan campos: {', '.join(missing)}", 400

    try:
        values = {
            key: float(data[key])
            for key in FEATURES
            if key not in ("waterfront", "condition")
        }
        values["waterfront"] = int(
            str(data["waterfront"]).lower() in ("yes", "1", "true")
        )
        values["condition"] = int(data["condition"])

        # Mantiene el mismo orden de características que espera el modelo.
        df = pd.DataFrame([[values[key] for key in FEATURES]], columns=FEATURES)
    except (TypeError, ValueError):
        return "Los valores enviados no son válidos", 400

    model_path = (
        Path(__file__).resolve().parents[1]
        / "submission"
        / "house_predictor.pkl"
    )
    with model_path.open("rb") as file:
        model = pickle.load(file)

    price = float(model.predict(df)[0])

    if request.is_json:
        return jsonify({"prediction": price})

    return render_template("index.html", prediction=price)


if __name__ == "__main__":
    app.run(debug=True)