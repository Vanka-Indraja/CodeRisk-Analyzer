from pathlib import Path
from flask import Flask, render_template, request

from src.features import extract_metrics
from src.model import load_model, predict_risk, review_suggestions

app = Flask(__name__)
MODEL_PATH = Path("models/defect_model.joblib")


def get_source_code():
    pasted = request.form.get("source_code", "").strip()
    uploaded = request.files.get("source_file")

    if uploaded and uploaded.filename:
        if not uploaded.filename.lower().endswith(".py"):
            raise ValueError("Please upload a Python (.py) file for this first version.")
        return uploaded.read().decode("utf-8", errors="replace")
    return pasted


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    try:
        code = get_source_code()
        metrics_obj = extract_metrics(code)
        metrics = metrics_obj.to_dict()
        model = load_model(MODEL_PATH)
        prediction = predict_risk(model, metrics)
        suggestions = review_suggestions(metrics)
        return render_template(
            "result.html",
            metrics=metrics,
            prediction=prediction,
            suggestions=suggestions,
            code=code,
        )
    except Exception as exc:
        return render_template("index.html", error=str(exc)), 400


if __name__ == "__main__":
    app.run(debug=True)
