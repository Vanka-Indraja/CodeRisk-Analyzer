from pathlib import Path
import json

from flask import Flask, render_template, request

from src.features import extract_metrics

from src.model import (
    load_model,
    predict_risk,
    review_suggestions
)


app = Flask(__name__)


MODEL_PATH = Path(
    "models/best_model.joblib"
)

METADATA_PATH = Path(
    "models/model_metadata.json"
)


def get_source_code():

    pasted = request.form.get(
        "source_code",
        ""
    ).strip()


    uploaded = request.files.get(
        "source_file"
    )


    if uploaded and uploaded.filename:

        if not uploaded.filename.lower().endswith(
            ".py"
        ):

            raise ValueError(
                "Please upload a Python (.py) file."
            )


        return uploaded.read().decode(
            "utf-8",
            errors="replace"
        )


    return pasted


def load_model_metadata():

    if not METADATA_PATH.exists():

        return {
            "best_model": "Unknown",
            "features": [],
            "best_metrics": {},
            "models_compared": []
        }


    with open(
        METADATA_PATH,
        "r"
    ) as file:

        return json.load(
            file
        )


@app.route(
    "/",
    methods=[
        "GET",
        "POST"
    ]
)
def index():

    if request.method == "GET":

        return render_template(
            "index.html"
        )


    try:

        code = get_source_code()


        if not code:

            raise ValueError(
                "Source code is empty."
            )


        metrics_object = extract_metrics(
            code
        )


        metrics = metrics_object.to_dict()


        model = load_model(
            MODEL_PATH
        )


        prediction = predict_risk(
            model,
            metrics
        )


        suggestions = review_suggestions(
            metrics
        )


        model_metadata = load_model_metadata()


        return render_template(
            "result.html",

            metrics=metrics,

            prediction=prediction,

            suggestions=suggestions,

            code=code,

            model_metadata=model_metadata
        )


    except Exception as error:

        return render_template(
            "index.html",
            error=str(error)
        ), 400


if __name__ == "__main__":

    app.run(
        debug=True
    )