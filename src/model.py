from pathlib import Path
import joblib
import pandas as pd


FEATURES = [
    "loc",
    "functions",
    "loops",
    "conditionals",
    "complexity",
    "nesting"
]


def load_model(model_path):
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run python train_model.py first."
        )

    return joblib.load(path)


def predict_risk(model, metrics):
    x = pd.DataFrame(
        [[metrics[name] for name in FEATURES]],
        columns=FEATURES
    )

    label = int(model.predict(x)[0])

    probability = None

    if hasattr(model, "predict_proba"):
        probability = float(
            model.predict_proba(x)[0][1]
        )

    if probability is None:
        risk_level = "High" if label == 1 else "Low"

    elif probability >= 0.70:
        risk_level = "High"

    elif probability >= 0.35:
        risk_level = "Medium"

    else:
        risk_level = "Low"

    return {
        "predicted_defect_prone": bool(label),
        "probability": probability,
        "risk_level": risk_level
    }


def review_suggestions(metrics):
    suggestions = []

    if metrics["complexity"] >= 15:
        suggestions.append(
            "Reduce code complexity by splitting complex logic into smaller functions."
        )

    if metrics["nesting"] >= 4:
        suggestions.append(
            "Reduce deep nesting using early returns or simpler conditions."
        )

    if metrics["conditionals"] >= 12:
        suggestions.append(
            "Simplify conditional logic and reduce unnecessary decision branches."
        )

    if metrics["loops"] >= 8:
        suggestions.append(
            "Review loop-heavy sections for simpler and more efficient logic."
        )

    if metrics["loc"] >= 200:
        suggestions.append(
            "Split large source files into smaller modules."
        )

    if metrics["functions"] >= 10:
        suggestions.append(
            "Review large modules and separate responsibilities into reusable functions."
        )

    if not suggestions:
        suggestions.append(
            "No major structural warning detected. Continue normal manual code review."
        )

    return suggestions