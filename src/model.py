from pathlib import Path
import joblib
import pandas as pd

FEATURES = ["loc", "functions", "loops", "conditionals", "complexity", "nesting"]


def load_model(model_path: str | Path):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python train_model.py` first."
        )
    return joblib.load(path)


def predict_risk(model, metrics: dict) -> dict:
    x = pd.DataFrame([[metrics[name] for name in FEATURES]], columns=FEATURES)
    label = int(model.predict(x)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(x)[0][1])

    if probability is None:
        risk_level = "High" if label == 1 else "Low"
    elif probability >= 0.70:
        risk_level = "High"
    elif probability >= 0.40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "predicted_defect_prone": bool(label),
        "probability": probability,
        "risk_level": risk_level,
    }


def review_suggestions(metrics: dict) -> list[str]:
    suggestions = []
    if metrics["complexity"] >= 10:
        suggestions.append("Reduce cyclomatic complexity by splitting large decision-heavy functions.")
    if metrics["nesting"] >= 4:
        suggestions.append("Reduce deep nesting using guard clauses or smaller helper functions.")
    if metrics["loc"] >= 200:
        suggestions.append("Break large modules into smaller, focused components.")
    if metrics["functions"] == 0 and metrics["loc"] > 40:
        suggestions.append("Improve modularity by grouping repeated logic into functions.")
    if metrics["loops"] >= 8:
        suggestions.append("Review repeated loops for simpler data-processing patterns.")
    if not suggestions:
        suggestions.append("No major structural warning from the selected metrics; continue normal manual review.")
    return suggestions
