"""Train the initial Project 136 classifier.

This starter uses a deterministic DEMO dataset so the whole application can run
from scratch. Replace `build_demo_dataset()` with a real labelled defect dataset
before reporting academic results.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from src.model import FEATURES

RANDOM_STATE = 42


def build_demo_dataset(n=1200):
    rng = np.random.default_rng(RANDOM_STATE)
    loc = rng.integers(10, 500, n)
    functions = rng.integers(0, 35, n)
    loops = rng.integers(0, 20, n)
    conditionals = rng.integers(0, 30, n)
    complexity = np.maximum(1, loops + conditionals + rng.integers(1, 8, n))
    nesting = rng.integers(0, 9, n)

    # Synthetic relationship for an executable prototype only.
    score = (
        0.004 * loc
        + 0.11 * loops
        + 0.09 * conditionals
        + 0.08 * complexity
        + 0.28 * nesting
        - 4.8
        + rng.normal(0, 1.1, n)
    )
    probability = 1 / (1 + np.exp(-score))
    defect = (rng.random(n) < probability).astype(int)

    return pd.DataFrame({
        "loc": loc,
        "functions": functions,
        "loops": loops,
        "conditionals": conditionals,
        "complexity": complexity,
        "nesting": nesting,
        "defect": defect,
    })


def main():
    df = build_demo_dataset()
    X = df[FEATURES]
    y = df["defect"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=250, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, class_weight="balanced", max_depth=8
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }

    results = []
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        trained[name] = model
        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
        })

    result_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    best_name = result_df.iloc[0]["model"]
    best_model = trained[best_name]

    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_model, "models/defect_model.joblib")
    result_df.to_csv("models/demo_evaluation.csv", index=False)

    print("DEMO evaluation only — do not report these as final project results.")
    print(result_df.to_string(index=False))
    print(f"\nSaved selected demo model: {best_name} -> models/defect_model.joblib")


if __name__ == "__main__":
    main()
