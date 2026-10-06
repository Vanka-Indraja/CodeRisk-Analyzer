import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


FEATURES = [
    "loc",
    "functions",
    "loops",
    "conditionals",
    "complexity",
    "nesting"
]


# -----------------------------
# Load Dataset
# -----------------------------

data = pd.read_csv(
    "data/defect_dataset.csv"
)

X = data[FEATURES]

y = data["defect"]


# -----------------------------
# Train / Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# -----------------------------
# Candidate Models
# -----------------------------

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    )
}


os.makedirs(
    "models",
    exist_ok=True
)


results = []

best_model = None
best_model_name = None
best_result = None

best_f1 = -1


# -----------------------------
# Train Models
# -----------------------------

for model_name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )


    result = {

        "model": model_name,

        "accuracy": round(
            float(accuracy),
            4
        ),

        "precision": round(
            float(precision),
            4
        ),

        "recall": round(
            float(recall),
            4
        ),

        "f1_score": round(
            float(f1),
            4
        )
    }


    results.append(
        result
    )


    print(
        "\n--------------------------"
    )

    print(
        model_name
    )

    print(
        "--------------------------"
    )

    print(
        "Accuracy :",
        round(accuracy, 3)
    )

    print(
        "Precision:",
        round(precision, 3)
    )

    print(
        "Recall   :",
        round(recall, 3)
    )

    print(
        "F1 Score :",
        round(f1, 3)
    )


    filename = (
        model_name
        .lower()
        .replace(" ", "_")
        + ".joblib"
    )


    joblib.dump(
        model,
        os.path.join(
            "models",
            filename
        )
    )


    if f1 > best_f1:

        best_f1 = f1

        best_model = model

        best_model_name = model_name

        best_result = result


# -----------------------------
# Save Comparison CSV
# -----------------------------

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)


# -----------------------------
# Save Best Model
# -----------------------------

joblib.dump(
    best_model,
    "models/best_model.joblib"
)


# -----------------------------
# Save Model Metadata
# -----------------------------

metadata = {

    "best_model": best_model_name,

    "features": FEATURES,

    "best_metrics": best_result,

    "models_compared": [
        model["model"]
        for model in results
    ],

    "all_results": results,

    "training_samples": int(
        len(X_train)
    ),

    "testing_samples": int(
        len(X_test)
    )
}


with open(
    "models/model_metadata.json",
    "w"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


print(
    "\n=========================="
)

print(
    "BEST MODEL"
)

print(
    "=========================="
)

print(
    best_model_name
)

print(
    "F1 Score:",
    round(best_f1, 3)
)

print(
    "\nModel metadata saved."
)

print(
    "Models saved successfully."
)