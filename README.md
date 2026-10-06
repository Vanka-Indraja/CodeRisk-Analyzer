# Project 136 — Automated Code Review and Defect Prediction

A from-scratch MVP implementing the architecture in the project review plan:
source code -> preprocessing -> feature extraction -> ML model -> defect-risk prediction -> risk classification -> review report.

## Current first version

- Python source-code upload/paste
- AST-based code metric extraction
- Metrics: LOC, functions, loops, conditionals, complexity, nesting
- ML candidates: Random Forest, Decision Tree, Logistic Regression
- Flask backend
- HTML/CSS web interface
- Low / Medium / High risk output
- Developer-oriented structural review suggestions

## Important academic note

`train_model.py` currently generates a deterministic **synthetic demo dataset** only so the complete application can run immediately. It is not a valid final evaluation dataset. Replace it with a real labelled defect dataset before reporting experimental results.

## Setup

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python train_model.py
python app.py
```

Open: http://127.0.0.1:5000

## Project structure

```text
project136_mvp/
├── app.py
├── train_model.py
├── requirements.txt
├── sample_code.py
├── src/
│   ├── features.py
│   └── model.py
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   └── style.css
├── data/
│   └── README.md
├── models/
└── tests/
    └── test_features.py
```

## Next development phases

1. Replace demo data with a real public defect dataset and map its metrics.
2. Add preprocessing, imbalance handling, cross-validation and confusion matrix.
3. Add model-comparison dashboard and feature importance.
4. Add MySQL analysis history.
5. Add authentication only if required.
6. Add exportable analysis report.
7. Add test cases and final deployment configuration.
