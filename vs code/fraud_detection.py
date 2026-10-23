"""Transaction fraud detection: train, evaluate, and predict.

Usage:
    python fraud_detection.py train   --data transactions.csv --model model.joblib
    python fraud_detection.py predict --model model.joblib \
        --json '{"amount": 950, "hour": 2, "distance_from_home": 300, ...}'
"""
import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "is_fraud"
RANDOM_STATE = 42


# ---------- 1. Load ----------
def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns from {path}")
    return df


# ---------- 2. Validate ----------
def validate_data(df):
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")

    # Target labels: must be binary 0/1, no missing
    if df[TARGET].isna().any():
        raise ValueError(f"'{TARGET}' contains {df[TARGET].isna().sum()} missing labels.")
    labels = set(df[TARGET].unique())
    if not labels <= {0, 1}:
        raise ValueError(f"'{TARGET}' must contain only 0/1, found {sorted(str(x) for x in labels)}.")
    if len(labels) < 2:
        raise ValueError("Target has only one class; cannot train a classifier.")
    df[TARGET] = df[TARGET].astype(int)

    # Data types: features must be numeric
    features = df.drop(columns=[TARGET])
    non_numeric = [c for c in features.columns if not pd.api.types.is_numeric_dtype(features[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns found: {non_numeric}")

    # Missing values & duplicates (reported; imputed later inside the pipeline)
    missing = features.isna().sum()
    missing = missing[missing > 0]
    print("\n--- Validation report ---")
    print("Missing values per column:", missing.to_dict() if len(missing) else "none")
    print("Duplicate rows:", int(df.duplicated().sum()))
    print("Class distribution:")
    print(df[TARGET].value_counts().rename({0: "LEGITIMATE (0)", 1: "FRAUD (1)"}).to_string())
    print(f"Fraud rate: {df[TARGET].mean():.2%}")
    return df


# ---------- 3. Split ----------
def split_data(df, test_size=0.25):
    X = df.drop(columns=[TARGET])  # target is never an input feature
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)


# ---------- 4. Models ----------
def build_model(kind="random_forest"):
    if kind == "logistic":
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    else:
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                     min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE)
    # Imputer + scaler are fitted on training data only (no leakage)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


# ---------- 5. Evaluate ----------
def evaluate(name, model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "pr_auc": average_precision_score(y_test, proba),
        "roc_auc": roc_auc_score(y_test, proba) if len(np.unique(proba)) > 1 else 0.5,
        "accuracy": (tp + tn) / cm.sum(),
    }
    print(f"\n=== {name} (threshold={threshold}) ===")
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(cm, index=["actual LEGIT", "actual FRAUD"],
                       columns=["pred LEGIT", "pred FRAUD"]).to_string())
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(classification_report(y_test, pred, target_names=["LEGITIMATE", "FRAUD"], zero_division=0))
    return metrics


# ---------- Train pipeline ----------
def train(data_path, model_path):
    df = validate_data(load_data(data_path))
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"\nTrain: {len(X_train)} rows (fraud {y_train.mean():.2%}) | "
          f"Test: {len(X_test)} rows (fraud {y_test.mean():.2%})")

    # 6. Baseline: always predicts the majority class (LEGITIMATE)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    results = {"Baseline (majority class)": evaluate("Baseline (majority class)", baseline, X_test, y_test)}

    models = {}
    for kind, label in [("logistic", "Logistic Regression"), ("random_forest", "Random Forest")]:
        m = build_model(kind).fit(X_train, y_train)
        models[label] = m
        results[label] = evaluate(label, m, X_test, y_test)

    summary = pd.DataFrame(results).T.round(4)
    print("\n=== Model comparison ===")
    print(summary.to_string())

    # Pick best real model by PR-AUC (more informative than accuracy under imbalance)
    best_name = max(models, key=lambda k: results[k]["pr_auc"])
    print(f"\nBest model by PR-AUC: {best_name}")
    best = models[best_name]
    joblib.dump({"model": best, "features": list(X_train.columns), "name": best_name}, model_path)
    print(f"Saved model to {model_path}")


# ---------- 7. Predict ----------
def predict_transaction(bundle, transaction, threshold=0.5):
    """transaction: dict of feature -> value. Returns predicted class + fraud probability."""
    features = bundle["features"]
    if TARGET in transaction:
        raise ValueError(f"'{TARGET}' must not be supplied as an input feature.")
    missing = [f for f in features if f not in transaction]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    row = pd.DataFrame([{f: transaction[f] for f in features}])
    prob = float(bundle["model"].predict_proba(row)[0, 1])
    return {"prediction": "FRAUD" if prob >= threshold else "LEGITIMATE",
            "fraud_probability": round(prob, 4)}


def main():
    p = argparse.ArgumentParser(description="Transaction fraud detection")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--model", default="model.joblib")
    q = sub.add_parser("predict")
    q.add_argument("--model", default="model.joblib")
    q.add_argument("--json", required=True, help="Transaction features as a JSON object")
    q.add_argument("--threshold", type=float, default=0.5)
    a = p.parse_args()

    if a.cmd == "train":
        train(a.data, a.model)
    else:
        bundle = joblib.load(a.model)
        print(json.dumps(predict_transaction(bundle, json.loads(a.json), a.threshold), indent=2))


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        sys.exit(f"Error: {e}")
