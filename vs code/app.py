"""Fraud Desk: transaction fraud detection web application (Flask + scikit-learn).

Run:   python app.py      then open http://127.0.0.1:5000 (a browser tab opens automatically)

static/index.html is the user interface. It talks to the JSON API below; all machine learning
(validation, stratified split, training, metrics, prediction) runs here in Python via fraud_detection.py.
State is kept in memory for a single user, which is fine for a demo or teaching app.
"""
import contextlib
import io
import os
import threading
import webbrowser

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score

import fraud_detection as fd
from generate_data import generate

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload cap
NAMES = {"base": "Baseline (always legitimate)", "lr": "Logistic regression", "rf": "Random forest"}
STATE = {"df": None, "report": None, "trained": None}
NAN = float("nan")


# ------------------------------------------------------------------ helpers
def num(v):
    """JSON-safe number: NaN -> None, whole floats -> int."""
    v = float(v)
    return None if v != v else (int(v) if v.is_integer() else round(v, 4))


def vals(d):
    return {k: num(v) for k, v in d.items()}


def fmt_tx(tx):
    return ", ".join(f"{k}={'empty' if v is None or v != v else '%g' % v}" for k, v in tx.items())


def parse_threshold(v, default=0.5):
    try:
        return min(max(float(v), 0.01), 0.99)
    except (TypeError, ValueError):
        return default


def need_training():
    return jsonify(ok=False, error="Train the models first."), 409


# ------------------------------------------------------------------ data
def run_validation(df):
    """Validate with fd.validate_data (silenced) and build the summary shown on the page."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            df = fd.validate_data(df)
    except ValueError as e:
        return None, str(e)
    n1 = int(df[fd.TARGET].sum())
    n0 = len(df) - n1
    if min(n0, n1) < 4:
        return None, f"Each class needs at least 4 examples (found {n0} legitimate, {n1} fraud)."
    feats = df.drop(columns=[fd.TARGET])
    if feats.shape[1] == 0:
        return None, "The CSV has no feature columns besides is_fraud."
    miss = feats.isna().sum()
    return df, {"rows": len(df), "n_features": feats.shape[1], "n_fraud": n1, "n_legit": n0,
                "rate": n1 / len(df), "missing": {k: int(v) for k, v in miss.items() if v},
                "duplicates": int(df.duplicated().sum()), "features": list(feats.columns)}


def set_dataset(df):
    df, rep = run_validation(df)
    STATE.update(df=df, report=rep if df is not None else None, trained=None)
    return jsonify(ok=True, report=rep) if df is not None else (jsonify(ok=False, error=rep), 400)


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/sample")
def api_sample():
    return set_dataset(generate())


@app.post("/api/upload")
def api_upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(ok=False, error="Choose a CSV file or use the sample data."), 400
    try:
        df = pd.read_csv(f)
    except Exception as e:  # unreadable / empty / malformed CSV
        return jsonify(ok=False, error=f"Could not read the CSV: {e}"), 400
    return set_dataset(df)


# ------------------------------------------------------------------ modelling
def train_all(df):
    X_train, X_test, y_train, y_test = fd.split_data(df)
    models = {
        "base": DummyClassifier(strategy="most_frequent").fit(X_train, y_train),
        "lr": fd.build_model("logistic").fit(X_train, y_train),
        "rf": fd.build_model("random_forest").fit(X_train, y_train),
    }
    return {"models": models, "features": list(X_train.columns), "y_test": y_test.to_numpy(),
            "probs": {k: m.predict_proba(X_test)[:, 1] for k, m in models.items()},
            "X_train": X_train, "y_train": y_train, "X_test": X_test.reset_index(drop=True),
            "n_train": len(X_train), "n_test": len(X_test)}


def metrics(p, y, t):
    pred = p >= t
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=pr, recall=rc, f1=f1, accuracy=(tp + tn) / len(y))


def profile(tr, cls):
    """Typical transaction of a class: feature means from the training data."""
    m = tr["X_train"][tr["y_train"] == cls].mean()
    return {c: int(round(m[c])) if any(s in c for s in ("hour", "txn", "age")) else round(float(m[c]), 2)
            for c in tr["features"]}


@app.post("/api/train")
def api_train():
    if STATE["df"] is None:
        return jsonify(ok=False, error="Load a valid dataset first."), 400
    tr = STATE["trained"] = train_all(STATE["df"])
    return jsonify(ok=True, n_train=tr["n_train"], n_test=tr["n_test"], features=tr["features"],
                   medians={k: num(v) for k, v in tr["X_train"].median().items()},
                   profiles={"0": profile(tr, 0), "1": profile(tr, 1)})


def held_out_cases(tr, model, t, per=3):
    """Real rows from the held-out test split, grouped by outcome, with their true labels."""
    p, y, X = tr["probs"][model], tr["y_test"], tr["X_test"]
    pred = p >= t
    groups = [("Caught fraud", pred & (y == 1)), ("Missed fraud", ~pred & (y == 1)),
              ("False alarm", pred & (y == 0)), ("Correct legitimate", ~pred & (y == 0))]
    out = []
    for label, mask in groups:
        pos = np.flatnonzero(mask)
        pos = pos[np.argsort(-p[pos], kind="stable")]  # most to least fraud-like
        pick = pos[np.unique(np.linspace(0, len(pos) - 1, min(per, len(pos))).round().astype(int))] if len(pos) else []
        rows = []
        for i in pick:
            v = vals(X.iloc[int(i)].to_dict())
            rows.append(dict(inputs=fmt_tx(v), values=v, actual="FRAUD" if y[i] else "LEGITIMATE",
                             predicted="FRAUD" if pred[i] else "LEGITIMATE", prob=round(float(p[i]), 3)))
        out.append(dict(label=label, total=int(mask.sum()), good=label in ("Caught fraud", "Correct legitimate"), rows=rows))
    return out


@app.get("/api/results")
def api_results():
    tr = STATE["trained"]
    if not tr:
        return need_training()
    t = parse_threshold(request.args.get("threshold"))
    sel = request.args.get("model") if request.args.get("model") in NAMES else "rf"
    hk = request.args.get("held") if request.args.get("held") in ("lr", "rf") else "rf"
    y, P = tr["y_test"], tr["probs"]
    rows = []
    for k, nm in NAMES.items():
        m = metrics(P[k], y, t)
        rows.append(dict(key=k, name=nm, precision=m["precision"], recall=m["recall"], f1=m["f1"],
                         accuracy=m["accuracy"],
                         pr_auc=float(y.mean()) if k == "base" else float(average_precision_score(y, P[k]))))
    real = sel if sel != "base" else "rf"
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    ms = [metrics(P[real], y, g) for g in grid]
    curve = {"t": grid.tolist(), **{k: [m[k] for m in ms] for k in ("precision", "recall", "f1")}}
    best = float(grid[int(np.argmax([m["f1"] for m in ms]))])
    cm = {k: metrics(P[sel], y, t)[k] for k in ("tn", "fp", "fn", "tp")}
    return jsonify(ok=True, rows=rows, cm=cm, curve=curve, best_t=best, n_test=tr["n_test"],
                   held=held_out_cases(tr, hk, t))


# ------------------------------------------------------------------ prediction
@app.post("/api/predict")
def api_predict():
    """Body: {"values": {feature: number or ""}, "model": "rf"|"lr", "threshold": 0.5}"""
    tr = STATE["trained"]
    if not tr:
        return need_training()
    body = request.get_json(silent=True) or {}
    model = body.get("model", "rf")
    if model not in ("lr", "rf"):
        return jsonify(ok=False, error="model must be 'rf' or 'lr'"), 400
    t = parse_threshold(body.get("threshold"))
    tx, bad = {}, []
    for f, v in (body.get("values") or {}).items():
        try:
            tx[f] = NAN if v is None or str(v).strip() == "" else float(v)  # empty -> imputed by the pipeline
        except (TypeError, ValueError):
            bad.append(f)
    if bad:
        return jsonify(ok=False, error="Not a number: " + ", ".join(bad)), 400
    try:
        res = fd.predict_transaction({"model": tr["models"][model], "features": tr["features"]}, tx, t)
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 400
    return jsonify(ok=True, model=NAMES[model], threshold=t, empty=sum(v != v for v in tx.values()), **res)


# ------------------------------------------------------------------ test cases
TEST_CASES = [
    ("Small daytime purchase near home", dict(amount=25, hour=14, distance_from_home=3, txn_last_24h=1, account_age_days=2500, avg_amount_30d=40, amount_ratio=0.62), "LEGITIMATE"),
    ("Regular grocery, old account", dict(amount=60, hour=18, distance_from_home=8, txn_last_24h=2, account_age_days=1800, avg_amount_30d=55, amount_ratio=1.09), "LEGITIMATE"),
    ("Normal weekday lunch", dict(amount=18, hour=12, distance_from_home=2, txn_last_24h=1, account_age_days=900, avg_amount_30d=30, amount_ratio=0.6), "LEGITIMATE"),
    ("Large night transaction, far away, new account", dict(amount=1200, hour=2, distance_from_home=300, txn_last_24h=7, account_age_days=40, avg_amount_30d=45, amount_ratio=26.7), "FRAUD"),
    ("Burst of transactions at 3am", dict(amount=800, hour=3, distance_from_home=150, txn_last_24h=9, account_age_days=60, avg_amount_30d=50, amount_ratio=16), "FRAUD"),
    ("Very new account, huge amount", dict(amount=2500, hour=1, distance_from_home=400, txn_last_24h=5, account_age_days=5, avg_amount_30d=35, amount_ratio=71), "FRAUD"),
    ("Borderline: late night, normal amount", dict(amount=40, hour=23, distance_from_home=10, txn_last_24h=2, account_age_days=700, avg_amount_30d=45, amount_ratio=0.9), "LEGITIMATE"),
    ("Missing value (amount left empty)", dict(amount=NAN, hour=2, distance_from_home=250, txn_last_24h=6, account_age_days=30, avg_amount_30d=45, amount_ratio=20), "FRAUD"),
]
SAMPLE_FEATURES = [c for c in generate(100).columns if c != fd.TARGET]


def features_match(tr):
    return bool(tr) and set(tr["features"]) == set(SAMPLE_FEATURES)


def run_prediction_tests(tr, model, t):
    bundle = {"model": tr["models"][model], "features": tr["features"]}
    out = []
    for _, tx, exp in TEST_CASES:
        r = fd.predict_transaction(bundle, tx, t)
        out.append(dict(predicted=r["prediction"], prob=r["fraud_probability"], passed=r["prediction"] == exp))
    return out


def run_validation_tests(tr):
    base = generate(2000, seed=1)
    bundle = {"model": tr["models"]["rf"], "features": tr["features"]}

    def rejected(fn):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fn()
        except ValueError as e:
            return True, str(e)
        return False, "Not rejected"

    good = TEST_CASES[0][1]
    checks = [
        ("Missing is_fraud column", lambda: fd.validate_data(base.drop(columns=fd.TARGET))),
        ("Label outside 0/1 (value 2)", lambda: fd.validate_data(base.assign(is_fraud=base.is_fraud.replace({1: 2})))),
        ("Missing labels (NaN)", lambda: fd.validate_data(base.assign(is_fraud=base.is_fraud.astype(float).where(base.index > 5)))),
        ("Only one class in target", lambda: fd.validate_data(base.assign(is_fraud=0))),
        ("Non-numeric feature column", lambda: fd.validate_data(base.assign(merchant="shop"))),
        ("is_fraud sent as a prediction input", lambda: fd.predict_transaction(bundle, {**good, fd.TARGET: 1})),
        ("Feature missing from a prediction", lambda: fd.predict_transaction(bundle, {"amount": 5})),
    ]
    out = []
    for name, fn in checks:
        ok, msg = rejected(fn)
        out.append(dict(name=name, expected="Rejected with an error", actual=msg, passed=ok))
    with contextlib.redirect_stdout(io.StringIO()):
        valid = fd.validate_data(base.copy())
    X_tr, X_te, y_tr, y_te = fd.split_data(valid)
    out.append(dict(name="Stratified split keeps the fraud rate", expected="Train and test rates match",
                    actual=f"{y_tr.mean():.2%} vs {y_te.mean():.2%}", passed=bool(abs(y_tr.mean() - y_te.mean()) < 0.005)))
    out.append(dict(name="Target is not a model input", expected="is_fraud absent from features",
                    actual="absent" if fd.TARGET not in X_tr.columns else "present", passed=fd.TARGET not in X_tr.columns))
    return out


@app.get("/api/testcases")
def api_testcases():
    return jsonify(ok=features_match(STATE["trained"]),
                   cases=[dict(name=n, inputs=fmt_tx(vals(tx)), expected=e, values=vals(tx)) for n, tx, e in TEST_CASES])


@app.post("/api/testcases/run")
def api_testcases_run():
    tr = STATE["trained"]
    if not tr:
        return need_training()
    body = request.get_json(silent=True) or {}
    model = body.get("model") if body.get("model") in ("lr", "rf") else "rf"
    t = parse_threshold(body.get("threshold"))
    pred = run_prediction_tests(tr, model, t) if features_match(tr) else []
    return jsonify(ok=True, model=NAMES[model], threshold=t, pred=pred, val=run_validation_tests(tr))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    if not os.environ.get("NO_BROWSER"):
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(host="127.0.0.1", port=port, debug=False)
