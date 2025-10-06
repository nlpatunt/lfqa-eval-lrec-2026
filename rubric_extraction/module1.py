# train_pairwise_lr.py
# Pairwise preference learning with fixed feature list and logistic regression (A − B diffs).

import json, pathlib
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

# -----------------------------
# Fixed feature list (in your requested order)
# -----------------------------
FEATURES: List[str] = [
    "completeness",
    "well_structured",
    "relevance",
    "veracity",
    "specificity",
    "example",
    "conciseness",
    "easy_to_understand",
    "grammar",
]

# Key patterns for dataset fields
KEY_PATTERNS = [
    "{feat}_score_answer_{which}",
    "{feat}_answer_{which}",
]

# -----------------------------
# IO utilities
# -----------------------------
def load_records(path: str) -> List[Dict]:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    text = p.read_text(encoding="utf-8", errors="ignore").strip()
    # Try JSON array first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    # Fallback to JSONL
    recs: List[Dict] = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs

# -----------------------------
# Resolve feature keys
# -----------------------------
def resolve_pair_keys(rec: Dict, feature: str) -> Tuple[str, str] | None:
    for pat in KEY_PATTERNS:
        k1 = pat.format(feat=feature, which="1")
        k2 = pat.format(feat=feature, which="2")
        if k1 in rec and k2 in rec:
            return k1, k2
    return None

# -----------------------------
# Build dataset (A − B diffs)
# -----------------------------
def build_X_y(records: List[Dict], features: List[str]) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    rows: List[Dict[str, float]] = []
    labels: List[int] = []

    skipped_ties, skipped_missing = 0, 0

    for r in records:
        hj = str(r.get("human_judgment", "")).strip().lower()
        if hj not in {"answer_1", "answer_2"}:
            skipped_ties += 1
            continue

        feats: Dict[str, float] = {}
        missing = False
        for f in features:
            pair = resolve_pair_keys(r, f)
            if pair is None:
                missing = True
                break
            k1, k2 = pair
            try:
                a1 = float(r[k1])
                a2 = float(r[k2])
            except Exception:
                missing = True
                break
            feats[f"{f}_diff"] = a1 - a2
        if missing:
            skipped_missing += 1
            continue

        rows.append(feats)
        labels.append(1 if hj == "answer_1" else 0)

    if not rows:
        raise ValueError("No usable rows after filtering ties/missing.")

    X = pd.DataFrame(rows)
    y = np.array(labels, dtype=int)
    print(f"[INFO] Built X with shape {X.shape}, y with {y.shape[0]} labels.")
    print(f"[INFO] Skipped ties: {skipped_ties}, skipped missing: {skipped_missing}")
    return X, y, list(X.columns)

# -----------------------------
# Train logistic regression
# -----------------------------
def train_logreg(X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        ))
    ])
    pipe.fit(X, y)
    return pipe

# -----------------------------
# Summarize weights
# -----------------------------
def summarize_weights(model: Pipeline, feature_names: List[str]) -> pd.DataFrame:
    lr: LogisticRegression = model.named_steps["logreg"]
    coefs = lr.coef_.ravel()
    intercept = lr.intercept_[0]
    df = pd.DataFrame({"feature": feature_names, "weight": coefs})
    df = df.sort_values("weight", ascending=False).reset_index(drop=True)
    df = pd.concat([df, pd.DataFrame([{"feature": "<intercept>", "weight": intercept}])],
                   ignore_index=True)
    return df

# -----------------------------
# Prediction helper
# -----------------------------
def predict_preference(model: Pipeline, rec: Dict, features: List[str]) -> Dict:
    feats = {}
    for f in features:
        pair = resolve_pair_keys(rec, f)
        if pair is None:
            raise ValueError(f"Missing keys for feature '{f}'")
        k1, k2 = pair
        feats[f"{f}_diff"] = float(rec[k1]) - float(rec[k2])
    x = pd.DataFrame([feats])
    pA = model.predict_proba(x)[0, 1]
    return {"P(A_preferred)": float(pA), "pred_label": int(pA >= 0.5)}

# -----------------------------
# Main function
# -----------------------------
def main():
    DATA_PATH = r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1__sample_100"

    records = load_records(DATA_PATH)
    X, y, feature_names = build_X_y(records, FEATURES)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = train_logreg(Xtr, ytr)

    # Evaluation
    yhat = model.predict(Xte)
    yproba = model.predict_proba(Xte)[:, 1]
    acc = accuracy_score(yte, yhat)
    auc = roc_auc_score(yte, yproba)
    print(f"\n[METRICS] Test Accuracy: {acc:.3f} | ROC-AUC: {auc:.3f}")
    print("\n[REPORT]\n" + classification_report(yte, yhat, digits=3))

    # Weights
    weights_df = summarize_weights(model, feature_names)
    print("\n=== Learned Weights (A−B diffs; positive => favors A) ===")
    print(weights_df.to_string(index=False))

    # Demo prediction
    demo_out = predict_preference(model, records[0], FEATURES)
    print("\n[DEMO] Prediction on first record:")
    print(demo_out)

# -----------------------------
if __name__ == "__main__":
    main()
