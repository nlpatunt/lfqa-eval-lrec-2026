# qa_rubric_pearson_all9.py
# Prints Pearson correlations for ALL 9 rubrics vs human judgment.
# - Loads JSON array or JSONL
# - Builds per-pair Δmetric = score_answer_1 - score_answer_2
# - EXCLUDING ties: Pearson r(Δmetric, y), y∈{0,1} (A1 preferred=1)
# - INCLUDING ties: Pearson r(Δmetric, y3), y3∈{-1,0,1} (A1=+1, tie=0, A2=-1)
# - Also prints Win-Rate per rubric (excluding ties): % where sign(Δ) matches human winner
#
# No files saved. No plotting. Only stdlib + numpy.

import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Map the 9 rubrics to their key prefixes in your JSON
RUBRIC_PREFIXES = {
    "specificity":        "specificity",
    "grammar":            "grammar",
    "easy_to_understand": "easy_to_understand",
    "veracity":           "veriscore_score",
    "well_structured":    "well_structure_score",
    "conciseness":        "conciseness_score",
    "relevance":          "relevance_score",
    "completeness":       "completeness_score",
    "example":            "example_score",
}

def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Input file is empty.")
    # Try JSON first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass
    # Fallback JSONL
    rows = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON on line {i}: {e}") from e
    return rows

def build_deltas_and_labels(rows: List[Dict[str, Any]]):
    """
    Returns:
      deltas: dict[rubric] -> np.array of Δmetric (A1 - A2) for each record (NaN if missing)
      y_bin: np.array of {0,1} for non-ties only (A1 preferred=1); same length as filtered records
      mask_non_tie: boolean mask over original rows indicating which are non-ties
      y3: np.array of {-1,0,1} over ALL rows (A1=+1, tie=0, A2=-1)
    """
    n = len(rows)
    # Build Δ arrays for each rubric over all rows
    deltas_all = {rubric: np.full(n, np.nan, dtype=float) for rubric in RUBRIC_PREFIXES}

    # Build labels
    y3 = np.zeros(n, dtype=float)  # -1 (A2), 0 (tie), +1 (A1)
    for i, obj in enumerate(rows):
        hj = obj.get("human_judgment")
        if hj == "answer_1":
            y3[i] = 1.0
        elif hj == "answer_2":
            y3[i] = -1.0
        else:
            y3[i] = 0.0  # tie or anything else

        for rubric, prefix in RUBRIC_PREFIXES.items():
            a1 = obj.get(f"{prefix}_answer_1", np.nan)
            a2 = obj.get(f"{prefix}_answer_2", np.nan)
            try:
                deltas_all[rubric][i] = float(a1) - float(a2)
            except Exception:
                deltas_all[rubric][i] = np.nan

    # Non-tie mask
    mask_non_tie = (y3 != 0.0)
    # Binary y for non-ties
    y_bin = (y3[mask_non_tie] == 1.0).astype(float)  # 1 if A1 preferred, else 0

    # Slice deltas for non-tie view as well
    deltas_non_tie = {k: v[mask_non_tie] for k, v in deltas_all.items()}

    return deltas_all, deltas_non_tie, y3, y_bin, mask_non_tie

def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """
    Pearson correlation with NaN safety.
    Returns np.nan if variance is zero or insufficient valid pairs.
    """
    # Mask finite pairs
    m = np.isfinite(x) & np.isfinite(y)
    x2 = x[m]
    y2 = y[m]
    if x2.size < 2:
        return np.nan
    if np.std(x2) == 0 or np.std(y2) == 0:
        return np.nan
    return float(np.corrcoef(x2, y2)[0, 1])

def win_rate(delta: np.ndarray, y_bin: np.ndarray) -> float:
    """
    Excluding ties only: y_bin∈{0,1} (1 means A1 preferred).
    Win if (delta > 0 and A1 wins) or (delta < 0 and A2 wins).
    If delta == 0 (tie on metric), it's not counted as a win.
    Returns fraction in [0,1] over valid, finite pairs.
    """
    m = np.isfinite(delta) & np.isfinite(y_bin)
    d = delta[m]
    y = y_bin[m]
    if d.size == 0:
        return np.nan
    # A1 picked and delta>0  OR  A2 picked and delta<0
    wins = ((y == 1) & (d > 0)) | ((y == 0) & (d < 0))
    # Only count cases where |delta|>0
    count_valid = np.sum(d != 0)
    if count_valid == 0:
        return np.nan
    return float(np.sum(wins & (d != 0)) / count_valid)

def main(path: str):
    rows = load_json_or_jsonl(path)
    deltas_all, deltas_non_tie, y3, y_bin, mask_non_tie = build_deltas_and_labels(rows)

    # 1) Pearson excluding ties (point-biserial with y in {0,1})
    print("\n=== Pearson r (EXCLUDING ties): corr(Δmetric, 1 if Answer1 preferred) ===")
    excl = []
    for rubric in RUBRIC_PREFIXES:
        r = pearson(deltas_non_tie[rubric], y_bin)
        excl.append((rubric, r))
    excl_sorted = sorted(excl, key=lambda t: (-(t[1] if np.isfinite(t[1]) else -999)))
    for rubric, r in excl_sorted:
        print(f"{rubric:>18s} : {r:.6f}" if np.isfinite(r) else f"{rubric:>18s} : NaN")

    # 2) Pearson including ties: y3 ∈ {+1, 0, −1}
    print("\n=== Pearson r (INCLUDING ties): corr(Δmetric, y3) with y3∈{+1 (A1), 0 (tie), −1 (A2)} ===")
    incl = []
    for rubric in RUBRIC_PREFIXES:
        r = pearson(deltas_all[rubric], y3)
        incl.append((rubric, r))
    incl_sorted = sorted(incl, key=lambda t: (-(t[1] if np.isfinite(t[1]) else -999)))
    for rubric, r in incl_sorted:
        print(f"{rubric:>18s} : {r:.6f}" if np.isfinite(r) else f"{rubric:>18s} : NaN")

    # 3) Simple Win-Rate per rubric (excluding ties)
    print("\n=== Win-Rate by rubric (EXCLUDING ties) ===")
    print("(fraction of non-tie comparisons where the higher-scoring answer on this rubric matches the human winner)")
    wrs = []
    for rubric in RUBRIC_PREFIXES:
        wr = win_rate(deltas_non_tie[rubric], y_bin)
        wrs.append((rubric, wr))
    wrs_sorted = sorted(wrs, key=lambda t: (-(t[1] if np.isfinite(t[1]) else -999)))
    for rubric, wr in wrs_sorted:
        if np.isfinite(wr):
            print(f"{rubric:>18s} : {wr*100:.1f}%")
        else:
            print(f"{rubric:>18s} : NaN")

if __name__ == "__main__":
    # Change this to your dataset path
    main(r"F:\PhD\Long form research question\Final Dataset\lfqa_pairwise_human_judgments_v1__sample_100_score")
