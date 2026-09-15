#!/usr/bin/env python3
"""
CyGro Phase 3 — Explainable AI for the frozen regime-aware Isolation Forest.

What this script explains
-------------------------
It explains the SAME continuous anomaly score used by the frozen detector:

    anomaly_score = -IsolationForest.score_samples(X_scaled)

Therefore:
    positive SHAP value  -> pushes the anomaly score upward (more anomalous)
    negative SHAP value  -> pushes the anomaly score downward (more normal)

Important:
SHAP explains the model's anomaly score. It does NOT prove physical causality.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import shap
except ImportError as exc:
    raise SystemExit(
        "SHAP is not installed.\n"
        "Activate the cygro_ml environment and run:\n"
        "  pip install shap\n"
    ) from exc


# =============================================================================
# PATHS
# =============================================================================

# =============================================================================
# PORTABLE PATH CONFIGURATION
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(
    os.environ.get(
        "ANOMALY_DATA_ROOT",
        REPO_ROOT / "data"
    )
)

MODEL_DIR = Path(
    os.environ.get(
        "ANOMALY_MODEL_DIR",
        REPO_ROOT / "models" / "regime_aware_if_v1_20260914"
    )
)

MODEL_FILE = MODEL_DIR / "model.joblib"
SCALER_FILE = MODEL_DIR / "scaler.joblib"
CONFIG_FILE = MODEL_DIR / "model_config.json"

NORMAL_FILE = Path(
    os.environ.get(
        "ANOMALY_NORMAL_FILE",
        MODEL_DIR / "normal_holdout_scored.csv"
    )
)

FAULT_FILE = Path(
    os.environ.get(
        "ANOMALY_FAULT_FILE",
        DATA_ROOT / "processed" / "hp_off_scored.csv"
    )
)

OUT = Path(
    os.environ.get(
        "ANOMALY_XAI_OUT",
        REPO_ROOT / "outputs" / "xai" / "hp_off_shap"
    )
)

OUT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SETTINGS
# =============================================================================

RANDOM_STATE = 42
BACKGROUND_N = 100
GLOBAL_MAX_N = 200
LOCAL_TOP_N = 5
LOCAL_BORDERLINE_N = 5
TOP_FEATURES_PER_SAMPLE = 8


# =============================================================================
# HELPERS
# =============================================================================

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recursive_find(obj, keys):
    """Find the first value under any candidate key in nested dict/list data."""
    keys = set(keys)

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in keys:
                return value
        for value in obj.values():
            result = recursive_find(value, keys)
            if result is not None:
                return result

    elif isinstance(obj, list):
        for item in obj:
            result = recursive_find(item, keys)
            if result is not None:
                return result

    return None


def resolve_features(scaler, config: dict) -> list[str]:
    # Best source: scikit-learn remembers DataFrame feature names when fitted.
    if hasattr(scaler, "feature_names_in_"):
        features = [str(x) for x in scaler.feature_names_in_]
        if features:
            return features

    candidates = recursive_find(
        config,
        {
            "features",
            "feature_names",
            "feature_columns",
            "continuous_features",
            "model_features",
            "input_features",
        },
    )

    if isinstance(candidates, list) and candidates:
        return [str(x) for x in candidates]

    raise RuntimeError(
        "Could not resolve the model feature list from scaler.feature_names_in_ "
        "or model_config.json."
    )


def resolve_threshold(config: dict) -> float | None:
    value = recursive_find(
        config,
        {
            "threshold",
            "anomaly_threshold",
            "decision_threshold",
            "score_threshold",
            "frozen_threshold",
        },
    )

    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def clean_feature_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(
            "Dataset is missing model features:\n  " + "\n  ".join(missing)
        )

    X = df[features].copy()

    for c in features:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    finite = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)

    if not finite.all():
        print(
            f"WARNING: dropping {(~finite).sum()} rows with NaN/inf "
            "in model features."
        )
        X = X.loc[finite].copy()

    return X


def choose_hp_active_normal_rows(
    normal: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    The frozen detector is regime-aware and was trained for HP-active operation.
    Prefer an explicit HP-active flag. Otherwise use a reference/simulated HP
    power signal if available. If neither exists, retain all finite rows and
    print a warning.
    """

    flag_candidates = [
        "hp_active",
        "is_hp_active",
        "hp_expected_active",
        "expected_hp_active",
        "hp_active_expected",
        "hp_ref_active",
    ]

    for col in flag_candidates:
        if col in normal.columns:
            values = normal[col]

            if values.dtype == bool:
                mask = values
            else:
                numeric = pd.to_numeric(values, errors="coerce")
                if numeric.notna().any():
                    mask = numeric.fillna(0).astype(float) > 0.5
                else:
                    text = values.astype(str).str.lower()
                    mask = text.isin({"1", "true", "yes", "on", "active"})

            selected = normal.loc[mask].copy()

            if len(selected) > 0:
                print(f"Normal background filtered by '{col}': {len(selected)} rows")
                return selected

    power_candidates = [
        "p_hp_el_ref_total_mw",
        "q_hp_th_ref_total_mw",
        "p_hp_el_expected_total_mw",
        "q_hp_th_expected_total_mw",
        "p_hp_el_sim_total_mw",
        "q_hp_th_sim_total_mw",
    ]

    for col in power_candidates:
        if col in normal.columns:
            v = pd.to_numeric(normal[col], errors="coerce").fillna(0.0)
            selected = normal.loc[v > 1e-9].copy()

            if len(selected) > 0:
                print(
                    f"Normal background filtered by positive '{col}': "
                    f"{len(selected)} rows"
                )
                return selected

    print(
        "WARNING: no HP-active indicator was found in normal_holdout_scored.csv. "
        "Using all finite normal rows as SHAP background."
    )
    return normal.copy()


def stratified_fault_sample(
    fault: pd.DataFrame,
    threshold: float | None,
    max_n: int,
) -> pd.DataFrame:
    if len(fault) <= max_n:
        return fault.copy()

    rng = np.random.default_rng(RANDOM_STATE)

    if threshold is None or "computed_anomaly_score" not in fault.columns:
        idx = rng.choice(fault.index.to_numpy(), size=max_n, replace=False)
        return fault.loc[idx].copy()

    detected = fault[fault["computed_anomaly_score"] >= threshold]
    missed = fault[fault["computed_anomaly_score"] < threshold]

    # Keep all detected cases when possible, then sample missed cases.
    if len(detected) >= max_n:
        idx = rng.choice(detected.index.to_numpy(), size=max_n, replace=False)
        return fault.loc[idx].copy()

    remaining = max_n - len(detected)
    missed_n = min(remaining, len(missed))

    if missed_n:
        missed_idx = rng.choice(
            missed.index.to_numpy(),
            size=missed_n,
            replace=False,
        )
        selected_idx = list(detected.index) + list(missed_idx)
    else:
        selected_idx = list(detected.index)

    return fault.loc[selected_idx].copy()


def save_current_figure(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


# =============================================================================
# LOAD FROZEN MODEL
# =============================================================================

for path in [
    MODEL_FILE,
    SCALER_FILE,
    CONFIG_FILE,
    NORMAL_FILE,
    FAULT_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(path)

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
config = load_json(CONFIG_FILE)

features = resolve_features(scaler, config)
threshold = resolve_threshold(config)

print("=" * 100)
print("CYGRO PHASE 3 — XAI FOR REGIME-AWARE ISOLATION FOREST")
print("=" * 100)
print("Model     :", MODEL_FILE)
print("Scaler    :", SCALER_FILE)
print("Fault data:", FAULT_FILE)
print("Features  :", len(features))
for i, f in enumerate(features, 1):
    print(f"  {i:2d}. {f}")
print("Threshold :", threshold)
print()


# =============================================================================
# LOAD DATA
# =============================================================================

normal = pd.read_csv(NORMAL_FILE)
fault = pd.read_csv(FAULT_FILE)

if "timestamp" in normal.columns:
    normal["timestamp"] = pd.to_datetime(normal["timestamp"], errors="coerce")

if "timestamp" in fault.columns:
    fault["timestamp"] = pd.to_datetime(fault["timestamp"], errors="coerce")

normal_active = choose_hp_active_normal_rows(normal, features)

X_normal = clean_feature_frame(normal_active, features)
X_fault = clean_feature_frame(fault, features)

# Align metadata after possible finite-row filtering.
normal_active = normal_active.loc[X_normal.index].copy()
fault = fault.loc[X_fault.index].copy()


# =============================================================================
# SAME ANOMALY SCORE AS THE FROZEN MODEL
# =============================================================================

def anomaly_score_from_raw(x) -> np.ndarray:
    """
    Raw feature values -> saved scaler -> frozen Isolation Forest.

    We negate sklearn's score_samples() because sklearn returns LOWER values
    for more abnormal observations, while the existing CyGro anomaly_score
    convention uses HIGHER values for more abnormal observations.
    """
    X_df = pd.DataFrame(x, columns=features)
    X_scaled = scaler.transform(X_df)
    return -model.score_samples(X_scaled)


normal_scores = anomaly_score_from_raw(X_normal.to_numpy())
fault_scores = anomaly_score_from_raw(X_fault.to_numpy())

normal_active["computed_anomaly_score"] = normal_scores
fault["computed_anomaly_score"] = fault_scores

if threshold is not None:
    normal_active["computed_anomaly"] = (
        normal_active["computed_anomaly_score"] >= threshold
    )
    fault["computed_anomaly"] = (
        fault["computed_anomaly_score"] >= threshold
    )


# =============================================================================
# SCORE CONSISTENCY CHECK
# =============================================================================

def score_consistency_report(df: pd.DataFrame, name: str):
    if "anomaly_score" not in df.columns:
        print(f"{name}: no saved 'anomaly_score' column; skipping consistency check.")
        return

    saved = pd.to_numeric(df["anomaly_score"], errors="coerce")
    calc = pd.to_numeric(df["computed_anomaly_score"], errors="coerce")
    mask = saved.notna() & calc.notna()

    if not mask.any():
        print(f"{name}: no comparable anomaly-score rows.")
        return

    max_abs = float(np.max(np.abs(saved[mask] - calc[mask])))
    mean_abs = float(np.mean(np.abs(saved[mask] - calc[mask])))

    print(
        f"{name} score consistency: "
        f"max_abs_diff={max_abs:.12g}, mean_abs_diff={mean_abs:.12g}"
    )


score_consistency_report(normal_active, "NORMAL")
score_consistency_report(fault, "HP_OFF")
print()


# =============================================================================
# SHAP BACKGROUND
# =============================================================================

rng = np.random.default_rng(RANDOM_STATE)

background_n = min(BACKGROUND_N, len(X_normal))
background_idx = rng.choice(
    X_normal.index.to_numpy(),
    size=background_n,
    replace=False,
)
background = X_normal.loc[background_idx].copy()

masker = shap.maskers.Independent(
    background,
    max_samples=background_n,
)

# Model-agnostic SHAP is deliberate here: it explains the actual CyGro
# anomaly-score function after scaling, instead of relying on a tree-internal
# output that is not the same quantity used by the deployed threshold.
explainer = shap.Explainer(
    anomaly_score_from_raw,
    masker,
    algorithm="permutation",
    feature_names=features,
)

min_evals = 2 * len(features) + 1
max_evals = max(min_evals, 101)


# =============================================================================
# GLOBAL XAI — REPRESENTATIVE HP_OFF FAULT SAMPLE
# =============================================================================

fault_for_global = fault.copy()
X_fault_global = X_fault.copy()

# Reindex feature frame to match sampled metadata.
sampled_fault = stratified_fault_sample(
    fault_for_global,
    threshold=threshold,
    max_n=GLOBAL_MAX_N,
)
X_global = X_fault_global.loc[sampled_fault.index].copy()

print(
    f"Explaining {len(X_global)} HP_OFF rows "
    f"with {background_n} normal HP-active background rows..."
)

global_exp = explainer(
    X_global,
    max_evals=max_evals,
)

global_abs = np.abs(global_exp.values).mean(axis=0)
global_signed = global_exp.values.mean(axis=0)

global_importance = pd.DataFrame(
    {
        "feature": features,
        "mean_abs_shap": global_abs,
        "mean_signed_shap": global_signed,
    }
).sort_values("mean_abs_shap", ascending=False)

total_abs = global_importance["mean_abs_shap"].sum()
if total_abs > 0:
    global_importance["importance_pct"] = (
        100.0 * global_importance["mean_abs_shap"] / total_abs
    )
else:
    global_importance["importance_pct"] = 0.0

global_importance.to_csv(
    OUT / "01_hp_off_global_shap_importance.csv",
    index=False,
)

# Global bar
plt.figure(figsize=(9, 6))
plot_df = global_importance.head(15).sort_values(
    "mean_abs_shap",
    ascending=True,
)
plt.barh(plot_df["feature"], plot_df["mean_abs_shap"])
plt.xlabel("Mean |SHAP value| on anomaly score")
plt.ylabel("Feature")
plt.title("CyGro HP-OFF — Global XAI feature importance")
save_current_figure(OUT / "01_hp_off_global_shap_importance.png")

# Beeswarm
shap.plots.beeswarm(
    global_exp,
    max_display=min(15, len(features)),
    show=False,
)
save_current_figure(OUT / "02_hp_off_global_shap_beeswarm.png")


# =============================================================================
# DETECTED VS MISSED — WHAT PUSHES SCORES DIFFERENTLY?
# =============================================================================

if threshold is not None:
    global_meta = sampled_fault.copy()
    global_meta["xai_row"] = np.arange(len(global_meta))
    global_meta["detected_by_frozen_threshold"] = (
        global_meta["computed_anomaly_score"] >= threshold
    )

    group_rows = []

    for label, mask in [
        ("detected", global_meta["detected_by_frozen_threshold"]),
        ("missed", ~global_meta["detected_by_frozen_threshold"]),
    ]:
        positions = global_meta.loc[mask, "xai_row"].to_numpy(dtype=int)

        if len(positions) == 0:
            continue

        vals = global_exp.values[positions]

        mean_abs = np.abs(vals).mean(axis=0)
        mean_signed = vals.mean(axis=0)

        for feature, a, s in zip(features, mean_abs, mean_signed):
            group_rows.append(
                {
                    "group": label,
                    "n_rows": len(positions),
                    "feature": feature,
                    "mean_abs_shap": float(a),
                    "mean_signed_shap": float(s),
                }
            )

    group_df = pd.DataFrame(group_rows)

    if not group_df.empty:
        group_df.to_csv(
            OUT / "03_detected_vs_missed_shap.csv",
            index=False,
        )

        pivot = (
            group_df.pivot(
                index="feature",
                columns="group",
                values="mean_abs_shap",
            )
            .fillna(0.0)
        )

        pivot["max"] = pivot.max(axis=1)
        pivot = pivot.sort_values("max", ascending=False).head(12)
        pivot = pivot.drop(columns="max")

        ax = pivot.sort_values(
            by=list(pivot.columns),
            ascending=True,
        ).plot(
            kind="barh",
            figsize=(9, 6),
        )
        ax.set_xlabel("Mean |SHAP value| on anomaly score")
        ax.set_ylabel("Feature")
        ax.set_title(
            "HP-OFF XAI — detected vs missed by frozen threshold"
        )
        plt.legend(title="Group")
        save_current_figure(
            OUT / "03_detected_vs_missed_shap.png"
        )


# =============================================================================
# LOCAL XAI — STRONGEST + BORDERLINE MISSED HP_OFF EVENTS
# =============================================================================

fault_sorted = fault.sort_values(
    "computed_anomaly_score",
    ascending=False,
)

local_parts = [
    fault_sorted.head(LOCAL_TOP_N)
]

if threshold is not None:
    borderline_missed = (
        fault[fault["computed_anomaly_score"] < threshold]
        .sort_values(
            "computed_anomaly_score",
            ascending=False,
        )
        .head(LOCAL_BORDERLINE_N)
    )
    local_parts.append(borderline_missed)

local_meta = (
    pd.concat(local_parts)
    .drop_duplicates()
    .copy()
)

X_local = X_fault.loc[local_meta.index].copy()

print(f"Explaining {len(X_local)} local HP_OFF events...")

local_exp = explainer(
    X_local,
    max_evals=max_evals,
)

local_rows = []

for pos, (idx, row) in enumerate(local_meta.iterrows()):
    values = local_exp.values[pos]
    base_value = float(np.asarray(local_exp.base_values[pos]).reshape(-1)[0])

    ordering = np.argsort(np.abs(values))[::-1]

    ts = row["timestamp"] if "timestamp" in row.index else idx
    score = float(row["computed_anomaly_score"])
    detected = (
        bool(score >= threshold)
        if threshold is not None
        else None
    )

    for rank, j in enumerate(
        ordering[:TOP_FEATURES_PER_SAMPLE],
        start=1,
    ):
        feature = features[j]
        feature_value = float(X_local.loc[idx, feature])
        shap_value = float(values[j])

        local_rows.append(
            {
                "timestamp": ts,
                "row_index": idx,
                "anomaly_score": score,
                "threshold": threshold,
                "detected": detected,
                "base_anomaly_score": base_value,
                "rank": rank,
                "feature": feature,
                "feature_value": feature_value,
                "shap_value": shap_value,
                "effect": (
                    "more_anomalous"
                    if shap_value > 0
                    else "more_normal"
                    if shap_value < 0
                    else "neutral"
                ),
            }
        )

    # Waterfall plot for each selected event.
    shap.plots.waterfall(
        local_exp[pos],
        max_display=min(12, len(features)),
        show=False,
    )

    safe_ts = (
        str(ts)
        .replace(":", "-")
        .replace(" ", "_")
        .replace("/", "-")
    )

    save_current_figure(
        OUT / f"local_waterfall_{pos+1:02d}_{safe_ts}.png"
    )


local_df = pd.DataFrame(local_rows)
local_df.to_csv(
    OUT / "04_local_hp_off_top_contributors.csv",
    index=False,
)


# =============================================================================
# HUMAN-READABLE SUMMARY TABLE
# =============================================================================

summary_rows = []

for (ts, row_index), group in local_df.groupby(
    ["timestamp", "row_index"],
    sort=False,
):
    first = group.iloc[0]

    positive = (
        group[group["shap_value"] > 0]
        .sort_values("shap_value", ascending=False)
        .head(3)
    )

    negative = (
        group[group["shap_value"] < 0]
        .sort_values("shap_value", ascending=True)
        .head(3)
    )

    positive_text = "; ".join(
        f"{r.feature}={r.feature_value:.6g} (SHAP {r.shap_value:+.4f})"
        for r in positive.itertuples()
    )

    negative_text = "; ".join(
        f"{r.feature}={r.feature_value:.6g} (SHAP {r.shap_value:+.4f})"
        for r in negative.itertuples()
    )

    summary_rows.append(
        {
            "timestamp": ts,
            "row_index": row_index,
            "anomaly_score": first["anomaly_score"],
            "threshold": first["threshold"],
            "detected": first["detected"],
            "top_features_increasing_anomaly_score": positive_text,
            "top_features_decreasing_anomaly_score": negative_text,
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    OUT / "05_local_xai_summary.csv",
    index=False,
)


# =============================================================================
# SAVE SCORED DATA USED BY XAI
# =============================================================================

fault.to_csv(
    OUT / "hp_off_test_with_recomputed_scores.csv",
    index=False,
)

normal_active.to_csv(
    OUT / "normal_hp_active_background_with_scores.csv",
    index=False,
)


# =============================================================================
# TERMINAL SUMMARY
# =============================================================================

print()
print("=" * 100)
print("XAI COMPLETE")
print("=" * 100)
print("Output:", OUT)
print()
print("Top global HP_OFF contributors:")
print(
    global_importance[
        [
            "feature",
            "mean_abs_shap",
            "mean_signed_shap",
            "importance_pct",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

if threshold is not None:
    n_detected = int(
        (fault["computed_anomaly_score"] >= threshold).sum()
    )
    print()
    print(
        f"Frozen-threshold detection on loaded HP_OFF rows: "
        f"{n_detected}/{len(fault)} "
        f"({100*n_detected/len(fault):.2f}%)"
    )

print()
print("Created:")
for p in sorted(OUT.iterdir()):
    print(" ", p.name)

print()
print(
    "Interpretation rule: positive SHAP -> raises anomaly score; "
    "negative SHAP -> lowers anomaly score."
)
print(
    "Scientific wording: these are model-attribution signals, not proof "
    "of physical causality."
)
