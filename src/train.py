#!/usr/bin/env python3

"""
Train the public demonstration anomaly-detection model.

IMPORTANT:
This script uses only independently generated synthetic data.
It does not use or reproduce confidential project data.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

from features import prepare_features


# ============================================================
# PATHS
# ============================================================

EXAMPLE_DIR = Path("examples")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


NORMAL_FILE = (
    EXAMPLE_DIR /
    "synthetic_normal.csv"
)

HP_OFF_FILE = (
    EXAMPLE_DIR /
    "synthetic_hp_off.csv"
)

HP_DOUBLED_FILE = (
    EXAMPLE_DIR /
    "synthetic_hp_doubled.csv"
)


# ============================================================
# GENERIC FEATURES
# ============================================================

FEATURES = [
    "heat_demand_mw",
    "base_electric_load_mw",
    "pv_generation_mw",
    "hp_electric_power_mw",
    "hp_thermal_power_mw",
    "cop_model",
    "p2h_electric_power_mw",
    "chp_electric_power_mw",
    "coupling_bus_voltage_pu",
    "grid_import_mw",
    "max_line_loading_pct",
    "unserved_heat_mw",
    "network_supply_temp_k",
    "network_return_temp_k",
]


# ============================================================
# LOAD DATA
# ============================================================

normal = pd.read_csv(
    NORMAL_FILE
)

hp_off = pd.read_csv(
    HP_OFF_FILE
)

hp_doubled = pd.read_csv(
    HP_DOUBLED_FILE
)


print("=" * 80)
print("PUBLIC DEMO — ANOMALY DETECTOR TRAINING")
print("=" * 80)

print(
    "Normal samples:",
    len(normal)
)

print(
    "HP_OFF samples:",
    len(hp_off)
)

print(
    "HP_DOUBLED samples:",
    len(hp_doubled)
)


# ============================================================
# NORMAL TRAIN / VALIDATION SPLIT
#
# No shuffle — preserve temporal ordering.
# ============================================================

split_index = int(
    len(normal) * 0.70
)

train = (
    normal
    .iloc[:split_index]
    .copy()
)

validation_normal = (
    normal
    .iloc[split_index:]
    .copy()
)


print(
    "\nTraining normal:",
    len(train)
)

print(
    "Validation normal:",
    len(validation_normal)
)


# ============================================================
# TRAINING COP MEDIAN
# ============================================================

valid_cop = train.loc[
    train["cop"].notna(),
    "cop"
]


cop_median = float(
    valid_cop.median()
)


print(
    f"\nTraining COP median: "
    f"{cop_median:.4f}"
)


# ============================================================
# TEMPORARY CONFIG FOR FEATURE PREPARATION
# ============================================================

config = {

    "features":
        FEATURES,

    "cop_median":
        cop_median,
}


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_dataframe(df):

    rows = []

    for _, row in df.iterrows():

        prepared = prepare_features(
            row,
            config
        )

        rows.append(
            prepared.iloc[0]
        )

    return pd.DataFrame(
        rows,
        columns=FEATURES
    )


X_train = prepare_dataframe(
    train
)

X_normal_val = prepare_dataframe(
    validation_normal
)

X_off = prepare_dataframe(
    hp_off
)

X_doubled = prepare_dataframe(
    hp_doubled
)


# ============================================================
# SCALER
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_normal_val_scaled = scaler.transform(
    X_normal_val
)

X_off_scaled = scaler.transform(
    X_off
)

X_doubled_scaled = scaler.transform(
    X_doubled
)


# ============================================================
# ISOLATION FOREST
# ============================================================

model = IsolationForest(
    n_estimators=300,
    max_samples="auto",
    contamination="auto",
    random_state=42,
    n_jobs=-1,
)


model.fit(
    X_train_scaled
)


# ============================================================
# ANOMALY SCORES
#
# Higher positive value = more anomalous
# ============================================================

normal_scores = (
    -model.score_samples(
        X_normal_val_scaled
    )
)

off_scores = (
    -model.score_samples(
        X_off_scaled
    )
)

doubled_scores = (
    -model.score_samples(
        X_doubled_scaled
    )
)


# ============================================================
# THRESHOLD CALIBRATION
#
# Rule:
#   healthy validation FPR <= 5%
#   maximize average F1 across both synthetic faults
# ============================================================

all_scores = np.unique(
    np.sort(
        np.concatenate([
            normal_scores,
            off_scores,
            doubled_scores,
        ])
    )
)


results = []


for threshold in all_scores:

    normal_pred = (
        normal_scores >= threshold
    ).astype(int)

    off_pred = (
        off_scores >= threshold
    ).astype(int)

    doubled_pred = (
        doubled_scores >= threshold
    ).astype(int)


    normal_fpr = float(
        normal_pred.mean()
    )


    # --------------------------------------------------------
    # HP_OFF
    # --------------------------------------------------------

    y_off_true = np.concatenate([
        np.zeros(
            len(normal_pred)
        ),
        np.ones(
            len(off_pred)
        ),
    ])


    y_off_pred = np.concatenate([
        normal_pred,
        off_pred,
    ])


    off_f1 = f1_score(
        y_off_true,
        y_off_pred,
        zero_division=0
    )


    # --------------------------------------------------------
    # HP_DOUBLED
    # --------------------------------------------------------

    y_double_true = np.concatenate([
        np.zeros(
            len(normal_pred)
        ),
        np.ones(
            len(doubled_pred)
        ),
    ])


    y_double_pred = np.concatenate([
        normal_pred,
        doubled_pred,
    ])


    doubled_f1 = f1_score(
        y_double_true,
        y_double_pred,
        zero_division=0
    )


    macro_f1 = np.mean([
        off_f1,
        doubled_f1,
    ])


    results.append({

        "threshold":
            threshold,

        "normal_fpr":
            normal_fpr,

        "off_f1":
            off_f1,

        "doubled_f1":
            doubled_f1,

        "macro_f1":
            macro_f1,
    })


threshold_results = pd.DataFrame(
    results
)


eligible = threshold_results[
    threshold_results[
        "normal_fpr"
    ] <= 0.05
].copy()


if eligible.empty:

    raise RuntimeError(
        "No threshold satisfies FPR <= 5%."
    )


best = (
    eligible
    .sort_values(
        [
            "macro_f1",
            "normal_fpr",
        ],
        ascending=[
            False,
            True,
        ]
    )
    .iloc[0]
)


threshold = float(
    best["threshold"]
)


# ============================================================
# FINAL DEMO VALIDATION METRICS
# ============================================================

normal_pred = (
    normal_scores
    >= threshold
).astype(int)

off_pred = (
    off_scores
    >= threshold
).astype(int)

doubled_pred = (
    doubled_scores
    >= threshold
).astype(int)


def fault_metrics(
    fault_pred
):

    y_true = np.concatenate([
        np.zeros(
            len(normal_pred)
        ),
        np.ones(
            len(fault_pred)
        ),
    ])

    y_pred = np.concatenate([
        normal_pred,
        fault_pred,
    ])

    return {

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),
    }


off_metrics = fault_metrics(
    off_pred
)

double_metrics = fault_metrics(
    doubled_pred
)


# ============================================================
# SAVE DEMO MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_DIR /
    "demo_isolation_forest.joblib"
)

joblib.dump(
    scaler,
    MODEL_DIR /
    "demo_scaler.joblib"
)


# ============================================================
# SAVE CONFIG
# ============================================================

model_config = {

    "model_name":
        "energy_anomaly_isolation_forest",

    "version":
        "demo-v1",

    "model_type":
        "IsolationForest",

    "features":
        FEATURES,

    "cop_median":
        cop_median,

    "threshold":
        threshold,

    "score_definition":
        "-model.score_samples(X_scaled)",

    "decision_rule":
        (
            "ANOMALY if anomaly_score "
            ">= threshold"
        ),

    "data_source":
        "independently generated synthetic demonstration data",

    "confidential_data_used":
        False,
}


with open(
    MODEL_DIR /
    "demo_model_config.json",
    "w"
) as f:

    json.dump(
        model_config,
        f,
        indent=2
    )


# ============================================================
# SAVE THRESHOLD RESULTS
# ============================================================

threshold_results.to_csv(
    MODEL_DIR /
    "demo_threshold_analysis.csv",
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)


print(
    f"\nSelected threshold: "
    f"{threshold:.4f}"
)

print(
    f"Healthy validation FPR: "
    f"{100 * normal_pred.mean():.2f}%"
)


print("\nSynthetic HP_OFF")

print(
    f"Precision: "
    f"{100 * off_metrics['precision']:.2f}%"
)

print(
    f"Recall: "
    f"{100 * off_metrics['recall']:.2f}%"
)

print(
    f"F1: "
    f"{100 * off_metrics['f1']:.2f}%"
)


print("\nSynthetic HP_DOUBLED")

print(
    f"Precision: "
    f"{100 * double_metrics['precision']:.2f}%"
)

print(
    f"Recall: "
    f"{100 * double_metrics['recall']:.2f}%"
)

print(
    f"F1: "
    f"{100 * double_metrics['f1']:.2f}%"
)


print("\nSaved:")

print(
    MODEL_DIR /
    "demo_isolation_forest.joblib"
)

print(
    MODEL_DIR /
    "demo_scaler.joblib"
)

print(
    MODEL_DIR /
    "demo_model_config.json"
)

