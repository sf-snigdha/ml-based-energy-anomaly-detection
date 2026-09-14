#!/usr/bin/env python3

"""
Run inference using the public synthetic-demo anomaly detector.
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from features import prepare_features


MODEL_DIR = Path("models")


class EnergyAnomalyDetector:

    def __init__(self):

        config_file = (
            MODEL_DIR /
            "demo_model_config.json"
        )

        model_file = (
            MODEL_DIR /
            "demo_isolation_forest.joblib"
        )

        scaler_file = (
            MODEL_DIR /
            "demo_scaler.joblib"
        )


        if not config_file.exists():
            raise FileNotFoundError(
                "Model config not found. "
                "Run: python src/train.py"
            )

        if not model_file.exists():
            raise FileNotFoundError(
                "Demo model not found. "
                "Run: python src/train.py"
            )

        if not scaler_file.exists():
            raise FileNotFoundError(
                "Demo scaler not found. "
                "Run: python src/train.py"
            )


        with open(
            config_file
        ) as f:
            self.config = json.load(f)


        self.threshold = float(
            self.config["threshold"]
        )


        self.model = joblib.load(
            model_file
        )

        self.scaler = joblib.load(
            scaler_file
        )


    def predict_one(self, record):

        if isinstance(record, pd.Series):
            raw = record.to_dict()
        else:
            raw = dict(record)


        # ----------------------------------------------------
        # Optional routing
        # ----------------------------------------------------

        hp_expected_active = int(
            raw.get(
                "hp_expected_active",
                1
            )
        )


        if hp_expected_active == 0:

            return {
                "anomaly_score": None,
                "threshold": self.threshold,
                "prediction": "NORMAL",
                "predicted_anomaly": 0,
                "router_status": "HP_EXPECTED_INACTIVE",
            }


        # ----------------------------------------------------
        # Feature preparation
        # ----------------------------------------------------

        X = prepare_features(
            raw,
            self.config
        )


        # ----------------------------------------------------
        # Scale
        # ----------------------------------------------------

        X_scaled = self.scaler.transform(
            X
        )


        # ----------------------------------------------------
        # Anomaly score
        #
        # Higher = more anomalous
        # ----------------------------------------------------

        anomaly_score = float(
            -self.model.score_samples(
                X_scaled
            )[0]
        )


        predicted_anomaly = int(
            anomaly_score
            >= self.threshold
        )


        prediction = (
            "ANOMALY"
            if predicted_anomaly
            else
            "NORMAL"
        )


        return {
            "anomaly_score": anomaly_score,
            "threshold": self.threshold,
            "prediction": prediction,
            "predicted_anomaly": predicted_anomaly,
            "router_status": "HP_ACTIVE_DETECTOR",
        }


    def predict_dataframe(self, df):

        results = []

        for _, row in df.iterrows():

            result = self.predict_one(
                row
            )

            result["timestamp"] = row.get(
                "timestamp",
                None
            )

            result["scenario"] = row.get(
                "scenario",
                None
            )

            result["label"] = row.get(
                "label",
                None
            )

            results.append(
                result
            )


        return pd.DataFrame(
            results
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="CSV file containing operating data"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV for predictions"
    )

    args = parser.parse_args()


    detector = EnergyAnomalyDetector()


    df = pd.read_csv(
        args.input
    )


    result = detector.predict_dataframe(
        df
    )


    result.to_csv(
        args.output,
        index=False
    )


    evaluated = result[
        result["router_status"]
        ==
        "HP_ACTIVE_DETECTOR"
    ]


    print("=" * 80)
    print("ENERGY ANOMALY DETECTION — INFERENCE")
    print("=" * 80)

    print(
        "Input rows:",
        len(result)
    )

    print(
        "Evaluated rows:",
        len(evaluated)
    )

    print(
        "Detected anomalies:",
        int(
            evaluated[
                "predicted_anomaly"
            ].sum()
        )
    )


    if len(evaluated):

        print(
            "Anomaly rate:",
            f"{100 * evaluated['predicted_anomaly'].mean():.2f}%"
        )


    print(
        "Threshold:",
        detector.threshold
    )

    print(
        "Saved:",
        args.output
    )


if __name__ == "__main__":
    main()
