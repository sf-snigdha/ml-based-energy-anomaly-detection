#!/usr/bin/env python3

"""
Feature preparation for ML-based anomaly detection
in a generic sector-coupled energy system.

This public version contains no project-specific paths,
network identifiers, locations, or confidential asset names.
"""

import numpy as np
import pandas as pd


HP_EPS = 1e-6


def prepare_features(record, config):
    """
    Convert one operating point into the exact feature format
    required by the trained anomaly-detection model.

    Parameters
    ----------
    record : dict or pandas.Series
        One energy-system operating point.

    config : dict
        Model configuration containing:
        - features
        - cop_median

    Returns
    -------
    pandas.DataFrame
        One-row model-ready feature matrix.
    """

    if isinstance(record, pd.Series):
        row = record.copy()
    else:
        row = pd.Series(record).copy()


    # ========================================================
    # HEAT-PUMP POWER
    # ========================================================

    hp_electric_power = float(
        row.get(
            "hp_electric_power_mw",
            np.nan
        )
    )

    hp_thermal_power = float(
        row.get(
            "hp_thermal_power_mw",
            np.nan
        )
    )


    # ========================================================
    # COP
    # ========================================================

    cop = row.get(
        "cop",
        np.nan
    )


    # Calculate COP when it is not provided
    # but the heat pump is operating.
    if (
        not np.isfinite(cop)
        and
        np.isfinite(hp_electric_power)
        and
        np.isfinite(hp_thermal_power)
        and
        hp_electric_power > HP_EPS
    ):

        cop = (
            hp_thermal_power
            /
            hp_electric_power
        )


    # If the heat pump is OFF, COP is undefined.
    # Use the frozen training median.
    if np.isfinite(cop):

        cop_model = float(cop)

    else:

        cop_model = float(
            config["cop_median"]
        )


    row["cop_model"] = cop_model


    # ========================================================
    # EXACT TRAINING FEATURE ORDER
    # ========================================================

    features = list(
        config["features"]
    )


    # ========================================================
    # CHECK REQUIRED FEATURES
    # ========================================================

    missing = [
        feature
        for feature in features
        if feature not in row.index
    ]


    if missing:

        raise ValueError(
            f"Missing required features: {missing}"
        )


    # ========================================================
    # VALIDATE NUMERIC VALUES
    # ========================================================

    values = {}


    for feature in features:

        try:

            value = float(
                row[feature]
            )

        except Exception as exc:

            raise ValueError(
                f"Feature '{feature}' "
                "must be numeric."
            ) from exc


        if not np.isfinite(value):

            raise ValueError(
                f"Feature '{feature}' "
                "contains NaN or infinity."
            )


        values[feature] = value


    # ========================================================
    # MODEL-READY ROW
    # ========================================================

    return pd.DataFrame(
        [values],
        columns=features
    )
