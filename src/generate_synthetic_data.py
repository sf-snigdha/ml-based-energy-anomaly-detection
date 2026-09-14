#!/usr/bin/env python3

"""
Generate fully synthetic demonstration data for the public
energy-system anomaly-detection repository.

These values are artificial and are not derived from any
confidential project dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd


RNG = np.random.default_rng(42)

OUT_DIR = Path("examples")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_normal(n=200):
    timestamps = pd.date_range(
        "2026-01-01",
        periods=n,
        freq="15min"
    )

    heat_demand = RNG.uniform(
        1.0,
        5.0,
        n
    )

    base_load = RNG.uniform(
        2.0,
        6.0,
        n
    )

    pv = RNG.uniform(
        0.0,
        3.0,
        n
    )

    hp_electric = RNG.uniform(
        0.3,
        1.4,
        n
    )

    cop = RNG.normal(
        3.0,
        0.15,
        n
    )

    hp_thermal = (
        hp_electric
        *
        cop
    )

    p2h = RNG.uniform(
        0.0,
        1.5,
        n
    )

    chp = RNG.uniform(
        0.5,
        2.5,
        n
    )

    voltage = RNG.normal(
        1.0,
        0.005,
        n
    )

    grid_import = (
        base_load
        +
        hp_electric
        +
        p2h
        -
        pv
        -
        chp
        +
        RNG.normal(
            0.0,
            0.15,
            n
        )
    )

    loading = RNG.uniform(
        20.0,
        60.0,
        n
    )

    unserved_heat = RNG.uniform(
        0.0,
        0.08,
        n
    )

    supply_temp = RNG.normal(
        345.0,
        2.0,
        n
    )

    return_temp = RNG.normal(
        320.0,
        2.0,
        n
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "scenario": "NORMAL",
        "label": 0,
        "hp_expected_active": 1,
        "heat_demand_mw": heat_demand,
        "base_electric_load_mw": base_load,
        "pv_generation_mw": pv,
        "hp_electric_power_mw": hp_electric,
        "hp_thermal_power_mw": hp_thermal,
        "cop": cop,
        "p2h_electric_power_mw": p2h,
        "chp_electric_power_mw": chp,
        "coupling_bus_voltage_pu": voltage,
        "grid_import_mw": grid_import,
        "max_line_loading_pct": loading,
        "unserved_heat_mw": unserved_heat,
        "network_supply_temp_k": supply_temp,
        "network_return_temp_k": return_temp,
    })

    return df


def make_hp_off(normal):
    df = normal.copy()

    df["scenario"] = "HP_OFF"
    df["label"] = 1

    df["hp_electric_power_mw"] = 0.0
    df["hp_thermal_power_mw"] = 0.0
    df["cop"] = np.nan

    df["grid_import_mw"] -= 0.8

    df["coupling_bus_voltage_pu"] += 0.003

    df["unserved_heat_mw"] += RNG.uniform(
        1.0,
        3.5,
        len(df)
    )

    df["network_supply_temp_k"] -= RNG.uniform(
        15.0,
        25.0,
        len(df)
    )

    df["network_return_temp_k"] -= RNG.uniform(
        8.0,
        18.0,
        len(df)
    )

    return df


def make_hp_doubled(normal):
    df = normal.copy()

    df["scenario"] = "HP_DOUBLED"
    df["label"] = 1

    df["hp_electric_power_mw"] *= 2.0
    df["hp_thermal_power_mw"] *= 2.0

    df["grid_import_mw"] += 0.8

    df["coupling_bus_voltage_pu"] -= 0.003

    df["network_supply_temp_k"] += RNG.uniform(
        4.0,
        10.0,
        len(df)
    )

    df["network_return_temp_k"] += RNG.uniform(
        3.0,
        8.0,
        len(df)
    )

    return df


def main():
    normal = make_normal(
        n=200
    )

    hp_off = make_hp_off(
        normal.iloc[
            :40
        ].copy()
    )

    hp_doubled = make_hp_doubled(
        normal.iloc[
            40:80
        ].copy()
    )

    normal.to_csv(
        OUT_DIR /
        "synthetic_normal.csv",
        index=False
    )

    hp_off.to_csv(
        OUT_DIR /
        "synthetic_hp_off.csv",
        index=False
    )

    hp_doubled.to_csv(
        OUT_DIR /
        "synthetic_hp_doubled.csv",
        index=False
    )

    combined = pd.concat(
        [
            normal,
            hp_off,
            hp_doubled,
        ],
        ignore_index=True
    )

    combined.to_csv(
        OUT_DIR /
        "synthetic_demo_dataset.csv",
        index=False
    )

    print(
        "Synthetic public demo data created."
    )

    print(
        "Normal rows:",
        len(normal)
    )

    print(
        "HP_OFF rows:",
        len(hp_off)
    )

    print(
        "HP_DOUBLED rows:",
        len(hp_doubled)
    )

    print(
        "Combined rows:",
        len(combined)
    )


if __name__ == "__main__":
    main()
