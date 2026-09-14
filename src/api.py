#!/usr/bin/env python3

from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .predict import EnergyAnomalyDetector


app = FastAPI(
    title="ML-Based Energy Anomaly Detection API",
    description="REST API for anomaly detection in synthetic energy-system data.",
    version="1.0.0",
)


detector = EnergyAnomalyDetector()


class EnergyOperatingPoint(BaseModel):
    hp_expected_active: Literal[0, 1]

    heat_demand_mw: float
    base_electric_load_mw: float
    pv_generation_mw: float

    hp_electric_power_mw: float
    hp_thermal_power_mw: float
    cop: Optional[float] = None

    p2h_electric_power_mw: float
    chp_electric_power_mw: float

    coupling_bus_voltage_pu: float
    grid_import_mw: float
    max_line_loading_pct: float

    unserved_heat_mw: float

    network_supply_temp_k: float
    network_return_temp_k: float


@app.get("/")
def root():
    return {
        "service": "ML-Based Energy Anomaly Detection API",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "threshold": detector.threshold
    }


@app.post("/predict")
def predict(data: EnergyOperatingPoint):

    try:
        result = detector.predict_one(
            data.model_dump()
        )

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed."
        ) from exc
