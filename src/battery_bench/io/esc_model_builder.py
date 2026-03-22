#esc_model_builder.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from battery_bench.io.mat_loader import load_model_data
from battery_bench.models.esc_model import ESCModel


def _optional_array(raw: dict, key: str):
    """
    Return a float numpy array for raw[key] if present, else None.
    """
    if key not in raw or raw[key] is None:
        return None
    return np.asarray(raw[key], dtype=float).reshape(-1)


def build_esc_model_from_model_data(model_data) -> ESCModel:
    """
    Build an ESCModel from the output of load_model_data(...).

    Expected source keys from PANmodel.mat include:
        OCV0, OCVrel, SOC, OCV, SOC0, SOCrel, OCVeta, OCVQ,
        name, temps, etaParam, QParam, GParam, M0Param, MParam,
        R0Param, RCParam, RParam, dOCV0, dOCVrel
    """
    raw = model_data.raw

    model = ESCModel(
        name=str(raw.get("name", "unknown_model")),
        temps_c=np.asarray(raw["temps"], dtype=float).reshape(-1),
        soc_grid=np.asarray(raw["SOC"], dtype=float).reshape(-1),

        # Inverse lookup grid for SOC(OCV, T)
        ocv_grid=np.asarray(raw["OCV"], dtype=float).reshape(-1),

        # Forward lookup fields for OCV(SOC, T)
        ocv0=_optional_array(raw, "OCV0"),
        ocvrel=_optional_array(raw, "OCVrel"),

        # Inverse lookup fields for SOC(OCV, T)
        soc0=_optional_array(raw, "SOC0"),
        socrel=_optional_array(raw, "SOCrel"),

        # Additional source-model fields
        ocveta=_optional_array(raw, "OCVeta"),
        ocvq=_optional_array(raw, "OCVQ"),
        docv0=_optional_array(raw, "dOCV0"),
        docvrel=_optional_array(raw, "dOCVrel"),

        # Temperature-dependent parameters
        q_param=_optional_array(raw, "QParam"),
        eta_param=_optional_array(raw, "etaParam"),
        g_param=_optional_array(raw, "GParam"),
        m0_param=_optional_array(raw, "M0Param"),
        m_param=_optional_array(raw, "MParam"),
        r0_param=_optional_array(raw, "R0Param"),
        rc_param=_optional_array(raw, "RCParam"),
        r_param=_optional_array(raw, "RParam"),
    )

    if hasattr(model, "validate"):
        model.validate()

    return model


def load_esc_model(path: str | Path) -> ESCModel:
    """
    Convenience wrapper:
        path -> load_model_data(path) -> ESCModel
    """
    model_data = load_model_data(path)
    return build_esc_model_from_model_data(model_data)