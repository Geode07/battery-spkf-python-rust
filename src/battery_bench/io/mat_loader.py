from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


# -----------------------------
# Dataclasses
# -----------------------------

@dataclass
class DynData:
    time_s: np.ndarray
    current_a: np.ndarray
    voltage_v: np.ndarray
    soc: np.ndarray | None = None

    @property
    def dt(self) -> float:
        if len(self.time_s) < 2:
            raise ValueError("Need at least two time points to compute dt.")
        return float(self.time_s[1] - self.time_s[0])


@dataclass
class ModelData:
    raw: dict[str, Any]


# -----------------------------
# MATLAB struct conversion
# -----------------------------

def _matobj_to_dict(obj: Any) -> Any:
    """
    Recursively convert scipy MATLAB objects into Python-native dict/list/scalars.
    """
    # MATLAB struct object
    if hasattr(obj, "_fieldnames"):
        return {field: _matobj_to_dict(getattr(obj, field)) for field in obj._fieldnames}

    # Object arrays from MATLAB cell arrays / nested structures
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            if obj.ndim == 0:
                return _matobj_to_dict(obj.item())
            return [_matobj_to_dict(x) for x in obj.flat]
        return np.asarray(obj)

    return obj


def load_mat_file(path: str | Path) -> dict[str, Any]:
    """
    Load a MATLAB .mat file into a Python dict with nested structs converted.
    Works for classic MAT files supported by scipy.io.loadmat.
    """
    path = Path(path)
    data = loadmat(path, struct_as_record=False, squeeze_me=True)

    # Drop MATLAB metadata keys
    cleaned = {k: v for k, v in data.items() if not k.startswith("__")}

    return {k: _matobj_to_dict(v) for k, v in cleaned.items()}


# -----------------------------
# Array helpers
# -----------------------------

def _as_1d_float_array(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    return arr


def _require_key(d: dict[str, Any], key: str, context: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing key '{key}' in {context}. Available keys: {list(d.keys())}")
    return d[key]


# -----------------------------
# Public loaders
# -----------------------------

def load_dyn_data(path: str | Path, script_name: str = "script1") -> DynData:
    """
    Load dynamic battery test data from a .mat file like PANdata_P45.mat.

    Expected MATLAB structure from your notebook:
        DYNData.script1.time
        DYNData.script1.current
        DYNData.script1.voltage
        DYNData.script1.soc
    """
    data = load_mat_file(path)

    dyn = _require_key(data, "DYNData", "top-level MAT file")

    if not isinstance(dyn, dict):
        raise TypeError(f"DYNData should be a dict after conversion, got {type(dyn)}")

    script = _require_key(dyn, script_name, "DYNData")

    if not isinstance(script, dict):
        raise TypeError(f"DYNData.{script_name} should be a dict after conversion, got {type(script)}")

    time_s = _as_1d_float_array(_require_key(script, "time", f"DYNData.{script_name}"), "time")
    current_a = _as_1d_float_array(_require_key(script, "current", f"DYNData.{script_name}"), "current")
    voltage_v = _as_1d_float_array(_require_key(script, "voltage", f"DYNData.{script_name}"), "voltage")

    soc = script.get("soc")
    soc_arr = _as_1d_float_array(soc, "soc") if soc is not None else None

    n = len(time_s)
    if len(current_a) != n or len(voltage_v) != n or (soc_arr is not None and len(soc_arr) != n):
        raise ValueError(
            "Time, current, voltage, and soc arrays must all have the same length. "
            f"Got len(time)={len(time_s)}, len(current)={len(current_a)}, "
            f"len(voltage)={len(voltage_v)}, len(soc)={None if soc_arr is None else len(soc_arr)}"
        )

    return DynData(
        time_s=time_s,
        current_a=current_a,
        voltage_v=voltage_v,
        soc=soc_arr,
    )


def load_model_data(path: str | Path, model_key: str = "model") -> ModelData:
    """
    Load model data from a .mat file like PANmodel.mat.

    Expected MATLAB structure from your notebook:
        model
    """
    data = load_mat_file(path)

    model = _require_key(data, model_key, "top-level MAT file")

    if not isinstance(model, dict):
        raise TypeError(f"{model_key} should be a dict after conversion, got {type(model)}")

    return ModelData(raw=model)


# -----------------------------
# Inspection helpers
# -----------------------------

def summarize_mat_file(path: str | Path) -> dict[str, Any]:
    """
    Return a lightweight summary of top-level keys and basic types/shapes.
    Useful when reverse-engineering unfamiliar .mat files.
    """
    data = load_mat_file(path)
    summary: dict[str, Any] = {}

    for key, value in data.items():
        if isinstance(value, dict):
            summary[key] = {
                "type": "dict",
                "keys": list(value.keys()),
            }
        elif isinstance(value, np.ndarray):
            summary[key] = {
                "type": "ndarray",
                "shape": value.shape,
                "dtype": str(value.dtype),
            }
        else:
            summary[key] = {
                "type": type(value).__name__,
            }

    return summary


def print_mat_summary(path: str | Path) -> None:
    summary = summarize_mat_file(path)
    print(f"Summary for: {Path(path)}")
    for key, info in summary.items():
        print(f"- {key}: {info}")