from __future__ import annotations

from typing import Any

import numpy as np

from battery_bench.models.esc_model import ESCModel


# -----------------------------
# Core interpolation helpers
# -----------------------------

def _interp1d(x: np.ndarray, y: np.ndarray, xq: float) -> np.ndarray:
    """
    Safe 1D linear interpolation.

    Parameters
    ----------
    x : (N,)
    y : (N,) or (N, k)
    xq : scalar

    Returns
    -------
    Interpolated value(s)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1:
        raise ValueError("x must be 1D")

    if xq <= x[0]:
        return y[0]
    if xq >= x[-1]:
        return y[-1]

    idx = np.searchsorted(x, xq) - 1
    idx = np.clip(idx, 0, len(x) - 2)

    x0, x1 = x[idx], x[idx + 1]
    w = (xq - x0) / (x1 - x0)

    return (1 - w) * y[idx] + w * y[idx + 1]


def _interp_param_over_temp(
    temps: np.ndarray,
    param: np.ndarray,
    temp_c: float,
) -> np.ndarray:
    """
    Interpolate parameter over temperature axis.

    Handles:
    - (N,) → scalar output
    - (N, k) → vector output
    - (k, N) → vector output (auto-detect orientation)
    """
    temps = np.asarray(temps, dtype=float)
    param = np.asarray(param, dtype=float)

    if param.ndim == 0:
        return param

    if param.ndim == 1:
        return _interp1d(temps, param, temp_c)

    if param.ndim == 2:
        # Case 1: shape (N_temps, k)
        if param.shape[0] == temps.size:
            return _interp1d(temps, param, temp_c)

        # Case 2: shape (k, N_temps)
        if param.shape[1] == temps.size:
            return _interp1d(temps, param.T, temp_c)

    raise ValueError(
        f"Unsupported param shape {param.shape} for temps shape {temps.shape}"
    )


# -----------------------------
# Public API
# -----------------------------

_PARAM_MAP = {
    # canonical python names
    "q_param": "q_param",
    "eta_param": "eta_param",
    "g_param": "g_param",
    "m0_param": "m0_param",
    "m_param": "m_param",
    "r0_param": "r0_param",
    "rc_param": "rc_param",
    "r_param": "r_param",

    # allow MATLAB-style names for convenience
    "QParam": "q_param",
    "etaParam": "eta_param",
    "GParam": "g_param",
    "M0Param": "m0_param",
    "MParam": "m_param",
    "R0Param": "r0_param",
    "RCParam": "rc_param",
    "RParam": "r_param",
}


def get_param_esc(
    model: ESCModel,
    param_name: str,
    temp_c: float,
) -> Any:
    """
    Retrieve ESC parameter at a given temperature.

    Parameters
    ----------
    model : ESCModel
    param_name : str
        Either Python-style ("q_param") or MATLAB-style ("QParam")
    temp_c : float

    Returns
    -------
    np.ndarray or scalar
    """
    if model.temps_c is None:
        raise ValueError("Model does not contain temperature grid (temps_c).")

    if param_name not in _PARAM_MAP:
        raise KeyError(
            f"Unknown parameter '{param_name}'. "
            f"Valid options: {list(_PARAM_MAP.keys())}"
        )

    attr_name = _PARAM_MAP[param_name]

    param = getattr(model, attr_name, None)

    if param is None:
        raise ValueError(f"Model is missing parameter '{attr_name}'")

    return _interp_param_over_temp(model.temps_c, param, temp_c)


# -----------------------------
# Convenience wrapper
# -----------------------------

def get_all_params_esc(model: ESCModel, temp_c: float) -> dict[str, Any]:
    """
    Retrieve all ESC parameters at a given temperature.

    Useful for debugging or bulk extraction.
    """
    results = {}

    for key in [
        "q_param",
        "eta_param",
        "g_param",
        "m0_param",
        "m_param",
        "r0_param",
        "rc_param",
        "r_param",
    ]:
        try:
            results[key] = get_param_esc(model, key, temp_c)
        except Exception:
            results[key] = None

    return results