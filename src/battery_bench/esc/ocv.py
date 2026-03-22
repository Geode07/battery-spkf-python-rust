from __future__ import annotations

import numpy as np

from battery_bench.models.esc_model import ESCModel


def _as_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _interp_clamped(x_grid: np.ndarray, y_grid: np.ndarray, xq):
    """
    1D linear interpolation with endpoint clamping.

    Parameters
    ----------
    x_grid : np.ndarray
        Monotone 1D grid.
    y_grid : np.ndarray
        1D values aligned with x_grid.
    xq : float or array-like
        Query point(s).

    Returns
    -------
    float or np.ndarray
        Interpolated value(s), clamped to endpoints outside the grid.
    """
    x_grid = _as_float_array(x_grid).reshape(-1)
    y_grid = _as_float_array(y_grid).reshape(-1)

    if x_grid.ndim != 1 or y_grid.ndim != 1:
        raise ValueError("x_grid and y_grid must be 1D.")
    if x_grid.size != y_grid.size:
        raise ValueError(
            f"x_grid and y_grid must have same length, got {x_grid.size} and {y_grid.size}."
        )
    if x_grid.size < 2:
        raise ValueError("Interpolation grid must contain at least 2 points.")

    xq_arr = _as_float_array(xq)
    xq_flat = xq_arr.reshape(-1)

    yq = np.interp(xq_flat, x_grid, y_grid, left=y_grid[0], right=y_grid[-1])
    yq = yq.reshape(xq_arr.shape)

    if np.isscalar(xq) or xq_arr.ndim == 0:
        return float(yq.reshape(-1)[0])
    return yq


def _require_forward_ocv_fields(model: ESCModel) -> None:
    missing = []
    if model.soc_grid is None:
        missing.append("soc_grid")
    if model.ocv0 is None:
        missing.append("ocv0")
    if model.ocvrel is None:
        missing.append("ocvrel")

    if missing:
        raise ValueError(
            f"ESCModel is missing forward OCV lookup fields: {missing}"
        )


def _require_inverse_ocv_fields(model: ESCModel) -> None:
    missing = []
    if model.ocv_grid is None:
        missing.append("ocv_grid")
    if model.soc0 is None:
        missing.append("soc0")
    if model.socrel is None:
        missing.append("socrel")

    if missing:
        raise ValueError(
            f"ESCModel is missing inverse OCV lookup fields: {missing}"
        )


def _require_docv_fields(model: ESCModel) -> None:
    missing = []
    if model.soc_grid is None:
        missing.append("soc_grid")
    if model.docv0 is None:
        missing.append("docv0")
    if model.docvrel is None:
        missing.append("docvrel")

    if missing:
        raise ValueError(
            f"ESCModel is missing dOCV lookup fields: {missing}"
        )


def ocv_from_soc_temp(model: ESCModel, soc, temp_c: float):
    """
    Forward lookup: OCV(SOC, T)

    Uses the matched forward grids:
    - model.soc_grid
    - model.ocv0
    - model.ocvrel

    Formula:
        OCV(SOC, T) = OCV0(SOC) + T * OCVrel(SOC)

    Parameters
    ----------
    model : ESCModel
    soc : float or array-like
        State of charge query point(s), typically nominally in [0, 1].
    temp_c : float
        Temperature in Celsius.

    Returns
    -------
    float or np.ndarray
        OCV estimate(s).
    """
    _require_forward_ocv_fields(model)

    soc_grid = _as_float_array(model.soc_grid).reshape(-1)
    ocv0 = _as_float_array(model.ocv0).reshape(-1)
    ocvrel = _as_float_array(model.ocvrel).reshape(-1)

    if not (soc_grid.size == ocv0.size == ocvrel.size):
        raise ValueError(
            "Forward OCV fields must have matching lengths: "
            f"soc_grid={soc_grid.size}, ocv0={ocv0.size}, ocvrel={ocvrel.size}"
        )

    ocv0_q = _interp_clamped(soc_grid, ocv0, soc)
    ocvrel_q = _interp_clamped(soc_grid, ocvrel, soc)

    return ocv0_q + float(temp_c) * ocvrel_q


def soc_from_ocv_temp(model: ESCModel, ocv, temp_c: float):
    """
    Inverse lookup: SOC(OCV, T)

    Uses the matched inverse grids:
    - model.ocv_grid
    - model.soc0
    - model.socrel

    Formula:
        SOC(OCV, T) = SOC0(OCV) + T * SOCrel(OCV)

    Parameters
    ----------
    model : ESCModel
    ocv : float or array-like
        OCV query point(s).
    temp_c : float
        Temperature in Celsius.

    Returns
    -------
    float or np.ndarray
        SOC estimate(s), typically near [0, 1], though table endpoints may extend slightly.
    """
    _require_inverse_ocv_fields(model)

    ocv_grid = _as_float_array(model.ocv_grid).reshape(-1)
    soc0 = _as_float_array(model.soc0).reshape(-1)
    socrel = _as_float_array(model.socrel).reshape(-1)

    if not (ocv_grid.size == soc0.size == socrel.size):
        raise ValueError(
            "Inverse OCV fields must have matching lengths: "
            f"ocv_grid={ocv_grid.size}, soc0={soc0.size}, socrel={socrel.size}"
        )

    soc0_q = _interp_clamped(ocv_grid, soc0, ocv)
    socrel_q = _interp_clamped(ocv_grid, socrel, ocv)

    return soc0_q + float(temp_c) * socrel_q


def docv_from_soc_temp(model: ESCModel, soc, temp_c: float):
    """
    Derivative lookup: dOCV/dSOC at (SOC, T)

    Uses:
    - model.soc_grid
    - model.docv0
    - model.docvrel

    Formula:
        dOCV/dSOC(SOC, T) = dOCV0(SOC) + T * dOCVrel(SOC)

    Parameters
    ----------
    model : ESCModel
    soc : float or array-like
        State of charge query point(s).
    temp_c : float
        Temperature in Celsius.

    Returns
    -------
    float or np.ndarray
        dOCV/dSOC estimate(s).
    """
    _require_docv_fields(model)

    soc_grid = _as_float_array(model.soc_grid).reshape(-1)
    docv0 = _as_float_array(model.docv0).reshape(-1)
    docvrel = _as_float_array(model.docvrel).reshape(-1)

    if not (soc_grid.size == docv0.size == docvrel.size):
        raise ValueError(
            "dOCV fields must have matching lengths: "
            f"soc_grid={soc_grid.size}, docv0={docv0.size}, docvrel={docvrel.size}"
        )

    docv0_q = _interp_clamped(soc_grid, docv0, soc)
    docvrel_q = _interp_clamped(soc_grid, docvrel, soc)

    return docv0_q + float(temp_c) * docvrel_q


def ocv_lookup_summary(model: ESCModel) -> dict:
    """
    Lightweight diagnostic summary of forward/inverse OCV lookup structure.
    """
    return {
        "soc_grid_shape": None if model.soc_grid is None else tuple(np.asarray(model.soc_grid).shape),
        "ocv0_shape": None if model.ocv0 is None else tuple(np.asarray(model.ocv0).shape),
        "ocvrel_shape": None if model.ocvrel is None else tuple(np.asarray(model.ocvrel).shape),
        "docv0_shape": None if model.docv0 is None else tuple(np.asarray(model.docv0).shape),
        "docvrel_shape": None if model.docvrel is None else tuple(np.asarray(model.docvrel).shape),
        "ocv_grid_shape": None if model.ocv_grid is None else tuple(np.asarray(model.ocv_grid).shape),
        "soc0_shape": None if model.soc0 is None else tuple(np.asarray(model.soc0).shape),
        "socrel_shape": None if model.socrel is None else tuple(np.asarray(model.socrel).shape),
    }