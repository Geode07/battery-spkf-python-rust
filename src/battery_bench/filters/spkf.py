from __future__ import annotations

from typing import Any

import numpy as np

from battery_bench.esc.get_param_esc import get_param_esc
from battery_bench.esc.ocv import docv_from_soc_temp, ocv_from_soc_temp
from battery_bench.models.spkf_state import SPKFState
from battery_bench.models.esc_model import ESCModel


def _safe_cholesky(a: np.ndarray, jitter0: float = 1e-12, max_tries: int = 8) -> np.ndarray:
    """
    Cholesky with diagonal jitter for near-PSD covariance matrices.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Cholesky input must be square, got shape {a.shape}")

    jitter = float(jitter0)
    eye = np.eye(a.shape[0], dtype=float)

    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(a + jitter * eye)
        except np.linalg.LinAlgError:
            jitter *= 10.0

    raise np.linalg.LinAlgError("Unable to compute Cholesky factor, even with jitter.")


def _normalize_sigma_w(sigma_w: np.ndarray, nx: int) -> np.ndarray:
    """
    Normalize process-noise std vector to length nx.
    """
    sigma_w = np.asarray(sigma_w, dtype=float).reshape(-1)

    if sigma_w.size == 1:
        return np.repeat(sigma_w[0], nx)

    if sigma_w.size != nx:
        raise ValueError(
            f"sigma_w must have length 1 or nx={nx}; got shape {sigma_w.shape}"
        )

    return sigma_w


def _make_weights(na: int, h: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Sigma-point weights following the Plett-style SPKF convention:
        X = x ± h * S columns
        W0 = (h^2 - na) / h^2
        Wi = 1 / (2 h^2)
    """
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}")

    n_sigma = 2 * na + 1
    w_m = np.full(n_sigma, 1.0 / (2.0 * h * h), dtype=float)
    w_c = np.full(n_sigma, 1.0 / (2.0 * h * h), dtype=float)

    w0 = (h * h - na) / (h * h)
    w_m[0] = w0
    w_c[0] = w0

    return w_m, w_c


def _hysteresis_sign(current_a: float, prev_sign: float, eps: float = 1e-9) -> float:
    """
    Keep previous sign when current is effectively zero.
    """
    if current_a > eps:
        return 1.0
    if current_a < -eps:
        return -1.0
    return float(prev_sign)


def _state_eqn_esc(
    x: np.ndarray,
    current_a: float,
    temp_c: float,
    deltat: float,
    model: ESCModel,
    prev_sign: float,
) -> np.ndarray:
    """
    ESC process model for a simple 1-RC + hysteresis + SOC state vector:
        x = [ir, hk, z]

    Notes
    -----
    This is a practical Python port of the common Plett-style ESC structure.
    Depending on the exact MATLAB notebook, small refinements may still be needed
    for sign conventions or hysteresis tuning.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != 3:
        raise ValueError(f"Expected state length 3, got {x.size}")

    ir, hk, z = x
    current_sign = _hysteresis_sign(current_a, prev_sign)

    q = float(get_param_esc(model, "q_param", temp_c))
    eta = float(get_param_esc(model, "eta_param", temp_c))
    g = float(get_param_esc(model, "g_param", temp_c))
    rc_param = float(get_param_esc(model, "rc_param", temp_c))

    # RC branch
    # Interpret rc_param as time constant tau (seconds) when positive.
    # If rc_param is extremely small/non-positive, fall back to no memory.
    if rc_param > 1e-12:
        a_rc = float(np.exp(-abs(deltat / rc_param)))
    else:
        a_rc = 0.0

    ir_next = a_rc * ir + (1.0 - a_rc) * float(current_a)

    # Hysteresis state
    # Ah form follows the common ESC/SPKF structure.
    if q <= 1e-12:
        raise ValueError(f"q_param is too small/non-positive: {q}")

    gamma = abs(eta * current_a * g * deltat / (3600.0 * q))
    a_h = float(np.exp(-gamma))
    hk_next = a_h * hk + (a_h - 1.0) * current_sign

    # SOC
    z_next = z - (eta * deltat / (3600.0 * q)) * float(current_a)
    z_next = float(np.clip(z_next, -0.05, 1.05))

    return np.array([ir_next, hk_next, z_next], dtype=float)


def _output_eqn_esc(
    x: np.ndarray,
    current_a: float,
    temp_c: float,
    model: ESCModel,
    prev_sign: float,
) -> float:
    """
    ESC output equation for terminal voltage:
        v = OCV(z, T) + M*h - M0*sign(i) - R*ir - R0*i
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != 3:
        raise ValueError(f"Expected state length 3, got {x.size}")

    ir, hk, z = x
    current_sign = _hysteresis_sign(current_a, prev_sign)

    ocv = float(ocv_from_soc_temp(model, z, temp_c))
    m = float(get_param_esc(model, "m_param", temp_c))
    m0 = float(get_param_esc(model, "m0_param", temp_c))
    r = float(get_param_esc(model, "r_param", temp_c))
    r0 = float(get_param_esc(model, "r0_param", temp_c))

    vk = ocv + m * hk - m0 * current_sign - r * ir - r0 * float(current_a)
    return float(vk)


def init_spkf(
    model: ESCModel,
    soc0: float,
    *,
    sigma_x0: np.ndarray | None = None,
    sigma_w: np.ndarray | float | None = None,
    sigma_v: float = 1e-3,
    h: float = 3.0,
    q_bump: float = 5.0,
    prior_i: float = 0.0,
) -> SPKFState:
    """
    Initialize the SPKF state for a basic cell SOC estimator.

    Parameters
    ----------
    model : ESCModel
    soc0 : float
        Initial SOC estimate.
    sigma_x0 : np.ndarray | None
        Initial state covariance, default diagonal.
    sigma_w : np.ndarray | float | None
        Process-noise std vector. If scalar, applied to all states.
    sigma_v : float
        Measurement-noise std for terminal voltage.
    h : float
        Sigma-point scaling parameter.
    q_bump : float
        Covariance inflation factor when current sign changes.
    prior_i : float
        Previous current for bookkeeping.

    Returns
    -------
    SPKFState
    """
    nx = 3
    ny = 1
    nu = 1
    nw = nx
    nv = 1
    na = nx + nw + nv

    xhat = np.array([0.0, 0.0, float(soc0)], dtype=float)

    if sigma_x0 is None:
        sigma_x0 = np.diag([1e-6, 1e-6, 1e-4]).astype(float)
    else:
        sigma_x0 = np.asarray(sigma_x0, dtype=float)

    if sigma_w is None:
        sigma_w = np.array([1e-5, 1e-5, 1e-6], dtype=float)
    sigma_w = _normalize_sigma_w(np.asarray(sigma_w, dtype=float), nx)

    # Augmented noise covariance uses state process noise + measurement noise
    noise_var = np.concatenate([sigma_w**2, np.array([float(sigma_v) ** 2])])
    s_noise = _safe_cholesky(np.diag(noise_var))

    w_m, w_c = _make_weights(na=na, h=h)

    return SPKFState(
        xhat=xhat,
        sigma_x=sigma_x0,
        sigma_v=float(sigma_v),
        sigma_w=sigma_w,
        s_noise=s_noise,
        w_m=w_m,
        w_c=w_c,
        prior_i=float(prior_i),
        sign_ik=_hysteresis_sign(float(prior_i), 0.0),
        ir_idx=0,
        hk_idx=1,
        zk_idx=2,
        nx=nx,
        ny=ny,
        nu=nu,
        nw=nw,
        nv=nv,
        na=na,
        h=float(h),
        q_bump=float(q_bump),
        model=model,
        meta={},
    )


def iter_spkf(
    state: SPKFState,
    measured_v: float,
    current_a: float,
    temp_c: float,
    deltat: float,
) -> tuple[SPKFState, dict[str, Any]]:
    """
    Run one SPKF update step.

    Parameters
    ----------
    state : SPKFState
    measured_v : float
        Measured terminal voltage.
    current_a : float
        Applied/measured current.
    temp_c : float
        Cell temperature in Celsius.
    deltat : float
        Time step in seconds.

    Returns
    -------
    (new_state, info)
        new_state : SPKFState
        info : dict
            Diagnostics including prediction, innovation, Kalman gain, etc.
    """
    if state.model is None:
        raise ValueError("SPKFState.model is required for iter_spkf.")

    model = state.model
    nx, nw, nv, na = state.nx, state.nw, state.nv, state.na

    # Optional covariance inflation when current sign flips.
    prev_sign = _hysteresis_sign(state.prior_i, state.sign_ik)
    curr_sign = _hysteresis_sign(current_a, prev_sign)
    if curr_sign != prev_sign and curr_sign != 0.0:
        sigma_x = state.sigma_x.copy()
        sigma_x[state.zk_idx, state.zk_idx] *= float(state.q_bump)
    else:
        sigma_x = state.sigma_x.copy()

    # Augmented covariance
    pa = np.zeros((na, na), dtype=float)
    pa[:nx, :nx] = sigma_x
    pa[nx:, nx:] = state.s_noise @ state.s_noise.T

    xa = np.zeros(na, dtype=float)
    xa[:nx] = state.xhat

    sa = _safe_cholesky(pa)
    n_sigma = 2 * na + 1

    x_sigma = np.zeros((na, n_sigma), dtype=float)
    x_sigma[:, 0] = xa
    for j in range(na):
        delta = state.h * sa[:, j]
        x_sigma[:, j + 1] = xa + delta
        x_sigma[:, j + 1 + na] = xa - delta

    # -----------------------------
    # Time update
    # -----------------------------
    x_pred_sigma = np.zeros((nx, n_sigma), dtype=float)

    for j in range(n_sigma):
        x_j = x_sigma[:nx, j]
        w_j = x_sigma[nx:nx + nw, j]

        x_prop = _state_eqn_esc(
            x=x_j,
            current_a=float(current_a),
            temp_c=float(temp_c),
            deltat=float(deltat),
            model=model,
            prev_sign=prev_sign,
        )

        # additive process noise
        x_pred_sigma[:, j] = x_prop + w_j

    xhat_minus = x_pred_sigma @ state.w_m

    pxx = np.zeros((nx, nx), dtype=float)
    for j in range(n_sigma):
        dx = (x_pred_sigma[:, j] - xhat_minus).reshape(-1, 1)
        pxx += state.w_c[j] * (dx @ dx.T)

    # -----------------------------
    # Measurement update
    # -----------------------------
    y_sigma = np.zeros(n_sigma, dtype=float)
    for j in range(n_sigma):
        v_j = x_sigma[nx + nw:, j][0]
        y_nom = _output_eqn_esc(
            x=x_pred_sigma[:, j],
            current_a=float(current_a),
            temp_c=float(temp_c),
            model=model,
            prev_sign=curr_sign,
        )
        y_sigma[j] = y_nom + v_j

    yhat = float(y_sigma @ state.w_m)

    pyy = 0.0
    pxy = np.zeros(nx, dtype=float)

    for j in range(n_sigma):
        dy = float(y_sigma[j] - yhat)
        dx = x_pred_sigma[:, j] - xhat_minus
        pyy += state.w_c[j] * dy * dy
        pxy += state.w_c[j] * dx * dy

    if pyy <= 0:
        raise ValueError(f"Non-positive innovation variance pyy={pyy}")

    k_gain = pxy / pyy
    innovation = float(measured_v - yhat)

    xhat_plus = xhat_minus + k_gain * innovation
    pxx_plus = pxx - np.outer(k_gain, k_gain) * pyy

    # Symmetrize / stabilize covariance
    pxx_plus = 0.5 * (pxx_plus + pxx_plus.T)
    pxx_plus += 1e-12 * np.eye(nx)

    # Clip SOC to slightly extended bounds for numerical stability
    xhat_plus[state.zk_idx] = float(np.clip(xhat_plus[state.zk_idx], -0.05, 1.05))

    new_state = state.copy()
    new_state.xhat = xhat_plus
    new_state.sigma_x = pxx_plus
    new_state.prior_i = float(current_a)
    new_state.sign_ik = float(curr_sign)

    # Optional derived diagnostic
    try:
        docv = float(docv_from_soc_temp(model, new_state.z, temp_c))
    except Exception:
        docv = np.nan

    info = {
        "predicted_voltage": yhat,
        "measured_voltage": float(measured_v),
        "innovation": innovation,
        "kalman_gain": k_gain.copy(),
        "predicted_state": xhat_minus.copy(),
        "updated_state": xhat_plus.copy(),
        "predicted_cov": pxx.copy(),
        "updated_cov": pxx_plus.copy(),
        "innovation_variance": float(pyy),
        "current_sign": float(curr_sign),
        "docv_dz": docv,
    }

    return new_state, info


def run_spkf(
    state: SPKFState,
    voltage_v: np.ndarray,
    current_a: np.ndarray,
    temp_c: float | np.ndarray,
    deltat_s: float | np.ndarray,
) -> tuple[SPKFState, dict[str, np.ndarray]]:
    """
    Convenience runner over a full time series.

    Parameters
    ----------
    state : SPKFState
        Initialized state.
    voltage_v : array-like
    current_a : array-like
    temp_c : float or array-like
    deltat_s : float or array-like

    Returns
    -------
    (final_state, outputs)
        outputs includes estimated states, predicted voltage, innovation,
        and variance traces.
    """
    voltage_v = np.asarray(voltage_v, dtype=float).reshape(-1)
    current_a = np.asarray(current_a, dtype=float).reshape(-1)

    if voltage_v.size != current_a.size:
        raise ValueError(
            f"voltage_v and current_a must have same length, got {voltage_v.size} and {current_a.size}"
        )

    n = voltage_v.size

    if np.isscalar(temp_c):
        temp_arr = np.full(n, float(temp_c), dtype=float)
    else:
        temp_arr = np.asarray(temp_c, dtype=float).reshape(-1)
        if temp_arr.size != n:
            raise ValueError(f"temp_c length must match n={n}, got {temp_arr.size}")

    if np.isscalar(deltat_s):
        dt_arr = np.full(n, float(deltat_s), dtype=float)
    else:
        dt_arr = np.asarray(deltat_s, dtype=float).reshape(-1)
        if dt_arr.size != n:
            raise ValueError(f"deltat_s length must match n={n}, got {dt_arr.size}")

    x_hist = np.zeros((n, state.nx), dtype=float)
    pred_v_hist = np.zeros(n, dtype=float)
    innov_hist = np.zeros(n, dtype=float)
    pyy_hist = np.zeros(n, dtype=float)

    curr_state = state
    for k in range(n):
        curr_state, info = iter_spkf(
            state=curr_state,
            measured_v=float(voltage_v[k]),
            current_a=float(current_a[k]),
            temp_c=float(temp_arr[k]),
            deltat=float(dt_arr[k]),
        )
        x_hist[k, :] = curr_state.xhat
        pred_v_hist[k] = float(info["predicted_voltage"])
        innov_hist[k] = float(info["innovation"])
        pyy_hist[k] = float(info["innovation_variance"])

    outputs = {
        "xhat": x_hist,
        "ir": x_hist[:, state.ir_idx],
        "hk": x_hist[:, state.hk_idx],
        "soc": x_hist[:, state.zk_idx],
        "predicted_voltage": pred_v_hist,
        "innovation": innov_hist,
        "innovation_variance": pyy_hist,
    }

    return curr_state, outputs