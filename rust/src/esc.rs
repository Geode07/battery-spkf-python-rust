use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::model::EscModel;
use crate::ocv::ocv_from_soc_temp;
use crate::params::{get_param_esc, EscParams};

/// Simple ESC state container:
///     x = [ir, hk, soc]
#[derive(Clone, Debug)]
pub struct EscState {
    pub ir: f64,
    pub hk: f64,
    pub soc: f64,
}

impl EscState {
    pub fn validate(&self) -> PyResult<()> {
        if !self.ir.is_finite() || !self.hk.is_finite() || !self.soc.is_finite() {
            return Err(PyValueError::new_err(
                "EscState contains non-finite values",
            ));
        }
        Ok(())
    }
}

fn hysteresis_sign(current_a: f64, prev_sign: f64) -> f64 {
    let eps = 1e-9_f64;
    if current_a > eps {
        1.0
    } else if current_a < -eps {
        -1.0
    } else {
        prev_sign
    }
}

/// Port of the Python `_state_eqn_esc(...)`.
///
/// State layout:
///     x = [ir, hk, soc]
///
/// Notes
/// -----
/// - Interprets `rc` as the RC time constant.
/// - Uses the same practical ESC hysteresis form as the Python baseline.
/// - Clamps SOC to a slightly extended range for numerical stability.
pub fn state_eqn_esc(
    state: &EscState,
    current_a: f64,
    temp_c: f64,
    dt_s: f64,
    model: &EscModel,
    prev_sign: f64,
) -> PyResult<EscState> {
    state.validate()?;

    if !current_a.is_finite() {
        return Err(PyValueError::new_err("current_a must be finite"));
    }
    if !temp_c.is_finite() {
        return Err(PyValueError::new_err("temp_c must be finite"));
    }
    if !dt_s.is_finite() || dt_s <= 0.0 {
        return Err(PyValueError::new_err("dt_s must be finite and > 0"));
    }

    let params: EscParams = get_param_esc(model, temp_c)?;
    let current_sign = hysteresis_sign(current_a, prev_sign);

    // RC branch update
    let a_rc = if params.rc > 1e-12 {
        (-((dt_s / params.rc).abs())).exp()
    } else {
        0.0
    };
    let ir_next = a_rc * state.ir + (1.0 - a_rc) * current_a;

    // Hysteresis update
    let gamma = (params.eta * current_a * params.g * dt_s / (3600.0 * params.q)).abs();
    let a_h = (-gamma).exp();
    let hk_next = a_h * state.hk + (a_h - 1.0) * current_sign;

    // SOC update
    let mut soc_next = state.soc - (params.eta * dt_s / (3600.0 * params.q)) * current_a;
    soc_next = soc_next.clamp(-0.05, 1.05);

    let next_state = EscState {
        ir: ir_next,
        hk: hk_next,
        soc: soc_next,
    };

    next_state.validate()?;
    Ok(next_state)
}

/// Port of the Python `_output_eqn_esc(...)`.
///
/// Terminal voltage:
///     v = OCV(z, T) + M*h - M0*sign(i) - R*ir - R0*i
pub fn output_eqn_esc(
    state: &EscState,
    current_a: f64,
    temp_c: f64,
    model: &EscModel,
    prev_sign: f64,
) -> PyResult<f64> {
    state.validate()?;

    if !current_a.is_finite() {
        return Err(PyValueError::new_err("current_a must be finite"));
    }
    if !temp_c.is_finite() {
        return Err(PyValueError::new_err("temp_c must be finite"));
    }

    let params: EscParams = get_param_esc(model, temp_c)?;
    let current_sign = hysteresis_sign(current_a, prev_sign);

    let ocv = ocv_from_soc_temp(model, state.soc, temp_c)?;

    let vk = ocv
        + params.m * state.hk
        - params.m0 * current_sign
        - params.r * state.ir
        - params.r0 * current_a;

    if !vk.is_finite() {
        return Err(PyValueError::new_err(
            "output_eqn_esc produced a non-finite voltage",
        ));
    }

    Ok(vk)
}

/// Convenience helper for creating state from `[ir, hk, soc]`.
pub fn esc_state_from_vec(x: &[f64]) -> PyResult<EscState> {
    if x.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "Expected state vector of length 3, got {}",
            x.len()
        )));
    }

    let state = EscState {
        ir: x[0],
        hk: x[1],
        soc: x[2],
    };
    state.validate()?;
    Ok(state)
}

/// Convenience helper for exporting state as `[ir, hk, soc]`.
pub fn esc_state_to_vec(state: &EscState) -> Vec<f64> {
    vec![state.ir, state.hk, state.soc]
}