use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::interp::interp_clamped;
use crate::model::EscModel;

/// Container for temperature-interpolated ESC parameters.
#[derive(Clone, Debug)]
pub struct EscParams {
    pub q: f64,
    pub eta: f64,
    pub g: f64,
    pub m0: f64,
    pub m: f64,
    pub r0: f64,
    pub rc: f64,
    pub r: f64,
}

impl EscParams {
    pub fn validate(&self) -> PyResult<()> {
        if self.q <= 0.0 || !self.q.is_finite() {
            return Err(PyValueError::new_err(format!(
                "Invalid capacity q: {}",
                self.q
            )));
        }
        if !self.eta.is_finite()
            || !self.g.is_finite()
            || !self.m0.is_finite()
            || !self.m.is_finite()
            || !self.r0.is_finite()
            || !self.rc.is_finite()
            || !self.r.is_finite()
        {
            return Err(PyValueError::new_err(
                "One or more ESC parameters are non-finite",
            ));
        }
        Ok(())
    }
}

/// Interpolate ESC parameters at a given temperature (°C).
///
/// Mirrors Python `get_param_esc(...)`:
/// - 1D interpolation over `model.temps_c`
/// - clamped outside bounds
pub fn get_param_esc(model: &EscModel, temp_c: f64) -> PyResult<EscParams> {
    if !temp_c.is_finite() {
        return Err(PyValueError::new_err("temp_c must be finite"));
    }

    let temps = &model.temps_c;

    if temps.len() < 2 {
        return Err(PyValueError::new_err(
            "model.temps_c must have at least 2 points",
        ));
    }

    // Interpolate each parameter independently
    let q = interp_clamped(temps, &model.q_param, temp_c)?;
    let eta = interp_clamped(temps, &model.eta_param, temp_c)?;
    let g = interp_clamped(temps, &model.g_param, temp_c)?;
    let m0 = interp_clamped(temps, &model.m0_param, temp_c)?;
    let m = interp_clamped(temps, &model.m_param, temp_c)?;
    let r0 = interp_clamped(temps, &model.r0_param, temp_c)?;
    let rc = interp_clamped(temps, &model.rc_param, temp_c)?;
    let r = interp_clamped(temps, &model.r_param, temp_c)?;

    let params = EscParams {
        q,
        eta,
        g,
        m0,
        m,
        r0,
        rc,
        r,
    };

    params.validate()?;
    Ok(params)
}