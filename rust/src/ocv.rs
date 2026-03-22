use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::interp::interp_clamped;
use crate::model::EscModel;

/// Forward lookup:
///     OCV(SOC, T) = OCV0(SOC) + T * OCVrel(SOC)
pub fn ocv_from_soc_temp(model: &EscModel, soc: f64, temp_c: f64) -> PyResult<f64> {
    if !soc.is_finite() {
        return Err(PyValueError::new_err("soc must be finite"));
    }
    if !temp_c.is_finite() {
        return Err(PyValueError::new_err("temp_c must be finite"));
    }

    if model.soc_grid.len() != model.ocv0.len() || model.soc_grid.len() != model.ocvrel.len() {
        return Err(PyValueError::new_err(format!(
            "Forward OCV arrays must have matching lengths: soc_grid={}, ocv0={}, ocvrel={}",
            model.soc_grid.len(),
            model.ocv0.len(),
            model.ocvrel.len()
        )));
    }

    let ocv0_q = interp_clamped(&model.soc_grid, &model.ocv0, soc)?;
    let ocvrel_q = interp_clamped(&model.soc_grid, &model.ocvrel, soc)?;

    Ok(ocv0_q + temp_c * ocvrel_q)
}

/// Inverse lookup:
///     SOC(OCV, T) = SOC0(OCV) + T * SOCrel(OCV)
pub fn soc_from_ocv_temp(model: &EscModel, ocv: f64, temp_c: f64) -> PyResult<f64> {
    if !ocv.is_finite() {
        return Err(PyValueError::new_err("ocv must be finite"));
    }
    if !temp_c.is_finite() {
        return Err(PyValueError::new_err("temp_c must be finite"));
    }

    if model.ocv_grid.len() != model.soc0.len() || model.ocv_grid.len() != model.socrel.len() {
        return Err(PyValueError::new_err(format!(
            "Inverse OCV arrays must have matching lengths: ocv_grid={}, soc0={}, socrel={}",
            model.ocv_grid.len(),
            model.soc0.len(),
            model.socrel.len()
        )));
    }

    let soc0_q = interp_clamped(&model.ocv_grid, &model.soc0, ocv)?;
    let socrel_q = interp_clamped(&model.ocv_grid, &model.socrel, ocv)?;

    Ok(soc0_q + temp_c * socrel_q)
}

/// Derivative lookup:
///     dOCV/dSOC(SOC, T) = dOCV0(SOC) + T * dOCVrel(SOC)
pub fn docv_from_soc_temp(model: &EscModel, soc: f64, temp_c: f64) -> PyResult<f64> {
    if !soc.is_finite() {
        return Err(PyValueError::new_err("soc must be finite"));
    }
    if !temp_c.is_finite() {
        return Err(PyValueError::new_err("temp_c must be finite"));
    }

    if model.soc_grid.len() != model.docv0.len() || model.soc_grid.len() != model.docvrel.len() {
        return Err(PyValueError::new_err(format!(
            "dOCV arrays must have matching lengths: soc_grid={}, docv0={}, docvrel={}",
            model.soc_grid.len(),
            model.docv0.len(),
            model.docvrel.len()
        )));
    }

    let docv0_q = interp_clamped(&model.soc_grid, &model.docv0, soc)?;
    let docvrel_q = interp_clamped(&model.soc_grid, &model.docvrel, soc)?;

    Ok(docv0_q + temp_c * docvrel_q)
}