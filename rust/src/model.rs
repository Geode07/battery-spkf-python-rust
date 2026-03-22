use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Flattened ESC model for Rust-side SPKF/ESC calculations.
///
/// This mirrors the Python ESCModel after export through a flat dict.
#[derive(Clone, Debug)]
pub struct EscModel {
    pub name: String,

    pub temps_c: Vec<f64>,
    pub soc_grid: Vec<f64>,
    pub ocv_grid: Vec<f64>,

    pub ocv0: Vec<f64>,
    pub ocvrel: Vec<f64>,

    pub soc0: Vec<f64>,
    pub socrel: Vec<f64>,

    pub docv0: Vec<f64>,
    pub docvrel: Vec<f64>,

    pub q_param: Vec<f64>,
    pub eta_param: Vec<f64>,
    pub g_param: Vec<f64>,
    pub m0_param: Vec<f64>,
    pub m_param: Vec<f64>,
    pub r0_param: Vec<f64>,
    pub rc_param: Vec<f64>,
    pub r_param: Vec<f64>,
}

impl EscModel {
    pub fn validate(&self) -> PyResult<()> {
        if self.temps_c.is_empty() {
            return Err(PyValueError::new_err("temps_c cannot be empty"));
        }
        if self.soc_grid.len() < 2 {
            return Err(PyValueError::new_err(
                "soc_grid must contain at least 2 points",
            ));
        }
        if self.ocv_grid.len() < 2 {
            return Err(PyValueError::new_err(
                "ocv_grid must contain at least 2 points",
            ));
        }

        // Forward OCV lookup arrays on soc_grid
        Self::require_same_len("ocv0", self.ocv0.len(), "soc_grid", self.soc_grid.len())?;
        Self::require_same_len(
            "ocvrel",
            self.ocvrel.len(),
            "soc_grid",
            self.soc_grid.len(),
        )?;
        Self::require_same_len(
            "docv0",
            self.docv0.len(),
            "soc_grid",
            self.soc_grid.len(),
        )?;
        Self::require_same_len(
            "docvrel",
            self.docvrel.len(),
            "soc_grid",
            self.soc_grid.len(),
        )?;

        // Inverse SOC lookup arrays on ocv_grid
        Self::require_same_len("soc0", self.soc0.len(), "ocv_grid", self.ocv_grid.len())?;
        Self::require_same_len(
            "socrel",
            self.socrel.len(),
            "ocv_grid",
            self.ocv_grid.len(),
        )?;

        // Temperature-dependent parameter arrays on temps_c
        let n_temps = self.temps_c.len();
        Self::require_same_len("q_param", self.q_param.len(), "temps_c", n_temps)?;
        Self::require_same_len("eta_param", self.eta_param.len(), "temps_c", n_temps)?;
        Self::require_same_len("g_param", self.g_param.len(), "temps_c", n_temps)?;
        Self::require_same_len("m0_param", self.m0_param.len(), "temps_c", n_temps)?;
        Self::require_same_len("m_param", self.m_param.len(), "temps_c", n_temps)?;
        Self::require_same_len("r0_param", self.r0_param.len(), "temps_c", n_temps)?;
        Self::require_same_len("rc_param", self.rc_param.len(), "temps_c", n_temps)?;
        Self::require_same_len("r_param", self.r_param.len(), "temps_c", n_temps)?;

        Ok(())
    }

    fn require_same_len(
        left_name: &str,
        left_len: usize,
        right_name: &str,
        right_len: usize,
    ) -> PyResult<()> {
        if left_len != right_len {
            return Err(PyValueError::new_err(format!(
                "{left_name} length {left_len} must match {right_name} length {right_len}"
            )));
        }
        Ok(())
    }

    pub fn n_temps(&self) -> usize {
        self.temps_c.len()
    }

    pub fn n_soc(&self) -> usize {
        self.soc_grid.len()
    }

    pub fn n_ocv(&self) -> usize {
        self.ocv_grid.len()
    }
}

/// SPKF config passed from Python into Rust.
///
/// Kept intentionally simple/flat for Phase 1.
#[derive(Clone, Debug)]
pub struct SpkfConfig {
    pub sigma_x0_diag: Vec<f64>,
    pub sigma_w: Vec<f64>,
    pub sigma_v: f64,
    pub h: f64,
    pub q_bump: f64,
    pub prior_i: f64,
}

impl SpkfConfig {
    pub fn validate(&self) -> PyResult<()> {
        if self.sigma_x0_diag.len() != 3 {
            return Err(PyValueError::new_err(format!(
                "sigma_x0_diag must have length 3, got {}",
                self.sigma_x0_diag.len()
            )));
        }
        if self.sigma_w.len() != 3 {
            return Err(PyValueError::new_err(format!(
                "sigma_w must have length 3, got {}",
                self.sigma_w.len()
            )));
        }
        if self.sigma_v <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "sigma_v must be positive, got {}",
                self.sigma_v
            )));
        }
        if self.h <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "h must be positive, got {}",
                self.h
            )));
        }
        if self.q_bump <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "q_bump must be positive, got {}",
                self.q_bump
            )));
        }
        Ok(())
    }
}

fn extract_required_vec_f64(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<f64>> {
    let obj = dict
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("Missing required key '{key}'")))?;

    obj.extract::<Vec<f64>>().map_err(|_| {
        PyValueError::new_err(format!(
            "Key '{key}' must be convertible to list[float]"
        ))
    })
}

fn extract_required_f64(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<f64> {
    let obj = dict
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("Missing required key '{key}'")))?;

    obj.extract::<f64>()
        .map_err(|_| PyValueError::new_err(format!("Key '{key}' must be a float")))
}

fn extract_optional_string(dict: &Bound<'_, PyDict>, key: &str, default: &str) -> PyResult<String> {
    match dict.get_item(key)? {
        Some(obj) => obj
            .extract::<String>()
            .map_err(|_| PyValueError::new_err(format!("Key '{key}' must be a string"))),
        None => Ok(default.to_string()),
    }
}

pub fn esc_model_from_pydict(dict: &Bound<'_, PyDict>) -> PyResult<EscModel> {
    let model = EscModel {
        name: extract_optional_string(dict, "name", "unknown_model")?,

        temps_c: extract_required_vec_f64(dict, "temps_c")?,
        soc_grid: extract_required_vec_f64(dict, "soc_grid")?,
        ocv_grid: extract_required_vec_f64(dict, "ocv_grid")?,

        ocv0: extract_required_vec_f64(dict, "ocv0")?,
        ocvrel: extract_required_vec_f64(dict, "ocvrel")?,

        soc0: extract_required_vec_f64(dict, "soc0")?,
        socrel: extract_required_vec_f64(dict, "socrel")?,

        docv0: extract_required_vec_f64(dict, "docv0")?,
        docvrel: extract_required_vec_f64(dict, "docvrel")?,

        q_param: extract_required_vec_f64(dict, "q_param")?,
        eta_param: extract_required_vec_f64(dict, "eta_param")?,
        g_param: extract_required_vec_f64(dict, "g_param")?,
        m0_param: extract_required_vec_f64(dict, "m0_param")?,
        m_param: extract_required_vec_f64(dict, "m_param")?,
        r0_param: extract_required_vec_f64(dict, "r0_param")?,
        rc_param: extract_required_vec_f64(dict, "rc_param")?,
        r_param: extract_required_vec_f64(dict, "r_param")?,
    };

    model.validate()?;
    Ok(model)
}

pub fn spkf_config_from_pydict(dict: &Bound<'_, PyDict>) -> PyResult<SpkfConfig> {
    let cfg = SpkfConfig {
        sigma_x0_diag: extract_required_vec_f64(dict, "sigma_x0_diag")?,
        sigma_w: extract_required_vec_f64(dict, "sigma_w")?,
        sigma_v: extract_required_f64(dict, "sigma_v")?,
        h: extract_required_f64(dict, "h")?,
        q_bump: extract_required_f64(dict, "q_bump")?,
        prior_i: extract_required_f64(dict, "prior_i")?,
    };

    cfg.validate()?;
    Ok(cfg)
}