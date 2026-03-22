use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::model::{esc_model_from_pydict, spkf_config_from_pydict, EscModel, SpkfConfig};
use crate::esc::{output_eqn_esc, state_eqn_esc, EscState};

#[pyclass]
pub struct BatterySpkfEngine {
    model: EscModel,
    config: SpkfConfig,

    // Core state: [ir, hk, soc]
    ir: f64,
    hk: f64,
    soc: f64,

    // Simple covariance placeholders for Phase 1
    sigma_x_diag: Vec<f64>,

    // Bookkeeping
    prior_i: f64,
    sign_ik: f64,
}

impl BatterySpkfEngine {
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

    fn validate_soc(soc0: f64) -> PyResult<()> {
        if !soc0.is_finite() {
            return Err(PyValueError::new_err("soc0 must be finite"));
        }
        Ok(())
    }
}

#[pymethods]
impl BatterySpkfEngine {
    #[new]
    pub fn new(model_dict: &Bound<'_, PyDict>, config_dict: &Bound<'_, PyDict>, soc0: f64) -> PyResult<Self> {
        Self::validate_soc(soc0)?;

        let model = esc_model_from_pydict(model_dict)?;
        let config = spkf_config_from_pydict(config_dict)?;

        let sigma_x_diag = config.sigma_x0_diag.clone();
        let prior_i = config.prior_i;
        let sign_ik = Self::hysteresis_sign(prior_i, 0.0);

        Ok(Self {
            model,
            config,
            ir: 0.0,
            hk: 0.0,
            soc: soc0,
            sigma_x_diag,
            prior_i,
            sign_ik,
        })
    }

    /// Phase 1 placeholder step:
    /// - preserves stateful engine shape
    /// - returns Python-friendly dict
    /// - keeps SOC bounded
    ///
    /// This is intentionally simple before full ESC/SPKF math is ported.
    pub fn step(
        &mut self,
        measured_v: f64,
        current_a: f64,
        temp_c: f64,
        dt_s: f64,
    ) -> PyResult<Py<PyDict>> {
        if !measured_v.is_finite() {
            return Err(PyValueError::new_err("measured_v must be finite"));
        }
        if !current_a.is_finite() {
            return Err(PyValueError::new_err("current_a must be finite"));
        }
        if !temp_c.is_finite() {
            return Err(PyValueError::new_err("temp_c must be finite"));
        }
        if !dt_s.is_finite() || dt_s <= 0.0 {
            return Err(PyValueError::new_err("dt_s must be finite and > 0"));
        }

        let prev_sign = self.sign_ik;

        let state = EscState {
            ir: self.ir,
            hk: self.hk,
            soc: self.soc,
        };

        // Predict next state using real ESC state equation
        let next_state = state_eqn_esc(
            &state,
            current_a,
            temp_c,
            dt_s,
            &self.model,
            prev_sign,
        )?;

        // Predict voltage using real ESC output equation
        let predicted_voltage = output_eqn_esc(
            &next_state,
            current_a,
            temp_c,
            &self.model,
            prev_sign,
        )?;

        let innovation = measured_v - predicted_voltage;

        // For now, state update = predicted state only
        // (full SPKF correction comes in the next phase)
        self.ir = next_state.ir;
        self.hk = next_state.hk;
        self.soc = next_state.soc;

        self.sign_ik = Self::hysteresis_sign(current_a, prev_sign);
        self.prior_i = current_a;

        Python::attach(|py| {
            let out = PyDict::new(py);
            out.set_item("soc", self.soc)?;
            out.set_item("predicted_voltage", predicted_voltage)?;
            out.set_item("innovation", innovation)?;
            out.set_item("innovation_variance", self.config.sigma_v * self.config.sigma_v)?;
            out.set_item("ir", self.ir)?;
            out.set_item("hk", self.hk)?;
            Ok(out.into())
        })
    }

    pub fn get_state(&self) -> (f64, f64, f64) {
        (self.ir, self.hk, self.soc)
    }

    pub fn get_sigma_x_diag(&self) -> Vec<f64> {
        self.sigma_x_diag.clone()
    }

    pub fn get_prior_i(&self) -> f64 {
        self.prior_i
    }

    pub fn get_sign_ik(&self) -> f64 {
        self.sign_ik
    }

    pub fn reset(&mut self, soc0: f64) -> PyResult<()> {
        Self::validate_soc(soc0)?;
        self.ir = 0.0;
        self.hk = 0.0;
        self.soc = soc0;
        self.sigma_x_diag = self.config.sigma_x0_diag.clone();
        self.prior_i = self.config.prior_i;
        self.sign_ik = Self::hysteresis_sign(self.prior_i, 0.0);
        Ok(())
    }

    pub fn summary(&self) -> PyResult<Py<PyDict>> {
        Python::attach(|py| {
            let d = PyDict::new(py);
            d.set_item("model_name", self.model.name.clone())?;
            d.set_item("n_temps", self.model.n_temps())?;
            d.set_item("n_soc", self.model.n_soc())?;
            d.set_item("n_ocv", self.model.n_ocv())?;
            d.set_item("ir", self.ir)?;
            d.set_item("hk", self.hk)?;
            d.set_item("soc", self.soc)?;
            d.set_item("sigma_x_diag", self.sigma_x_diag.clone())?;
            d.set_item("sigma_w", self.config.sigma_w.clone())?;
            d.set_item("sigma_v", self.config.sigma_v)?;
            d.set_item("h", self.config.h)?;
            d.set_item("q_bump", self.config.q_bump)?;
            d.set_item("prior_i", self.prior_i)?;
            d.set_item("sign_ik", self.sign_ik)?;
            Ok(d.into())
        })
    }
}