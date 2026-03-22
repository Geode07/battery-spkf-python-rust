use pyo3::prelude::*;

mod model;
mod engine;
mod interp;
mod params;
mod ocv;
mod esc;

use crate::engine::BatterySpkfEngine;

#[pymodule]
fn spkf_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BatterySpkfEngine>()?;
    Ok(())
}