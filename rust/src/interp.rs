use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Clamped 1D linear interpolation.
///
/// Behavior matches the Python-side helper pattern:
/// - requires x_grid.len() == y_grid.len()
/// - requires at least 2 points
/// - clamps to endpoints outside the grid
///
/// Assumes x_grid is monotone increasing.
pub fn interp_clamped(x_grid: &[f64], y_grid: &[f64], xq: f64) -> PyResult<f64> {
    if x_grid.len() != y_grid.len() {
        return Err(PyValueError::new_err(format!(
            "x_grid and y_grid must have same length, got {} and {}",
            x_grid.len(),
            y_grid.len()
        )));
    }

    if x_grid.len() < 2 {
        return Err(PyValueError::new_err(
            "Interpolation grid must contain at least 2 points",
        ));
    }

    if !xq.is_finite() {
        return Err(PyValueError::new_err("xq must be finite"));
    }

    if !is_monotone_increasing(x_grid) {
        return Err(PyValueError::new_err(
            "x_grid must be monotone increasing",
        ));
    }

    // Clamp to endpoints
    if xq <= x_grid[0] {
        return Ok(y_grid[0]);
    }
    if xq >= x_grid[x_grid.len() - 1] {
        return Ok(y_grid[y_grid.len() - 1]);
    }

    // Find interval [i, i+1] such that x_grid[i] <= xq < x_grid[i+1]
    let mut i = 0usize;
    for k in 0..(x_grid.len() - 1) {
        if xq >= x_grid[k] && xq < x_grid[k + 1] {
            i = k;
            break;
        }
    }

    let x0 = x_grid[i];
    let x1 = x_grid[i + 1];
    let y0 = y_grid[i];
    let y1 = y_grid[i + 1];

    let dx = x1 - x0;
    if dx.abs() < 1e-15 {
        return Err(PyValueError::new_err(format!(
            "Degenerate interpolation interval at indices {} and {}",
            i,
            i + 1
        )));
    }

    let w = (xq - x0) / dx;
    Ok((1.0 - w) * y0 + w * y1)
}

/// Vectorized clamped interpolation for multiple query points.
pub fn interp_clamped_vec(x_grid: &[f64], y_grid: &[f64], xq: &[f64]) -> PyResult<Vec<f64>> {
    let mut out = Vec::with_capacity(xq.len());
    for &x in xq {
        out.push(interp_clamped(x_grid, y_grid, x)?);
    }
    Ok(out)
}

fn is_monotone_increasing(x: &[f64]) -> bool {
    if x.len() < 2 {
        return true;
    }

    for i in 0..(x.len() - 1) {
        if !x[i].is_finite() || !x[i + 1].is_finite() {
            return false;
        }
        if x[i + 1] < x[i] {
            return false;
        }
    }
    true
}