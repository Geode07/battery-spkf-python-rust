from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------
# Fix import path for src layout
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# -----------------------------
# Backend toggle
# -----------------------------
USE_RUST = True  # set True once rust_bridge/spkf_engine.py is ready

# -----------------------------
# Imports
# -----------------------------
from battery_bench.io.mat_loader import load_dyn_data
from battery_bench.io.esc_model_builder import load_esc_model
from battery_bench.filters.spkf import init_spkf, run_spkf
from battery_bench.models.dyn_data import DynData
from battery_bench.viz.plots import save_all_spkf_plots

# Optional Rust bridge import
if USE_RUST:
    try:
        from battery_bench.rust_bridge.spkf_engine import RustBatterySpkfEngine
    except Exception as e:
        RustBatterySpkfEngine = None
        print(f"[warn] Rust backend requested but unavailable: {e}")
        print("[warn] Falling back to Python backend.")
        USE_RUST = False


# -----------------------------
# Paths
# -----------------------------
DATA_DIR = BASE_DIR / "data" / "raw"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PAN_DATA_PATH = DATA_DIR / "PANdata_P45.mat"
PAN_MODEL_PATH = DATA_DIR / "PANmodel.mat"


def build_dyn_data_from_mat(path: str | Path, script_name: str = "script1") -> DynData:
    raw_dyn = load_dyn_data(path, script_name=script_name)

    return DynData(
        time_s=np.asarray(raw_dyn.time_s, dtype=float).reshape(-1),
        current_a=np.asarray(raw_dyn.current_a, dtype=float).reshape(-1),
        voltage_v=np.asarray(raw_dyn.voltage_v, dtype=float).reshape(-1),
        soc=None if raw_dyn.soc is None else np.asarray(raw_dyn.soc, dtype=float).reshape(-1),
        temperature_c=None,
    )


def compute_metrics(
    soc_est: np.ndarray,
    soc_true: np.ndarray | None,
    predicted_voltage: np.ndarray,
    measured_voltage: np.ndarray,
    innovation: np.ndarray,
) -> dict[str, float]:
    metrics = {
        "voltage_rmse": float(np.sqrt(np.mean((predicted_voltage - measured_voltage) ** 2))),
        "voltage_mae": float(np.mean(np.abs(predicted_voltage - measured_voltage))),
        "innovation_rmse": float(np.sqrt(np.mean(innovation**2))),
        "innovation_mae": float(np.mean(np.abs(innovation))),
    }

    if soc_true is not None:
        abs_err = np.abs(soc_est - soc_true)
        sq_err = (soc_est - soc_true) ** 2
        metrics.update(
            {
                "soc_mae": float(np.mean(abs_err)),
                "soc_rmse": float(np.sqrt(np.mean(sq_err))),
                "soc_max_abs_error": float(np.max(abs_err)),
            }
        )

    return metrics


def build_results_dataframe(
    dyn: DynData,
    outputs: dict[str, np.ndarray],
) -> pd.DataFrame:
    n_samples = len(outputs["soc"])

    df = pd.DataFrame(
        {
            "time_s": dyn.time_s[:n_samples],
            "current_a": dyn.current_a[:n_samples],
            "voltage_v": dyn.voltage_v[:n_samples],
            "soc_est": outputs["soc"][:n_samples],
            "predicted_voltage": outputs["predicted_voltage"][:n_samples],
            "innovation": outputs["innovation"][:n_samples],
            "innovation_variance": outputs["innovation_variance"][:n_samples],
            "ir_state": outputs["ir"][:n_samples],
            "hk_state": outputs["hk"][:n_samples],
        }
    )

    if dyn.soc is not None:
        df["soc_true"] = dyn.soc[:n_samples]
        df["soc_abs_error"] = np.abs(df["soc_est"] - df["soc_true"])

    return df


def run_spkf_python_backend(
    model,
    dyn_run: DynData,
    temp_input: float | np.ndarray,
    soc0: float,
) -> tuple[object, dict[str, np.ndarray]]:
    state = init_spkf(
        model=model,
        soc0=soc0,
        sigma_x0=np.diag([1e-6, 1e-6, 1e-4]),
        sigma_w=np.array([1e-5, 1e-5, 1e-6]),
        sigma_v=1e-3,
        h=3.0,
        q_bump=5.0,
        prior_i=float(dyn_run.current_a[0]),
    )

    print(f"Initial SOC estimate: {state.z}")

    final_state, outputs = run_spkf(
        state=state,
        voltage_v=dyn_run.voltage_v,
        current_a=dyn_run.current_a,
        temp_c=temp_input,
        deltat_s=dyn_run.dt,
    )
    return final_state, outputs


def run_spkf_rust_backend(
    model,
    dyn_run: DynData,
    temp_input: float | np.ndarray,
    soc0: float,
) -> tuple[object | None, dict[str, np.ndarray]]:
    if RustBatterySpkfEngine is None:
        raise RuntimeError("RustBatterySpkfEngine is not available.")

    engine = RustBatterySpkfEngine(
        model=model,
        soc0=soc0,
        sigma_x0_diag=[1e-6, 1e-6, 1e-4],
        sigma_w=[1e-5, 1e-5, 1e-6],
        sigma_v=1e-3,
        h=3.0,
        q_bump=5.0,
        prior_i=float(dyn_run.current_a[0]),
    )

    n = dyn_run.n
    soc_hist = np.zeros(n, dtype=float)
    pred_v_hist = np.zeros(n, dtype=float)
    innov_hist = np.zeros(n, dtype=float)
    innov_var_hist = np.full(n, np.nan, dtype=float)
    ir_hist = np.zeros(n, dtype=float)
    hk_hist = np.zeros(n, dtype=float)

    for k in range(n):
        temp_k = float(temp_input if np.isscalar(temp_input) else temp_input[k])

        step_out = engine.step(
            measured_v=float(dyn_run.voltage_v[k]),
            current_a=float(dyn_run.current_a[k]),
            temp_c=temp_k,
            dt_s=float(dyn_run.dt),
        )

        soc_hist[k] = step_out["soc"]
        pred_v_hist[k] = step_out["predicted_voltage"]
        innov_hist[k] = step_out["innovation"]
        ir_hist[k] = step_out["ir"]
        hk_hist[k] = step_out["hk"]

        if "innovation_variance" in step_out and step_out["innovation_variance"] is not None:
            innov_var_hist[k] = step_out["innovation_variance"]

    outputs = {
        "xhat": np.column_stack([ir_hist, hk_hist, soc_hist]),
        "ir": ir_hist,
        "hk": hk_hist,
        "soc": soc_hist,
        "predicted_voltage": pred_v_hist,
        "innovation": innov_hist,
        "innovation_variance": innov_var_hist,
    }

    return None, outputs


def main() -> None:
    script_name = "script1"
    n_samples = 2000
    default_temp_c = 25.0
    output_prefix = "spkf_soc_rust" if USE_RUST else "spkf_soc_python"

    print("=== Loading model and dynamic data ===")
    model = load_esc_model(PAN_MODEL_PATH)
    dyn = build_dyn_data_from_mat(PAN_DATA_PATH, script_name=script_name)

    print(f"Model name       : {model.name}")
    print(f"Script           : {script_name}")
    print(f"Total samples    : {dyn.n}")
    print(f"Using n_samples  : {n_samples}")
    print(f"dt               : {dyn.dt}")
    print(f"Backend          : {'Rust' if USE_RUST else 'Python'}")

    if n_samples > dyn.n:
        n_samples = dyn.n

    dyn_run = dyn.slice(n_samples)

    if dyn_run.temperature_c is None:
        temp_input: float | np.ndarray = float(default_temp_c)
        print(f"Assumed temp_c   : {default_temp_c}")
    else:
        temp_input = dyn_run.temperature_c
        if np.isscalar(temp_input):
            print(f"Using dyn temp_c : {float(temp_input)}")
        else:
            print(f"Using dyn temperature array with shape: {np.asarray(temp_input).shape}")

    soc0 = float(dyn_run.soc[0]) if dyn_run.soc is not None else 1.0

    print("\n=== Running SPKF ===")
    if USE_RUST:
        final_state, outputs = run_spkf_rust_backend(
            model=model,
            dyn_run=dyn_run,
            temp_input=temp_input,
            soc0=soc0,
        )
    else:
        final_state, outputs = run_spkf_python_backend(
            model=model,
            dyn_run=dyn_run,
            temp_input=temp_input,
            soc0=soc0,
        )

    metrics = compute_metrics(
        soc_est=outputs["soc"],
        soc_true=dyn_run.soc,
        predicted_voltage=outputs["predicted_voltage"],
        measured_voltage=dyn_run.voltage_v,
        innovation=outputs["innovation"],
    )

    print("\n=== Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\n=== Final state ===")
    if final_state is not None:
        print(f"Final ir  : {final_state.ir}")
        print(f"Final hk  : {final_state.hk}")
        print(f"Final soc : {final_state.z}")
    else:
        print(f"Final ir  : {outputs['ir'][-1]}")
        print(f"Final hk  : {outputs['hk'][-1]}")
        print(f"Final soc : {outputs['soc'][-1]}")

    results_df = build_results_dataframe(dyn=dyn_run, outputs=outputs)

    output_csv = RESULTS_DIR / f"{output_prefix}_results.csv"
    results_df.to_csv(output_csv, index=False)

    plot_paths = save_all_spkf_plots(
        results_df,
        RESULTS_DIR,
        prefix=output_prefix,
    )

    print("\n=== Output saved ===")
    print(f"Saved results to: {output_csv}")
    for label, path in plot_paths.items():
        print(f"Saved {label} plot to: {path}")

    print("\n=== Preview ===")
    print(results_df.head())


if __name__ == "__main__":
    main()