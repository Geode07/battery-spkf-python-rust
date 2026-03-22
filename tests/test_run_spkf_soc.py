import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------
# Fix import path for src layout
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# -----------------------------
# Imports
# -----------------------------
from battery_bench.io.mat_loader import load_dyn_data
from battery_bench.io.esc_model_builder import load_esc_model
from battery_bench.filters.spkf import init_spkf, run_spkf
from battery_bench.viz.plots import save_all_spkf_plots


# -----------------------------
# Paths
# -----------------------------
DATA_DIR = BASE_DIR / "data" / "raw"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PAN_DATA_PATH = DATA_DIR / "PANdata_P45.mat"
PAN_MODEL_PATH = DATA_DIR / "PANmodel.mat"


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
    dyn,
    outputs: dict[str, np.ndarray],
    n_samples: int,
) -> pd.DataFrame:
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


def main() -> None:
    script_name = "script1"
    n_samples = 200
    temp_c = 25.0
    output_prefix = "test_spkf_soc"

    print("=== Loading model and dynamic data ===")
    model = load_esc_model(PAN_MODEL_PATH)
    dyn = load_dyn_data(PAN_DATA_PATH, script_name=script_name)

    print(f"Model name      : {model.name}")
    print(f"Script          : {script_name}")
    print(f"Total samples   : {len(dyn.time_s)}")
    print(f"Using n_samples : {n_samples}")
    print(f"dt              : {dyn.dt}")
    print(f"Assumed temp_c  : {temp_c}")

    if n_samples > len(dyn.time_s):
        n_samples = len(dyn.time_s)

    voltage_v = dyn.voltage_v[:n_samples]
    current_a = dyn.current_a[:n_samples]
    soc_true = dyn.soc[:n_samples] if dyn.soc is not None else None

    soc0 = float(soc_true[0]) if soc_true is not None else 1.0

    print("\n=== Initializing SPKF ===")
    state = init_spkf(
        model=model,
        soc0=soc0,
        sigma_x0=np.diag([1e-6, 1e-6, 1e-4]),
        sigma_w=np.array([1e-5, 1e-5, 1e-6]),
        sigma_v=1e-3,
        h=3.0,
        q_bump=5.0,
        prior_i=float(current_a[0]),
    )

    print(f"Initial SOC estimate: {state.z}")

    print("\n=== Running SPKF ===")
    final_state, outputs = run_spkf(
        state=state,
        voltage_v=voltage_v,
        current_a=current_a,
        temp_c=temp_c,
        deltat_s=dyn.dt,
    )

    metrics = compute_metrics(
        soc_est=outputs["soc"],
        soc_true=soc_true,
        predicted_voltage=outputs["predicted_voltage"],
        measured_voltage=voltage_v,
        innovation=outputs["innovation"],
    )

    print("\n=== Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\n=== Final state ===")
    print(f"Final ir  : {final_state.ir}")
    print(f"Final hk  : {final_state.hk}")
    print(f"Final soc : {final_state.z}")

    print("\n=== Basic sanity checks ===")
    print(f"Any NaN in outputs['xhat']?              {np.isnan(outputs['xhat']).any()}")
    print(f"Any NaN in outputs['soc']?               {np.isnan(outputs['soc']).any()}")
    print(f"Any NaN in outputs['predicted_voltage']? {np.isnan(outputs['predicted_voltage']).any()}")
    print(f"Any NaN in outputs['innovation']?        {np.isnan(outputs['innovation']).any()}")

    results_df = build_results_dataframe(dyn=dyn, outputs=outputs, n_samples=n_samples)

    print("\n=== Results dataframe summary ===")
    print(f"shape: {results_df.shape}")
    print(f"columns: {list(results_df.columns)}")
    print(results_df.head())

    if soc_true is not None:
        print("\n=== SOC error preview ===")
        print(results_df[["time_s", "soc_true", "soc_est", "soc_abs_error"]].head())

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


if __name__ == "__main__":
    main()