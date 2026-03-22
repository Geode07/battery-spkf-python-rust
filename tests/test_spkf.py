import sys
from pathlib import Path

import numpy as np

# -----------------------------
# Fix import path for src layout
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# -----------------------------
# Imports
# -----------------------------
from battery_bench.io.mat_loader import load_dyn_data, load_model_data
from battery_bench.models.esc_model import ESCModel
from battery_bench.filters.spkf import init_spkf, iter_spkf, run_spkf


# -----------------------------
# Paths
# -----------------------------
DATA_DIR = BASE_DIR / "data" / "raw"
PAN_DATA_PATH = DATA_DIR / "PANdata_P45.mat"
PAN_MODEL_PATH = DATA_DIR / "PANmodel.mat"


def build_esc_model_from_model_data(model_data) -> ESCModel:
    raw = model_data.raw

    model = ESCModel(
        name=str(raw.get("name", "unknown_model")),
        temps_c=np.asarray(raw["temps"], dtype=float).reshape(-1),
        soc_grid=np.asarray(raw["SOC"], dtype=float).reshape(-1),
        ocv_grid=np.asarray(raw["OCV"], dtype=float).reshape(-1),
        ocv0=np.asarray(raw["OCV0"], dtype=float).reshape(-1) if "OCV0" in raw else None,
        ocvrel=np.asarray(raw["OCVrel"], dtype=float).reshape(-1) if "OCVrel" in raw else None,
        soc0=np.asarray(raw["SOC0"], dtype=float).reshape(-1) if "SOC0" in raw else None,
        socrel=np.asarray(raw["SOCrel"], dtype=float).reshape(-1) if "SOCrel" in raw else None,
        ocveta=np.asarray(raw["OCVeta"], dtype=float).reshape(-1) if "OCVeta" in raw else None,
        ocvq=np.asarray(raw["OCVQ"], dtype=float).reshape(-1) if "OCVQ" in raw else None,
        docv0=np.asarray(raw["dOCV0"], dtype=float).reshape(-1) if "dOCV0" in raw else None,
        docvrel=np.asarray(raw["dOCVrel"], dtype=float).reshape(-1) if "dOCVrel" in raw else None,
        q_param=np.asarray(raw["QParam"], dtype=float).reshape(-1) if "QParam" in raw else None,
        eta_param=np.asarray(raw["etaParam"], dtype=float).reshape(-1) if "etaParam" in raw else None,
        g_param=np.asarray(raw["GParam"], dtype=float).reshape(-1) if "GParam" in raw else None,
        m0_param=np.asarray(raw["M0Param"], dtype=float).reshape(-1) if "M0Param" in raw else None,
        m_param=np.asarray(raw["MParam"], dtype=float).reshape(-1) if "MParam" in raw else None,
        r0_param=np.asarray(raw["R0Param"], dtype=float).reshape(-1) if "R0Param" in raw else None,
        rc_param=np.asarray(raw["RCParam"], dtype=float).reshape(-1) if "RCParam" in raw else None,
        r_param=np.asarray(raw["RParam"], dtype=float).reshape(-1) if "RParam" in raw else None,
    )

    # Optional if your ESCModel has validate()
    if hasattr(model, "validate"):
        model.validate()

    return model


def print_array_info(name: str, value, n: int = 5) -> None:
    arr = np.asarray(value)
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
    if arr.size > 0:
        flat = arr.reshape(-1)
        n_show = min(n, flat.size)
        print(f"  first {n_show} values: {flat[:n_show]}")


def main() -> None:
    print("=== Loading model and dynamic data ===")
    model_data = load_model_data(PAN_MODEL_PATH)
    dyn = load_dyn_data(PAN_DATA_PATH)
    model = build_esc_model_from_model_data(model_data)

    print(f"model name: {model.name}")
    print(f"dyn samples: {len(dyn.time_s)}")
    print(f"dt from dyn: {dyn.dt}")

    # Use a short slice first for sanity testing
    n_test = 50
    voltage_v = dyn.voltage_v[:n_test]
    current_a = dyn.current_a[:n_test]
    soc_true = dyn.soc[:n_test] if dyn.soc is not None else None
    deltat_s = dyn.dt
    temp_c = 25.0

    print("\n=== Data slice summary ===")
    print_array_info("voltage_v", voltage_v)
    print_array_info("current_a", current_a)
    if soc_true is not None:
        print_array_info("soc_true", soc_true)

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

    print("Initial state summary:")
    summary = state.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\n=== Single-step SPKF test ===")
    state_1, info_1 = iter_spkf(
        state=state,
        measured_v=float(voltage_v[0]),
        current_a=float(current_a[0]),
        temp_c=temp_c,
        deltat=float(deltat_s),
    )

    print_array_info("state_1.xhat", state_1.xhat)
    print(f"predicted_voltage: {info_1['predicted_voltage']}")
    print(f"measured_voltage : {info_1['measured_voltage']}")
    print(f"innovation       : {info_1['innovation']}")
    print_array_info("kalman_gain", info_1["kalman_gain"])
    print(f"innovation_variance: {info_1['innovation_variance']}")
    print(f"docv_dz: {info_1['docv_dz']}")

    print("\n=== Multi-step SPKF run ===")
    final_state, outputs = run_spkf(
        state=state,
        voltage_v=voltage_v,
        current_a=current_a,
        temp_c=temp_c,
        deltat_s=deltat_s,
    )

    print_array_info("outputs['xhat']", outputs["xhat"])
    print_array_info("outputs['soc']", outputs["soc"])
    print_array_info("outputs['predicted_voltage']", outputs["predicted_voltage"])
    print_array_info("outputs['innovation']", outputs["innovation"])
    print_array_info("outputs['innovation_variance']", outputs["innovation_variance"])

    print("\n=== Final state ===")
    print_array_info("final_state.xhat", final_state.xhat)
    print_array_info("final_state.sigma_x", final_state.sigma_x)

    if soc_true is not None:
        soc_est = outputs["soc"]
        soc_true_used = soc_true[:len(soc_est)]
        abs_err = np.abs(soc_est - soc_true_used)

        print("\n=== SOC error summary ===")
        print_array_info("soc_true_used", soc_true_used)
        print_array_info("soc_est", soc_est)
        print_array_info("abs_err", abs_err)
        print(f"mean_abs_error: {float(np.mean(abs_err))}")
        print(f"max_abs_error : {float(np.max(abs_err))}")

    print("\n=== Basic sanity checks ===")
    print(f"Any NaN in state history? {np.isnan(outputs['xhat']).any()}")
    print(f"Any NaN in innovations?   {np.isnan(outputs['innovation']).any()}")
    print(f"Final SOC estimate       : {final_state.z}")
    if soc_true is not None:
        print(f"Initial true SOC         : {soc_true[0]}")
        print(f"Final true SOC (slice)   : {soc_true[min(len(soc_true)-1, n_test-1)]}")


if __name__ == "__main__":
    main()