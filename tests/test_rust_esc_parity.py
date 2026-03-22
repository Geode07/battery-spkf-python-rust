import sys
from pathlib import Path

import numpy as np
import spkf_rust

# -----------------------------
# Fix import path for src layout
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# -----------------------------
# Imports
# -----------------------------
from battery_bench.io.esc_model_builder import load_esc_model
from battery_bench.rust_bridge.model_export import esc_model_to_dict
from battery_bench.filters.spkf import _state_eqn_esc, _output_eqn_esc


def print_diff(name: str, py_val: float, rust_val: float) -> None:
    abs_diff = abs(py_val - rust_val)
    print(f"{name}:")
    print(f"  python = {py_val}")
    print(f"  rust   = {rust_val}")
    print(f"  absdiff= {abs_diff}")


def run_case(
    model,
    model_dict,
    soc0: float,
    ir0: float,
    hk0: float,
    measured_v: float,
    current_a: float,
    temp_c: float,
    dt_s: float,
    prev_sign: float,
) -> None:
    print("\n" + "=" * 60)
    print(
        f"CASE: soc0={soc0}, ir0={ir0}, hk0={hk0}, "
        f"measured_v={measured_v}, current_a={current_a}, temp_c={temp_c}, dt_s={dt_s}"
    )

    # -----------------------------
    # Python ESC prediction
    # -----------------------------
    x0 = np.array([ir0, hk0, soc0], dtype=float)

    py_next = _state_eqn_esc(
        x=x0,
        current_a=current_a,
        temp_c=temp_c,
        deltat=dt_s,
        model=model,
        prev_sign=prev_sign,
    )

    py_pred_v = _output_eqn_esc(
        x=py_next,
        current_a=current_a,
        temp_c=temp_c,
        model=model,
        prev_sign=prev_sign,
    )

    py_innov = measured_v - py_pred_v

    # -----------------------------
    # Rust ESC prediction
    # -----------------------------
    config_dict = {
        "sigma_x0_diag": [1e-6, 1e-6, 1e-4],
        "sigma_w": [1e-5, 1e-5, 1e-6],
        "sigma_v": 1e-3,
        "h": 3.0,
        "q_bump": 5.0,
        "prior_i": 0.0,
    }

    engine = spkf_rust.BatterySpkfEngine(model_dict, config_dict, soc0)

    # Optional: once Rust reset/set_state exists, use it.
    # For now, this parity test assumes ir0=0 and hk0=0 for exact match.
    if abs(ir0) > 1e-15 or abs(hk0) > 1e-15:
        print("[warn] Rust engine currently initializes ir=0, hk=0, so this case is not exact parity.")

    rust_out = engine.step(
        measured_v=measured_v,
        current_a=current_a,
        temp_c=temp_c,
        dt_s=dt_s,
    )
    rust_state = engine.get_state()

    # -----------------------------
    # Compare
    # -----------------------------
    print_diff("ir", float(py_next[0]), float(rust_state[0]))
    print_diff("hk", float(py_next[1]), float(rust_state[1]))
    print_diff("soc", float(py_next[2]), float(rust_state[2]))
    print_diff("predicted_voltage", float(py_pred_v), float(rust_out["predicted_voltage"]))
    print_diff("innovation", float(py_innov), float(rust_out["innovation"]))


def main() -> None:
    model = load_esc_model(BASE_DIR / "data" / "raw" / "PANmodel.mat")
    model_dict = esc_model_to_dict(model)

    print("Loaded model:", model.name)

    # Best first parity cases: ir0=0, hk0=0 because Rust engine initializes that way
    cases = [
        {
            "soc0": 1.0,
            "ir0": 0.0,
            "hk0": 0.0,
            "measured_v": 4.09543,
            "current_a": 0.0,
            "temp_c": 25.0,
            "dt_s": 1.0,
            "prev_sign": 0.0,
        },
        {
            "soc0": 0.8,
            "ir0": 0.0,
            "hk0": 0.0,
            "measured_v": 3.8,
            "current_a": 1.0,
            "temp_c": 25.0,
            "dt_s": 1.0,
            "prev_sign": 0.0,
        },
        {
            "soc0": 0.8,
            "ir0": 0.0,
            "hk0": 0.0,
            "measured_v": 3.8,
            "current_a": -1.0,
            "temp_c": 25.0,
            "dt_s": 1.0,
            "prev_sign": 0.0,
        },
        {
            "soc0": 0.5,
            "ir0": 0.0,
            "hk0": 0.0,
            "measured_v": 3.65,
            "current_a": 2.0,
            "temp_c": 15.0,
            "dt_s": 1.0,
            "prev_sign": 1.0,
        },
    ]

    for case in cases:
        run_case(model=model, model_dict=model_dict, **case)


if __name__ == "__main__":
    main()