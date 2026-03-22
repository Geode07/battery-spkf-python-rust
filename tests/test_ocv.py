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
from battery_bench.io.mat_loader import load_model_data
from battery_bench.models.esc_model import ESCModel
from battery_bench.esc.ocv import (
    ocv_from_soc_temp,
    soc_from_ocv_temp,
    docv_from_soc_temp,
    ocv_lookup_summary,
)


# -----------------------------
# Paths
# -----------------------------
DATA_DIR = BASE_DIR / "data" / "raw"
PAN_MODEL_PATH = DATA_DIR / "PANmodel.mat"


def build_esc_model_from_model_data(model_data) -> ESCModel:
    raw = model_data.raw

    return ESCModel(
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


def print_array_info(name: str, value) -> None:
    arr = np.asarray(value)
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
    if arr.size > 0:
        flat = arr.reshape(-1)
        n = min(10, flat.size)
        print(f"  first {n} values: {flat[:n]}")


def main() -> None:
    model_data = load_model_data(PAN_MODEL_PATH)
    model = build_esc_model_from_model_data(model_data)

    print("=== OCV lookup summary ===")
    summary = ocv_lookup_summary(model)
    for key, value in summary.items():
        print(f"{key}: {value}")

    temp_mid = float(model.temps_c[len(model.temps_c) // 2])
    print(f"\nUsing temp_mid = {temp_mid} C")

    # -----------------------------
    # Forward OCV lookup checks
    # -----------------------------
    print("\n=== Forward OCV(SOC, T) checks ===")
    soc_test = np.array([0.0, 0.1, 0.5, 0.9, 1.0], dtype=float)
    ocv_test = ocv_from_soc_temp(model, soc_test, temp_mid)

    print_array_info("soc_test", soc_test)
    print_array_info("ocv_from_soc_temp(soc_test, temp_mid)", ocv_test)

    # Scalar check
    soc_scalar = 0.5
    ocv_scalar = ocv_from_soc_temp(model, soc_scalar, temp_mid)
    print_array_info("ocv_from_soc_temp(0.5, temp_mid)", ocv_scalar)

    # Endpoint/clamping checks
    soc_outside = np.array([-0.1, 0.0, 1.0, 1.1], dtype=float)
    ocv_outside = ocv_from_soc_temp(model, soc_outside, temp_mid)
    print_array_info("soc_outside", soc_outside)
    print_array_info("ocv_from_soc_temp(soc_outside, temp_mid)", ocv_outside)

    # -----------------------------
    # Inverse SOC lookup checks
    # -----------------------------
    print("\n=== Inverse SOC(OCV, T) checks ===")
    ocv_grid = np.asarray(model.ocv_grid, dtype=float).reshape(-1)
    ocv_test_inv = np.array(
        [
            ocv_grid[0],
            ocv_grid[len(ocv_grid) // 4],
            ocv_grid[len(ocv_grid) // 2],
            ocv_grid[3 * len(ocv_grid) // 4],
            ocv_grid[-1],
        ],
        dtype=float,
    )
    soc_from_ocv = soc_from_ocv_temp(model, ocv_test_inv, temp_mid)

    print_array_info("ocv_test_inv", ocv_test_inv)
    print_array_info("soc_from_ocv_temp(ocv_test_inv, temp_mid)", soc_from_ocv)

    # Scalar inverse check
    ocv_scalar_inv = float(ocv_grid[len(ocv_grid) // 2])
    soc_scalar_inv = soc_from_ocv_temp(model, ocv_scalar_inv, temp_mid)
    print_array_info("soc_from_ocv_temp(mid_ocv, temp_mid)", soc_scalar_inv)

    # Inverse endpoint/clamping checks
    ocv_outside = np.array([ocv_grid[0] - 0.1, ocv_grid[0], ocv_grid[-1], ocv_grid[-1] + 0.1], dtype=float)
    soc_outside_inv = soc_from_ocv_temp(model, ocv_outside, temp_mid)
    print_array_info("ocv_outside", ocv_outside)
    print_array_info("soc_from_ocv_temp(ocv_outside, temp_mid)", soc_outside_inv)

    # -----------------------------
    # dOCV/dSOC checks
    # -----------------------------
    print("\n=== dOCV/dSOC checks ===")
    docv_test = docv_from_soc_temp(model, soc_test, temp_mid)
    print_array_info("docv_from_soc_temp(soc_test, temp_mid)", docv_test)

    docv_scalar = docv_from_soc_temp(model, 0.5, temp_mid)
    print_array_info("docv_from_soc_temp(0.5, temp_mid)", docv_scalar)

    # -----------------------------
    # Round-trip sanity check
    # -----------------------------
    print("\n=== Round-trip sanity check ===")
    soc_roundtrip_in = np.array([0.05, 0.2, 0.5, 0.8, 0.95], dtype=float)
    ocv_roundtrip = ocv_from_soc_temp(model, soc_roundtrip_in, temp_mid)
    soc_roundtrip_out = soc_from_ocv_temp(model, ocv_roundtrip, temp_mid)

    print_array_info("soc_roundtrip_in", soc_roundtrip_in)
    print_array_info("ocv_roundtrip", ocv_roundtrip)
    print_array_info("soc_roundtrip_out", soc_roundtrip_out)
    print_array_info("roundtrip_abs_error", np.abs(soc_roundtrip_out - soc_roundtrip_in))


if __name__ == "__main__":
    main()