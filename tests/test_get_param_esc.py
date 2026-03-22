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
from battery_bench.esc.get_param_esc import get_param_esc, get_all_params_esc


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
        ocv_grid=np.asarray(raw["OCV"], dtype=float),

        ocv0=np.asarray(raw["OCV0"], dtype=float) if "OCV0" in raw else None,
        ocvrel=np.asarray(raw["OCVrel"], dtype=float) if "OCVrel" in raw else None,
        soc0=np.asarray(raw["SOC0"], dtype=float) if "SOC0" in raw else None,
        socrel=np.asarray(raw["SOCrel"], dtype=float) if "SOCrel" in raw else None,
        ocveta=np.asarray(raw["OCVeta"], dtype=float) if "OCVeta" in raw else None,
        ocvq=np.asarray(raw["OCVQ"], dtype=float) if "OCVQ" in raw else None,
        docv0=np.asarray(raw["dOCV0"], dtype=float) if "dOCV0" in raw else None,
        docvrel=np.asarray(raw["dOCVrel"], dtype=float) if "dOCVrel" in raw else None,

        q_param=np.asarray(raw["QParam"], dtype=float) if "QParam" in raw else None,
        eta_param=np.asarray(raw["etaParam"], dtype=float) if "etaParam" in raw else None,
        g_param=np.asarray(raw["GParam"], dtype=float) if "GParam" in raw else None,
        m0_param=np.asarray(raw["M0Param"], dtype=float) if "M0Param" in raw else None,
        m_param=np.asarray(raw["MParam"], dtype=float) if "MParam" in raw else None,
        r0_param=np.asarray(raw["R0Param"], dtype=float) if "R0Param" in raw else None,
        rc_param=np.asarray(raw["RCParam"], dtype=float) if "RCParam" in raw else None,
        r_param=np.asarray(raw["RParam"], dtype=float) if "RParam" in raw else None,
    )


def print_array_info(name: str, value) -> None:
    arr = np.asarray(value)
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
    if arr.size > 0:
        flat = arr.reshape(-1)
        print(f"  first 5 values: {flat[:5]}")


def main() -> None:
    model_data = load_model_data(PAN_MODEL_PATH)
    model = build_esc_model_from_model_data(model_data)

    print("=== ESCModel summary ===")
    print(f"name: {model.name}")
    print(f"temps_c shape: {model.temps_c.shape}")
    print(f"temps_c: {model.temps_c}")
    print(f"soc_grid shape: {model.soc_grid.shape}")
    print(f"ocv_grid shape: {model.ocv_grid.shape}")

    print("\n=== Raw parameter shapes ===")
    for field_name in [
        "q_param",
        "eta_param",
        "g_param",
        "m0_param",
        "m_param",
        "r0_param",
        "rc_param",
        "r_param",
    ]:
        value = getattr(model, field_name)
        if value is None:
            print(f"{field_name}: None")
        else:
            print_array_info(field_name, value)

    temps_to_test = [
        float(model.temps_c[0]),
        float(model.temps_c[len(model.temps_c) // 2]),
        float(model.temps_c[-1]),
        float(model.temps_c[0] - 5.0),
        float(model.temps_c[-1] + 5.0),
    ]

    print("\n=== Interpolated parameter checks ===")
    for temp_c in temps_to_test:
        print(f"\n--- temp_c = {temp_c} ---")
        params = get_all_params_esc(model, temp_c)

        for key, value in params.items():
            if value is None:
                print(f"{key}: None")
            else:
                print_array_info(key, value)

    print("\n=== Individual parameter spot checks ===")
    for param_name in ["q_param", "r0_param", "rc_param", "r_param"]:
        value = get_param_esc(model, param_name, float(model.temps_c[0]))
        print_array_info(f"{param_name} @ {model.temps_c[0]}C", value)

    for key in ["OCV", "OCV0", "OCVrel", "SOC", "SOC0", "SOCrel", "dOCV0", "dOCVrel"]:
        arr = np.asarray(model_data.raw[key])
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
        flat = arr.reshape(-1)
        print(f"  first 10: {flat[:10]}")


if __name__ == "__main__":
    main()