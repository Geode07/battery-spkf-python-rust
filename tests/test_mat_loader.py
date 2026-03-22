import sys
from pathlib import Path

# -----------------------------
# Fix import path for src layout
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# -----------------------------
# Imports
# -----------------------------
from battery_bench.io.mat_loader import (
    load_dyn_data,
    load_model_data,
    print_mat_summary,
)

# -----------------------------
# Data paths
# -----------------------------
DATA_DIR = BASE_DIR / "data" / "raw"

PAN_DATA_PATH = DATA_DIR / "PANdata_P45.mat"
PAN_MODEL_PATH = DATA_DIR / "PANmodel.mat"


def main() -> None:
    print("=== MAT file summaries ===")
    print_mat_summary(PAN_DATA_PATH)
    print_mat_summary(PAN_MODEL_PATH)

    print("\n=== Loading dynamic data ===")
    dyn = load_dyn_data(PAN_DATA_PATH)

    print(f"time_s shape: {dyn.time_s.shape}")
    print(f"current_a shape: {dyn.current_a.shape}")
    print(f"voltage_v shape: {dyn.voltage_v.shape}")
    print(f"soc shape: {None if dyn.soc is None else dyn.soc.shape}")

    print("time_s[:5]    =", dyn.time_s[:5])
    print("current_a[:5] =", dyn.current_a[:5])
    print("voltage_v[:5] =", dyn.voltage_v[:5])
    print("soc[:5]       =", dyn.soc[:5] if dyn.soc is not None else "No SOC present")

    print("\n=== Loading model data ===")
    model = load_model_data(PAN_MODEL_PATH)

    print("Top-level model keys:")
    for key in model.raw.keys():
        print(f"  - {key}")


if __name__ == "__main__":
    main()