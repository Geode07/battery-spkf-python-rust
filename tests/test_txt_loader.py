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
from battery_bench.io.txt_loader import (
    load_profile_txt,
    load_txt_numeric,
    print_txt_summary,
)

# -----------------------------
# Data paths
# -----------------------------
DATA_DIR = BASE_DIR / "data" / "raw"
UDDS_PATH = DATA_DIR / "udds.txt"


def main() -> None:
    print("=== TXT file summary ===")
    print_txt_summary(
        UDDS_PATH,
        sep=r"\s+",
        header=None,
    )

    print("\n=== Loading generic numeric TXT data ===")
    txt_data = load_txt_numeric(
        UDDS_PATH,
        sep=r"\s+",
        header=None,
    )

    print(f"shape: {txt_data.shape}")
    print(f"columns: {txt_data.columns}")
    print(txt_data.data.head())

    print("\n=== Loading profile TXT data ===")
    profile = load_profile_txt(
        UDDS_PATH,
        value_name="value",
        sep=r"\s+",
        header=None,
    )

    print(f"time_s shape: {profile.time_s.shape}")
    print(f"value shape: {profile.value.shape}")
    print(f"value_name: {profile.value_name}")
    print(f"dt: {profile.dt}")

    print("time_s[:5] =", profile.time_s[:5])
    print("value[:5]  =", profile.value[:5])


if __name__ == "__main__":
    main()