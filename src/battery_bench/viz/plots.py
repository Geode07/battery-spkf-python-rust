from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_soc(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    time_col: str = "time_s",
    soc_est_col: str = "soc_est",
    soc_true_col: str = "soc_true",
    title: str = "SOC: Estimated vs True",
) -> Path:
    output_path = _ensure_parent_dir(output_path)

    plt.figure(figsize=(10, 5))
    plt.plot(df[time_col], df[soc_est_col], label="Estimated SOC")

    if soc_true_col in df.columns:
        plt.plot(df[time_col], df[soc_true_col], label="True SOC")

    plt.xlabel("Time (s)")
    plt.ylabel("SOC")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def plot_voltage(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    time_col: str = "time_s",
    measured_col: str = "voltage_v",
    predicted_col: str = "predicted_voltage",
    title: str = "Voltage: Measured vs Predicted",
) -> Path:
    output_path = _ensure_parent_dir(output_path)

    plt.figure(figsize=(10, 5))
    plt.plot(df[time_col], df[measured_col], label="Measured Voltage")
    plt.plot(df[time_col], df[predicted_col], label="Predicted Voltage")

    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def plot_innovation(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    time_col: str = "time_s",
    innovation_col: str = "innovation",
    title: str = "Voltage Innovation Over Time",
) -> Path:
    output_path = _ensure_parent_dir(output_path)

    plt.figure(figsize=(10, 5))
    plt.plot(df[time_col], df[innovation_col])

    plt.xlabel("Time (s)")
    plt.ylabel("Innovation (V)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def plot_soc_abs_error(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    time_col: str = "time_s",
    error_col: str = "soc_abs_error",
    title: str = "SOC Absolute Error Over Time",
) -> Path | None:
    if error_col not in df.columns:
        return None

    output_path = _ensure_parent_dir(output_path)

    plt.figure(figsize=(10, 5))
    plt.plot(df[time_col], df[error_col])

    plt.xlabel("Time (s)")
    plt.ylabel("Absolute SOC Error")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def save_all_spkf_plots(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    prefix: str = "spkf",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}

    saved["soc"] = plot_soc(df, output_dir / f"{prefix}_soc.png")
    saved["voltage"] = plot_voltage(df, output_dir / f"{prefix}_voltage.png")
    saved["innovation"] = plot_innovation(df, output_dir / f"{prefix}_innovation.png")

    soc_err_path = plot_soc_abs_error(df, output_dir / f"{prefix}_soc_abs_error.png")
    if soc_err_path is not None:
        saved["soc_abs_error"] = soc_err_path

    return saved