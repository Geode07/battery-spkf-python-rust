from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TxtData:
    data: pd.DataFrame
    source_path: Path

    @property
    def columns(self) -> list[str]:
        return list(self.data.columns)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape


@dataclass
class ProfileData:
    time_s: np.ndarray
    value: np.ndarray
    value_name: str
    source_path: Path

    @property
    def dt(self) -> float:
        if len(self.time_s) < 2:
            raise ValueError("Need at least two time points to compute dt.")
        return float(self.time_s[1] - self.time_s[0])


def _validate_file_exists(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def _default_column_names(n_cols: int) -> list[str]:
    if n_cols == 1:
        return ["value"]
    if n_cols == 2:
        return ["time_s", "value"]
    return [f"col_{i}" for i in range(n_cols)]


def load_txt_numeric(
    path: str | Path,
    *,
    sep: str | None = None,
    header: int | None = None,
    column_names: list[str] | None = None,
    comment: str | None = None,
) -> TxtData:
    """
    Load a numeric text file into a pandas DataFrame.

    Parameters
    ----------
    path
        Path to .txt, .csv, or whitespace-delimited numeric file.
    sep
        Delimiter. If None, pandas will infer; for whitespace-delimited files,
        you can pass sep=r"\\s+".
    header
        Row number to use as header. Use None for headerless files.
    column_names
        Optional column names to assign.
    comment
        Optional comment character, e.g. '#'.
    """
    path = _validate_file_exists(path)

    if sep is None:
        df = pd.read_csv(path, header=header, comment=comment, sep=None, engine="python")
    else:
        df = pd.read_csv(path, header=header, comment=comment, sep=sep, engine="python")

    if column_names is not None:
        if len(column_names) != df.shape[1]:
            raise ValueError(
                f"column_names length {len(column_names)} does not match "
                f"file column count {df.shape[1]}"
            )
        df.columns = column_names
    elif header is None:
        df.columns = _default_column_names(df.shape[1])

    return TxtData(data=df, source_path=path)


def load_whitespace_txt(
    path: str | Path,
    *,
    header: int | None = None,
    column_names: list[str] | None = None,
    comment: str | None = None,
) -> TxtData:
    """
    Convenience wrapper for whitespace-delimited numeric text files.
    """
    return load_txt_numeric(
        path,
        sep=r"\s+",
        header=header,
        column_names=column_names,
        comment=comment,
    )


def load_profile_txt(
    path: str | Path,
    *,
    value_name: str = "value",
    assume_unit_timestep: bool = True,
    sep: str | None = None,
    header: int | None = None,
    comment: str | None = None,
) -> ProfileData:
    """
    Load a simple profile text file as a time series.

    Supported cases:
    - 1 column: interpreted as value only; time is generated as 0,1,2,...
    - 2 columns: interpreted as time_s, value

    Returns
    -------
    ProfileData
    """
    txt = load_txt_numeric(
        path,
        sep=sep,
        header=header,
        column_names=None,
        comment=comment,
    )

    df = txt.data

    if df.shape[1] == 1:
        if not assume_unit_timestep:
            raise ValueError(
                "File has only one column and assume_unit_timestep=False, "
                "so time cannot be inferred."
            )
        value = df.iloc[:, 0].to_numpy(dtype=float)
        time_s = np.arange(len(value), dtype=float)
        return ProfileData(
            time_s=time_s,
            value=value,
            value_name=value_name,
            source_path=txt.source_path,
        )

    if df.shape[1] == 2:
        time_s = df.iloc[:, 0].to_numpy(dtype=float)
        value = df.iloc[:, 1].to_numpy(dtype=float)
        return ProfileData(
            time_s=time_s,
            value=value,
            value_name=value_name,
            source_path=txt.source_path,
        )

    raise ValueError(
        f"Expected 1 or 2 columns for a profile text file, got {df.shape[1]} columns."
    )


def summarize_txt_file(
    path: str | Path,
    *,
    sep: str | None = None,
    header: int | None = None,
    comment: str | None = None,
    max_preview_rows: int = 5,
) -> dict[str, Any]:
    """
    Return a lightweight summary of a numeric text file.
    """
    txt = load_txt_numeric(
        path,
        sep=sep,
        header=header,
        comment=comment,
    )
    df = txt.data

    summary: dict[str, Any] = {
        "path": str(txt.source_path),
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "head": df.head(max_preview_rows).to_dict(orient="list"),
    }

    return summary


def print_txt_summary(
    path: str | Path,
    *,
    sep: str | None = None,
    header: int | None = None,
    comment: str | None = None,
    max_preview_rows: int = 5,
) -> None:
    summary = summarize_txt_file(
        path,
        sep=sep,
        header=header,
        comment=comment,
        max_preview_rows=max_preview_rows,
    )

    print(f"Summary for: {summary['path']}")
    print(f"- shape: {summary['shape']}")
    print(f"- columns: {summary['columns']}")
    print(f"- dtypes: {summary['dtypes']}")
    print("- head:")
    for i in range(len(next(iter(summary["head"].values()), []))):
        row = {col: summary["head"][col][i] for col in summary["columns"]}
        print(f"  {row}")