from dataclasses import dataclass
import numpy as np


@dataclass
class DynData:
    time_s: np.ndarray
    current_a: np.ndarray
    voltage_v: np.ndarray

    soc: np.ndarray | None = None
    temperature_c: float | np.ndarray | None = None

    @property
    def dt(self) -> float:
        if self.time_s.size < 2:
            return 1.0
        return float(self.time_s[1] - self.time_s[0])

    @property
    def n(self) -> int:
        return int(self.time_s.size)

    def slice(self, n: int) -> "DynData":
        return DynData(
            time_s=self.time_s[:n],
            current_a=self.current_a[:n],
            voltage_v=self.voltage_v[:n],
            soc=None if self.soc is None else self.soc[:n],
            temperature_c=self.temperature_c,
        )