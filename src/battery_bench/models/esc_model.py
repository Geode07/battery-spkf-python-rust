from dataclasses import dataclass
import numpy as np


@dataclass
class ESCModel:
    # Metadata
    name: str

    # Independent grids
    temps_c: np.ndarray
    soc_grid: np.ndarray

    # Inverse OCV lookup grid (used with soc0/socrel)
    ocv_grid: np.ndarray

    # Forward OCV lookup tables on soc_grid
    ocv0: np.ndarray | None = None
    ocvrel: np.ndarray | None = None

    # Inverse SOC lookup tables on ocv_grid
    soc0: np.ndarray | None = None
    socrel: np.ndarray | None = None

    # Additional source-model fields
    ocveta: np.ndarray | None = None
    ocvq: np.ndarray | None = None
    docv0: np.ndarray | None = None
    docvrel: np.ndarray | None = None

    # Temperature-dependent / model parameters
    q_param: np.ndarray | None = None
    eta_param: np.ndarray | None = None
    g_param: np.ndarray | None = None
    m0_param: np.ndarray | None = None
    m_param: np.ndarray | None = None
    r0_param: np.ndarray | None = None
    rc_param: np.ndarray | None = None
    r_param: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.temps_c = np.asarray(self.temps_c, dtype=float).reshape(-1)
        self.soc_grid = np.asarray(self.soc_grid, dtype=float).reshape(-1)
        self.ocv_grid = np.asarray(self.ocv_grid, dtype=float).reshape(-1)

        for field_name in [
            "ocv0", "ocvrel", "soc0", "socrel", "ocveta", "ocvq",
            "docv0", "docvrel", "q_param", "eta_param", "g_param",
            "m0_param", "m_param", "r0_param", "rc_param", "r_param"
        ]:
            value = getattr(self, field_name)
            if value is not None:
                setattr(self, field_name, np.asarray(value, dtype=float))

    @property
    def n_temps(self) -> int:
        return int(self.temps_c.size)

    @property
    def n_soc(self) -> int:
        return int(self.soc_grid.size)

    @property
    def n_ocv(self) -> int:
        return int(self.ocv_grid.size)

    @property
    def n_rc_branches(self) -> int | None:
        if self.r_param is None:
            return None
        arr = np.asarray(self.r_param)
        if arr.ndim == 1:
            return 1
        return None

    def validate(self) -> None:
        if self.ocv0 is not None and self.ocv0.reshape(-1).size != self.n_soc:
            raise ValueError("ocv0 length must match soc_grid length")
        if self.ocvrel is not None and self.ocvrel.reshape(-1).size != self.n_soc:
            raise ValueError("ocvrel length must match soc_grid length")
        if self.docv0 is not None and self.docv0.reshape(-1).size != self.n_soc:
            raise ValueError("docv0 length must match soc_grid length")
        if self.docvrel is not None and self.docvrel.reshape(-1).size != self.n_soc:
            raise ValueError("docvrel length must match soc_grid length")

        if self.soc0 is not None and self.soc0.reshape(-1).size != self.n_ocv:
            raise ValueError("soc0 length must match ocv_grid length")
        if self.socrel is not None and self.socrel.reshape(-1).size != self.n_ocv:
            raise ValueError("socrel length must match ocv_grid length")

        for field_name in [
            "q_param", "eta_param", "g_param", "m0_param",
            "m_param", "r0_param", "rc_param", "r_param"
        ]:
            value = getattr(self, field_name)
            if value is not None:
                arr = np.asarray(value)
                if arr.ndim == 1 and arr.size != self.n_temps:
                    raise ValueError(
                        f"{field_name} length {arr.size} must match temps_c length {self.n_temps}"
                    )

    def summary(self) -> dict:
        return {
            "name": self.name,
            "temps_c_shape": self.temps_c.shape,
            "soc_grid_shape": self.soc_grid.shape,
            "ocv_grid_shape": self.ocv_grid.shape,
            "q_param_shape": None if self.q_param is None else self.q_param.shape,
            "eta_param_shape": None if self.eta_param is None else self.eta_param.shape,
            "g_param_shape": None if self.g_param is None else self.g_param.shape,
            "m0_param_shape": None if self.m0_param is None else self.m0_param.shape,
            "m_param_shape": None if self.m_param is None else self.m_param.shape,
            "r0_param_shape": None if self.r0_param is None else self.r0_param.shape,
            "rc_param_shape": None if self.rc_param is None else self.rc_param.shape,
            "r_param_shape": None if self.r_param is None else self.r_param.shape,
            "ocv0_shape": None if self.ocv0 is None else self.ocv0.shape,
            "ocvrel_shape": None if self.ocvrel is None else self.ocvrel.shape,
            "soc0_shape": None if self.soc0 is None else self.soc0.shape,
            "socrel_shape": None if self.socrel is None else self.socrel.shape,
            "docv0_shape": None if self.docv0 is None else self.docv0.shape,
            "docvrel_shape": None if self.docvrel is None else self.docvrel.shape,
        }