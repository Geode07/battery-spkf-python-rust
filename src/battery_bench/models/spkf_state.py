from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from battery_bench.models.esc_model import ESCModel


@dataclass
class SPKFState:
    """
    Container for SPKF state, covariance, noise settings, weights, and model.

    Intended state layout for the basic ESC/SPKF SoC estimator:
        xhat = [ir, hk, z]

    where:
        ir : RC branch current state
        hk : hysteresis state
        z  : state of charge

    This dataclass does not implement filter logic; it only stores state
    and provides lightweight validation/introspection helpers.
    """

    # -----------------------------
    # Core filter state
    # -----------------------------
    xhat: np.ndarray
    sigma_x: np.ndarray

    # -----------------------------
    # Noise settings
    # -----------------------------
    sigma_v: float
    sigma_w: np.ndarray
    s_noise: np.ndarray

    # -----------------------------
    # Sigma-point weights
    # -----------------------------
    w_m: np.ndarray
    w_c: np.ndarray

    # -----------------------------
    # Previous/current bookkeeping
    # -----------------------------
    prior_i: float = 0.0
    sign_ik: float = 0.0

    # -----------------------------
    # State indices
    # -----------------------------
    ir_idx: int = 0
    hk_idx: int = 1
    zk_idx: int = 2

    # -----------------------------
    # Dimensions
    # -----------------------------
    nx: int = 3
    ny: int = 1
    nu: int = 1
    nw: int = 1
    nv: int = 1
    na: int = 0

    # -----------------------------
    # Tuning / numerical settings
    # -----------------------------
    h: float = 3.0
    q_bump: float = 5.0

    # -----------------------------
    # Model reference
    # -----------------------------
    model: ESCModel | None = None

    # -----------------------------
    # Optional metadata
    # -----------------------------
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.xhat = np.asarray(self.xhat, dtype=float).reshape(-1)
        self.sigma_x = np.asarray(self.sigma_x, dtype=float)
        self.sigma_w = np.asarray(self.sigma_w, dtype=float).reshape(-1)
        self.s_noise = np.asarray(self.s_noise, dtype=float)
        self.w_m = np.asarray(self.w_m, dtype=float).reshape(-1)
        self.w_c = np.asarray(self.w_c, dtype=float).reshape(-1)

        self._validate_basic_shapes()

    def _validate_basic_shapes(self) -> None:
        if self.xhat.ndim != 1:
            raise ValueError(f"xhat must be 1D, got shape {self.xhat.shape}")

        if self.sigma_x.ndim != 2:
            raise ValueError(f"sigma_x must be 2D, got shape {self.sigma_x.shape}")

        if self.sigma_x.shape[0] != self.sigma_x.shape[1]:
            raise ValueError(
                f"sigma_x must be square, got shape {self.sigma_x.shape}"
            )

        if self.sigma_x.shape[0] != self.xhat.size:
            raise ValueError(
                "sigma_x dimension must match xhat length, "
                f"got sigma_x shape {self.sigma_x.shape}, xhat length {self.xhat.size}"
            )

        if self.w_m.ndim != 1 or self.w_c.ndim != 1:
            raise ValueError("w_m and w_c must both be 1D")

        if self.w_m.size != self.w_c.size:
            raise ValueError(
                f"w_m and w_c must have same length, got {self.w_m.size} and {self.w_c.size}"
            )

        if self.s_noise.ndim != 2:
            raise ValueError(f"s_noise must be 2D, got shape {self.s_noise.shape}")

        if self.s_noise.shape[0] != self.s_noise.shape[1]:
            raise ValueError(
                f"s_noise must be square, got shape {self.s_noise.shape}"
            )

        expected_na = self.nx + self.nw + self.nv
        expected_noise_dim = self.nw + self.nv

        if self.na == 0:
            self.na = expected_na

        if self.na != expected_na:
            raise ValueError(
                f"na should equal nx + nw + nv = {expected_na}, got {self.na}"
            )

        if self.s_noise.shape[0] != expected_noise_dim:
            raise ValueError(
                f"s_noise shape {self.s_noise.shape} incompatible with nw+nv={expected_noise_dim}"
            )

        if self.xhat.size != self.nx:
            raise ValueError(
                f"xhat length {self.xhat.size} does not match nx={self.nx}"
            )

        if self.sigma_w.size not in (1, self.nx):
            raise ValueError(
                "sigma_w should usually be length 1 or nx for this implementation, "
                f"got shape {self.sigma_w.shape}"
            )

        if not np.isscalar(self.sigma_v):
            raise ValueError(f"sigma_v must be scalar-like, got {type(self.sigma_v)}")
        
    @property
    def ir(self) -> float:
        return float(self.xhat[self.ir_idx])

    @property
    def hk(self) -> float:
        return float(self.xhat[self.hk_idx])

    @property
    def z(self) -> float:
        return float(self.xhat[self.zk_idx])

    def set_ir(self, value: float) -> None:
        self.xhat[self.ir_idx] = float(value)

    def set_hk(self, value: float) -> None:
        self.xhat[self.hk_idx] = float(value)

    def set_z(self, value: float) -> None:
        self.xhat[self.zk_idx] = float(value)

    def copy(self) -> "SPKFState":
        return SPKFState(
            xhat=self.xhat.copy(),
            sigma_x=self.sigma_x.copy(),
            sigma_v=float(self.sigma_v),
            sigma_w=self.sigma_w.copy(),
            s_noise=self.s_noise.copy(),
            w_m=self.w_m.copy(),
            w_c=self.w_c.copy(),
            prior_i=float(self.prior_i),
            sign_ik=float(self.sign_ik),
            ir_idx=int(self.ir_idx),
            hk_idx=int(self.hk_idx),
            zk_idx=int(self.zk_idx),
            nx=int(self.nx),
            ny=int(self.ny),
            nu=int(self.nu),
            nw=int(self.nw),
            nv=int(self.nv),
            na=int(self.na),
            h=float(self.h),
            q_bump=float(self.q_bump),
            model=self.model,
            meta=self.meta.copy(),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "xhat": self.xhat.copy(),
            "sigma_x": self.sigma_x.copy(),
            "sigma_v": float(self.sigma_v),
            "sigma_w": self.sigma_w.copy(),
            "s_noise": self.s_noise.copy(),
            "w_m": self.w_m.copy(),
            "w_c": self.w_c.copy(),
            "prior_i": float(self.prior_i),
            "sign_ik": float(self.sign_ik),
            "ir_idx": int(self.ir_idx),
            "hk_idx": int(self.hk_idx),
            "zk_idx": int(self.zk_idx),
            "nx": int(self.nx),
            "ny": int(self.ny),
            "nu": int(self.nu),
            "nw": int(self.nw),
            "nv": int(self.nv),
            "na": int(self.na),
            "h": float(self.h),
            "q_bump": float(self.q_bump),
            "has_model": self.model is not None,
            "meta": self.meta.copy(),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "xhat_shape": tuple(self.xhat.shape),
            "sigma_x_shape": tuple(self.sigma_x.shape),
            "sigma_w_shape": tuple(self.sigma_w.shape),
            "s_noise_shape": tuple(self.s_noise.shape),
            "w_m_shape": tuple(self.w_m.shape),
            "w_c_shape": tuple(self.w_c.shape),
            "ir": self.ir,
            "hk": self.hk,
            "z": self.z,
            "prior_i": float(self.prior_i),
            "sign_ik": float(self.sign_ik),
            "nx": int(self.nx),
            "ny": int(self.ny),
            "nu": int(self.nu),
            "nw": int(self.nw),
            "nv": int(self.nv),
            "na": int(self.na),
            "h": float(self.h),
            "q_bump": float(self.q_bump),
            "has_model": self.model is not None,
        }