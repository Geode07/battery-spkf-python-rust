from __future__ import annotations

import spkf_rust

from battery_bench.rust_bridge.model_export import esc_model_to_dict


class RustBatterySpkfEngine:
    def __init__(
        self,
        model,
        soc0: float,
        sigma_x0_diag: list[float],
        sigma_w: list[float],
        sigma_v: float,
        h: float,
        q_bump: float,
        prior_i: float = 0.0,
    ):
        model_dict = esc_model_to_dict(model)

        config_dict = {
            "sigma_x0_diag": list(sigma_x0_diag),
            "sigma_w": list(sigma_w),
            "sigma_v": float(sigma_v),
            "h": float(h),
            "q_bump": float(q_bump),
            "prior_i": float(prior_i),
        }

        self._engine = spkf_rust.BatterySpkfEngine(
            model_dict,
            config_dict,
            float(soc0),
        )

    def step(
        self,
        measured_v: float,
        current_a: float,
        temp_c: float,
        dt_s: float,
    ) -> dict:
        out = self._engine.step(
            float(measured_v),
            float(current_a),
            float(temp_c),
            float(dt_s),
        )

        return {
            "soc": float(out["soc"]),
            "predicted_voltage": float(out["predicted_voltage"]),
            "innovation": float(out["innovation"]),
            "innovation_variance": (
                None
                if out.get("innovation_variance") is None
                else float(out["innovation_variance"])
            ),
            "ir": float(out["ir"]),
            "hk": float(out["hk"]),
        }

    def get_state(self) -> tuple[float, float, float]:
        ir, hk, soc = self._engine.get_state()
        return float(ir), float(hk), float(soc)

    def summary(self) -> dict:
        return dict(self._engine.summary())

    def reset(self, soc0: float) -> None:
        self._engine.reset(float(soc0))