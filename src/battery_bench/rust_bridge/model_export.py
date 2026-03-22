from __future__ import annotations

import numpy as np

from battery_bench.models.esc_model import ESCModel


def esc_model_to_dict(model: ESCModel) -> dict:
    def arr(x):
        if x is None:
            return None
        return np.asarray(x, dtype=float).reshape(-1).tolist()

    return {
        "name": model.name,
        "temps_c": arr(model.temps_c),
        "soc_grid": arr(model.soc_grid),
        "ocv_grid": arr(model.ocv_grid),
        "ocv0": arr(model.ocv0),
        "ocvrel": arr(model.ocvrel),
        "soc0": arr(model.soc0),
        "socrel": arr(model.socrel),
        "docv0": arr(model.docv0),
        "docvrel": arr(model.docvrel),
        "q_param": arr(model.q_param),
        "eta_param": arr(model.eta_param),
        "g_param": arr(model.g_param),
        "m0_param": arr(model.m0_param),
        "m_param": arr(model.m_param),
        "r0_param": arr(model.r0_param),
        "rc_param": arr(model.rc_param),
        "r_param": arr(model.r_param),
    }