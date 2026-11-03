"""冷子行程開雙精度，算地毯那一筆的無規入射吸音係數：每帶要落在擬合的允收窗內，也不准比加墊的地毯更吸音。"""
from aosr import runtime
runtime.configure_calculation_jax()

import jax.numpy as jnp
import numpy as np

from aosr.config.construction_presets import load_construction_presets
from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.materials.construction import random_incidence_absorption
from aosr.physics.report_surfaces import WholeWallZone

# OSHA Technical Manual（技術手冊）Sec III Ch 5 Table V-2（材料吸音表），125／250／500 Hz。
# https://www.osha.gov/otm/section-3-health-hazards/chapter-5
# 無規入射二手彙編：Carpet, heavy, on concrete（厚地毯直接鋪混凝土）與加墊那列。
OSHA_FREQUENCIES_HZ = (125., 250., 500.)
OSHA_CARPET_ON_CONCRETE = (.02, .06, .14)
OSHA_CARPET_WITH_UNDERLAYMENT = (.08, .24, .57)
# 擬合允收窗：#990 地毯施工單訂的挑選條件，每帶跟目標的相對差不超過一半。
FIT_WINDOW = .5


def main() -> None:
    raw = next(p.material for p in load_construction_presets(config_path("construction_presets.toml")) if p.id == "carpet_on_concrete")
    material = WholeWallZone.model_validate({"area": "whole_wall", "material": raw}).material
    air = load_physics_constants(config_path("physics_constants.toml"))
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    absorption, _ = random_incidence_absorption(material, jnp.asarray(OSHA_FREQUENCIES_HZ), air=air, ranges=ranges)
    for frequency, actual, direct, padded in zip(OSHA_FREQUENCIES_HZ, np.asarray(absorption),
            OSHA_CARPET_ON_CONCRETE, OSHA_CARPET_WITH_UNDERLAYMENT, strict=True):
        assert 0 < actual < padded, (frequency, actual, direct, padded)
        assert abs(actual / direct - 1) <= FIT_WINDOW, (frequency, actual, direct)


if __name__ == "__main__":
    main()
