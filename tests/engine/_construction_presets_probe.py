"""冷子行程開双精度，逐筆經完整方案驗證，再走晚期的正式牆面回應。"""
import sys

from aosr import runtime
runtime.configure_calculation_jax()

from aosr.config.construction_presets import load_construction_presets
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Wall
from aosr.physics import report_io
from aosr.physics.wall_surfaces import wall_surface_response
from aosr.reporting.validation import checked_inputs, validate_scheme
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES
from tests.engine.test_construction_presets import preset_document


def main() -> None:
    for preset in load_construction_presets(config_path("construction_presets.toml")):
        document = preset_document(preset.material)
        assert not validate_scheme(document, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
        scheme, pairs = checked_inputs(document, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
        inputs = next(iter(pairs.values()))[1]
        materials = report_io.solver_inputs(inputs).materials_by_wall
        response = wall_surface_response(materials, (20., 63., 125., 1000., 8000., 11166.),
            density_kg_m3=scheme.scene.density_kg_m3, sound_speed_m_s=scheme.scene.sound_speed_m_s)
        assert response.absorption_by_wall[Wall.FLOOR]


def fitted_vpr() -> None:
    """用材料層重算，和寫入端記在預設備註的全部小數位逐字比。"""
    import jax.numpy as jnp
    from aosr.config.product_air import product_air_environment
    from aosr.materials.construction import random_incidence_absorption, surface_impedance
    from aosr.materials.construction_models import Assembly
    from aosr.materials.transfer_matrix import reflection_and_absorption_coefficient
    from tests.engine._construction_preset_answers import FITTED_VPR_IDS

    air, ranges = product_air_environment()
    frequencies = jnp.asarray([34., 42., 57., 69., 85.])
    presets = {preset.id: preset for preset in load_construction_presets(config_path("construction_presets.toml"))}
    for key in FITTED_VPR_IDS:
        preset = presets[key]
        material = Assembly.model_validate(preset.material)
        z, _ = surface_impedance(material, frequencies, air=air, ranges=ranges)
        normal = reflection_and_absorption_coefficient(z, air=air)[1]
        diffuse, _ = random_incidence_absorption(material, frequencies, air=air, ranges=ranges)
        for name, values in (("垂直", normal), ("無規", diffuse)):
            reference = f"擬合後{name}入射（34、42、57、69、85 赫茲）：" + "、".join(f"{v:.12f}" for v in values)
            assert reference in preset.notes, (key, reference)


if __name__ == "__main__":
    if sys.argv[1:] == ["fitted_vpr"]:
        fitted_vpr()
    else:
        main()
