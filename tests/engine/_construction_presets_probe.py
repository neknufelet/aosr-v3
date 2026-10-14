"""冷子行程開双精度，逐筆經完整方案驗證，再走晚期的正式牆面回應。"""
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


if __name__ == "__main__":
    main()
