"""第二區構造的正式雙精度註記控制；不改考卷父行程的精度。"""
from aosr.runtime import configure_calculation_jax

configure_calculation_jax()

from aosr.geometry.shoebox import Wall
from pathlib import Path
import sys
from aosr.materials.construction_models import Assembly, ConstructionNote
from aosr.physics.wall_surfaces import zoned_wall_normal_impedances, zoned_wall_surface_response
from aosr.reporting.scheme import Scheme


def check_notes() -> None:
    """只在 x0 第二區放超出樣品頻率範圍的多孔構造。"""
    scheme = Scheme.model_validate_json(Path(sys.argv[1]).read_bytes())
    zones = dict(scheme.scene.surface_zones_by_wall or {})
    a, b = zones["x0"]
    material = Assembly.model_validate({"kind": "assembly", "layers": [{"kind": "porous",
        "thickness_m": 0.05, "flow_resistivity_pa_s_m2": 60000.0, "material_class": "foam"}], "backing": "rigid"})
    zones["x0"] = (a, b.model_copy(update={"material": material}))
    notes: dict[Wall, tuple[ConstructionNote, ...]] = {}
    arguments = dict(density_kg_m3=scheme.scene.density_kg_m3, sound_speed_m_s=scheme.scene.sound_speed_m_s)
    zoned_wall_normal_impedances(scheme.scene.room_m, zones, (20.0, 200.0), notes_by_wall=notes, **arguments)
    diffuse = zoned_wall_surface_response(scheme.scene.room_m, zones, (20.0, 200.0), **arguments)
    expected = diffuse.equivalent.notes_by_wall[Wall.X0]
    assert expected
    assert notes.get(Wall.X0) == expected, notes
    assert all(not value for wall, value in notes.items() if wall != Wall.X0), notes


if __name__ == "__main__":
    check_notes()
