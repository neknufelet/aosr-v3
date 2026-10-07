"""#559 家具列與表頭輸出契約；無反射的家具仍在表頭、欄位四件事不漏。"""
from __future__ import annotations

from dataclasses import asdict

import pytest

from aosr.physics import report_io
from aosr.physics.report_io import PathRow, PathTableSection
from tests.engine.test_furniture_path_table import built_table


def furniture_row() -> dict[str, object]:
    # 獨立手定輸出範例；不拿被測 builder 製造驗證器的答案。
    return dict(order=1, wall_sequence=("furniture",), delay_s=1.5 / 343.0, distance_m=1.5,
        direction_vector=(-0.8, 0.0, -0.6), direction_angles={"azimuth_deg": 180.0, "elevation_deg": -36.86989764584402},
        departure_off_axis_deg=None, relative_direct_energy=(0.1, 0.2, 0.3),
        furniture_id="desk", furniture_face="top", reflection_point_m=(3.0, 2.0, 0.6))


def header(**changes: object) -> dict[str, object]:
    return dict(reflection_order_k=1, frequencies_hz=(125.0, 500.0, 2000.0), scattering_coefficient=(0.36, 0.36, 0.36),
        source_model_kind="omnidirectional", rows=(furniture_row(),), furniture_ids=("desk",),
        furniture_model="single_bounce_finite_size_v1", blocked_wall_paths=(("floor",),),
        furniture_materials=({"furniture_id": "desk", "material": "wood", "unknown_bands_hz": (63.0, 8000.0)},),
    ) | changes


@pytest.mark.parametrize("face", ("top", "bottom", "+x", "-x", "+y", "-y"))
def test_valid_furniture_row_round_trips_with_structured_face(face: str) -> None:
    row = PathRow.model_validate(furniture_row() | {"furniture_face": face})
    assert row.wall_sequence == ("furniture",)
    assert row.furniture_face == face
    assert PathRow.model_validate_json(row.model_dump_json()) == row


@pytest.mark.parametrize("change", ({"order": 2}, {"wall_sequence": ("x0",)},
    {"wall_sequence": ("furniture", "x0")}, {"furniture_id": None}, {"furniture_face": None}, {"reflection_point_m": None}))
def test_furniture_row_rejects_invalid_shape(change: dict[str, object]) -> None:
    PathRow.model_validate(furniture_row())
    with pytest.raises(ValueError, match="家具|牆面"):
        PathRow.model_validate(furniture_row() | change)


@pytest.mark.parametrize("field,value", (("furniture_id", "desk"), ("furniture_face", "top"),
    ("reflection_point_m", (3.0, 2.0, 0.6))))
def test_wall_row_rejects_any_furniture_field(field: str, value: object) -> None:
    row = furniture_row() | {"wall_sequence": ("floor",), "furniture_id": None,
        "furniture_face": None, "reflection_point_m": None}
    PathRow.model_validate(row)
    with pytest.raises(ValueError, match="牆面"):
        PathRow.model_validate(row | {field: value})


def test_header_without_furniture_rejects_furniture_rows() -> None:
    PathTableSection.model_validate(header())
    with pytest.raises(ValueError, match="家具"):
        PathTableSection.model_validate(header(furniture_ids=None, furniture_model=None,
            blocked_wall_paths=None, furniture_materials=None))


@pytest.mark.parametrize("change", ({"furniture_ids": ("other",)}, {"furniture_ids": ("desk", "desk")},
    {"furniture_model": None}, {"furniture_materials": None}, {"blocked_wall_paths": None},
    {"rows": (furniture_row() | {"furniture_id": "other"},)},
    {"furniture_materials": ({"furniture_id": "other", "material": "wood", "unknown_bands_hz": ()},)}))
def test_header_rejects_mismatched_input_materials_or_row_ids(change: dict[str, object]) -> None:
    PathTableSection.model_validate(header())
    with pytest.raises(ValueError, match="家具"):
        PathTableSection.model_validate(header(**change))


def test_input_furniture_with_no_visible_reflections_stays_in_header() -> None:
    section = PathTableSection.model_validate(header(rows=()))
    assert section.furniture_ids == ("desk",)
    assert section.furniture_model == "single_bounce_finite_size_v1"
    assert PathTableSection.model_validate_json(section.model_dump_json()) == section


def test_unfurnished_rows_and_header_omit_every_new_field() -> None:
    section = PathTableSection.model_validate(asdict(built_table(items=None)))
    payload = section.model_dump(mode="json")
    assert not {"furniture_ids", "furniture_model", "blocked_wall_paths", "furniture_materials"} & payload.keys()
    for row in payload["rows"]:
        assert not {"furniture_id", "furniture_face", "reflection_point_m"} & row.keys()


def test_furniture_facts_include_material_unknown_bands_and_coordinate_basis() -> None:
    table = report_io.quantity_table()
    fields = {
        "path_table.rows.furniture_id", "path_table.rows.furniture_face", "path_table.rows.reflection_point_m",
        "path_table.furniture_ids", "path_table.furniture_model", "path_table.blocked_wall_paths", "path_table.furniture_materials",
        "path_table.furniture_materials.furniture_id", "path_table.furniture_materials.material", "path_table.furniture_materials.unknown_bands_hz",
    }
    assert fields <= table.keys()
    for name in fields:
        assert all(table[name])
    assert table["path_table.rows.reflection_point_m"].unit == "m"
    assert "原點" in table["path_table.rows.reflection_point_m"].reference
    assert table["path_table.furniture_materials.unknown_bands_hz"].unit == "Hz"
    assert "相鄰頻帶延伸代算" in table["path_table.furniture_materials.unknown_bands_hz"].reference

