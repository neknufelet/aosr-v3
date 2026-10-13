"""正式三入口拒收、正規化、完整材料傳遞與固定場景逐位控制。"""
import json
from pathlib import Path
from typing import Literal, cast

import pytest
from pydantic import ValidationError

from aosr.physics import report_io
from aosr.physics.report_validation import ReportInputError
from aosr.reporting.modal_lookup import key_from_scheme
from aosr.reporting.scheme import Scene, Scheme
from aosr.reporting.validation import SchemeValidationError, checked_inputs, validated_scheme
from aosr.search.ledger import header_for
from aosr.search.placement_stability_record import scheme_hash
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES, pair, reference_document
from tests.engine._surface_zone_cases import NEW, OLD, invalid_scene, mixed_scheme, zone_document
from tests.engine.test_scheme_furniture import _store


@pytest.mark.parametrize("case", ["both", "neither", "null-old", "null-new", "missing", "extra", "two", "area", "measured", "layer"])
@pytest.mark.parametrize("entry", ["scheme", "report", "search"])
def test_bad_zones_are_located_at_all_three_entries(tmp_path: Path, case: str, entry: str) -> None:
    scene, path, hint = invalid_scene(case)
    document = reference_document() | {"scene": scene}
    with pytest.raises((SchemeValidationError, ValidationError, ReportInputError)) as caught:
        if entry == "scheme":
            validated_scheme(document)
        elif entry == "report":
            old_pair = pair(Scheme.model_validate(reference_document()))
            for name in Scene.model_fields:
                old_pair.pop(name, None)
            report_io.load_input_document(old_pair | scene, CAPABILITIES, DIRECTIVITY)
        else:
            # 故意繞過模型檢查，證明搜尋建立自己的再驗證會在落盤之前拒收。
            project = Scheme.model_validate(reference_document()).model_copy(update={
                "scene": Scene.model_construct(_fields_set=set(scene), **scene)})
            _store(tmp_path / "search", project)
    text = str(caught.value)
    assert path in text and hint in text, text
    assert "union_tag_invalid" not in text
    assert not (tmp_path / "search").exists()


def test_all_fixed_new_format_has_every_original_fingerprint(tmp_path: Path) -> None:
    old, old_pairs = checked_inputs(reference_document(), capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    new, new_pairs = checked_inputs(zone_document(), capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    assert old == new
    assert old.model_dump_json() == new.model_dump_json()
    assert old_pairs == new_pairs
    for _, inputs in new_pairs.values():
        assert report_io.scene_fingerprint(inputs) == "ef718eb5ecfa13d9d0c84320d18aeb99566db02021bdc16d7b7cbb2644b74de2"
    store = _store(tmp_path, new)
    assert header_for(store).project_fingerprint == "0cd0e215f3fa0fe6d0de2798098fbc57b283caa35e946c68160690406c2b4d1f"
    assert json.loads((store.path / "project.json").read_text()) == old.model_dump(mode="json")
    assert scheme_hash(old) == scheme_hash(new)
    old_key, new_key = key_from_scheme(old), key_from_scheme(new)
    assert old_key is not None and new_key is not None
    assert old_key.digest == new_key.digest


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("mode", ["python", "json"])
def test_zone_normalization_is_idempotent_and_order_independent(mixed: bool, mode: Literal["python", "json"]) -> None:
    document = zone_document(mixed=mixed)
    scheme = Scheme.model_validate(document)
    dumped = scheme.model_dump(mode=mode)
    assert Scheme.model_validate(dumped) == scheme
    scene = cast(dict[str, object], document["scene"])
    zones = cast(dict[str, object], scene[NEW])
    scene[NEW] = dict(reversed(list(zones.items())))
    reordered = Scheme.model_validate(document)
    assert reordered == scheme
    inputs = report_io.load_input_document(pair(scheme), CAPABILITIES, DIRECTIVITY)
    assert report_io.load_input_document(inputs.model_dump(mode=mode), CAPABILITIES, DIRECTIVITY) == inputs
    if mixed:
        assert scheme.scene.impedance_pa_s_per_m_by_wall is None
        assert OLD not in dumped["scene"]
    else:
        assert NEW not in dumped["scene"]


def test_mixed_solver_keeps_validated_material_objects_without_fake_impedances() -> None:
    inputs = report_io.load_input_document(pair(mixed_scheme()), CAPABILITIES, DIRECTIVITY)
    solved = report_io.solver_inputs(inputs)
    assert solved.impedance_by_wall is None
    assert inputs.surface_zones_by_wall is not None
    for wall, material in solved.materials_by_wall.items():
        assert material is inputs.surface_zones_by_wall[wall.wall_name()][0].material


def test_zone_schema_expresses_exclusive_inputs_and_complete_wall_shape() -> None:
    schema = report_io.input_schema()
    assert schema["oneOf"] == [
        {"required": [OLD], "not": {"required": [NEW]}},
        {"required": [NEW], "not": {"required": [OLD]}}]
    properties = schema["properties"]
    assert isinstance(properties, dict)
    field = properties[NEW]
    assert isinstance(field, dict)
    assert set(field["required"]) == {"x0", "xL", "y0", "yL", "floor", "ceiling"}
    assert field["additionalProperties"] is False
    assert NEW in report_io.ReportInput.quantity_table()


def test_fixed_zones_cli_matches_original_sha(tmp_path: Path) -> None:
    from tests.engine.test_material_flow import _child

    _child(tmp_path, "constant-zones")


@pytest.mark.parametrize("model,parameters", [("komatsu", {}), ("miki", {}),
    ("wilson_2015", {"porosity": 0.98, "tortuosity": 1.1}), ("mechel_1976", {"porosity": 0.98})])
def test_all_porous_models_round_trip_without_inventing_unused_parameters(model: str, parameters: dict[str, float]) -> None:
    document = zone_document(mixed=True)
    scene = cast(dict[str, object], document["scene"])
    zones = cast(dict[str, list[dict[str, object]]], scene[NEW])
    zones["floor"][0]["material"] = {"kind": "assembly", "backing": "rigid", "layers": [
        {"kind": "porous", "thickness_m": 0.1, "flow_resistivity_pa_s_m2": 10000.0,
         "material_class": "fibrous", "model": model, **parameters}]}
    scheme = Scheme.model_validate(document)
    for mode in ("python", "json"):
        assert Scheme.model_validate(scheme.model_dump(mode=mode)) == scheme
        inputs = report_io.load_input_document(pair(scheme), CAPABILITIES, DIRECTIVITY)
        assert report_io.load_input_document(inputs.model_dump(mode=mode), CAPABILITIES, DIRECTIVITY) == inputs


def test_null_area_keeps_zone_rejection_instead_of_generic_required() -> None:
    scene, path, _ = invalid_scene("area")
    zones = cast(dict[str, list[dict[str, object]]], scene[NEW])
    zones["floor"][0]["area"] = None
    with pytest.raises(SchemeValidationError) as caught:
        validated_scheme(reference_document() | {"scene": scene})
    assert path in str(caught.value) and "矩形分區是 #309 第二步" in str(caught.value)


def test_supplied_null_alternative_is_still_two_fields() -> None:
    document = zone_document(mixed=True)
    cast(dict[str, object], document["scene"])[OLD] = None
    with pytest.raises(SchemeValidationError, match="恰好"):
        validated_scheme(document)


def test_formal_construction_input_reaches_three_lane_preparation(tmp_path: Path) -> None:
    from tests.engine.test_material_flow import _child

    _child(tmp_path, "formal-zones")
