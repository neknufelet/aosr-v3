"""#561：矩形分區只進格式，計算入口拒收；不呼叫物理解算。"""
import hashlib
import json
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from aosr.physics import report_io, report_surfaces
from aosr.reporting.scheme import Scene, Scheme, load_scheme
from aosr.search.placement_stability_record import scheme_hash
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES, pair
from tests.engine._surface_zone_cases import NEW, OLD, rectangle_document, rectangle_scene, zone_document


@pytest.mark.parametrize("mode", ["python", "json"])
def test_rectangles_round_trip_in_scene_report_and_file(tmp_path: Path, mode: str) -> None:
    """抓只放寬一個入口、存回丟區或不冪等；共邊的 2×2 必須收。"""
    scheme = Scheme.model_validate(rectangle_document())
    dumped = scheme.model_dump(mode="json" if mode == "json" else "python")
    assert Scheme.model_validate(dumped).model_dump(mode="json") == scheme.model_dump(mode="json")
    assert Scene.model_validate(rectangle_scene()) == scheme.scene
    inputs = report_io.load_input_document(pair(scheme), CAPABILITIES, DIRECTIVITY)
    assert inputs.surface_zones_by_wall is not None
    assert report_io.load_input_document(inputs.model_dump(), CAPABILITIES, DIRECTIVITY).model_dump() == inputs.model_dump()
    path = tmp_path / "rectangles.json"
    path.write_text(scheme.model_dump_json(), encoding="utf-8")
    assert load_scheme(path).model_dump_json() == scheme.model_dump_json()
    schema = json.loads(Path("blueprint/schemas/three_lane_report_input.schema.json").read_text())
    assert schema == report_io.input_schema()


@pytest.mark.parametrize("entry", ["scene", "report"])
@pytest.mark.parametrize("case,hint", [("overlap", "重疊"), ("gap", "蓋滿"), ("outside", "範圍"),
    ("reverse", "起點"), ("duplicate", "重複"), ("dot", "代號"), ("mixed", "整面一區"), ("missing", "必填")])
def test_bad_rectangle_partition_names_wall_and_id(entry: str, case: str, hint: str) -> None:
    """抓重疊、縫、越界、倒序、重名、不合法代號、混寫、漏座標；兩個文件入口都要守。"""
    scene = rectangle_scene()
    zones = cast(dict[str, list[dict[str, object]]], scene[NEW])
    first = zones["x0"][0]
    zone_id = "r0_c0"
    if case == "overlap":
        first["u_m"] = [0.0, 3.0]
    elif case == "gap":
        first["u_m"] = [0.0, 1.0]
    elif case == "outside":
        first["u_m"] = [-1.0, 2.0]
    elif case == "reverse":
        first["u_m"] = [2.0, 2.0]
    elif case == "duplicate":
        zones["x0"][1]["id"] = zone_id
    elif case == "dot":
        zone_id = "r0.c0"
        first["id"] = zone_id
    elif case == "mixed":
        zones["x0"].append({"area": "whole_wall", "material": first["material"]})
    else:
        first.pop("u_m")
    with pytest.raises(ValidationError) as caught:
        if entry == "scene":
            Scene.model_validate(scene)
        else:
            base = pair(Scheme.model_validate(zone_document()))
            for name in Scene.model_fields:
                base.pop(name, None)
            report_io.ReportInput.model_validate(base | scene, context={"table": CAPABILITIES, "directivity": DIRECTIVITY})
    text = str(caught.value)
    assert "x0" in text and zone_id in text and hint in text, text


def test_constant_rectangles_never_fold_to_six_numbers() -> None:
    """抓全固定材料就折回、抹掉矩形；一個矩形也不是整面寫法。"""
    scene = Scene.model_validate(rectangle_scene(only_wall="x0"))
    assert scene.impedance_pa_s_per_m_by_wall is None
    assert OLD not in scene.model_dump(mode="json")
    assert len(scene.model_dump(mode="json")[NEW]["x0"]) == 4


# 2026-11-04，乾淨起點 8fd6562513c3a71c1e204bba22dba56203523395 親量；未改產品程式前存證。
CONTROLS = [
    (False, "4db605563e8bba46252ca4d6eecb0bba837f588ddc6690a5ab141f45e468d8c5",
     "40403e6dda09bebd4cbe9e8a31b5fc6f9c8a241ff75ffb5f93ca27ad83ea5025",
     "4c01224256f69a943501f9fa684c7c911396d6af443aa5d38d2ac9ea988034fa"),
    (True, "e18aec2c8063812451bba1801dd384dc746524b6ecad89f3409bb9b8461ffbb6",
     "7c2865df5d3fd98e034f820cfebf883d85f8cd808eaa03e578fe9915d344d48b",
     "9a6e4d3c89bc26318d1ab33fc535955f970916f3498b6b21f33fe807471153e4")]


@pytest.mark.parametrize("mixed,scheme_sha,report_sha,search_sha", CONTROLS)
def test_whole_wall_mainline_bytes(mixed: bool, scheme_sha: str, report_sha: str, search_sha: str) -> None:
    """抓整面寫法的存檔、正規化文件與搜尋身分任一位元漂移。"""
    scheme = Scheme.model_validate(zone_document(mixed=mixed))
    inputs = report_io.load_input_document(pair(scheme), CAPABILITIES, DIRECTIVITY)
    assert hashlib.sha256(scheme.model_dump_json().encode()).hexdigest() == scheme_sha
    assert hashlib.sha256(inputs.model_dump_json().encode()).hexdigest() == report_sha
    assert scheme_hash(scheme) == search_sha


def test_solver_explicitly_refuses_rectangles() -> None:
    inputs = report_io.load_input_document(pair(Scheme.model_validate(rectangle_document())), CAPABILITIES, DIRECTIVITY)
    with pytest.raises(report_surfaces.RectangleZonesUnsupported, match=r"ceiling.*這一段還不支援矩形分區（#561 之後幾步）"):
        report_io.solver_inputs(inputs)


@pytest.mark.parametrize("wall,u,v", [("floor", 6.0, 4.0), ("ceiling", 6.0, 4.0),
    ("x0", 4.0, 3.0), ("xL", 4.0, 3.0), ("y0", 6.0, 3.0), ("yL", 6.0, 3.0)])
def test_single_full_rectangle_is_preserved_but_not_calculable(wall: str, u: float, v: float) -> None:
    """矩形一區也不折回；六面各自依正確軸向驗邊界。"""
    from tests.engine._surface_zone_cases import zone_scene

    scene = zone_scene()
    zones = cast(dict[str, list[dict[str, object]]], scene[NEW])
    zones[wall] = [{"area": "rectangle", "id": "full-A_1", "u_m": [0, u], "v_m": [0, v],
                    "material": zones[wall][0]["material"]}]
    parsed = Scene.model_validate(scene)
    assert parsed.surface_zones_by_wall is not None and parsed.impedance_pa_s_per_m_by_wall is None
    with pytest.raises(report_surfaces.RectangleZonesUnsupported) as caught:
        report_surfaces.whole_wall_materials(parsed.surface_zones_by_wall)
    assert str(caught.value) == f"牆面 {wall}：這一段還不支援矩形分區（#561 之後幾步）"


@pytest.mark.parametrize("overlap", [False, True])
def test_one_float_step_gap_or_overlap_is_never_tolerated(overlap: bool) -> None:
    """抓以容差忽略一個浮點步長的細縫或重疊。"""
    import math

    scene = rectangle_scene()
    zones = cast(dict[str, list[dict[str, object]]], scene[NEW])
    zones["x0"][0]["u_m"] = [0.0, math.nextafter(2.0, math.inf if overlap else 0.0)]
    with pytest.raises(ValidationError, match="重疊" if overlap else "蓋滿"):
        Scene.model_validate(scene)


def test_zone_schema_has_two_exclusive_array_shapes() -> None:
    """抓綱要仍限一區，或放行空列、兩整面、混寫；覆蓋檢查由程式守。"""
    schema = report_io.input_schema()
    field = cast(dict[str, object], cast(dict[str, object], schema["properties"])[NEW])
    walls = cast(dict[str, dict[str, object]], field["properties"])
    assert set(walls) == {"floor", "ceiling", "x0", "xL", "y0", "yL"}
    for cell in walls.values():
        assert cell["minItems"] == 1 and "maxItems" not in cell
        whole, rectangle = cast(list[dict[str, object]], cell["anyOf"])
        assert whole == {"items": {"$ref": "#/$defs/WholeWallZone"}, "maxItems": 1}
        assert rectangle == {"items": {"$ref": "#/$defs/RectangleZone"}}
        assert "蓋滿與不重疊由程式" in str(cell["description"])


@pytest.mark.parametrize("coordinate", [False, "0"])
def test_rectangle_coordinates_are_numbers_not_coerced_values(coordinate: object) -> None:
    """JSON 綱要的數字不包含真假值或文字；格式入口也必須一致拒收。"""
    scene = rectangle_scene()
    zones = cast(dict[str, list[dict[str, object]]], scene[NEW])
    zones["x0"][0]["u_m"] = [coordinate, 2.0]
    with pytest.raises(ValidationError) as caught:
        Scene.model_validate(scene)
    text = str(caught.value)
    assert "x0" in text and "r0_c0" in text and "數字" in text
