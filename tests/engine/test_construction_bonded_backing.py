"""#1171 黏住鋼板的背襯（多孔層填骨架楊氏模數與損耗）：格式、位置規則、網頁欄位與白話名、分區換材料重組。

算數（對獨立轉接逐位、彈簧解析差、實測峰位）在 test_construction 的隔離雙精度子行程；梯度在 test_construction_diff。
這一支只驗不需要算數的部分，在本行程跑。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from aosr.gui.material_names import layers_name
from aosr.gui.problem_text import material_form_spec
from aosr.gui.zone_limits import range_target
from aosr.gui.zone_modes import Origin, zone_material
from aosr.materials.construction_models import Assembly, ConstantImpedance, MountedOnWall, PorousLayer, SurfaceMaterial
from aosr.materials.library import FixedEntry, OptimizableEntry
from tests.engine._bonded_backing_cases import (BONDED_VPR, E_MID, LOSS, MELAMINE_50, MELAMINE_100, MELAMINE_BONDED,
                                                SLOTTED_FRONT, STEEL_2_5MM)

_ADAPTER: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
_AIR: dict[str, object] = {"kind": "air", "thickness_m": 0.05}
_WALL: dict[str, object] = {"kind": "constant_impedance", "impedance_pa_s_m": 1.0e9}


def _assembly(*layers: dict[str, object], backing: str = "rigid") -> dict[str, object]:
    return {"kind": "assembly", "backing": backing, "layers": list(layers)}


def _rejected(document: dict[str, object]) -> str:
    with pytest.raises(ValidationError) as caught:
        _ADAPTER.validate_python(document)
    return str(caught.value)


@pytest.mark.parametrize("given", [{"frame_young_modulus_pa": E_MID}, {"frame_loss_factor": LOSS},
                                   {"frame_young_modulus_pa": E_MID, "frame_loss_factor": None}])
def test_frame_fields_come_in_pairs(given: dict[str, object]) -> None:
    """抓只填一格（或一格填空值）就被收下：彈簧少了楊氏模數或損耗會靜靜算錯。"""
    with pytest.raises(ValidationError, match="骨架楊氏模數與骨架損耗必須一起填"):
        PorousLayer.model_validate({**MELAMINE_100, **given})


def test_zero_loss_is_accepted_and_negative_is_not() -> None:
    """損耗可以是零（純彈簧），不能是負的；楊氏模數必須是正數。"""
    assert PorousLayer.model_validate({**MELAMINE_BONDED, "frame_loss_factor": 0.0}).bonded()
    for field, value in (("frame_loss_factor", -0.01), ("frame_young_modulus_pa", 0.0)):
        with pytest.raises(ValidationError):
            PorousLayer.model_validate({**MELAMINE_BONDED, field: value})


def test_plain_porous_dump_has_no_frame_keys() -> None:
    """沒填骨架兩格的多孔層存檔不多出欄位：既有材料的指紋與存檔一個字都不變。"""
    dumped = PorousLayer.model_validate(MELAMINE_100).model_dump()
    assert not {"frame_young_modulus_pa", "frame_loss_factor"} & set(dumped)
    assert not PorousLayer.model_validate(MELAMINE_100).bonded()


@pytest.mark.parametrize("document,words", [
    (_assembly(MELAMINE_BONDED), "前面一層必須是薄板層"),
    (_assembly(_AIR, MELAMINE_BONDED), "前面一層必須是薄板層"),
    (_assembly(STEEL_2_5MM, MELAMINE_BONDED, _AIR), "必須是最後一層"),
    (_assembly(STEEL_2_5MM, MELAMINE_BONDED, backing="air"), "背後負載必須是剛背"),
    ({"kind": "mounted_on_wall", "layers": [STEEL_2_5MM, MELAMINE_BONDED, _AIR], "wall": _WALL}, "必須是最後一層"),
    ({"kind": "measured_impedance", "source_and_conditions": "考卷用的假表",
      "points": [{"frequency_hz": 50.0, "real_pa_s_m": 400.0, "imag_pa_s_m": 0.0},
                 {"frequency_hz": 100.0, "real_pa_s_m": 400.0, "imag_pa_s_m": 0.0}],
      "base_assembly": _assembly(STEEL_2_5MM, MELAMINE_BONDED, _AIR)}, "必須是最後一層"),
])
def test_bonded_backing_place_is_enforced(document: dict[str, object], words: str) -> None:
    """抓黏住鋼板的背襯放錯位置也照算：骨架彈簧頂著牆，前面要是板、後面不能再有層、背後不能是空氣。"""
    message = _rejected(document)
    assert words in message and "骨架楊氏模數" in message, message


def test_layers_in_front_of_the_steel_are_free() -> None:
    """老闆 11-15「前面加長縫板、圓孔板」：鋼板前面疊什麼都收；貼在原牆上的也收。"""
    front = _ADAPTER.validate_python(_assembly(SLOTTED_FRONT, MELAMINE_50, STEEL_2_5MM, MELAMINE_BONDED))
    assert isinstance(front, Assembly) and front.layers[-1] == PorousLayer.model_validate(MELAMINE_BONDED)
    mounted = _ADAPTER.validate_python({"kind": "mounted_on_wall", "layers": [SLOTTED_FRONT, _AIR, STEEL_2_5MM, MELAMINE_BONDED],
                                        "wall": _WALL})
    assert isinstance(mounted, MountedOnWall)
    # 沒黏的多孔層照舊放哪裡都可以（不是新規則管的）。
    assert isinstance(_ADAPTER.validate_python(_assembly(STEEL_2_5MM, MELAMINE_100, _AIR, backing="air")), Assembly)


def test_input_schema_lists_the_pair() -> None:
    """產生的輸入綱要有兩格、而且成對（一格都不給或兩格都給）。"""
    schema = json.loads((Path(__file__).resolve().parents[2] / "blueprint" / "schemas" / "three_lane_report_input.schema.json").read_text())
    porous = schema["$defs"]["PorousLayer"]
    assert {"frame_young_modulus_pa", "frame_loss_factor"} <= set(porous["properties"])
    pairs = [rule for rule in porous["allOf"] if "oneOf" in rule]
    assert pairs == [{"oneOf": [
        {"not": {"anyOf": [{"required": ["frame_young_modulus_pa"]}, {"required": ["frame_loss_factor"]}]}},
        {"required": ["frame_young_modulus_pa", "frame_loss_factor"],
         "properties": {"frame_young_modulus_pa": {"not": {"type": "null"}}, "frame_loss_factor": {"not": {"type": "null"}}}}]}]


def test_web_form_has_the_two_fields() -> None:
    """網頁材料表單照模型描述多兩格，欄名與單位是中文（老闆在網頁上填）。"""
    fields = material_form_spec()["porous"]["fields"]
    assert isinstance(fields, dict)
    assert {name: (fields[name]["label"], fields[name]["unit"]) for name in ("frame_young_modulus_pa", "frame_loss_factor")} == {
        "frame_young_modulus_pa": ("骨架楊氏模數", "Pa"), "frame_loss_factor": ("骨架損耗因數", "無量綱")}


def test_plain_name_says_vpr() -> None:
    """網頁白話名把黏住鋼板的背襯講成 VPR，並寫出骨架彈性；沒黏的照舊。"""
    material = _ADAPTER.validate_python(BONDED_VPR)
    assert isinstance(material, Assembly)
    name = layers_name(material.layers, ()).text
    assert "VPR 複合板共振背襯" in name and "0.05125 MPa" in name, name
    plain = _ADAPTER.validate_python(_assembly(STEEL_2_5MM, MELAMINE_100))
    assert isinstance(plain, Assembly) and "VPR" not in layers_name(plain.layers, ()).text


def test_optimizable_porous_cannot_be_bonded() -> None:
    """抓可最佳化材料的多孔層填了骨架兩格也存得進去（11-15 測試網頁實測過存得進）：多孔層＋空腔前面沒有薄板，一排出構造就壞。"""
    entry: dict[str, object] = {
        "kind": "optimizable", "id": "test_opt", "label": "考卷 可最佳化", "provenance": "model", "source": "考卷",
        "porous": MELAMINE_BONDED, "porous_thickness": {"adjustable": False, "initial_m": 0.1, "range_m": [0.1, 0.1]},
        "cavity": {"adjustable": True, "initial_m": 0.05, "range_m": [0.0, 0.1]}}
    with pytest.raises(ValidationError, match="可最佳化材料的多孔層不能填骨架楊氏模數與骨架損耗"):
        OptimizableEntry.model_validate(entry)
    assert OptimizableEntry.model_validate({**entry, "porous": MELAMINE_100}).porous == PorousLayer.model_validate(MELAMINE_100)


def test_zone_rebuilds_keep_the_bonded_backing() -> None:
    """分區換材料把固定材料貼到原牆上、可信範圍把貼上的層拆回剛背：兩條重組路都不能擋或弄丟背襯。"""
    entry = FixedEntry.model_validate({"id": "test_vpr", "label": "考卷 VPR", "provenance": "model", "source": "考卷",
                                       "kind": "fixed", "material": BONDED_VPR})
    wall = ConstantImpedance(impedance_pa_s_m=1.0e9)
    chosen = entry.surface_material()
    mounted, _ = zone_material(wall, Origin(wall, "current"), entry)
    assert isinstance(chosen, Assembly) and isinstance(mounted, MountedOnWall) and mounted.layers == chosen.layers
    target = range_target(mounted)
    assert isinstance(target, Assembly) and target.backing == "rigid" and target.layers == mounted.layers
