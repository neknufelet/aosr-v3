"""空腔範圍的下限介於 0 和匯出最小空腔之間（#1234）：收下的範圍，匯出的構造一定守得住它。

匯出規則（辛二，``endpoint.kept_cavity``，這一支不改）：空腔小於設定檔 ``[export] min_cavity_m`` 就整層拿掉。範圍的下限要是 0
（可以沒有空腔）或至少那個數，不然搜尋落在下限附近時，交出去的構造沒有空氣層，違反使用者寫的範圍。兩個來源：分區自己的範圍
（``ZoneRange``，起跑設計那一步收窄）與材料頁那一份材料自己的空腔範圍（使用者材料庫接成搜尋家族那一步）。

數字全是考卷自己編的：分區那一組用內建 48K（設定檔：產品厚度 0.05／0.1／0.2、空腔 0～0.25、總深 0.3；樣板 50 mm 棉＋50 mm 空腔），
x0 那一面收窄成棉 0.05～0.1、只換空腔範圍；材料頁那一組是這裡編的一筆可最佳化材料（棉厚產品 0.05／0.1、總深 0.2，只換空腔範圍）。
最小空腔從設定檔讀；另用一份自訂設定（最小空腔加倍）核判法跟著這一場的設定走，不是寫死一個數。
"""
from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from aosr.config.material_gradient import ExportSettings, MaterialGradientSettings, load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Wall
from aosr.gradient import endpoint
from aosr.gradient.depth_fit import ZoneRange
from aosr.gradient.start_config import designs_for, presets_table
from aosr.gradient.surface_design import SurfaceDesign
from aosr.materials import library
from aosr.materials.construction_models import AirLayer, Assembly, ConstantImpedance, SurfaceMaterial
from aosr.materials.search_library import read_user_entries, with_user_materials

SETTINGS = load_material_gradient_settings(config_path("material_gradient.toml"))
WOOL = "glass_wool_48k"
PRESET = "glass_wool_48k_50mm_gap50"
PAGE = "demo_thin_gap_wool"
FACE = "x0"
# 拒收句裡一定有的那幾個字（講匯出時整層拿掉）；別的原因拒收不算數。
REFUSED_WORDS = "整層拿掉"
_ENTRY: TypeAdapter[library.LibraryEntry] = TypeAdapter(library.LibraryEntry)
Designs = Callable[[MaterialGradientSettings, tuple[float, float], Path], tuple[SurfaceDesign, ...]]


def _doubled() -> MaterialGradientSettings:
    """自訂設定：最小空腔是設定檔那個數的兩倍，其餘照設定檔。"""
    return SETTINGS.model_copy(update={"export": ExportSettings(min_cavity_m=SETTINGS.export.min_cavity_m * 2.0)})


def _cases(minimum: float) -> tuple[tuple[str, float, float], ...]:
    """（名字, 下限, 上限），全照這一場的最小空腔換算：下限 0 與恰好最小空腔照收，介於中間的拒收（固定深度是下限＝上限）。"""
    return (("兩端都比最小空腔薄", minimum / 2.0, minimum * 0.8),
            ("下限 0、上限比最小空腔薄", 0.0, minimum * 0.8),
            ("下限恰是最小空腔", minimum, minimum * 2.0),
            ("下限比最小空腔薄、上限寬", minimum / 2.0, 0.05),
            ("固定深度比最小空腔薄", minimum / 2.0, minimum / 2.0))


def _kept() -> dict[Wall, SurfaceMaterial]:
    return {wall: ConstantImpedance(impedance_pa_s_m=12348.0) for wall in Wall.all()}


def _zone_designs(settings: MaterialGradientSettings, cavity: tuple[float, float], _folder: Path) -> tuple[SurfaceDesign, ...]:
    """分區自己的範圍：六面都起跑 48K，x0 那一面寫範圍（命令列 ``--range`` 與網頁開跑都走這一支起跑設計）。"""
    ranges = {FACE: {WOOL: ZoneRange(porous_m=(0.05, 0.1), cavity_m=cavity)}}
    return designs_for((PRESET,) * 6, settings, kept=_kept(), ranges=ranges)


def _page_entry(cavity: tuple[float, float]) -> library.LibraryEntry:
    """材料頁那一份：自編一筆可最佳化材料，空腔範圍照給的；兩端相等就是固定深度（不能調），否則能調、初值取上限。"""
    low, high = cavity
    return _ENTRY.validate_python({
        "kind": "optimizable", "id": PAGE, "label": "示範薄空腔棉（自編）", "provenance": "model", "source": "考卷自編",
        "porous": {"kind": "porous", "thickness_m": 0.05, "flow_resistivity_pa_s_m2": 17500.0, "material_class": "fibrous"},
        "porous_thickness": {"adjustable": True, "range_m": [0.05, 0.1], "choices_m": [0.05, 0.1], "initial_m": 0.05},
        "cavity": {"adjustable": low < high, "range_m": [low, high], "initial_m": high}, "max_total_depth_m": 0.2})


def _page_designs(settings: MaterialGradientSettings, cavity: tuple[float, float], folder: Path) -> tuple[SurfaceDesign, ...]:
    """材料頁那一份材料自己的範圍：材料庫寫成檔、開跑那一步讀回來接成家族（``read_user_entries`` → ``with_user_materials``），
    六面都起跑這一種、不寫分區範圍。"""
    path = folder / f"{cavity[0]!r}-{cavity[1]!r}-{settings.export.min_cavity_m!r}.json"
    path.write_text(library.dump_user_library((_page_entry(cavity),)), encoding="utf-8")
    materials = with_user_materials(settings, presets_table(), read_user_entries(path))
    return designs_for((PAGE,) * 6, materials.settings, kept=_kept(), table=materials.presets)


def _theta(designs: tuple[SurfaceDesign, ...], cavity: float) -> np.ndarray:
    """每面棉取最薄的合法厚度；x0 的空腔放 ``cavity``，其他面放自己的空腔下界。"""
    values: list[float] = []
    for design in designs:
        if design.material is None:
            continue
        values.append(design.legal_thicknesses_m[0])
        if design.cavity_variable is not None:
            values.append(cavity if design.surface_id == FACE else design.lower[1])
    return np.asarray(values, dtype=np.float64)


def _exported_cavity(designs: tuple[SurfaceDesign, ...], cavity: float, settings: MaterialGradientSettings) -> float:
    """真的走終點匯出（``materials_from_surfaces``），回 x0 那一面匯出構造裡的空腔（沒有空氣層是 0）。"""
    material = endpoint.materials_from_surfaces(designs, _theta(designs, cavity), settings)[Wall.from_name(FACE)]
    assert isinstance(material, Assembly)
    return sum(layer.thickness_m for layer in material.layers if isinstance(layer, AirLayer))


def test_accepted_cavity_ranges_are_kept_by_the_exported_material(tmp_path: Path) -> None:
    """抓 #1234：空腔範圍的下限介於 0 和最小空腔之間照樣收下，匯出時整層拿掉、交出去的構造違反範圍。

    兩個來源（分區範圍、材料頁那一份）× 兩份設定（設定檔、最小空腔加倍）× 五種範圍：收下的每一個，θ 的空腔放在範圍兩端走
    ``materials_from_surfaces``，x0 匯出構造的空腔要在範圍裡（沒有空氣層算 0，只有下限是 0 才合）；開跑前拒收的要是
    下限介於 0 和那一場最小空腔之間的那幾個，而且拒收句講的是匯出整層拿掉，其餘一個都不准拒收。"""
    sources: tuple[tuple[str, Designs], ...] = (("分區範圍", _zone_designs), ("材料頁", _page_designs))
    violations: list[str] = []
    refused: set[tuple[str, float, str]] = set()
    expected: set[tuple[str, float, str]] = set()
    for settings in (SETTINGS, _doubled()):
        minimum = settings.export.min_cavity_m
        for name, low, high in _cases(minimum):
            for source, build in sources:
                key = (source, minimum, name)
                if 0.0 < low < minimum:
                    expected.add(key)
                try:
                    designs = build(settings, (low, high), tmp_path)
                except ValueError as error:
                    if REFUSED_WORDS not in str(error):
                        violations.append(f"{source}（最小空腔 {minimum}）{name} {low}～{high}：別的原因拒收 {error}")
                    refused.add(key)
                    continue
                for end in dict.fromkeys((low, high)):
                    exported = _exported_cavity(designs, end, settings)
                    if not low <= exported <= high:
                        violations.append(f"{source}（最小空腔 {minimum}）{name} {low}～{high} 收下了，"
                                          f"空腔 {end} 匯出成 {exported}")
    assert violations == []
    assert refused == expected


# ---- 判法與拒收句（一份判法，四個地方叫） ----

def _sentence(low_cm: str, floor_cm: str) -> str:
    return f"空腔下限 {low_cm} cm 要寫 0（可以沒有空腔）或至少 {floor_cm} cm，因為比 {floor_cm} cm 薄的空腔匯出時會整層拿掉"


def _fixed_sentence(depth_cm: str, floor_cm: str) -> str:
    """不能調的空腔（材料頁那一格只有「深度」）那一句；「有空腔這一格」是材料頁那個勾選的名字（``materials.js``）。"""
    return f"空腔深度 {depth_cm} cm 要至少 {floor_cm} cm，或取消勾選「有空腔這一格」，因為比 {floor_cm} cm 薄的空腔匯出時會整層拿掉"


def test_cavity_floor_takes_zero_or_the_minimum_and_says_why_in_centimeters() -> None:
    """抓邊界放錯或數字寫死：0 與恰好最小空腔照收，介於中間（含差最小空腔一個浮點格）拒收；句子是公分、最小空腔照設定。
    不能調的空腔只換字（講深度、叫人取消勾選「有空腔這一格」），收不收照同一條。"""
    export, doubled = SETTINGS.export, _doubled().export
    minimum = export.min_cavity_m
    assert export.cavity_floor_problem(0.0) is None and export.cavity_floor_problem(minimum) is None
    assert export.cavity_floor_problem(minimum * 3.0) is None
    assert export.cavity_floor_problem(minimum / 2.0) == _sentence("0.05", "0.1")
    assert export.cavity_floor_problem(float(np.nextafter(minimum, 0.0))) is not None
    assert doubled.cavity_floor_problem(minimum) == _sentence("0.1", "0.2")
    assert doubled.cavity_floor_problem(minimum * 2.0) is None
    assert export.cavity_floor_problem(minimum / 2.0, adjustable=False) == _fixed_sentence("0.05", "0.1")
    assert export.cavity_floor_problem(minimum, adjustable=False) is None
    assert export.cavity_floor_problem(0.0, adjustable=False) is None


def test_zone_page_red_and_command_line_range_refuse_with_the_same_sentence() -> None:
    """抓分區頁標紅或命令列 ``--range`` 沒接上（只有起跑設計那一步擋）：同一份範圍，立面圖那一區紅、句子點名材料；
    命令列開跑前核範圍（``check_ranges``）點名哪一面哪一族；兩邊的最小空腔都照這一場的設定（加倍後原本的最小空腔也拒收）。"""
    from aosr.gradient import fixed_surfaces
    from aosr.gui.gradient_plan import builtin_depth_targets
    from aosr.gui.zone_limits import zone_range_problems
    from tests.engine.test_zone_range import _limits, _ranged, _red

    minimum = SETTINGS.export.min_cavity_m
    given = ZoneRange(porous_m=(0.05, 0.1), cavity_m=(minimum / 2.0, 0.05))
    lines = _red(_limits(_ranged({"y0@s": {"ranges": {WOOL: given.model_dump(mode="json")}}})), "y0", "s")
    label = next(entry.label for entry in library.builtin_entries() if entry.id == WOOL)
    assert [(line["code"], line["text"]) for line in lines] == [("range_invalid", f"{label}的可調範圍：{_sentence('0.05', '0.1')}")]
    surfaces = tuple((wall.wall_name(), PRESET) for wall in Wall.all())
    with pytest.raises(ValueError) as caught:
        fixed_surfaces.check_ranges({FACE: {WOOL: given}}, surfaces, SETTINGS, allowed={})
    assert str(caught.value) == f"面 x0 的 glass_wool_48k：{_sentence('0.05', '0.1')}"
    at_minimum = ZoneRange(porous_m=(0.05, 0.1), cavity_m=(minimum, 0.05))
    assert zone_range_problems({WOOL: at_minimum}, builtin_depth_targets(SETTINGS))[0] == []
    fixed_surfaces.check_ranges({FACE: {WOOL: at_minimum}}, surfaces, SETTINGS, allowed={})
    problems, valid = zone_range_problems({WOOL: at_minimum}, builtin_depth_targets(_doubled()))
    assert [problem.text for problem in problems] == [f"{label}的可調範圍：{_sentence('0.1', '0.2')}"] and valid == {}
    with pytest.raises(ValueError, match="面 x0 的 glass_wool_48k：空腔下限 0.1 cm 要寫 0（可以沒有空腔）或至少 0.2 cm"):
        fixed_surfaces.check_ranges({FACE: {WOOL: at_minimum}}, surfaces, _doubled(), allowed={})


def test_search_start_names_the_material_whose_own_cavity_floor_is_too_thin(tmp_path: Path) -> None:
    """抓材料庫接成家族那一步只講一個籠統的錯：拒收句點名是哪一份材料、叫人到材料頁改（網頁開跑那一步照這一句擋）。
    固定深度（不能調、上下限相等）的空腔一樣點名擋，句子照材料頁那一格講深度。"""
    from aosr.materials.search_library import MaterialsRefused

    minimum = SETTINGS.export.min_cavity_m
    with pytest.raises(MaterialsRefused) as caught:
        with_user_materials(SETTINGS, presets_table(), (_page_entry((minimum / 2.0, 0.05)),))
    assert str(caught.value) == (f"材料「示範薄空腔棉（自編）」（{PAGE}）的{_sentence('0.05', '0.1')}；"
                                 "請到材料頁改那份材料的空腔範圍，搜尋不開")
    assert with_user_materials(SETTINGS, presets_table(), (_page_entry((0.0, 0.05)),)).settings.family[-1].id == PAGE
    with pytest.raises(MaterialsRefused) as fixed:
        with_user_materials(SETTINGS, presets_table(), (_page_entry((minimum / 2.0, minimum / 2.0)),))
    assert str(fixed.value) == (f"材料「示範薄空腔棉（自編）」（{PAGE}）的{_fixed_sentence('0.05', '0.1')}；"
                                "請到材料頁改那份材料的空腔，搜尋不開")
    assert with_user_materials(SETTINGS, presets_table(), (_page_entry((minimum, minimum)),)).settings.family[-1].id == PAGE
    # 最小空腔照這一場的設定：加倍後設定檔那個最小空腔也點名拒收（不是等合併設定時丟一個籠統的檢查錯）。
    with pytest.raises(MaterialsRefused, match=re.escape(f"（{PAGE}）的{_sentence('0.1', '0.2')}；")):
        with_user_materials(_doubled(), presets_table(), (_page_entry((minimum, 0.05)),))


def _custom_settings_file(folder: Path, **replace: str) -> Path:
    """產品設定檔照抄一份，只換掉給的那幾行（鍵 → 那一行新的值；同名的行只換第一行）。"""
    text = config_path("material_gradient.toml").read_text(encoding="utf-8")
    for key, value in replace.items():
        text, count = re.subn(rf"^{key} = .*$", f"{key} = {value}", text, count=1, flags=re.MULTILINE)
        assert count == 1, key
    path = folder / "material_gradient.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_settings_file_family_with_a_too_thin_cavity_floor_is_refused_on_load(tmp_path: Path) -> None:
    """抓設定檔家族（材料頁的內建材料就是它們）沒核：第一個家族（48K）的空腔下限改成最小空腔的一半，讀設定就拒收、點名家族。"""
    minimum = SETTINGS.export.min_cavity_m
    path = _custom_settings_file(tmp_path, cavity_range_m=f"[{minimum / 2.0!r}, 0.250]")
    with pytest.raises(ValidationError, match=f"家族 glass_wool_48k 的{re.escape(_sentence('0.05', '0.1'))}"):
        load_material_gradient_settings(path)
    good = _custom_settings_file(tmp_path, cavity_range_m=f"[{minimum!r}, 0.250]")
    assert load_material_gradient_settings(good).family[0].cavity_range_m == (minimum, 0.25)


def _page_document(low: float, high: float = 0.05) -> dict[str, object]:
    """材料頁送來的那一份；``high`` 等於 ``low`` 就是固定深度（不能調）。"""
    return cast(dict[str, object], _page_entry((low, high)).model_dump(mode="json"))


def test_material_page_refuses_to_save_a_too_thin_cavity_floor_and_writes_nothing(tmp_path: Path) -> None:
    """抓材料頁照存、等開跑才講：可最佳化材料的空腔下限介於 0 和最小空腔之間就不存（檔一個位元組都不寫）、旁邊那一句是公分；
    0 與恰好最小空腔照存。固定深度（不能調、上下限相等）一樣不存，句子照那一格講深度。
    網頁開著的搜尋設定另給了一份（最小空腔加倍）時照那一份核。"""
    from starlette.testclient import TestClient

    from aosr.gui.app import GuiSettings, create_app
    from aosr.gui.material_editor import MaterialEditError, MaterialStore, user_library_path

    minimum = SETTINGS.export.min_cavity_m
    store = MaterialStore(tmp_path)
    with pytest.raises(MaterialEditError) as caught:
        store.save(PAGE, _page_document(minimum / 2.0), create=True)
    assert str(caught.value) == _sentence("0.05", "0.1") and caught.value.status == 400
    with pytest.raises(MaterialEditError) as fixed:
        store.save(PAGE, _page_document(minimum / 2.0, minimum / 2.0), create=True)
    assert str(fixed.value) == _fixed_sentence("0.05", "0.1") and fixed.value.status == 400
    assert not user_library_path(tmp_path).exists()
    assert store.save(PAGE, _page_document(minimum, minimum), create=True).id == PAGE
    assert store.save(PAGE, _page_document(minimum)).id == PAGE
    assert store.save(PAGE, _page_document(0.0)).id == PAGE
    web = tmp_path / "web"
    web.mkdir()
    settings = GuiSettings(engine_commit="a" * 40, data_dir=web,
                           gradient_settings=_custom_settings_file(tmp_path, min_cavity_m=repr(minimum * 2.0)))
    with TestClient(create_app(settings), base_url="http://localhost") as client:
        refused = client.put(f"/api/materials/{PAGE}", json=_page_document(minimum), headers={"If-None-Match": "*"})
        assert (refused.status_code, refused.json()) == (400, {"error": _sentence("0.1", "0.2")})
        assert not user_library_path(web).exists()
        saved = client.put(f"/api/materials/{PAGE}", json=_page_document(minimum * 2.0), headers={"If-None-Match": "*"})
        assert saved.status_code == 200, saved.text
