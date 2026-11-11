"""深度上限跟個案走、跟材料組自己的限制取較嚴（#1082 段 2-4）的引擎考卷（父行程、不求值）。

- 判法一份（``aosr.gradient.depth_fit``）：家族生效上限取兩邊小、一樣大算個案定的；比最淺排法（最薄棉＋空腔下界）還淺不可用；
  固定材料比整塊總深、看不出多深也不可用（限制是硬的）。判「可用」跟換過去之後可行（``swap`` 成功且 ``is_feasible``）是同一件事。
- 個案上限比家族上限大不再拒收；起跑那一面的家族放不下才拒收；不貼的面不跟任何家族比（2-3a 交接）。
- 換樣板候選濾掉放不下的家族；每個家族都放得下時候選逐位跟沒給上限一樣。
- 開跑前兩條：清單裡放不下的固定材料拒收；一面可選的材料組全放不下拒收（不貼不算救、清單只有不貼不判）。
- 報告寫出生效上限與是哪一條擋住的；沒上限的面那一行字不變。
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from pydantic import TypeAdapter

from aosr.config.material_gradient import MaterialGradientSettings, load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Wall
from aosr.gradient import material_gradient as search
from aosr.gradient import run
from aosr.gradient import template_rounds as rounds
from aosr.gradient.depth_fit import (
    family_fit, family_shallowest_m, fixed_fit, fixed_material_depth_m, target_fit,
)
from aosr.gradient.fixed_surfaces import check_depth_targets
from aosr.gradient.report import render_report
from aosr.materials import library
from aosr.materials import search_library as materials
from aosr.materials.construction_models import Assembly, ConstantImpedance, SurfaceMaterial
from tests.engine.test_material_gradient_mount import _project

GAP50 = "glass_wool_48k_50mm_gap50"
USER = "deep_rock_wool"
_ENTRY: TypeAdapter[library.LibraryEntry] = TypeAdapter(library.LibraryEntry)
_MATERIAL: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)


def _base() -> MaterialGradientSettings:
    return load_material_gradient_settings(config_path("material_gradient.toml"))


def _user_entry() -> library.OptimizableEntry:
    """最薄 0.05 m、空腔下界 0.03 m（最淺排法 0.08 m）、自己的總深上限 0.15 m 的使用者可最佳化材料。"""
    return cast(library.OptimizableEntry, _ENTRY.validate_python({
        "kind": "optimizable", "id": USER, "label": "深背岩棉（自填）", "provenance": "model", "source": "自填",
        "porous": {"kind": "porous", "thickness_m": 0.05, "flow_resistivity_pa_s_m2": 20000.0, "material_class": "fibrous"},
        "porous_thickness": {"adjustable": True, "range_m": [0.05, 0.1], "choices_m": [0.05, 0.1], "initial_m": 0.05},
        "cavity": {"adjustable": True, "range_m": [0.03, 0.1], "initial_m": 0.04}, "max_total_depth_m": 0.15}))


def _user() -> materials.SearchMaterials:
    return materials.with_user_materials(_base(), run.presets_table(), (_user_entry(),))


def _kept() -> dict[Wall, SurfaceMaterial]:
    return {wall: ConstantImpedance(impedance_pa_s_m=12348.0) for wall in Wall.all()}


def _preset_material(preset_id: str) -> SurfaceMaterial:
    return _MATERIAL.validate_python(next(p for p in run.presets_table() if p.id == preset_id).material)


# ---- 判法 ----

def test_family_fit_takes_the_stricter_and_names_who_blocks() -> None:
    """抓取大不取小、一樣大時說成材料組擋的、比最淺排法還淺仍說可用。"""
    family = _base().family_by_id("glass_wool_48k")
    assert family.max_total_depth_m == 0.3 and family_shallowest_m(family) == 0.05
    tight = family_fit(family, 0.1)
    assert (tight.usable, tight.effective_m, tight.blocker) == (True, 0.1, "case")
    loose = family_fit(family, 0.5)
    assert (loose.usable, loose.effective_m, loose.blocker) == (True, 0.3, "material")
    tie = family_fit(family, 0.3)
    assert (tie.usable, tie.effective_m, tie.blocker) == (True, 0.3, "case")
    edge = family_fit(family, 0.05)
    assert edge.usable and edge.effective_m == 0.05
    shallow = family_fit(family, 0.049)
    assert (shallow.usable, shallow.reason, shallow.effective_m, shallow.shallowest_m) == (False, "shallower_than_family", None, 0.05)


def test_shallowest_counts_the_cavity_lower_bound() -> None:
    """抓最淺排法只看最薄棉、漏掉空腔下界（使用者材料空腔下界可以不是 0）。"""
    family = _user().settings.family_by_id(USER)
    assert family_shallowest_m(family) == pytest.approx(0.08)
    assert not family_fit(family, 0.07).usable and family_fit(family, 0.08).usable


@pytest.mark.parametrize("cap", [0.03, 0.049, 0.05, 0.07, 0.08, 0.1, 0.2, 0.5])
def test_usable_means_the_swap_is_feasible(cap: float) -> None:
    """判可用 ⇔ 換過去成功且可行：每一個家族（含使用者的）都核，判法跟搜尋本身不能各說各話。"""
    chosen = _user()
    settings, kept = chosen.settings, {wall.wall_name(): m for wall, m in _kept().items()}
    presets: list[str | None] = [GAP50] * 6
    presets[2] = None  # x0 起跑不貼：不跟任何家族比
    designs = run.designs_for(presets, settings, kept=_kept(), max_depths={"x0": cap}, table=chosen.presets)
    context = rounds.TemplateContext(settings, chosen.presets, kept, lambda d: pytest.fail("不應求值"), max_depths={"x0": cap})
    theta = search.baseline_theta(designs)
    for family in settings.family:
        usable = family_fit(family, cap).usable
        try:
            new, values = rounds.swap(designs, theta, "x0", family.id, context)
            swapped = search.is_feasible(values, new)
        except ValueError:
            swapped = False
        assert usable == swapped, (family.id, cap)


def test_fixed_fit_measures_assemblies_and_refuses_what_it_cannot_measure() -> None:
    """抓固定材料厚度加總的浮點尾數被當成超過、看不出多深的被當成放得下。"""
    deep = _preset_material("glass_wool_48k_50mm_gap100")
    assert fixed_material_depth_m(deep) == pytest.approx(0.15)
    assert fixed_fit("fixed:a", deep, 0.15).usable  # 照寫下來的十進位相加：0.05＋0.1 就是 0.15，不帶浮點尾數
    over = fixed_fit("fixed:a", deep, 0.149)
    assert (over.usable, over.reason) == (False, "fixed_too_deep") and over.material_m == pytest.approx(0.15)
    flat = fixed_fit("fixed:b", ConstantImpedance(impedance_pa_s_m=5000.0), 1.0)
    assert (flat.usable, flat.reason) == (False, "fixed_not_comparable")
    assert target_fit("none", 0.01, _base(), {}).usable
    with pytest.raises(ValueError, match="不在這一場的材料清單"):
        target_fit("fixed:absent", 0.1, _base(), {})


# ---- 起跑設計 ----

def test_caps_deeper_than_the_family_take_the_family_limit() -> None:
    """抓個案上限比家族上限大還拒收（#1114 交接）、生效上限沒取小。"""
    settings = _base()
    designs = run.designs_for((GAP50,) * 6, settings, kept=_kept(), max_depths={"x0": 0.5})
    assert designs[2].max_depth_m == 0.5 and designs[2].max_total_depth_m == 0.3
    assert search.depth_limits(designs)[2].max_total_m == 0.3


def test_start_family_too_deep_for_the_cap_is_refused_but_a_bare_face_is_not() -> None:
    """抓起跑家族放不下照樣開、不貼的面因為某個家族放不下就拒收（2-3a 交接：不貼面不比家族）。"""
    chosen = _user()
    with pytest.raises(ValueError, match="x0 的深度上限 0.06 m 放不下家族 deep_rock_wool 最淺排法.*最薄棉"):
        presets: list[str | None] = [GAP50] * 6
        presets[2] = USER
        run.designs_for(presets, chosen.settings, kept=_kept(), max_depths={"x0": 0.06}, table=chosen.presets)
    bare: list[str | None] = [GAP50] * 6
    bare[2] = None
    for cap in (0.06, 0.5):  # 比使用者家族最淺還淺、比它自己的上限還深，都照收
        designs = run.designs_for(bare, chosen.settings, kept=_kept(), max_depths={"x0": cap}, table=chosen.presets)
        assert designs[2].max_depth_m == cap and designs[2].material is None


# ---- 換樣板候選 ----

def test_alternatives_drop_families_that_do_not_fit_and_nothing_else() -> None:
    """抓放不下的家族照樣進快速評分（中途丟錯）、或都放得下時候選順序與個數被動到。"""
    chosen = _user()
    surfaces: tuple[tuple[str, str | None], ...] = tuple((name, GAP50 if name != "x0" else None)
                                                         for name in search.wall_surface_ids())
    plain = rounds.alternatives(surfaces, chosen.settings)
    assert rounds.alternatives(surfaces, chosen.settings, max_depths={"x0": 0.2}) == plain
    filtered = rounds.alternatives(surfaces, chosen.settings, max_depths={"x0": 0.06})
    assert filtered == tuple(item for item in plain if item != ("x0", USER))
    listed = rounds.alternatives(surfaces, chosen.settings, allowed={"x0": (USER, "glass_wool_24k", "none")},
                                 max_depths={"x0": 0.06})
    assert [target for face, target in listed if face == "x0"] == ["glass_wool_24k"]
    assert rounds.next_round_bound(surfaces, chosen.settings, max_depths={"x0": 0.06}) < rounds.next_round_bound(surfaces, chosen.settings)


# ---- 開跑前 ----

def _surfaces(x0: str | None = GAP50) -> tuple[tuple[str, str | None], ...]:
    return tuple((name, x0 if name == "x0" else GAP50) for name in search.wall_surface_ids())


def _fixed(*ids: str) -> dict[str, SurfaceMaterial]:
    return {f"fixed:{item}": _preset_material(item) for item in ids}


def test_start_refuses_listed_fixed_materials_that_do_not_fit() -> None:
    """抓有上限的面列了放不下（太深或看不出多深）的固定材料照樣開——它整塊不變，放不下就是清單寫錯。"""
    deep, carpet = "glass_wool_48k_50mm_gap100", "carpet_on_concrete"
    settings = _base()
    check_depth_targets(_surfaces(), settings, fixed_surfaces={}, max_depths={"x0": 0.1},
                        allowed={"x0": ("glass_wool_48k", f"fixed:{carpet}")}, fixed_materials=_fixed(carpet))
    with pytest.raises(ValueError, match="x0 的准用清單：固定材料 fixed:glass_wool_48k_50mm_gap100 總深 0.15 m 超過深度上限 0.1 m"):
        check_depth_targets(_surfaces(), settings, fixed_surfaces={}, max_depths={"x0": 0.1},
                            allowed={"x0": ("glass_wool_48k", f"fixed:{deep}")}, fixed_materials=_fixed(deep))
    with pytest.raises(ValueError, match="看不出多深"):
        check_depth_targets(_surfaces(), settings, fixed_surfaces={}, max_depths={"x0": 0.1},
                            allowed={"x0": ("glass_wool_48k", "fixed:flat")},
                            fixed_materials={"fixed:flat": ConstantImpedance(impedance_pa_s_m=5000.0)})


def test_start_refuses_a_face_where_no_material_group_fits() -> None:
    """抓一組都放不下還開（交給搜尋卻什麼都貼不上）、把「不貼」當成救、清單只有不貼也拒收。"""
    settings = _base()
    with pytest.raises(ValueError, match="x0 可選的材料組一組都放不進深度上限"):
        check_depth_targets(_surfaces(None), settings, fixed_surfaces={}, max_depths={"x0": 0.01}, allowed={}, fixed_materials={})
    with pytest.raises(ValueError, match="一組都放不進"):
        check_depth_targets(_surfaces(None), settings, fixed_surfaces={}, max_depths={"x0": 0.01},
                            allowed={"x0": ("glass_wool_48k", "none")}, fixed_materials={})
    check_depth_targets(_surfaces(None), settings, fixed_surfaces={}, max_depths={"x0": 0.01},
                        allowed={"x0": ("none",)}, fixed_materials={})
    check_depth_targets(_surfaces(), settings, fixed_surfaces={}, max_depths={}, allowed={}, fixed_materials={})


# ---- 報告 ----

def test_report_names_which_limit_blocks(tmp_path: Path) -> None:
    """抓報告只寫數字、沒寫是哪一條擋住；沒上限的面那一行被動到。"""
    settings = _base()
    project = run.GradientProject.model_validate({
        **_project().model_dump(), "settings_path": str(config_path("material_gradient.toml")),
        "settings_fingerprint": settings.fingerprint, "max_depths": {"x0": 0.08, "xL": 0.5, "ceiling": 0.02}})
    run.write_json_atomic(tmp_path / run.PROJECT_FILE, project)
    text = render_report(tmp_path)
    assert "x0：總深度上限 0.08 m（個案上限擋；家族 glass_wool_48k 自己 0.3 m）" in text
    assert "xL：總深度上限 0.3 m（家族 glass_wool_48k 自己的上限擋；個案上限 0.5 m）" in text
    assert "floor：總深度上限 0.3 m\n" in text
    assert "ceiling：總深度上限 0.02 m（目前不貼，換家族仍套用）；放不下、不換過去：glass_wool_48k（最淺 0.05 m）" in text


def test_assembly_type_is_what_is_measured() -> None:
    assert isinstance(_preset_material("carpet_on_concrete"), Assembly)
    assert fixed_material_depth_m(_preset_material("carpet_on_concrete")) == pytest.approx(0.007)
