"""#1082 段 2-3c 替身涵蓋閘門：這一場的家族要在替身訓練過的範圍裡，不然拒收、要重建（不只警告）。

純設定與檔案，不載入 JAX、不建替身；真的建模把涵蓋寫進訓練說明、開跑閘門收下，在 ``test_gui_surrogate_builds.py`` 真跑。
- 只有產品設定檔那幾個家族：不寫涵蓋（訓練說明逐位照舊）、舊替身（沒有訓練說明、或 #1098 那一版）照舊放行。
- 有使用者家族：舊替身拒收；有涵蓋紀錄的照家族代號、構造內容、厚度、空腔、總深度逐格核，哪一格超出就點名哪一格。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from aosr.config.material_gradient import MaterialGradientSettings, load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.gradient import run
from aosr.gradient.surrogate_coverage import (
    COVERAGE_VERSION, NOT_TRAINED_TEXT, TRAINING_VERSION, FixedTarget, SurrogateNotTrainedOn, check_covers, coverage_for,
    fixed_targets, only_product_families,
)
from aosr.gradient.surrogate_files import TRAINING_FILE
from aosr.materials import library
from aosr.materials.search_library import SearchMaterials, fixed_candidates, with_user_materials
from tests.engine._surrogate_coverage import trained_on
from tests.engine.test_material_gradient_library import _ENTRY, _fixed, _optimizable

BASE = load_material_gradient_settings(config_path("material_gradient.toml"))


def _materials(*entries: library.LibraryEntry, settings: MaterialGradientSettings = BASE) -> SearchMaterials:
    return with_user_materials(settings, run.presets_table(), entries)


def _legacy(directory: Path) -> Path:
    """#1098 那一版的訓練說明：沒有 coverage 這一節。"""
    path = directory / TRAINING_FILE
    path.write_text(json.dumps({"version": TRAINING_VERSION, "families": [family.id for family in BASE.family]}) + "\n",
                    encoding="utf-8")
    return path


def _refused(training: Path, materials: SearchMaterials, *words: str) -> None:
    with pytest.raises(SurrogateNotTrainedOn) as caught:
        check_covers(training, materials)
    message = str(caught.value)
    assert message.startswith(f"{NOT_TRAINED_TEXT}："), message
    assert all(word in message for word in words), message


def test_product_families_write_nothing_and_old_surrogates_still_pass(tmp_path: Path) -> None:
    """抓材料庫空的（或只有固定材料）也寫涵蓋（訓練說明位元組變了），或舊替身——旁邊沒有訓練說明、或 #1098 那一版——
    在只有產品家族的一場被拒收（舊替身行為要照舊）。"""
    for materials in (_materials(), _materials(_fixed())):
        assert only_product_families(materials) and coverage_for(materials) is None
        check_covers(tmp_path / "absent" / TRAINING_FILE, materials)
        check_covers(_legacy(tmp_path), materials)


def test_old_surrogate_is_refused_once_the_library_adds_a_family(tmp_path: Path) -> None:
    """抓材料庫有可最佳化材料、替身卻是舊的（沒記訓練過什麼）時照樣放行——2-3c 要修的就是這一格。"""
    materials = _materials(_fixed(), _optimizable())
    assert not only_product_families(materials)
    _refused(tmp_path / "absent" / TRAINING_FILE, materials, "家族 rock_wool_60k 沒訓練過")
    _refused(_legacy(tmp_path), materials, "家族 rock_wool_60k 沒訓練過")


def test_old_surrogate_is_refused_when_a_product_family_is_widened(tmp_path: Path) -> None:
    """抓換了一份家族範圍比產品寬的搜尋設定、舊替身照樣放行（它只用產品那份範圍訓練過）。"""
    wide = BASE.model_copy(update={"family": (BASE.family[0].model_copy(update={"max_total_depth_m": 0.4}),
                                              *BASE.family[1:])})
    _refused(_legacy(tmp_path), _materials(settings=wide), "glass_wool_48k 的總深度上限 0.4 m 比訓練過的 0.3 m 大")


def test_recorded_coverage_accepts_the_same_or_narrower_families(tmp_path: Path) -> None:
    """抓同一份材料、或只把範圍收窄、或材料庫多了一筆固定材料（固定材料不接成家族）也被拒收（會白白逼人重建）。"""
    training = trained_on(tmp_path / "surrogate.npz", _materials(_optimizable()))
    check_covers(training, _materials(_optimizable()))
    check_covers(training, _materials(_fixed(), _optimizable()))
    check_covers(training, _materials())
    narrow = _optimizable(porous_thickness={"adjustable": True, "range_m": [0.05, 0.075], "choices_m": [0.05, 0.075],
                                            "initial_m": 0.05},
                          cavity={"adjustable": True, "range_m": [0.0, 0.05], "initial_m": 0.04}, max_total_depth_m=0.1)
    check_covers(training, _materials(narrow))
    document = json.loads(training.read_text(encoding="utf-8"))
    assert document["version"] == COVERAGE_VERSION
    assert [family["id"] for family in document["coverage"]["families"]] == [*(family.id for family in BASE.family),
                                                                              "rock_wool_60k"]


@pytest.mark.parametrize(("changes", "words"), [
    ({"porous": {"kind": "porous", "thickness_m": 0.05, "flow_resistivity_pa_s_m2": 30000.0, "material_class": "fibrous"}},
     ("rock_wool_60k 的材料內容跟訓練時不同",)),
    ({"porous_thickness": {"adjustable": True, "range_m": [0.05, 0.2], "choices_m": [0.05, 0.1, 0.2], "initial_m": 0.05},
      "max_total_depth_m": 0.3},
     ("rock_wool_60k 的厚度範圍 [0.05, 0.2] 超出訓練過的 [0.05, 0.1]", "總深度上限 0.3 m 比訓練過的 0.15 m 大")),
    ({"cavity": {"adjustable": True, "range_m": [0.0, 0.12], "initial_m": 0.04}},
     ("rock_wool_60k 的空腔範圍 [0.0, 0.12] 超出訓練過的 [0.0, 0.1]",)),
])
def test_recorded_coverage_names_what_went_outside(tmp_path: Path, changes: dict[str, object],
                                                   words: tuple[str, ...]) -> None:
    """抓材料內容（流阻）換了、代號沒變就放行，或厚度、空腔、總深度超出訓練過的範圍沒擋、擋了沒講是哪一格。"""
    training = trained_on(tmp_path / "surrogate.npz", _materials(_optimizable()))
    _refused(training, _materials(_optimizable(**changes)), *words)


def test_a_family_the_surrogate_never_saw_and_a_broken_note_are_refused(tmp_path: Path) -> None:
    """抓訓練說明裡沒有的家族被放行、或訓練說明壞掉時當成舊替身（只有產品家族的一場也要講清楚、拒收）。"""
    training = trained_on(tmp_path / "surrogate.npz", _materials(_optimizable()))
    other = _optimizable(id="rock_wool_100k", label="100K 岩棉（自填）")
    _refused(training, _materials(_optimizable(), other), "家族 rock_wool_100k 沒訓練過")
    training.write_text("{", encoding="utf-8")
    _refused(training, _materials(), "讀不懂")


def test_template_without_cavity_does_not_cover_members_with_cavity(tmp_path: Path) -> None:
    """抓換樣板沒有空氣層（訓練一個空腔都沒抽）、家族裡別的構造卻有空腔（起跑選它就會調空腔）時，
    涵蓋照設定檔記了空腔範圍、開跑閘門放行（#1107 唯讀審查重現的漏洞）。期望值手寫，不由 ``coverage_for`` 生成。"""
    raw = BASE.model_dump(mode="json")
    raw["family"][0].update(template_preset="glass_wool_48k_50mm", initial_cavity_m=None)
    settings = MaterialGradientSettings.model_validate(raw)
    assert settings.family[0].presets.count("glass_wool_48k_50mm_gap50") == 1
    materials = _materials(settings=settings)
    coverage = coverage_for(materials)
    assert coverage is not None and coverage.families[0].id == "glass_wool_48k" and coverage.families[0].cavity_m is None
    training = tmp_path / TRAINING_FILE
    training.write_text(json.dumps({"version": COVERAGE_VERSION, "coverage": coverage.model_dump(mode="json")}),
                        encoding="utf-8")
    _refused(training, materials, "家族 glass_wool_48k 的空腔沒訓練過")


def test_a_member_with_another_porous_layer_is_not_covered_by_its_template(tmp_path: Path) -> None:
    """抓家族成員的多孔層（流阻）跟換樣板不同、替身只用換樣板訓練過，開跑照樣放行；產品設定檔的家族不准中這一條
    （中了就是每一場預設搜尋都被拒）。"""
    training = trained_on(tmp_path / "surrogate.npz", _materials(_optimizable()))
    check_covers(training, _materials(_optimizable()))
    joined = _materials(_optimizable())
    # 設定檔驗證不准一個構造掛兩個家族；這裡繞過驗證直接塞，只考閘門自己擋不擋得住
    mixed = joined.settings.model_copy(update={"family": (joined.settings.family[0].model_copy(
        update={"presets": (*joined.settings.family[0].presets, "glass_wool_24k_50mm")}), *joined.settings.family[1:])})
    _refused(training, joined._replace(settings=mixed),
             "家族 glass_wool_48k 的構造 glass_wool_24k_50mm 的多孔層跟換樣板 glass_wool_48k_50mm_gap50 不同")


# ---- 每區准用清單的固定材料候選（主對話 11-12 加進涵蓋）----

ZONE = "x0@r0_c0"


def _fixed_targets(allowed: dict[str, tuple[str, ...]], *entries: library.FixedEntry) -> tuple[FixedTarget, ...]:
    targets = sorted({target for listed in allowed.values() for target in listed if target.startswith("fixed:")})
    return fixed_targets(allowed, fixed_candidates(targets, run.presets_table(), entries))


def _refused_fixed(training: Path, fixed: tuple[FixedTarget, ...], words: str) -> None:
    with pytest.raises(SurrogateNotTrainedOn, match=words) as caught:
        check_covers(training, _materials(), fixed)
    assert str(caught.value).startswith(f"{NOT_TRAINED_TEXT}：")


def _written(directory: Path, fixed: tuple[FixedTarget, ...]) -> Path:
    coverage = coverage_for(_materials(), fixed)
    assert coverage is not None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TRAINING_FILE
    path.write_text(json.dumps({"version": COVERAGE_VERSION, "coverage": coverage.model_dump(mode="json")}),
                    encoding="utf-8")
    return path


def test_fixed_candidates_need_a_surrogate_trained_with_them(tmp_path: Path) -> None:
    """抓准用清單有固定材料、替身卻沒訓練過它（舊替身：沒有訓練說明或 #1098 那一版）時照樣放行；
    只有產品家族加固定材料時沒寫涵蓋（建了也核不出來）。"""
    fixed = _fixed_targets({ZONE: ("glass_wool_48k", "fixed:panel_fixed")}, _fixed())
    assert fixed[0].surfaces == (ZONE,) and fixed[0].material == {"kind": "constant_impedance", "impedance_pa_s_m": 5000.0}
    _refused_fixed(_legacy(tmp_path), fixed, "固定材料 fixed:panel_fixed 沒訓練過")
    _refused_fixed(tmp_path / "absent" / TRAINING_FILE, fixed, "固定材料 fixed:panel_fixed 沒訓練過")
    coverage = coverage_for(_materials(), fixed)
    assert coverage is not None and coverage.fixed == (fixed[0].coverage(),)
    training = _written(tmp_path / "trained", fixed)
    check_covers(training, _materials(), fixed)
    check_covers(training, _materials())


def test_fixed_candidate_content_and_surfaces_must_match_what_was_trained(tmp_path: Path) -> None:
    """抓固定材料換了內容（阻抗）、或准用到訓練沒放過的面片、或多了一個沒訓練過的候選時照樣放行。"""
    training = _written(tmp_path, _fixed_targets({ZONE: ("glass_wool_48k", "fixed:panel_fixed")}, _fixed()))
    changed = cast(library.FixedEntry, _ENTRY.validate_python(
        {**_fixed().model_dump(mode="json"), "material": {"kind": "constant_impedance", "impedance_pa_s_m": 6000.0}}))
    _refused_fixed(training, _fixed_targets({ZONE: ("fixed:panel_fixed",)}, changed),
                   "固定材料 fixed:panel_fixed 的內容跟訓練時不同")
    _refused_fixed(training, _fixed_targets({ZONE: ("fixed:panel_fixed",), "x0@r1_c1": ("fixed:panel_fixed",)}, _fixed()),
                   r"固定材料 fixed:panel_fixed 准用在 \['x0@r1_c1'\]，訓練只放過 \['x0@r0_c0'\]")
    _refused_fixed(training, _fixed_targets({ZONE: ("fixed:panel_fixed", "fixed:glass_wool_24k_50mm")}, _fixed()),
                   "固定材料 fixed:glass_wool_24k_50mm 沒訓練過")
