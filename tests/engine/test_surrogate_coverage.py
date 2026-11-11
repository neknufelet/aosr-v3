"""#1082 段 2-3c 替身涵蓋閘門：這一場的家族要在替身訓練過的範圍裡，不然拒收、要重建（不只警告）。

純設定與檔案，不載入 JAX、不建替身；真的建模把涵蓋寫進訓練說明、開跑閘門收下，在 ``test_gui_surrogate_builds.py`` 真跑。
- 只有產品設定檔那幾個家族：不寫涵蓋（訓練說明逐位照舊）、舊替身（沒有訓練說明、或 #1098 那一版）照舊放行。
- 有使用者家族：舊替身拒收；有涵蓋紀錄的照家族代號、構造內容、厚度、空腔、總深度逐格核，哪一格超出就點名哪一格。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aosr.config.material_gradient import MaterialGradientSettings, load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.gradient import run
from aosr.gradient.surrogate_coverage import (
    COVERAGE_VERSION, NOT_TRAINED_TEXT, TRAINING_VERSION, SurrogateNotTrainedOn, check_covers, coverage_for,
    only_product_families,
)
from aosr.gradient.surrogate_files import TRAINING_FILE
from aosr.materials import library
from aosr.materials.search_library import SearchMaterials, with_user_materials
from tests.engine._surrogate_coverage import trained_on
from tests.engine.test_material_gradient_library import _fixed, _optimizable

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
