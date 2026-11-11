"""#1098 任意切法的替身訓練／留出材料：組數、固定種子、盒界與總深度、家族與兩種安裝方式都抽得到、留出跟訓練獨立、
方案存檔那一份只在跟原牆不同時才多一組；抽法設定與搜尋家族對不上要拒收。小房間分區案例，不建替身。

阻抗（走 JAX、要雙精度）不在同程式算（#309 冷子行程規矩）：真的命令列在 ``test_gui_surrogate_builds.py`` 的子行程裡算，
有限數與面片序由命令列自己核（``TrainingSets.impedances`` 與建模的 ``impedance_rows``）。"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from aosr.config.construction_environment import construction_environment
from aosr.config.frequency_axis import low_frequency_axis_frequencies
from aosr.config.material_gradient import MOUNT_ON_WALL, MOUNT_RIGID, load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.config.surrogate_training import SurrogateTrainingSettings, load_surrogate_training_settings
from aosr.gradient import run
from aosr.gradient.boundary import scheme_patch_keys
from aosr.gradient.surrogate_coverage import coverage_for, fixed_targets
from aosr.gradient.surrogate_training import TrainingSets, original_walls
from aosr.materials import library
from aosr.materials.search_library import SearchMaterials, fixed_candidates, with_user_materials
from aosr.materials.construction_models import MountedOnWall
from aosr.reporting.scheme import Scheme
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES
from tests.engine._material_gradient_zones_case import scheme
from tests.engine.test_material_gradient_library import _fixed, _optimizable

SEARCH = load_material_gradient_settings(config_path("material_gradient.toml"))
SAMPLING = load_surrogate_training_settings(config_path("surrogate_training.toml"))


def _sets(case: Scheme | None = None, sampling: SurrogateTrainingSettings = SAMPLING) -> TrainingSets:
    base = case if case is not None else scheme()
    assert base.scene.low_frequency_axis is not None
    axis = low_frequency_axis_frequencies(base.scene.low_frequency_axis)[0]
    saved = run.scene_wall_materials(base, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    air, ranges = construction_environment(density_kg_m3=base.scene.density_kg_m3,
                                           sound_speed_m_s=base.scene.sound_speed_m_s)
    return TrainingSets(saved, SEARCH, sampling, air=air, ranges=ranges, frequencies_hz=axis)


def _mounted_scheme() -> Scheme:
    """x0 右區存成「貼在原牆上的構造」（像固定區放了一組材料），其餘不變。"""
    base = scheme()
    sets = _sets(base)
    preset = SEARCH.family[0].template_preset
    designs = run.designs_for([preset if index == 2 else None for index in range(len(sets.keys))], SEARCH,
                              kept=sets.originals, mount=MOUNT_ON_WALL)
    theta = np.asarray([0.05, 0.05])
    from aosr.gradient.endpoint import materials_from_surfaces
    mounted = materials_from_surfaces(designs, theta, SEARCH)[sets.keys[2]]
    assert isinstance(mounted, MountedOnWall)
    document = base.model_dump(mode="json")
    zones = document["scene"]["surface_zones_by_wall"]["x0"]
    zones[0]["material"] = mounted.model_dump(mode="json")
    return Scheme.model_validate(document)


def test_counts_order_and_every_set_is_a_full_patch_list() -> None:
    """抓組數跟設定對不上、面片數不對、或第一組不是全部原牆（建模起點平均挑會挑到第 0 組）。"""
    sets = _sets()
    training, holdout = sets.training_sets(), sets.holdout_sets()
    corners = 5 + 2 + 2  # 48K 有空腔：最薄三點＋最厚兩點；24K、96K 沒有空腔：最薄、最厚
    assert len(training) == 1 + corners + 1 + SAMPLING.training.random_sets
    assert len(holdout) == 1 + 1 + SAMPLING.holdout.random_sets
    assert all(len(item.faces) == len(scheme_patch_keys(scheme())) == 7 for item in training + holdout)
    assert training[0].label == holdout[0].label == "全部原牆"
    assert all(face.kind == "original" for face in training[0].faces)


def test_same_inputs_give_the_same_sets_and_holdouts_are_not_training() -> None:
    """抓種子沒固定（兩次不一樣）、或留出跟訓練用同一個種子抽出同樣的組（驗證就不是留出了）。"""
    first, second = _sets(), _sets()
    assert first.training_sets() == second.training_sets() and first.holdout_sets() == second.holdout_sets()
    training = {item.faces for item in first.training_sets()}
    random_holdouts = [item for item in first.holdout_sets() if "隨機" in item.label]
    assert random_holdouts and all(item.faces not in training for item in random_holdouts)


def test_faces_stay_inside_the_search_box_and_depth_cap_and_cover_families_and_mounts() -> None:
    """抓抽到盒界外或厚度加空腔超過家族總深度、某個家族或剛背從來沒抽到（替身沒見過那種阻抗）。"""
    sets = _sets()
    every = sets.training_sets() + sets.holdout_sets()
    families = {face.family for item in every for face in item.faces if face.kind == "mounted"}
    assert families == {family.id for family in SEARCH.family}
    assert {item.mount for item in every} == {MOUNT_ON_WALL, MOUNT_RIGID}
    for item in every:
        for face in item.faces:
            if face.kind != "mounted":
                continue
            assert face.family is not None and face.thickness_m is not None
            box = sets.boxes[face.family]
            assert box.thickness[0] <= face.thickness_m <= box.thickness[1]
            if box.cavity is None:
                assert face.cavity_m is None
            else:
                assert face.cavity_m is not None and box.cavity[0] <= face.cavity_m <= box.room(face.thickness_m)
        sets.materials(item)  # 產品的盒界與總深度檢查（checked_theta、depth_limits）再核一次


def test_saved_set_only_when_it_differs_and_originals_unwrap_mounted_walls() -> None:
    """抓方案存檔跟原牆一樣時還多算一組、或存的是貼在原牆上的構造時「原牆」沒拆回貼之前那面牆。"""
    plain = _sets()
    assert "方案存檔那一份" not in [item.label for item in plain.training_sets()]
    case = _mounted_scheme()
    sets = _sets(case)
    key = sets.keys[2]
    saved = sets.saved[key]
    assert isinstance(saved, MountedOnWall) and sets.originals[key] == saved.wall
    assert original_walls({key: saved}) == {key: saved.wall}
    for collection in (sets.training_sets(), sets.holdout_sets()):
        assert collection[1].label == "方案存檔那一份"
        assert sets.materials(collection[1])[key] == saved
        assert sets.materials(collection[0])[key] == saved.wall


def test_document_lists_every_set_with_its_materials() -> None:
    """抓訓練說明漏組、漏片、或沒寫材料（之後要查替身見過什麼得靠它）。"""
    sets = _sets()
    training, holdout = sets.training_sets(), sets.holdout_sets()
    document = sets.document(training, holdout)
    assert document["patches"] == list(sets.names)
    for key, items in (("training", training), ("holdout", holdout)):
        entries = document[key]
        assert isinstance(entries, list) and len(entries) == len(items)
        assert all(set(entry["faces"]) == set(entry["materials"]) == set(sets.names) for entry in entries)


def test_whole_wall_scheme_uses_the_six_walls() -> None:
    """抓沒有分區的方案（六面整面）面片鍵不是六面牆、或材料匯出的鍵型跟方案對不上。"""
    from aosr.config.frequency_axis import FEM_LANE_FREQUENCIES_HZ
    from aosr.geometry.shoebox import Wall
    from tests.engine.test_gui_gradient_jobs import _scheme as gui_scheme

    case = gui_scheme("whole")
    saved = run.scene_wall_materials(case, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    air, ranges = construction_environment(density_kg_m3=case.scene.density_kg_m3,
                                           sound_speed_m_s=case.scene.sound_speed_m_s)
    sets = TrainingSets(saved, SEARCH, SAMPLING, air=air, ranges=ranges, frequencies_hz=FEM_LANE_FREQUENCIES_HZ)
    assert sets.keys == tuple(Wall.all()) and sets.names == ("floor", "ceiling", "x0", "xL", "y0", "yL")
    for item in sets.training_sets()[:3] + sets.holdout_sets()[:3]:
        assert tuple(sets.materials(item)) == tuple(Wall.all())


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


BASE = config_path("surrogate_training.toml").read_text(encoding="utf-8")


@pytest.mark.parametrize(("old", "new", "message"), [
    ("glass_wool_96k = 0.25", "glass_wool_99k = 0.25", "家族"),
    ("seed = 1098202611199", "seed = 1098202611110", "種子"),
    ("glass_wool_96k = 0.25", "glass_wool_96k = 0.5", "加起來"),
    ("bare_probabilities = [0.0, 0.2, 0.4, 0.6, 0.8]", "bare_probabilities = [1.0]", "bare_probabilities"),
])
def test_settings_refusals(tmp_path: Path, old: str, new: str, message: str) -> None:
    """抓家族對不上搜尋設定、訓練與留出同種子、權重加起來不是 1、原牆機率 1 時照樣收下。"""
    assert old in BASE
    path = _write(tmp_path / "s.toml", BASE.replace(old, new, 1))
    with pytest.raises(ValueError, match=message):
        _sets(sampling=load_surrogate_training_settings(path))


# ---- 材料庫的可最佳化材料（#1082 段 2-3c）----

# 改動前（116426a7）這間分區小房間的訓練說明（``document()`` 的 JSON，``ensure_ascii=False`` 加換行，跟建模命令列寫檔同一個寫法）
# 整份位元組的 SHA-256；改動前的分支實跑存的那一份在 #1107 工作區 evidence 底下 baseline 那一夾（分區那份 document）。
MAIN_DOCUMENT_SHA256 = "8492876d9277ed54a3404b17f30add0f9042a882ede5491891758d87085599a1"
USER = "rock_wool_60k"
ONE_VARIABLE = "panel_one_variable"


def _one_variable() -> library.LibraryEntry:
    """只有厚度能調、沒有空腔的使用者材料（一個連續變數）。"""
    return _optimizable(id=ONE_VARIABLE, label="一個變數的板（自填）",
                        porous_thickness={"adjustable": True, "range_m": [0.05, 0.1], "choices_m": [0.05, 0.1],
                                          "initial_m": 0.05},
                        cavity=None, max_total_depth_m=None)


def _user_sets(*entries: library.LibraryEntry) -> tuple[TrainingSets, SearchMaterials]:
    base = scheme()
    assert base.scene.low_frequency_axis is not None
    axis = low_frequency_axis_frequencies(base.scene.low_frequency_axis)[0]
    saved = run.scene_wall_materials(base, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    air, ranges = construction_environment(density_kg_m3=base.scene.density_kg_m3,
                                           sound_speed_m_s=base.scene.sound_speed_m_s)
    materials = with_user_materials(SEARCH, run.presets_table(), entries)
    users = tuple(entry.id for entry in materials.entries)
    sets = TrainingSets(saved, materials.settings, SAMPLING, air=air, ranges=ranges, frequencies_hz=axis,
                        presets=materials.presets if users else None, user_families=users)
    return sets, materials


def test_without_user_families_the_document_and_weights_are_the_old_ones() -> None:
    """抓材料庫空的（或只有固定材料）時訓練說明跟改動前差一個位元（包括它記的每一組每一片的材料），
    或抽樣權重被重新算過（隨機組就跟著變了）。"""
    for entries in ((), (_fixed(),)):
        sets, materials = _user_sets(*entries)
        training, holdout = sets.training_sets(), sets.holdout_sets()
        text = json.dumps(sets.document(training, holdout, coverage_for(materials)), ensure_ascii=False,
                          allow_nan=False) + "\n"
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == MAIN_DOCUMENT_SHA256
        assert sets.weights == tuple(SAMPLING.sampling.family_weights[family] for family in sets.families)
        assert training == _sets().training_sets() and holdout == _sets().holdout_sets()


def test_user_family_weights_give_every_continuous_variable_the_same_share() -> None:
    """抓使用者家族沒抽到、或權重規則不是「每個連續變數分一樣多」（設定檔 48K 兩個變數 0.5、24K／96K 一個 0.25 的同一條），
    或加起來不是 1。"""
    sets, _ = _user_sets(_optimizable(), _one_variable())
    assert sets.families[-2:] == (USER, ONE_VARIABLE)
    assert (sets.boxes[USER].variables, sets.boxes[ONE_VARIABLE].variables) == (2, 1)
    weights = dict(zip(sets.families, sets.weights, strict=True))
    total = 1.0 + 0.25 * 2 + 0.25 * 1
    expected = {"glass_wool_48k": 0.5 / total, "glass_wool_24k": 0.25 / total, "glass_wool_96k": 0.25 / total,
                USER: 0.5 / total, ONE_VARIABLE: 0.25 / total}
    assert weights == pytest.approx(expected, rel=0.0, abs=1e-15)
    assert math.isclose(sum(sets.weights), 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_every_user_family_is_trained_and_held_out_inside_its_box() -> None:
    """抓使用者家族只進了訓練、留出一片都沒驗到（混搭輪不到、隨機組沒抽到），或抽到它自己盒界與總深度外。"""
    sets, _ = _user_sets(_optimizable(), _one_variable())
    training, holdout = sets.training_sets(), sets.holdout_sets()
    for family in (USER, ONE_VARIABLE):
        middle = [item for item in holdout if item.label.startswith(f"{family} 中間值")]
        assert len(middle) == 1 and set(middle[0].faces) == {sets._middle(family)}
    sets.require_every_family(training, holdout)
    with pytest.raises(ValueError, match="留出材料一片都沒貼過家族"):
        sets.require_every_family(training, holdout[:1])
    for item in training + holdout:
        for face in item.faces:
            if face.kind == "mounted" and face.family in (USER, ONE_VARIABLE):
                assert face.thickness_m is not None
                box = sets.boxes[face.family]
                assert box.thickness[0] <= face.thickness_m <= box.thickness[1]
                assert (face.cavity_m is None) == (box.cavity is None)
        sets.materials(item)


def test_document_records_what_was_trained_only_with_user_families() -> None:
    """抓有使用者家族時訓練說明沒換版、沒記涵蓋（開跑閘門就看不出替身訓練過它），或記的涵蓋跟這一次真的抽的對不上。"""
    sets, materials = _user_sets(_fixed(), _optimizable())
    coverage = coverage_for(materials)
    assert coverage is not None and coverage.materials_sha256 is not None
    training, holdout = sets.training_sets(), sets.holdout_sets()
    document = sets.document(training, holdout, coverage)
    assert document["version"] == "aosr.gradient.surrogate_training.v2" and document["families"] == list(sets.families)
    assert sets.families[-1] == USER
    assert document["coverage"] == coverage.model_dump(mode="json")
    assert document["family_weights"] == dict(zip(sets.families, sets.weights, strict=True))
    other = coverage_for(with_user_materials(SEARCH, run.presets_table(), (_one_variable(),)))
    assert other is not None
    with pytest.raises(ValueError, match="涵蓋那一節的家族"):
        sets.document(training, holdout, other)
    wider = coverage.model_copy(update={"families": (*coverage.families[:-1], coverage.families[-1].model_copy(
        update={"thickness_m": (0.05, 0.2)}))})
    with pytest.raises(ValueError, match="範圍跟這一次真的抽的盒界不同"):
        sets.document(training, holdout, wider)


def test_fixed_candidates_are_trained_and_held_out_on_their_allowed_patches() -> None:
    """抓准用清單的固定材料沒進訓練或留出、放到沒准用它的面片、匯出的材料不是那一份，或訓練說明沒記它（開跑閘門就核不出）。"""
    base_sets, materials = _user_sets()
    zone = base_sets.names[2]
    fixed = fixed_targets({zone: ("glass_wool_48k", "fixed:panel_fixed")},
                          fixed_candidates(["fixed:panel_fixed"], run.presets_table(), (_fixed(),)))
    sets = TrainingSets(base_sets.saved, SEARCH, SAMPLING, air=base_sets.air, ranges=base_sets.ranges,
                        frequencies_hz=base_sets.axis, fixed=fixed)
    training, holdout = sets.training_sets(), sets.holdout_sets()
    placed_training = [item for item in training if any(face.kind == "fixed" for face in item.faces)]
    placed_holdout = [item for item in holdout if any(face.kind == "fixed" for face in item.faces)]
    assert len(placed_training) == 2 and len(placed_holdout) == 1
    for item in placed_training + placed_holdout:
        assert [name for name, face in zip(sets.names, item.faces, strict=True) if face.kind == "fixed"] == [zone]
        assert sets.materials(item)[sets.keys[2]] == _fixed().surface_material()
    coverage = coverage_for(materials, fixed)
    assert coverage is not None
    document = sets.document(training, holdout, coverage)
    assert document["version"] == "aosr.gradient.surrogate_training.v2"
    assert document["coverage"] == coverage.model_dump(mode="json")
    with pytest.raises(ValueError, match="固定材料候選跟這一次真的放的不同"):
        sets.document(training, holdout, coverage_for(materials, ()) or coverage.model_copy(update={"fixed": ()}))
    with pytest.raises(ValueError, match="留出材料一片都沒放過固定材料"):
        sets.require_every_family(training, [item for item in holdout if item not in placed_holdout])
