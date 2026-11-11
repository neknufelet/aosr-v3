"""#1098 任意切法的替身訓練／留出材料：組數、固定種子、盒界與總深度、家族與兩種安裝方式都抽得到、留出跟訓練獨立、
方案存檔那一份只在跟原牆不同時才多一組；抽法設定與搜尋家族對不上要拒收。小房間分區案例，不建替身。

阻抗（走 JAX、要雙精度）不在同程式算（#309 冷子行程規矩）：真的命令列在 ``test_gui_surrogate_builds.py`` 的子行程裡算，
有限數與面片序由命令列自己核（``TrainingSets.impedances`` 與建模的 ``impedance_rows``）。"""
from __future__ import annotations

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
from aosr.gradient.surrogate_training import MaterialSet, TrainingSets, original_walls
from aosr.materials.construction_models import MountedOnWall
from aosr.reporting.scheme import Scheme
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES
from tests.engine._material_gradient_zones_case import scheme

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


def test_shrunk_holdout_count_only_drops_random_sets(tmp_path: Path) -> None:
    """考卷縮組數用的設定只少隨機組：固定那幾組（全部原牆、混搭）照舊在。"""
    path = _write(tmp_path / "s.toml", BASE.replace("random_sets = 7", "random_sets = 1", 1))
    holdout = _sets(sampling=load_surrogate_training_settings(path)).holdout_sets()
    assert [item.label for item in holdout][:2] == ["全部原牆", "混搭（原牆與各家族厚度、空腔都取中間值輪流）"]
    assert len(holdout) == 3 and isinstance(holdout[2], MaterialSet)
