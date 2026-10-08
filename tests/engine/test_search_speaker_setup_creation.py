"""建資料夾前拒收輸入；訊息答案來自方案字面與手算幾何。"""
from pathlib import Path

import pytest

from aosr.reporting.scheme import Scheme
from aosr.reporting.validation import SchemeValidationError
from aosr.search import constraints, store as store_module
from tests.engine._search_speaker_setup_cases import flow_project, geometric, store_for
from tests.engine._speaker_setup_cases import document
from tests.engine._furniture_cases import relative_item


@pytest.mark.parametrize("mount", ["stand", "desk", "floor"])
def test_all_mounts_create_with_matching_settings(tmp_path: Path, mount: str) -> None:
    store, _ = store_for(tmp_path, flow_project(mount))
    assert store.path.is_dir()
    assert store.settings.layout.speaker_height_m == store.project.speakers["left"].z
    assert store.project.speaker_setup is not None
    assert store.settings.layout.cabinet.model_dump() == store.project.speaker_setup.cabinet.model_dump()


def test_setting_height_must_match_exactly(tmp_path: Path) -> None:
    with pytest.raises(SchemeValidationError) as caught:
        store_for(tmp_path, flow_project("desk"), layout_changes={"speaker_height_m": 0.9349999999999999})
    assert str(caught.value) == "settings.layout.speaker_height_m：搜尋設定的喇叭高度 0.9349999999999999 m 跟方案的 0.935 m 不同"
    assert not (tmp_path / "searches").exists()


@pytest.mark.parametrize("field,value", [("width_m", 0.22), ("acoustic_center_above_bottom_m", None)])
def test_each_cabinet_field_is_explicit_and_exact(tmp_path: Path, field: str, value: float | None) -> None:
    project = flow_project("stand")
    assert project.speaker_setup is not None
    cabinet = project.speaker_setup.cabinet.model_dump() | {field: value}
    expected = {"width_m": "settings.layout.cabinet.width_m：搜尋設定的箱寬 0.22 m，跟方案的 0.21 m 不同",
                "acoustic_center_above_bottom_m": "settings.layout.cabinet.acoustic_center_above_bottom_m："
                "搜尋設定的聲學中心離箱底沒寫（照箱高一半），跟方案的 0.205 m 不同"}[field]
    with pytest.raises(SchemeValidationError) as caught:
        store_for(tmp_path, project, layout_changes={"cabinet": cabinet})
    assert str(caught.value) == expected
    assert not (tmp_path / "searches").exists()


def test_left_and_right_height_must_match_exactly(tmp_path: Path) -> None:
    project = Scheme.model_validate(document(left_z=1.2, right_z=1.2000000000000002))
    with pytest.raises(SchemeValidationError) as caught:
        store_for(tmp_path, project)
    assert str(caught.value) == "speakers：搜尋只用一個喇叭高度：左 1.2 m、右 1.2000000000000002 m 不同"
    assert not (tmp_path / "searches").exists()


def test_floor_height_validation_runs_without_furniture(tmp_path: Path) -> None:
    project = Scheme.model_validate(document("floor", left_z=0.9, right_z=0.9))
    with pytest.raises(SchemeValidationError) as caught:
        store_for(tmp_path, project)
    assert str(caught.value) == "speakers.left.z：喇叭 left 的高度 0.9 m 跟擺法推出值不同：落地喇叭聲學中心離地 0.8 m；speakers.right.z：喇叭 right 的高度 0.9 m 跟擺法推出值不同：落地喇叭聲學中心離地 0.8 m"
    assert not (tmp_path / "searches").exists()


@pytest.mark.parametrize("mount,forward,center,reason,amount", [
    ("desk", 0.7, 0.2, "箱體超出桌面", 0.9 - 0.74),
    ("floor", 0.7, 0.9, "箱體穿入家具", 1.06 - 0.9),
    ("stand", 0.6, 0.2, "腳架下方有家具", 1.06 - (2.0 - 0.6 - 0.4)),
])
def test_impossible_original_placement_is_input_error(tmp_path: Path, mount: str, forward: float,
        center: float, reason: str, amount: float) -> None:
    project, _ = geometric(mount, forward=forward, center=center)
    with pytest.raises(SchemeValidationError) as caught:
        store_for(tmp_path, project)
    assert all(problem.message == f"原方案{reason}：家具 table，{amount!r} m" for problem in caught.value.problems)
    assert not (tmp_path / "searches").exists()


def test_original_keep_out_is_not_a_baseline_constraint(tmp_path: Path) -> None:
    project, _ = geometric("stand", forward=0.3)
    sofa = relative_item(furniture_id="sofa", width_m=1.8, depth_m=0.75, height_m=0.65,
        placement={"forward_m": -1.0, "left_m": 0.0, "bottom_height_m": 0.0, "yaw_deg": 0})
    project = Scheme.model_validate(project.model_dump() | {"furniture": [sofa]})
    region = {"x": {"low": 2.0, "high": 3.0}, "y": {"low": 0.0, "high": 1.3},
              "z": {"low": 0.0, "high": 2.1}}
    store, registry = store_for(tmp_path, project, layout_changes={"keep_out": [region]})
    assert store.path.is_dir()
    from aosr.search import furniture_prefilter
    assert furniture_prefilter.check(project, contact_rel=2**-40) == ()
    assert {v.reason for v in furniture_prefilter.check_candidate(project, store.settings.layout,
            contact_rel=2**-40)} == {constraints.Reason.FURNITURE_IN_KEEP_OUT}
    from aosr.search.run import start_search
    from tests.engine._search_furniture_cases import FurnitureFlowCompute
    from tests.engine._search_run_cases import ENGINE, RUN_DATE
    status = start_search(store, compute=FurnitureFlowCompute(store), probe=lambda: store.identity,
                          registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)
    assert status.baseline_outcome == "scored"
    assert status.baseline_reason_codes == ()


def test_legacy_project_does_not_read_furniture_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.engine._furniture_cases import reference_document
    def forbidden() -> tuple[object, ...]:
        raise AssertionError("舊案不准多讀登記簿")
    monkeypatch.setattr(store_module, "default_precision_contracts_path", forbidden)
    store_module.check_project_furniture_layout(Scheme.model_validate(reference_document()))


def _renamed(project: Scheme) -> Scheme:
    """喇叭代號不叫 left、right：左右照聲道角色找，不准寫死代號。"""
    document = project.model_dump(mode="json")
    names = {"left": "spk-L", "right": "spk-R"}
    document["speakers"] = {names[key]: value for key, value in document["speakers"].items()}
    for channel in document["channel_group"]["channels"]:
        channel["speaker_id"] = names[channel["speaker_id"]]
    return Scheme.model_validate(document)


def test_speaker_height_follows_channel_roles_not_ids(tmp_path: Path) -> None:
    store, _ = store_for(tmp_path / "same", _renamed(flow_project("stand")))
    assert store.settings.layout.speaker_height_m == store.project.speakers["spk-L"].z
    with pytest.raises(SchemeValidationError) as caught:
        store_for(tmp_path / "different", _renamed(Scheme.model_validate(document(left_z=1.2, right_z=1.3))))
    assert str(caught.value) == "speakers：搜尋只用一個喇叭高度：左 1.2 m、右 1.3 m 不同"
