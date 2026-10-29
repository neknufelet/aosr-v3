"""網頁輸入關、中文欄名與選項：答案照施工單第 2、3、6 條。"""
import json
from pathlib import Path
from typing import cast

import pytest

from aosr.gui.problem_text import field_name, plain_problems
from aosr.gui.compare_view import scheme_differences
from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme
from aosr.reporting.validation import SchemeProblem
from tests.engine import _speaker_setup_cases as cases
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_gui_compare_routes import _client as compare_client, _csv_rows, _files
from tests.engine.test_gui_problem_text import _client
from tests.engine.test_scheme_pipeline import shared_control_result

# 比較頁喇叭差異列的手寫答案（#559 施工單）：A 書架喇叭放腳架（代表模型），B 落地喇叭放地面（實際型號）。
SPEAKER_ROWS = {
    "喇叭類型": ("書架喇叭", "落地喇叭"), "喇叭擺法": ("腳架", "地面"),
    "代表模型": ("代表模型，非實際型號", "實際型號"),
    "箱寬": ("0.21 公尺", "0.25 公尺"), "箱深": ("0.28 公尺", "0.38 公尺"),
    "箱高": ("0.355 公尺", "1.605 公尺"), "聲學中心離前面板": ("0 公尺", "0.01 公尺"),
    "聲學中心離箱底": ("0.205 公尺", "1.25 公尺"),
}


def compare_setups() -> tuple[dict[str, object], dict[str, object]]:
    """比較兩邊的喇叭設定：左邊預設書架喇叭，右邊落地喇叭換成手寫箱體尺寸。"""
    left_setup = cases.setup()
    right_setup = cases.setup("floorstanding", "floor", representative=False)
    cabinet = cast(dict[str, object], right_setup["cabinet"])
    cabinet.update(width_m=0.25, depth_m=0.38, height_m=1.605,
                   acoustic_center_behind_front_m=0.01, acoustic_center_above_bottom_m=1.25)
    return left_setup, right_setup


def changed_result(original: SchemeResult, identifier: str, setup: dict[str, object] | None) -> SchemeResult:
    # 只改不進逐對物理輸入的 metadata（方案描述）；沿用實跑結果與座標，不偽造求解答案。
    document = original.scheme.model_dump(mode="json") | {"scheme_id": identifier, "speaker_setup": setup}
    scheme = Scheme.model_validate(document)
    return original.model_copy(update={"scheme": scheme, "candidate": original.candidate.model_copy(
        update={"candidate_id": identifier, "purpose": scheme.purpose})})


@pytest.mark.parametrize("mount", ["desk", "floor"])
def test_all_gui_calculation_and_save_entries_reject_bad_height(tmp_path: Path, mount: str) -> None:
    document = cases.document(mount, left_z=0.81 if mount == "floor" else 0.936)
    document["scheme_id"] = "bad-height"
    run_id = "b" * 32
    with _client(tmp_path) as client:
        for response in (client.post("/api/plan", json=document),
                         client.put("/api/schemes/bad-height", json=document),
                         client.put("/api/schemes/copy", json=document | {"scheme_id": "copy"},
                                    headers={"If-None-Match": "*"})):
            assert response.status_code == 422
            problem, = response.json()["problems"]
            assert problem["paths"] == ["speakers.left.z"]
            assert problem["fields"] == ["左聲道喇叭 z 座標"]
            assert "跟擺法推出值不同" in problem["text"]
        assert not (tmp_path / "schemes" / "bad-height.json").exists()
        assert not (tmp_path / "schemes" / "copy.json").exists()
        (tmp_path / "schemes" / "bad-height.json").write_text(json.dumps(document))
        (tmp_path / "results" / f"{run_id}.json").write_text(json.dumps({"scheme": document}))
        for response in (client.post("/api/runs", json={"scheme_id": "bad-height"}),
                         client.post(f"/api/results/{run_id}/rerun", json={})):
            assert response.status_code == 422
            problem, = response.json()["problems"]
            assert problem["paths"] == ["speakers.left.z"]
        checked = client.post("/api/validate", json=document)
        assert checked.status_code == 200
        problem, = checked.json()["problems"]
        assert problem["fields"] == ["左聲道喇叭 z 座標"]
        assert not list((tmp_path / "runs").iterdir())


@pytest.mark.parametrize("path,title", [
    ("speaker_setup", "喇叭類型與擺法"), ("speaker_setup.kind", "喇叭類型"),
    ("speaker_setup.mount", "喇叭擺法"), ("speaker_setup.representative", "代表模型"),
    ("speaker_setup.cabinet", "喇叭箱體"), ("speaker_setup.cabinet.width_m", "箱寬"),
    ("speaker_setup.cabinet.depth_m", "箱深"), ("speaker_setup.cabinet.height_m", "箱高"),
    ("speaker_setup.cabinet.acoustic_center_behind_front_m", "聲學中心離前面板"),
    ("speaker_setup.cabinet.acoustic_center_above_bottom_m", "聲學中心離箱底"),
])
def test_speaker_field_names_are_chinese(path: str, title: str) -> None:
    assert field_name(path, {}) == title


@pytest.mark.parametrize("field,value,message", [
    ("kind", "bad", "只能選「書架喇叭」或「落地喇叭」"),
    ("mount", "bad", "只能選「腳架」或「桌面」或「地面」"),
    ("representative", "bad", "要填真假值"),
])
def test_speaker_options_are_chinese(tmp_path: Path, field: str, value: object, message: str) -> None:
    document = cases.document()
    cast(dict[str, object], document["speaker_setup"])[field] = value
    with _client(tmp_path) as client:
        problem, = client.post("/api/validate", json=document).json()["problems"]
    assert problem["message"] == message


def test_whole_scheme_table_error_is_mapped_back_to_speaker_field() -> None:
    problem, = plain_problems((SchemeProblem("scheme", "喇叭放桌面必須剛好一件茶几或書桌，現在有 2 件"),), {})
    assert problem["fields"] == ["喇叭類型與擺法"]
    assert problem["paths"] == ["scheme"]


def test_speaker_difference_survives_simultaneous_purpose_change() -> None:
    # 方案描述差異不依賴求解，手寫答案也不從差異收集器取。
    left = Scheme.model_validate(cases.document())
    other = cases.document() | {"purpose": "other_purpose"}
    cast(dict[str, object], other["speaker_setup"])["representative"] = False
    rows = {row.path: (row.label, row.a_text, row.b_text) for row in scheme_differences(left, Scheme.model_validate(other))}
    assert rows["speaker_setup.representative"] == ("代表模型", "代表模型，非實際型號", "實際型號")
    assert rows["purpose"][0] == "方案用途"
    assert "other_settings" not in rows


def test_compare_route_marks_unset_speaker_side_in_rows_and_csv(
        tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> None:
    """只有一邊設了喇叭：比較端點的差異列與匯出的摘要 CSV 每一格都是「未設定」對另一邊的值（#972 由
    test_gui_speaker_setup_browser 的 compare[one] 搬來；網頁照印差異列這一段由那支瀏覽器考卷的 [both] 守）。"""
    original = shared_control_result(tmp_path_factory, worker_id, "wall-1")
    left = changed_result(original, "speaker-a", None)
    right = changed_result(original, "speaker-b", compare_setups()[1])
    expected = {title: ("未設定", values[1]) for title, values in SPEAKER_ROWS.items()}
    a_id, b_id = "b" * 32, "c" * 32
    with compare_client(tmp_path) as client:
        _files(tmp_path, left, a_id)
        _files(tmp_path, right, b_id)
        response = client.get(f"/api/compare/{a_id}/{b_id}")
        exported = client.get(f"/api/compare/{a_id}/{b_id}/export/summary")
    assert response.status_code == 200, response.text
    rows = {row["label"]: (row["a_text"], row["b_text"]) for row in response.json()["changes"]}
    for title, values in expected.items():
        assert rows[title] == values, title
    assert "其他設定（未逐項列出）" not in rows
    assert exported.status_code == 200, exported.text
    csv_rows = _csv_rows(exported)
    for title, values in expected.items():
        assert any(title in row and list(values) == row[-2:] for row in csv_rows), title


def test_furniture_kind_choices_do_not_borrow_speaker_words(tmp_path: Path) -> None:
    # desk 也是家具種類；家具種類打錯時選項整組不是喇叭選項，不准把 desk 印成喇叭擺法的「桌面」。
    document = cases.document("desk")
    cast(list[dict[str, object]], document["furniture"])[0]["kind"] = "bad"
    with _client(tmp_path) as client:
        problem, = client.post("/api/validate", json=document).json()["problems"]
    assert "只能選" in problem["message"] and "「桌面」" not in problem["message"]


def test_height_problem_names_speaker_with_dotted_id(tmp_path: Path) -> None:
    # 喇叭代號可含點：欄名從原句的代號取，不從路徑 speakers.spk.L.z 用點切（#722 同一類）。
    document = cases.document("floor", left_z=0.81)
    speakers = cast(dict[str, object], document["speakers"])
    speakers["spk.L"] = speakers.pop("left")
    channels = cast(list[dict[str, object]], cast(dict[str, object], document["channel_group"])["channels"])
    channels[0]["speaker_id"] = "spk.L"
    with _client(tmp_path) as client:
        problem, = client.post("/api/validate", json=document).json()["problems"]
    assert problem["fields"] == ["左聲道喇叭 z 座標"]
    # 第八支第三步之一：網頁顯示重排，喇叭名用中文欄名；驗證原句不變、仍留在明細。
    assert problem["text"] == ("左聲道喇叭 z 座標：高度 0.81 m 跟擺法推出值不同："
                               "落地喇叭聲學中心離地 0.8 m")
    assert problem["details"] == [
        "speakers.spk.L.z：喇叭 spk.L 的高度 0.81 m 跟擺法推出值不同：落地喇叭聲學中心離地 0.8 m"]


def test_unknown_speaker_setup_key_prefix_is_chinese(tmp_path: Path) -> None:
    document = cases.document()
    setup = cast(dict[str, object], document["speaker_setup"])
    cast(dict[str, object], setup["cabinet"])["extra"] = 1.0
    with _client(tmp_path) as client:
        problem, = client.post("/api/validate", json=document).json()["problems"]
    assert "speaker_setup" not in problem["text"] and "cabinet" not in problem["text"]
    assert "喇叭類型與擺法" in problem["text"] and "喇叭箱體" in problem["text"]
