"""第一階段網頁投影；數值與提醒答案由原始細算文件獨立重算。"""
from pathlib import Path
from datetime import datetime, timezone, tzinfo
from typing import Self
import fcntl
import json
import os
import re
import time
import uuid

import pytest

from aosr.gui import search_view as view
from aosr.reporting.stage_one_result import wall_assumption_text
from aosr.runtime import configure_calculation_jax
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.scoring.low_frequency_decay_reference import load_reference
from aosr.search import ledger
from aosr.search.run import SearchStatus
from aosr.search.store import SearchStore
from aosr.search.stage_one_shortlist import NOT_FINAL, RANK_AND_QUALITY, DELTA_NOTICE, VARIABLE_ROOM, FIXED_ROOM
from aosr.search.room_proportions import PROPORTION_EXPLANATION
from tests.engine.test_stage_one_shortlist import _finished, _numbers
from tests.engine.test_room_proportions_shortlist import _expected, WARNING, LABELS
from tests.engine._stage_one_post_check_cases import prepared
from tests.engine.test_search_stage_one_post_check import run
from tests.engine._search_run_cases import make_store
from tests.engine._seat_locked_cases import LOCKED


HEADINGS = ("試算編號", "角色", "搜尋名次", "細算名次", "主項單獨名次", "細算狀態", "房間 長×寬×高（公尺）",
    "地板面積 m²", "容積 m³", "主項 dB", "座位穩健性 dB", "合分", "主項單獨排序鍵 dB",
    "Δ主項 dB／Δ座位穩健性 dB／合分差／Δ地板面積 m²／Δ容積 m³", "比例提醒")


def blocks(store: SearchStore) -> dict[str, view.Block]:
    result = view.build_search_view(store.path, server_physics=store.identity.physics_identity,
        server_program=store.identity.program_fingerprint)
    return {block.key: block for block in result.blocks}


def proportion(values: tuple[float, ...]) -> str:
    hits = _expected(values[:3])
    details = []
    for kind, ratio, value, integer in hits:
        label = f"{LABELS[ratio]} {float(value):.6f}"
        if kind == "band":
            details.append(label + " 在建議比例帶 1.1·w/h ≤ l/h ≤ 4.5·w/h − 4 之外（ITU-R BS.1116-3；EBU Tech 3276）")
        elif kind == "below":
            details.append(label + " 達到建議上限 3（建議小於此值）（ITU-R BS.1116-3；EBU Tech 3276）")
        else:
            details.append(label + f" 在整數 {integer} 的 ±5% 內（EBU Tech 3276）")
    return "；".join((WARNING, *details)) if hits else "無"


@pytest.mark.parametrize("variable", [True, False])
def test_shortlist_every_cell_from_raw_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variable: bool) -> None:
    store = _finished(tmp_path, monkeypatch, variable=variable)
    raw = json.loads((store.path / "stage_one_refine.json").read_bytes())
    seats = {seat["trial_number"]: seat for seat in raw["selection"]["candidates"]}
    ranks = {seat["trial_number"]: seat["refine_rank"] for seat in raw["results"]}
    top = [n for n, seat in seats.items() if "top_n" in seat["roles"]]
    order = sorted(top, key=lambda n: ranks[n]) + [n for n in seats if n not in top]
    principal = sorted(top, key=lambda n: (_numbers(store, n)[-1], n))
    base = _numbers(store, order[0])
    expected = []
    for n in order:
        seat, values = seats[n], _numbers(store, n)
        names = {"top_n": f"前 N（搜尋第 {seat['search_rank']} 名）", "big_peak_control": "大峰對照", "far_control": "較遠對照"}
        expected.append((f"試算 {n}", "、".join(names[role] for role in seat["roles"]), str(seat["search_rank"]),
            str(ranks[n]) if n in top else "無", str(principal.index(n) + 1) if n in top else "無", "已計分",
            "×".join(f"{v:.6f}" for v in values[:3]), *(f"{v:.6f}" for v in values[3:]),
            "／".join(f"{values[i] - base[i]:+.6f}" for i in (5, 6, 7, 3, 4)), proportion(values)))
    shown = blocks(store)["shortlist"]
    assert shown.title == f"入選清單（細算後；前 {raw['selection']['top_n']} 席照細算名次，對照列出來、不算入選）"
    assert shown.lines == (NOT_FINAL, RANK_AND_QUALITY, DELTA_NOTICE, VARIABLE_ROOM if variable else FIXED_ROOM,
        PROPORTION_EXPLANATION)
    table = shown.model_dump()["table"]
    assert table == {"headings": HEADINGS, "rows": tuple(expected)}
    result = view.build_search_view(store.path, server_physics="測試", server_program="測試")
    keys = [block.key for block in result.blocks]
    assert keys[keys.index("search-best") + 1:keys.index("crossover")] == ["shortlist"]
    assert result.scope_text == ("第一階段：" + ("房間尺寸與" if variable else "")
        + "喇叭、座位擺位一起搜尋，只算 20～300 Hz 有限元素")


@pytest.mark.parametrize("state,held,label", [("absent", False, "未開始"), ("selected", True, "進行中"),
    ("selected", False, "細算中斷："), ("done", False, "完成"), ("interrupted", False, "中斷"),
    ("failed", False, "失敗"), ("broken", False, "讀不到：")])
def test_refine_states_counts_reasons_and_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   state: str, held: bool, label: str) -> None:
    store = _finished(tmp_path, monkeypatch)
    path = store.path / "stage_one_refine.json"
    raw = json.loads(path.read_bytes())
    if state == "absent":
        path.unlink()
    elif state == "broken":
        path.write_text("{")
    else:
        raw.update(state=state, message="第一階段細算原始原因")
        if state == "selected":
            raw["results"] = []
        path.write_text(json.dumps(raw))
    descriptor = store.open_folder_lock()
    try:
        if held:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        shown = blocks(store)
    finally:
        os.close(descriptor)
    assert next(line for line in shown["stage"].lines if line.startswith("細算：")).startswith("細算：" + label)
    result = view.build_search_view(store.path, server_physics="測試", server_program="測試")
    if not held:
        assert "細算：" + label in result.stage_text
    assert shown["stage"].warning == (state == "broken")
    assert shown["counts"].warning == (state == "broken")
    assert shown["reasons"].warning == (state == "broken")
    assert shown["shortlist"].warning == (state == "broken")
    assert not shown["modal"].warning and not shown["identity"].warning
    if state == "absent":
        assert shown["shortlist"].lines == ("還沒細算",)
        assert "table" not in shown["shortlist"].model_dump()
        assert "細算：未開始" in shown["counts"].lines
        assert "細算訊息原文：未開始" in shown["reasons"].lines
    elif state == "broken":
        assert shown["shortlist"].lines[0].startswith("入選清單讀不回：")
        assert "table" not in shown["shortlist"].model_dump()
        assert any("讀不到" in line for line in shown["counts"].lines)
        assert any("讀不到" in line for line in shown["reasons"].lines)
    else:
        assert "細算訊息原文：第一階段細算原始原因" in shown["reasons"].lines
        assert f"第一階段細算：前 {raw['selection']['top_n']} 席設定；共 {len(raw['selection']['candidates'])} 席（含對照）" in shown["counts"].lines
        for status, text in (("pending", "尚未完成"), ("scored", "已計分"), ("unassessed", "不可估"),
                             ("incomparable", "不能同表"), ("unreadable", "讀不回")):
            count = sum(seat["state"] == status for seat in raw["results"])
            if status == "pending":
                count += len(raw["selection"]["candidates"]) - len(raw["results"])
            assert f"{text} {count} 席" in shown["counts"].lines[-1]
    assert ("細算時間：未開始" if state == "absent" else "第一階段細算的時間還沒上網頁") in shown["timings"].lines
    assert not any("各輪細算" in line or "搜尋＋細算合計" in line for line in shown["timings"].lines)


@pytest.mark.parametrize("broken_wall", [False, True])
def test_assumption_order_and_wall_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broken_wall: bool) -> None:
    store = _finished(tmp_path, monkeypatch)
    wall = wall_assumption_text(store.project)
    if broken_wall:
        def broken(project: object) -> str:
            raise ValueError("牆面假設損壞")
        monkeypatch.setattr(view, "wall_assumption_text", broken)
    shown = blocks(store)
    lines = shown["stage"].lines
    ideal = lines.index("未加入實際喇叭的低頻衰減，排名只適用於本次理想聲源基準；換播放系統後可能不同")
    assert lines[ideal + 1] == ("牆面假設讀不到：牆面假設損壞" if broken_wall else wall)
    assert lines[ideal + 2] == "低頻拖尾不進自動排名，入選後要另外檢查"
    assert shown["stage"].warning == broken_wall
    assert all(not block.warning for key, block in shown.items() if key != "stage")


def test_unknown_process_does_not_claim_refine_is_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _finished(tmp_path, monkeypatch)
    path = store.path / "stage_one_refine.json"
    raw = json.loads(path.read_bytes())
    raw.update(state="selected", results=[])
    path.write_text(json.dumps(raw))
    monkeypatch.setattr(view, "folder_process", lambda folder: view.Process(None, "判不出有沒有行程在跑：考卷原因"))
    shown = blocks(store)
    assert "細算：判不出有沒有行程在跑：考卷原因" in shown["stage"].lines
    assert "細算：進行中" not in "\n".join(shown["stage"].lines)
    assert shown["stage"].warning
    assert not shown["counts"].warning and not shown["shortlist"].warning


@pytest.mark.parametrize("damage", ["content", "unreadable"])
def test_invalid_refine_summary_does_not_hide_other_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    store = _finished(tmp_path, monkeypatch)
    path = store.path / "stage_one_refine.json"
    if damage == "content":
        path.write_text("{}")
    else:
        path.unlink()
        path.mkdir()
    shown = blocks(store)
    assert {key for key, block in shown.items() if block.warning} == {"stage", "counts", "shortlist", "reasons"}
    assert shown["shortlist"].lines[0].startswith("入選清單讀不回：")


def test_unscored_shortlist_cells_have_no_fabricated_numbers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _finished(tmp_path, monkeypatch)
    path = store.path / "stage_one_refine.json"
    raw = json.loads(path.read_bytes())
    raw.update(state="selected", results=[])
    path.write_text(json.dumps(raw))
    table = blocks(store)["shortlist"].model_dump()["table"]
    for row in table["rows"]:
        assert row[3:6] == ("無", "無", "尚未完成；細算結果尚未完成")
        assert row[6:13] == tuple("無" for heading in HEADINGS[6:13])
        assert row[13:] == ("沒有細算第一名，不算差值", "無數值，不判")


def test_second_stage_refine_artifacts_cannot_override_first_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _finished(tmp_path, monkeypatch)
    before = blocks(store)
    raw = json.loads(store.status_path.read_bytes())
    raw["refine"] = {"state": "failed", "message": "不應顯示的第二階段細算原因"}
    store.status_path.write_text(json.dumps(raw))
    (store.path / "refine.jsonl").write_text("{")
    after = blocks(store)
    for key in ("stage", "counts", "timings", "reasons", "shortlist", "modal"):
        assert after[key] == before[key]


def test_counts_include_each_nonzero_seat_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _finished(tmp_path, monkeypatch)
    path = store.path / "stage_one_refine.json"
    original = path.read_bytes()
    labels = {"pending": "尚未完成", "scored": "已計分", "unassessed": "不可估", "incomparable": "不能同表", "unreadable": "讀不回"}
    for state in labels:
        raw = json.loads(original)
        if state != "scored":
            raw["results"][0].update(state=state, score=None, refine_rank=None, reason="考卷席次狀態")
        path.write_text(json.dumps(raw))
        expected = "；".join(f"{label} {sum(seat['state'] == code for seat in raw['results'])} 席" for code, label in labels.items())
        assert blocks(store)["counts"].lines[-1] == expected


@pytest.mark.parametrize("state", ["absent", "running", "done", "broken", "one_file"])
def test_post_check_summary_from_diagnosis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    # 真網頁 main 先做同一設定，避免在行程內直接組頁時誤認為舊模態程式。
    configure_calculation_jax()
    store = prepared(tmp_path, monkeypatch)
    if state != "absent":
        run(store, tmp_path / "child")
        path = store.path / "stage-one-check" / "summary.json"
        raw = json.loads(path.read_bytes())
        if state == "broken":
            path.write_text("{")
        elif state == "running":
            raw.update(state="running", message="正在逐席檢查")
            raw["seats"][0].update(state="running", diagnosis_file=None)
            path.write_text(json.dumps(raw))
        elif state == "one_file":
            (path.parent / raw["seats"][0]["diagnosis_file"]).write_text("{")
    shown = blocks(store)
    modal = shown["modal"]
    assert modal.title == "入選後檢查（低頻模態診斷）"
    assert modal.warning == (state in ("broken", "one_file"))
    assert all(not block.warning for key, block in shown.items() if key != "modal")
    assert not any(word in "\n".join(modal.lines) for word in ("原方案：", "搜尋第一名：", "細算第一名：", "舊版模態程式"))
    if state == "absent":
        assert modal.lines == ("入選後檢查還沒做",)
    elif state == "broken":
        assert "摘要讀不回" in modal.lines[0]
    else:
        assert modal.lines[0].startswith("入選後檢查狀態：" + ("開始過、沒有收尾紀錄" if state == "running" else "已收尾"))
        reference = load_reference(load_quality_targets(config_path("quality_targets.toml")).purpose(store.settings.purpose))
        for seat in raw["seats"]:
            line = next(line for line in modal.lines if line.startswith(f"試算 {seat['trial_number']}／"))
            if seat["state"] == "running":
                assert "開始過、沒有收尾紀錄" in line and "未評估" in line
                continue
            if state == "one_file" and seat == raw["seats"][0]:
                assert "診斷文件讀不回" in line
                continue
            document = json.loads((path.parent / seat["diagnosis_file"]).read_bytes())
            modes = document["room_layer"]["modes"]
            exceeding = [mode for mode in modes if (target := reference.target_at(mode["frequency_hz"]).t60_s) is not None
                and mode["t60_s"] is not None and mode["t60_s"] > target]
            scheme = json.loads((store.path / seat["scheme_file"]).read_bytes())
            room = scheme["scene"]["room_m"]
            assert "×".join(f"{room[key]:.6f}" for key in ("Lx", "Ly", "Lz")) in line
            assert f"房間層：共振 {len(modes)} 個；超過參考線 {len(exceeding)} 個" in line
        assert modal.lines[-1] == "每個超線共振的頻率與 T60 清單還沒上網頁"


def second_stage(tmp_path: Path, locked: bool) -> SearchStore:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=2 if locked else 1))
        store, _ = make_store(tmp_path, layout_changes=LOCKED if locked else None)
    ledger.create_for(store)
    store.status_path.write_text(SearchStatus(state="budget_exhausted", message="考卷停止原因").model_dump_json())
    for path in (store.status_path, store.ledger_path):
        os.utime(path, (1700000000, 1700000000))
    return store


class UtcClock(datetime):
    """顯示時區固定在輸入端；不遮 JSON 裡的任何時間文字。"""

    def astimezone(self, tz: tzinfo | None = None) -> Self:
        return super().astimezone(timezone.utc if tz is None else tz)


def masked_json(store: SearchStore, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(time, "time", lambda: 1700000060.)
    monkeypatch.setattr(view, "datetime", UtcClock)
    monkeypatch.setattr(view, "folder_process", lambda path: view.Process(False, "沒有計算行程拿著這個資料夾"))
    output = view.build_search_view(store.path, server_physics="現在物理", server_program="現在程式").model_dump_json()
    return re.sub(r"-?\d+\.\d+", "#", output)


# dc475264 的 git archive 副本，兩種自建第二階段資料各獨立跑兩次相同才釘。
# 副本放在 PYTHONPATH 最前端，呼叫 second_stage 建一般與座位鎖定資料夾；
# 兩個新行程讀同一份資料，呼叫 masked_json，兩份 JSON 用 cmp 核相同後才錄入下表。
# 代號、檔案時間、顯示時鐘與時區在輸入端固定；輸出只遮小數，其餘逐字比。
SECOND_STAGE = {'False': '{"search_id":"00000000000000000000000000000001","name":"reference-room-original","fetched_text":"22:14:20","stage_text":"搜尋：因預算停止；第 1 輪；細算：未開始","blocks":[{"key":"stage","title":"目前階段","lines":["沒有計算行程拿著這個資料夾","搜尋：因預算停止；第 1 輪","細算：未開始","外圈結論：未判定","品質合格：本頁未判定；停止或細算完成不代表品質合格，也不是最終推薦","未加入實際喇叭的低頻衰減，排名只適用於本次理想聲源基準；換播放系統後可能不同"],"warning":false},{"key":"counts","title":"候選處理數量","lines":["問過 0 題（到狀態最後存檔為止）","算完 0 個；不合法 0 個","不合法各原因數：沒有","被淘汰或排除 0 個；各區數：沒有","細算帳：還沒有（細算還沒開始）"],"warning":false},{"key":"timings","title":"已用時間","lines":["到最後一次存檔為止；不是目前行程的即時計時","這個搜尋資料夾沒有時間紀錄（加上時間紀錄之前開的搜尋）","各輪搜尋花的時間：沒有紀錄（還沒跑，或是加上時間紀錄之前的程式跑的）","各輪細算花的時間：沒有紀錄（還沒跑，或是加上時間紀錄之前的程式跑的）","這個搜尋資料夾沒有時間紀錄（加上時間紀錄之前開的搜尋）"],"warning":false},{"key":"updated","title":"最後更新時間","lines":["最後更新：2023-11-14 22:13:20 UTC；距今 1 分 0 秒"],"warning":false},{"key":"search-best","title":"搜尋最佳","lines":["本次預算內最佳：沒有取得篩選分數的候選"],"warning":false},{"key":"refine-best","title":"細算最佳","lines":["尚無可排名的細算第一名"],"warning":false},{"key":"crossover","title":"300 Hz 交接敏感度（不改名次）","lines":["只看換接法時名次會不會變，正式接法、分數與名次不變","交接影響尚未驗證","未開始（搜尋正常收尾後才補）","距離近不等於聲音接近；這裡只看算法選擇對排名的影響，不是誤差界，也不是聽感比較"],"warning":false},{"key":"stability","title":"擺位穩定性（±2 公分，不改名次）","lines":["這場搜尋沒有擺位穩定性結果（這一段在 #699 合併後才有）"],"warning":false},{"key":"reasons","title":"停止或失敗原因","lines":["搜尋訊息原文：考卷停止原因","細算訊息原文：細算未開始：還沒有任何候選用驗證軸細算"],"warning":false},{"key":"modal","title":"低頻模態診斷（不計分）","lines":["不計分、不改名次；搜尋結果與外圈結論不受診斷影響","原方案：未開始（搜尋正常收尾後才補）","搜尋第一名：未開始（搜尋正常收尾後才補）","細算第一名：未開始（搜尋正常收尾後才補）","其他座位與完整排序放進結果清單後到網頁結果頁看"],"warning":false},{"key":"identity","title":"身分","lines":["伺服器現在的物理身分：現在物理","伺服器程式指紋：現在程式","資料夾物理身分：phys-v1:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","資料夾程式指紋：calc-v1:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","這場搜尋用的是較舊的程式（指紋與伺服器不同；這是備查資訊）","這場搜尋的物理身分與伺服器不同（備查資訊）"],"warning":false}],"best_default":"search","best_versions":{"search":{"which":"search","candidate":"","result_file":"","version":"讀不到：尚無搜尋第一名","error":"讀不到：尚無搜尋第一名"},"refine":{"which":"refine","candidate":"","result_file":"","version":"讀不到：尚無細算第一名","error":"讀不到：尚無細算第一名"}}}', 'True': '{"search_id":"00000000000000000000000000000002","name":"reference-room-original","fetched_text":"22:14:20","stage_text":"搜尋：因預算停止；第 1 輪；細算：未開始","blocks":[{"key":"stage","title":"目前階段","lines":["沒有計算行程拿著這個資料夾","搜尋：因預算停止；第 1 輪","細算：未開始","外圈結論：未判定","品質合格：本頁未判定；停止或細算完成不代表品質合格，也不是最終推薦","未加入實際喇叭的低頻衰減，排名只適用於本次理想聲源基準；換播放系統後可能不同","座位：鎖定在原方案主位，只搜離前牆與間距，聆聽距離由座位推出"],"warning":false},{"key":"counts","title":"候選處理數量","lines":["問過 0 題（到狀態最後存檔為止）","算完 0 個；不合法 0 個","不合法各原因數：沒有","被淘汰或排除 0 個；各區數：沒有","細算帳：還沒有（細算還沒開始）"],"warning":false},{"key":"timings","title":"已用時間","lines":["到最後一次存檔為止；不是目前行程的即時計時","這個搜尋資料夾沒有時間紀錄（加上時間紀錄之前開的搜尋）","各輪搜尋花的時間：沒有紀錄（還沒跑，或是加上時間紀錄之前的程式跑的）","各輪細算花的時間：沒有紀錄（還沒跑，或是加上時間紀錄之前的程式跑的）","這個搜尋資料夾沒有時間紀錄（加上時間紀錄之前開的搜尋）"],"warning":false},{"key":"updated","title":"最後更新時間","lines":["最後更新：2023-11-14 22:13:20 UTC；距今 1 分 0 秒"],"warning":false},{"key":"search-best","title":"搜尋最佳","lines":["本次預算內最佳：沒有取得篩選分數的候選"],"warning":false},{"key":"refine-best","title":"細算最佳","lines":["尚無可排名的細算第一名"],"warning":false},{"key":"crossover","title":"300 Hz 交接敏感度（不改名次）","lines":["只看換接法時名次會不會變，正式接法、分數與名次不變","交接影響尚未驗證","未開始（搜尋正常收尾後才補）","距離近不等於聲音接近；這裡只看算法選擇對排名的影響，不是誤差界，也不是聽感比較"],"warning":false},{"key":"stability","title":"擺位穩定性（±2 公分，不改名次）","lines":["這場搜尋沒有擺位穩定性結果（這一段在 #699 合併後才有）"],"warning":false},{"key":"reasons","title":"停止或失敗原因","lines":["搜尋訊息原文：考卷停止原因","細算訊息原文：細算未開始：還沒有任何候選用驗證軸細算"],"warning":false},{"key":"modal","title":"低頻模態診斷（不計分）","lines":["不計分、不改名次；搜尋結果與外圈結論不受診斷影響","原方案：未開始（搜尋正常收尾後才補）","搜尋第一名：未開始（搜尋正常收尾後才補）","細算第一名：未開始（搜尋正常收尾後才補）","其他座位與完整排序放進結果清單後到網頁結果頁看"],"warning":false},{"key":"identity","title":"身分","lines":["伺服器現在的物理身分：現在物理","伺服器程式指紋：現在程式","資料夾物理身分：phys-v1:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","資料夾程式指紋：calc-v1:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","這場搜尋用的是較舊的程式（指紋與伺服器不同；這是備查資訊）","這場搜尋的物理身分與伺服器不同（備查資訊）"],"warning":false}],"best_default":"search","best_versions":{"search":{"which":"search","candidate":"","result_file":"","version":"讀不到：尚無搜尋第一名","error":"讀不到：尚無搜尋第一名"},"refine":{"which":"refine","candidate":"","result_file":"","version":"讀不到：尚無細算第一名","error":"讀不到：尚無細算第一名"}}}'}


@pytest.mark.parametrize("state", ["skipped", "failed"])
def test_post_check_red_follows_states_not_reason_words(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    # 寫入端照規矩跳過細算結果讀不回的那一席，原因原文寫「讀不回」：不是這一塊的檢查失敗；診斷真的失敗才標紅。
    configure_calculation_jax()
    store = prepared(tmp_path, monkeypatch)
    if state == "skipped":
        store.refine_result_path(8).unlink()
    run(store, tmp_path / "child", state="failed" if state == "failed" else "diagnosed_not_scored")
    raw = json.loads((store.path / "stage-one-check" / "summary.json").read_bytes())
    modal = blocks(store)["modal"]
    assert modal.warning == (state == "failed")
    for seat in raw["seats"]:
        line = next(line for line in modal.lines if line.startswith(f"試算 {seat['trial_number']}／"))
        scheme = json.loads((store.path / seat["scheme_file"]).read_bytes())
        dimensions = "×".join(f"{scheme['scene']['room_m'][key]:.6f}" for key in ("Lx", "Ly", "Lz"))
        if state == "failed":
            assert seat["state"] == "failed"
            assert line.endswith(f"：失敗；{seat['reason_text']}；房間 長×寬×高（公尺）：{dimensions}")
        elif seat == raw["seats"][0]:
            assert seat["state"] == "skipped" and "讀不回" in seat["reason_text"]
            assert line == f"{line.split('：')[0]}：跳過；{seat['reason_text']}；房間層尚無可讀診斷，未評估"
        else:
            assert seat["state"] == "diagnosed_not_scored" and f"房間 長×寬×高（公尺）：{dimensions}；房間層：共振 " in line


def test_updated_follows_stage_one_refine_and_post_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 細算與入選後檢查各跑幾十分鐘、只寫自己的檔：最後更新時間要跟著它們動，不能停在搜尋收尾那一刻。
    configure_calculation_jax()
    store = prepared(tmp_path, monkeypatch)
    run(store, tmp_path / "child")
    refine, summary = store.path / "stage_one_refine.json", store.path / "stage-one-check" / "summary.json"
    base = int(time.time()) - 3600
    for name, offset in (("status.json", 0), ("ledger.jsonl", 0), ("stage_one_refine.json", 600),
                         ("stage-one-check/summary.json", 1200)):
        os.utime(store.path / name, (base + offset, base + offset))
    # 兩個檔依序拿掉（還沒做的檔本來就不存在）：最新時間跟著退回，也不寫讀不到。
    for newest, later in ((base + 1200, summary), (base + 600, refine), (base, None)):
        updated = blocks(store)["updated"]
        assert updated.lines[0].startswith(f"最後更新：{datetime.fromtimestamp(newest).astimezone():%Y-%m-%d %H:%M:%S %Z}")
        assert len(updated.lines) == 1 and not updated.warning
        if later is not None:
            later.unlink()


def test_search_list_reads_stage_one_refine_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _finished(tmp_path, monkeypatch)
    searches = view.list_searches(store.path.parent)["searches"]
    assert isinstance(searches, list)
    item = next(item for item in searches if item["search_id"] == store.path.name)
    assert "細算：完成" in item["stage_text"] and "細算：未開始" not in item["stage_text"]
    assert item["stage_text"] == "；".join(blocks(store)["stage"].lines[:3])


@pytest.mark.parametrize("locked", [False, True])
def test_second_stage_json_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, locked: bool) -> None:
    store = second_stage(tmp_path, locked)
    assert masked_json(store, monkeypatch) == SECOND_STAGE[str(locked)]
