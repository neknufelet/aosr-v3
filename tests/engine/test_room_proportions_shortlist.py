"""入選清單的比例提醒只追加一欄；從細算結果獨立核對提醒與原有欄位。"""
from dataclasses import fields
from fractions import Fraction
import json
import math
from pathlib import Path

import pytest

from aosr.config.paths import config_path
from aosr.search.stage_one_shortlist import ShortlistRow, stage_one_shortlist, stage_one_shortlist_lines
from tests.engine._stage_one_refine_cases import snapshot
from tests.engine.test_stage_one_shortlist import _finished, _numbers


EXPLANATION = "比例提醒照 ITU-R BS.1116-3 與 EBU Tech 3276 對參考聆聽室的建議，只提醒、不改名次與分數，也不拿來評比例的高下"
WARNING = "這個房間的比例可能產生模態重合，請仔細檢查低頻結果"
LABELS = {"l/w": "長寬比", "l/h": "長高比", "w/h": "寬高比"}


def _expected(room: tuple[float, ...]) -> list[tuple[str, str, Fraction, int | None]]:
    from aosr.config.room_proportions import load_room_proportions
    rules = load_room_proportions(config_path("room_proportions.toml"))
    l, w, h = max(map(Fraction, room[:2])), min(map(Fraction, room[:2])), Fraction(room[2])
    ratios = {"l/w": l / w, "l/h": l / h, "w/h": w / h}
    expected: list[tuple[str, str, Fraction, int | None]] = []
    for entry in rules.entries:
        limits = entry.thresholds
        if entry.kind == "band":
            assert limits.lower_factor is not None and limits.upper_factor is not None and limits.upper_offset is not None
            if l < limits.lower_factor * w or l > limits.upper_factor * w - limits.upper_offset * h:
                expected.append((entry.kind, "l/h", ratios["l/h"], None))
        elif entry.kind == "below":
            assert limits.upper is not None
            for name in ("l/h", "w/h"):
                if ratios[name] >= limits.upper:
                    expected.append((entry.kind, name, ratios[name], None))
        else:
            width = limits.relative_width
            assert width is not None
            for name, value in ratios.items():
                for n in range(1, math.ceil(value / (1 - width)) + 1):
                    if abs(value - n) <= width * n:
                        expected.append((entry.kind, name, value, n))
    return expected


def _check_notice_text(notice: str, expected: list[tuple[str, str, Fraction, int | None]]) -> None:
    if not expected:
        assert notice == "無"
        return
    assert notice.startswith(WARNING + "；")
    for kind, ratio, value, integer in expected:
        assert f"{LABELS[ratio]} {float(value):.6f}" in notice
        if kind == "near_integer":
            assert f"在整數 {integer} 的 ±5% 內（EBU Tech 3276）" in notice
        else:
            assert "ITU-R BS.1116-3" in notice and "EBU Tech 3276" in notice


def test_each_room_notice_and_legacy_columns_are_independent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _finished(tmp_path, monkeypatch)
    raw = json.loads((store.path / "stage_one_refine.json").read_bytes())
    selection = {seat["trial_number"]: seat for seat in raw["selection"]["candidates"]}
    refined = {seat["trial_number"]: seat for seat in raw["results"]}
    top = [n for n, seat in selection.items() if "top_n" in seat["roles"]]
    order = sorted(top, key=lambda n: refined[n]["refine_rank"]) + [n for n in selection if n not in top]
    principal_order = sorted(top, key=lambda n: (_numbers(store, n)[-1], n))
    assert len({_numbers(store, n)[:3] for n in top}) >= raw["selection"]["top_n"]
    before = snapshot(store)
    shortlist = stage_one_shortlist(store)
    assert shortlist is not None
    assert [row.trial_number for row in shortlist.rows] == order
    assert fields(ShortlistRow)[-1].name == "proportion" and fields(ShortlistRow)[-1].default is None
    lines = stage_one_shortlist_lines(store)
    assert EXPLANATION in lines
    header = next(line for line in lines if line.startswith("試算編號｜"))
    assert header == ("試算編號｜角色｜搜尋名次｜細算名次｜主項單獨名次｜細算狀態｜房間 長×寬×高（公尺）｜"
        "地板面積 m²｜容積 m³｜主項 dB｜座位穩健性 dB｜合分｜主項單獨排序鍵 dB｜"
        "Δ主項 dB／Δ座位穩健性 dB／合分差／Δ地板面積 m²／Δ容積 m³｜比例提醒")
    base = _numbers(store, order[0])
    for row in shortlist.rows:
        n = row.trial_number
        values = _numbers(store, n)
        expected = _expected(values[:3])
        assert row.proportion is not None
        assert [(hit.rule.kind, hit.ratio, hit.value, hit.near_integer) for hit in row.proportion.hits] == expected
        delta = tuple(values[index] - base[index] for index in (5, 6, 7, 3, 4))
        assert row.values is not None and row.delta is not None
        assert tuple(getattr(row.values, field.name) for field in fields(row.values)) == values
        assert tuple(getattr(row.delta, field.name) for field in fields(row.delta)) == delta
        rank = str(refined[n]["refine_rank"]) if n in top else "無"
        principal_rank = str(principal_order.index(n) + 1) if n in top else "無"
        expected_columns = [f"試算 {n}", "、".join(row.roles), str(selection[n]["search_rank"]), rank, principal_rank,
            "已計分", "×".join(f"{v:.6f}" for v in values[:3]), *(f"{v:.6f}" for v in values[3:]),
            "／".join(f"{v:+.6f}" for v in delta)]
        line = next(line for line in lines if line.startswith(f"試算 {n}｜"))
        legacy, _, notice = line.rpartition("｜")
        assert legacy == "｜".join(expected_columns)
        _check_notice_text(notice, expected)
    assert snapshot(store) == before
    text = "\n".join(lines)
    assert all(word not in text for word in ("不合格", "不符標準", "違規", "比例較好", "最佳比例", "黃金比例"))


def test_missing_values_are_not_judged_in_all_states(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _finished(tmp_path, monkeypatch, variable=False)
    path = store.path / "stage_one_refine.json"
    original = path.read_bytes()
    for state in ("pending", "unassessed", "incomparable", "unreadable"):
        raw = json.loads(original)
        target = raw["results"][0]
        number = target["trial_number"]
        target.update(state=state, score=None, refine_rank=None, reason="合成原因")
        path.write_text(json.dumps(raw))
        shortlist = stage_one_shortlist(store)
        assert shortlist is not None
        row = next(row for row in shortlist.rows if row.trial_number == number)
        assert row.values is None and row.proportion is None
        line = next(line for line in stage_one_shortlist_lines(store) if line.startswith(f"試算 {number}｜"))
        assert line.endswith("｜無數值，不判")
    path.write_bytes(original)
    raw = json.loads(original)
    number = raw["results"][0]["trial_number"]
    store.refine_result_path(number).write_text("{")
    shortlist = stage_one_shortlist(store)
    assert shortlist is not None
    row = next(row for row in shortlist.rows if row.trial_number == number)
    assert row.state == "unreadable" and row.proportion is None
    assert any(line.startswith(f"試算 {number}｜") and line.endswith("｜無數值，不判")
               for line in stage_one_shortlist_lines(store))
