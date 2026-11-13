"""exemptions-need-expiry 的 ``[[by_design]]`` 樣本：每一份只紅它要證的那一件，控制組照登記寫的道具一處都不咬。

後設測試（tests/test_fixture_runner.py）只看離開碼。這幾份樣本各自要證的是「哪一件錯會紅」：離開碼對、
紅的那一件不對，樣本就證不了它說的事。#1164 審查實測過兩個這種洞——把「比對函式名」拿掉，
「同檔別的函式」那一份仍因處數對不上回 1；把類別從函式範圍拿掉，控制組多咬兩筆照樣回 1。
所以這裡逐份釘「剛好一筆、而且是那一件」。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from governance.checks import exemptions_need_expiry as subject

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "governance" / "fixtures" / "exemptions-need-expiry"
RIDING = "多搭了別的代碼或另一種抑制標記"
# 樣本樹裡的路徑用零件拼：它們只住在樣本迷你樹裡，寫成整段字面會被「路徑要解析得到」那張卡當死路徑。
PROBE = "/".join(("tests", "test_probe.py"))
GONE = "/".join(("tests", "test_gone.py"))
SAMPLE_CARD = "/".join(("governance", "rules", "sample-guard.toml"))

# 樣本目錄 → 那唯一一筆紅的開頭（哪支檔哪一行，或卡上哪一筆）與它該講的那一件錯
ONE_FAULT = {
    "case-by-design-line-in-unlisted-function": (f"{PROBE}:12 ", "缺 expires"),
    "case-by-design-file-gone": (f"卡 exemptions-need-expiry 頂層的 [[by_design]] 准許 {GONE} ", "掃描面裡沒有這支檔"),
    "case-by-design-count-grew": (f"{PROBE} 的 probe 有 2 處 ", "登記 1 處"),
    "case-by-design-line-carries-expires": (f"{PROBE}:8 ", "卻還帶 expires="),
    "case-by-design-codes-widened": (f"{PROBE}:10 ", RIDING),
    "case-by-design-codes-appended-with-space": (f"{PROBE}:10 ", RIDING),
    "case-by-design-marker-riding-along": (f"{PROBE}:10 ", RIDING),
    "case-by-design-marker-not-at-start": (f"{PROBE}:8 ", RIDING),
}


def _hits(case: str) -> list[str]:
    root = FIXTURES / case
    return subject.check(root, sorted(p for p in root.rglob("*") if p.is_file()))


def test_every_by_design_fixture_is_pinned_here() -> None:
    # 新加一份 [[by_design]] 樣本卻沒在這裡釘「紅的是哪一件」，就又回到只看離開碼。
    on_disk = {p.name for p in FIXTURES.iterdir() if p.is_dir() and p.name.startswith("case-by-design-")}
    assert on_disk == set(ONE_FAULT)


@pytest.mark.parametrize("case", sorted(ONE_FAULT))
def test_fixture_is_red_for_exactly_its_own_fault(case: str) -> None:
    where, fault = ONE_FAULT[case]
    hits = _hits(case)
    assert len(hits) == 1, hits
    assert hits[0].startswith(where) and fault in hits[0], hits


def test_control_designed_prop_bites_nowhere() -> None:
    # 控制組唯一該紅的是那一張卡的放行缺 expires；照登記寫的道具（類別方法、模組層各一處）一處都不准咬。
    hits = _hits("control")
    assert len(hits) == 1, hits
    assert hits[0].startswith(f"{SAMPLE_CARD} "), hits
