"""推之前一行跑完全部規矩檢查（#1071）：清單從卡來、先鏡像再並行、離開碼照約定、收據不在的那一支不算。"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

from governance import run_checks as subject
from governance.exit_codes import CLEAN, TOOL_BROKEN, VIOLATION, ToolBroken
from governance.loader import load_all_cards

REPO = Path(__file__).resolve().parents[1]


def outcome(code: int, receipt: str = "", output: str = "") -> subject.Outcome:
    return subject.Outcome(subject.Step(f"step-{code}-{receipt}", ("x",), receipt), code, 0.1, output)


def test_plan_has_one_step_per_card_then_ruff() -> None:
    """抓清單另寫一份會漂：每張卡一步、名字是卡的 id、叫的是卡上 check 那一支；最後一步是 ruff。"""
    cards = load_all_cards(REPO)
    steps = subject.plan(cards, python="py")
    assert [step.name for step in steps] == [*sorted(card.id for card in cards), "ruff"]
    by_id = {card.id: card for card in cards}
    for step in steps[:-1]:
        assert step.argv == ("py", "-m", by_id[step.name].check_module, "--scan-root", ".")
        assert (REPO / by_id[step.name].check).is_file()
    assert steps[-1].argv == ("py", "-m", "ruff", "check")
    assert {step.name for step in steps if step.receipt} == {card.id for card in cards if card.junit}


def test_plan_without_cards_is_tool_broken() -> None:
    with pytest.raises(ToolBroken, match="一張卡都沒有"):
        subject.plan([])


@pytest.mark.parametrize(("codes", "expected"), [
    ((0, 0), CLEAN),
    ((0, 1, 2), VIOLATION),
    ((0, 2), TOOL_BROKEN),
    ((0, -9), TOOL_BROKEN),
    ((0, 124), TOOL_BROKEN),
])
def test_verdict_follows_exit_code_contract(tmp_path: Path, codes: tuple[int, ...], expected: int) -> None:
    """抓把 2 或訊號當成乾淨：有 1 回 1；沒有 1 但有非 0 回 2；全 0 才回 0。"""
    assert subject.verdict([outcome(code) for code in codes], tmp_path) == expected


def test_missing_receipt_step_is_shown_but_not_counted(tmp_path: Path) -> None:
    """抓本機沒跑全套就永遠紅、或收據在了還不算：收據不在那一支不進離開碼、照印並標明；收據在就照算。"""
    steps = [outcome(CLEAN), outcome(VIOLATION, receipt="junit-receipt", output="HIT: 沒有收據")]
    assert subject.verdict(steps, tmp_path) == CLEAN
    lines = subject.report(steps, tmp_path)
    assert any("本機沒有全套收據 junit-receipt" in line for line in lines)
    assert any("HIT: 沒有收據" in line for line in lines)
    (tmp_path / "junit-receipt").write_text("<testsuites/>", encoding="utf-8")
    assert subject.verdict(steps, tmp_path) == VIOLATION


def test_report_shows_whole_output_only_for_nonzero(tmp_path: Path) -> None:
    lines = subject.report([outcome(CLEAN, output="乾淨的輸出"), outcome(TOOL_BROKEN, output="第一行\n第二行")], tmp_path)
    assert not any("乾淨的輸出" in line for line in lines)
    assert [line.strip() for line in lines[-2:]] == ["第一行", "第二行"]


def test_run_keeps_step_order_while_running_in_parallel(tmp_path: Path) -> None:
    """先完成的不准排前面：結果照清單順序回，同時跑的確實重疊。"""
    steps = [subject.Step(name, ()) for name in ("slow", "fast")]
    running: list[int] = []
    peak = 0
    lock = threading.Lock()

    def call(step: subject.Step, root: Path) -> tuple[int, str]:
        nonlocal peak
        with lock:
            running.append(1)
            peak = max(peak, len(running))
        time.sleep(0.3 if step.name == "slow" else 0.0)
        time.sleep(0.05)
        with lock:
            running.pop()
        return CLEAN, step.name

    outcomes = subject.run(steps, tmp_path, jobs=2, call=call)
    assert [item.output for item in outcomes] == ["slow", "fast"]
    assert peak == 2


def test_main_mirrors_first_then_runs_every_card(capsys: pytest.CaptureFixture[str]) -> None:
    """抓讀收據的檢查比鏡像先跑：鏡像跑完，其他每一支才開始；任何一支回 1，整支回 1。"""
    order: list[str] = []
    early: list[str] = []
    mirrored = threading.Event()
    lock = threading.Lock()

    def call(step: subject.Step, root: Path) -> tuple[int, str]:
        with lock:
            order.append(step.name)
        if step.name == "mirror_receipts":
            time.sleep(0.2)
            mirrored.set()
        elif not mirrored.is_set():
            early.append(step.name)
        return (VIOLATION, "HIT: 植的錯") if step.name == "ruff" else (CLEAN, "")

    assert subject.main(["--jobs", "4"], call=call) == VIOLATION
    cards = load_all_cards(REPO)
    assert order[0] == "mirror_receipts" and early == []
    assert sorted(order[1:]) == sorted([*(card.id for card in cards), "ruff"])
    err = capsys.readouterr().err
    assert "HIT: 植的錯" in err and f"共 {len(cards) + 2} 支" in err


def test_real_subprocess_exit_code_and_output_come_back(tmp_path: Path) -> None:
    step = subject.Step("probe", (sys.executable, "-c", "import sys; print('看得到'); sys.exit(3)"))
    code, output = subject._call(step, tmp_path)
    assert code == 3 and "看得到" in output
