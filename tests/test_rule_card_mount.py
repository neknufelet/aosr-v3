"""掛載只採信同一份 workflow 的一層依賴與第⑤條認得出的完整步驟。"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from governance.checks import rule_card_required_fields as cards
from governance.exit_codes import ToolBroken
from governance.loader import RULES_DIR, Card, load_card

REPO = Path(__file__).resolve().parents[1]
MODULE = "governance.checks.stub_check"
CALL = (
    "uv run python -m governance.status.record_step --name sample -- "
    "uv run python -m governance.checks.stub_check --scan-root ."
)


def _tree(tmp_path: Path, jobs: dict[str, object], other: dict[str, object] | None = None) -> Card:
    """從版控樣本複製資料到暫存樹；workflow 與必要名單均在暫存樹內。"""
    source = REPO / "governance/fixtures/rule-card-required-fields-green/case-new-card-legal"
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    workflows = tmp_path / ".github/workflows"
    for workflow in workflows.iterdir():
        workflow.unlink()
    (workflows / "verify.yml").write_text(yaml.safe_dump({"jobs": jobs}), encoding="utf-8")
    if other is not None:
        (workflows / "other.yml").write_text(yaml.safe_dump({"jobs": other}), encoding="utf-8")
    shutil.copy2(REPO / "governance/rules/ci-jobs-cannot-die-quietly.toml", tmp_path / RULES_DIR)
    return load_card(tmp_path / RULES_DIR / "sample-card.toml", tmp_path)


def _calls(body: str = CALL) -> dict[str, object]:
    """組出一個具有真 run 欄位的 job。"""
    return {"steps": [{"run": body}]}


def _mount(card: Card, tmp_path: Path) -> list[str]:
    return cards._mount_problems(card, tmp_path, [p for p in tmp_path.rglob("*") if p.is_file()])


@pytest.mark.parametrize("needs", ["shard", ["shard"], []])
def test_mount_accepts_self_or_direct_needs(tmp_path: Path, needs: object) -> None:
    """拿掉本 job 或直接 needs 的放行路徑，就會拒絕合規掛載。"""
    for location in ("verify", "shard"):
        if needs == [] and location == "shard":
            continue
        jobs: dict[str, object] = {"verify": {"needs": needs}, "shard": {}}
        jobs[location] = {**_calls(), **({"needs": needs} if location == "verify" else {})}
        card = _tree(tmp_path, jobs)
        assert _mount(card, tmp_path) == []


def test_mount_rejects_card_job_in_two_workflows(tmp_path: Path) -> None:
    """退回挑第一份 workflow，會把跨份同名宣告誤放行。"""
    card = _tree(tmp_path, {"verify": _calls()}, {"verify": _calls()})
    bad = _mount(card, tmp_path)
    assert any("多份 workflow" in hit and "verify.yml" in hit and "other.yml" in hit for hit in bad)


@pytest.mark.parametrize("other", [None, {"shard": _calls()}])
def test_mount_needs_must_exist_in_same_workflow(tmp_path: Path, other: dict[str, object] | None) -> None:
    """忽略未知 needs 或借別份 workflow 的 job，即使本 job 有掛載也必須紅。"""
    card = _tree(tmp_path, {"verify": {**_calls(), "needs": "shard"}}, other)
    assert any("needs" in hit and "不存在" in hit and "verify.yml" in hit for hit in _mount(card, tmp_path))


def test_mount_rejects_only_unrelated_job(tmp_path: Path) -> None:
    """把所有同份 job 都納入搜尋，會讓無關 job 冒充掛載。"""
    card = _tree(tmp_path, {"verify": {}, "unrelated": _calls()})
    assert any("沒有呼叫" in hit for hit in _mount(card, tmp_path))


def test_mount_does_not_recurse_needs(tmp_path: Path) -> None:
    """遞迴搜尋 needs 會讓第二層冒充直接掛載。"""
    card = _tree(tmp_path, {"verify": {"needs": "shard"}, "shard": {"needs": "upstream"}, "upstream": _calls()})
    assert any("沒有呼叫" in hit for hit in _mount(card, tmp_path))


@pytest.mark.parametrize("needs", [None, 7, True, {"expression": "shard"}, ["shard", 7], "${{ inputs.job }}"])
def test_mount_rejects_unknown_needs_shape(tmp_path: Path, needs: object) -> None:
    """形狀驗證不能因本 job 已呼叫檢查而短路。"""
    card = _tree(tmp_path, {"verify": {**_calls(), "needs": needs}, "shard": _calls()})
    assert any("needs" in hit for hit in _mount(card, tmp_path))


def test_mount_requires_card_job_in_required_list(tmp_path: Path) -> None:
    """只列分片不能取代卡宣告的必要檢查名。"""
    card = _tree(tmp_path, {"verify": {"needs": "shard"}, "shard": _calls()})
    (tmp_path / cards.REQUIRED_CHECKS_FILE).write_text("shard\n", encoding="utf-8")
    assert any("沒被列在" in hit for hit in _mount(card, tmp_path))


@pytest.mark.parametrize("body", [f"cat <<D\n{CALL}\nD\n", f'echo "\n{CALL}\n"\n', f"{CALL}\nprintf bad\n"])
def test_mount_rejects_entire_step_unreadable_by_receipt_rule(tmp_path: Path, body: str) -> None:
    """拿掉第⑤條或只過濾壞行，就會誤採資料行或不合規整步。"""
    _tree(tmp_path, {"verify": {}})
    assert cards._job_calls_check(_calls(body), MODULE, tmp_path) is False


@pytest.mark.parametrize("body", [CALL, CALL.replace(" -- uv", " \\\n -- uv")])
def test_mount_accepts_valid_receipt_and_continuation(tmp_path: Path, body: str) -> None:
    _tree(tmp_path, {"verify": {}})
    assert cards._job_calls_check(_calls(body), MODULE, tmp_path) is True


def test_mount_finds_valid_step_after_unreadable_step(tmp_path: Path) -> None:
    _tree(tmp_path, {"verify": {}})
    job = {"steps": [{"run": f"cat <<D\n{CALL}\nD"}, {"run": CALL}]}
    assert cards._job_calls_check(job, MODULE, tmp_path) is True


@pytest.mark.parametrize("content", ["not toml", '[settings]\nwrapper_command = "anything"\n'])
def test_mount_broken_receipt_settings_raise_tool_broken(tmp_path: Path, content: str) -> None:
    _tree(tmp_path, {"verify": {}})
    (tmp_path / RULES_DIR / "ci-jobs-cannot-die-quietly.toml").write_text(content, encoding="utf-8")
    with pytest.raises(ToolBroken):
        cards._job_calls_check(_calls(), MODULE, tmp_path)


def test_mount_missing_borrowed_receipt_settings_raise_tool_broken(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _tree(tmp_path, {"verify": {}})
    shutil.copy2(REPO / "governance/rules/rule-card-required-fields.toml", tmp_path / RULES_DIR)
    (tmp_path / RULES_DIR / "ci-jobs-cannot-die-quietly.toml").unlink()
    monkeypatch.setattr(cards, "__file__", str(tmp_path / "runtime" / "governance" / "checks" / "mount.py"))
    with pytest.raises(ToolBroken, match="ci-jobs-cannot-die-quietly"):
        cards._job_calls_check(_calls(), MODULE, tmp_path)
