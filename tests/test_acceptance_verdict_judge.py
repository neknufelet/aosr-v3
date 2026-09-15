"""驗收宣告那支檢查的純判準：內文對 head，各種寫法的固定回歸。

第 2 回合的樣本樹只蓋四種內文；這裡補的是找碴席點名靠手驗的邊界：大小寫、前後空白、
短 id、沒 id、壞 JSON、沒有 pull_request、事件檔不存在。判準物件直接造，不碰真事件檔。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from governance.checks import acceptance_verdict_points_at_head as subject
from governance.exit_codes import ToolBroken
from governance.loader import clean_tree_unset_env_problems
from tests.conftest import GitSandbox

HEAD = "3f2a9c1e8b7d6a5f4e3d2c1b0a9f8e7d6c5b4a39"
OTHER = "9ceb2224d1c0b9a8f7e6d5c4b3a2918f7e6d5c4b"
SETTINGS = subject.Settings("驗收：通過", "驗收：不通過", len(HEAD), "governance/fixture-pr-event.json")


def test_matching_head_is_clean() -> None:
    assert subject.judge(HEAD, f"改了什麼\n\n驗收：通過 {HEAD}\n", SETTINGS) == []


def test_uppercase_and_padding_still_match() -> None:
    body = f"   驗收：通過    {HEAD.upper()}   \n"
    assert subject.judge(HEAD, body, SETTINGS) == []


def test_older_commit_is_red() -> None:
    hits = subject.judge(HEAD, f"驗收：通過 {OTHER}", SETTINGS)
    assert hits and "修完沒再驗" in hits[0]


def test_short_id_is_red() -> None:
    hits = subject.judge(HEAD, f"驗收：通過 {HEAD[:7]}", SETTINGS)
    assert hits and "短 id" in hits[0]


def test_missing_id_is_red() -> None:
    hits = subject.judge(HEAD, "驗收：通過", SETTINGS)
    assert hits and "沒有提交 id" in hits[0]


def test_trailing_words_after_id_are_red() -> None:
    hits = subject.judge(HEAD, f"驗收：通過 {HEAD} 但其實不通過", SETTINGS)
    assert hits and "夾了字" in hits[0]


def test_no_line_is_red() -> None:
    hits = subject.judge(HEAD, "驗收席說沒問題。", SETTINGS)
    assert hits and "沒有「驗收：通過" in hits[0]


def test_fail_line_is_red_even_with_head() -> None:
    hits = subject.judge(HEAD, f"驗收：不通過 {HEAD}", SETTINGS)
    assert hits and "不准合" in hits[0]


def test_two_pass_lines_are_red() -> None:
    hits = subject.judge(HEAD, f"驗收：通過 {OTHER}\n驗收：通過 {HEAD}", SETTINGS)
    assert hits and "不知道信哪一行" in hits[0]


def test_prose_mentioning_the_prefix_mid_line_is_not_a_declaration() -> None:
    body = f"監督補記：昨天的 驗收：通過 {OTHER} 已作廢。\n驗收：通過 {HEAD}\n"
    assert subject.judge(HEAD, body, SETTINGS) == []


def test_event_without_pull_request_is_tool_broken() -> None:
    with pytest.raises(ToolBroken, match="pull_request"):
        subject._head_and_body({"action": "push"}, SETTINGS)


def test_event_with_short_head_is_tool_broken() -> None:
    event: dict[str, object] = {"pull_request": {"head": {"sha": HEAD[:7]}, "body": ""}}
    with pytest.raises(ToolBroken, match="完整提交 id"):
        subject._head_and_body(event, SETTINGS)


def test_event_with_null_body_counts_as_empty() -> None:
    event: dict[str, object] = {"pull_request": {"head": {"sha": HEAD}, "body": None}}
    head, body = subject._head_and_body(event, SETTINGS)
    assert head == HEAD
    assert body == ""


def test_broken_json_event_file_is_tool_broken(tmp_path: Path) -> None:
    path = tmp_path / "event.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ToolBroken, match="JSON"):
        subject._load_event(path, "事件檔")


def test_missing_event_file_is_tool_broken(tmp_path: Path) -> None:
    with pytest.raises(ToolBroken, match="讀不開"):
        subject._load_event(tmp_path / "nope.json", "事件檔")


def test_event_file_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "event.json"
    path.write_text(json.dumps({"pull_request": {"head": {"sha": HEAD}, "body": f"驗收：通過 {HEAD}"}}), encoding="utf-8")
    head, body = subject._head_and_body(subject._load_event(path, "事件檔"), SETTINGS)
    assert subject.judge(head, body, SETTINGS) == []


def test_real_tree_root_with_fixture_event_file_is_tool_broken(git_sandbox: GitSandbox) -> None:
    """防繞過：真的 git 工作樹的根出現樣本事件檔，必須回 2，不准走樣本那條路。"""
    root = git_sandbox.root
    (root / "governance").mkdir()
    (root / "governance" / "fixture-pr-event.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ToolBroken, match="真的 git 工作樹的根"):
        subject._resolve_event(root, SETTINGS)


def test_clean_tree_unset_env_only_accepts_event_variables() -> None:
    """卡上那一格只准列事件變數；列 PATH 之類的會把第 1 回的紅藏掉，載入器要擋。"""
    assert clean_tree_unset_env_problems({"clean_tree_unset_env": ["GITHUB_EVENT_NAME"]}) == []
    assert clean_tree_unset_env_problems({}) == []
    assert clean_tree_unset_env_problems({"clean_tree_unset_env": ["PATH"]})
    assert clean_tree_unset_env_problems({"clean_tree_unset_env": ["GITHUB_EVENT_NAME", "AOSR_RANGE_BASE"]})
    assert clean_tree_unset_env_problems({"clean_tree_unset_env": []})
    assert clean_tree_unset_env_problems({"clean_tree_unset_env": "GITHUB_EVENT_NAME"})
