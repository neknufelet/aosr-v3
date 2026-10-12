"""收尾例外只接受登記的完整形狀；每次只改一個條件。"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import cast

import pytest
import yaml

from governance.checks import ci_jobs_cannot_die_quietly as ci
from governance.exit_codes import ToolBroken

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "case-finalizer-without-if": "收尾 job 層 if 必須整串等於",
    "case-finalizer-if-not-cancelled": "收尾 job 層 if 必須整串等於",
    "case-finalizer-step-without-always": "步驟 if 必須整串等於",
    "case-finalizer-needs-missing-shard": "收尾 needs 集合必須等於",
    "case-finalizer-needs-extra-job": "收尾 needs 集合必須等於",
    "case-finalizer-verdict-step-missing": "必須恰好一步包抄寫員",
    "case-finalizer-verdict-env-not-whole-needs": "的判決 env.",
    "case-verify-defined-twice": "必須恰好出現一次",
    "case-shard-step-not-wrapped": "開頭不是抄寫員",
    "case-shard-continue-on-error": "continue-on-error: True",
    "case-shard-has-strategy": "卻有 strategy:",
    "case-shard-has-needs": "卻有 needs:",
    "case-extra-job-in-finalizer-workflow": "只准有收尾和分片",
    "case-upload-overwrite-true": "不是 false——證據不准被覆蓋",
    "case-finalizer-only-needs": "收尾 job 層 if 必須整串等於",
}


def _tree(tmp_path: Path, family: str, case: str) -> Path:
    source = ROOT / "governance/fixtures" / family / case
    root = tmp_path / "tree"
    shutil.copytree(source, root)
    return root


def _check(root: Path) -> list[str]:
    return ci.check(root, [p for p in root.rglob("*") if p.is_file()])


@pytest.mark.parametrize("case,reason", CASES.items())
def test_finalizer_fixture_names_its_only_fault(tmp_path: Path, case: str, reason: str) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly", case)
    bad = _check(root)
    assert bad and all(reason in hit for hit in bad), bad


def test_registered_finalizer_and_shards_pass(tmp_path: Path) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    assert _check(root) == []


@pytest.mark.parametrize("text", [None, "[broken", "other = true", 'shards = "shard-a"',
                                  "shards = []", 'shards = ["shard-a", "shard-a"]',
                                  'shards = ["verify-sharded"]', 'shards = ["", "shard-b"]',
                                  'shards = ["  ", "shard-b"]', 'shards = ["shard-a", 1]', b"\xff\xfe"])
def test_unreadable_or_malformed_shard_registry_is_two(tmp_path: Path, text: str | bytes | None) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    registry = root / "governance" / "verify-shards.toml"
    if text is None:
        registry.unlink()
    elif isinstance(text, bytes):
        registry.write_bytes(text)
    else:
        registry.write_text(text, encoding="utf-8")
    with pytest.raises(ToolBroken, match="登記簿"):
        _check(root)


@pytest.mark.parametrize("setting", ["registry", "job_if", "step_if", "verdict_env_name", "verdict_env_value",
                                     "verdict_run"])
@pytest.mark.parametrize("value", [None, "", "  ", False])
def test_finalizer_settings_require_whole_strings(tmp_path: Path, setting: str, value: object) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    files = [p for p in root.rglob("*") if p.is_file()]
    settings = ci._card_settings(root, files)
    entry = cast(dict[str, dict[str, object]], settings["finalizer_jobs"])["verify-sharded"]
    if value is None:
        del entry[setting]
    else:
        entry[setting] = value
    with pytest.raises(ToolBroken, match="finalizer_jobs"):
        ci._finalizer_settings_problems(settings, "card.toml")


@pytest.mark.parametrize("value", [None, [], {"": {}}, {"  ": {}}, {"end": "wrong"}])
def test_finalizer_table_shape_is_two(tmp_path: Path, value: object) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    files = [p for p in root.rglob("*") if p.is_file()]
    settings = ci._card_settings(root, files)
    settings["finalizer_jobs"] = value
    with pytest.raises(ToolBroken, match="finalizer_jobs"):
        ci._finalizer_settings_problems(settings, "card.toml")


def _workflow(root: Path) -> tuple[Path, dict[str, dict[str, dict[str, object]]]]:
    path = root / ".github" / "workflows" / "verify-shards.yml"
    return path, cast(dict[str, dict[str, dict[str, object]]], yaml.safe_load(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize("holder,key,value,reason", [
    ("verify-sharded", "strategy", {}, "strategy"),
    ("verify-sharded", "name", "other", "name"),
    ("shard-a", "if", "false", "job 層 if"),
    ("shard-a", "name", "other", "name"),
])
def test_protected_identity_keeps_other_bans(tmp_path: Path, holder: str, key: str,
                                           value: object, reason: str) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    path, data = _workflow(root)
    data["jobs"][holder][key] = value
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert any(reason in hit for hit in _check(root))


@pytest.mark.parametrize("key,value", [("paths", ["src/**"]), ("types", ["opened"])])
def test_protected_triggers_apply_to_shards_and_finalizer(tmp_path: Path, key: str, value: list[str]) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    path, data = _workflow(root)
    data["on"]["pull_request"][key] = value
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    bad = _check(root)
    for job in ("shard-a", "shard-b", "verify-sharded"):
        assert any(job in hit for hit in bad), bad


def test_registry_other_keys_are_outside_this_step(tmp_path: Path) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    registry = root / "governance" / "verify-shards.toml"
    registry.write_text(registry.read_text(encoding="utf-8") + '\nunknown = false\n', encoding="utf-8")
    assert _check(root) == []


def test_registered_registry_is_in_targets(tmp_path: Path) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    files = [p for p in root.rglob("*") if p.is_file()]
    assert root / "governance" / "verify-shards.toml" in ci.targets(root, files)


def test_finalizer_missing_id_is_red(tmp_path: Path) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    path, data = _workflow(root)
    del data["jobs"]["verify-sharded"]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert any("verify-sharded" in hit and "恰好出現一次" in hit for hit in _check(root))


def test_duplicate_yaml_keys_win_over_missing_registry(tmp_path: Path) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    (root / "governance" / "verify-shards.toml").unlink()
    path, _ = _workflow(root)
    path.write_text(path.read_text(encoding="utf-8") + "jobs: {other: {}}\n", encoding="utf-8")
    assert any("重複的鍵" in hit for hit in _check(root))


@pytest.mark.parametrize("change,reason", [("env-missing", "判決 env"), ("two-verdicts", "恰好一步"),
                                           ("step-false", "步驟 if"), ("job-false", "job 層 if")])
def test_finalizer_rejects_incomplete_verdict(tmp_path: Path, change: str, reason: str) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    path, data = _workflow(root)
    finalizer = data["jobs"]["verify-sharded"]
    steps = cast(list[dict[str, object]], finalizer["steps"])
    if change == "env-missing":
        del steps[1]["env"]
    elif change == "two-verdicts":
        steps.append(dict(steps[1]))
    elif change == "step-false":
        steps[0]["if"] = False
    else:
        finalizer["if"] = False
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert any(reason in hit for hit in _check(root))


@pytest.mark.parametrize("value", [True, "true", "TRUE", "${{ true }}"])
@pytest.mark.parametrize("job", ["acceptance", "shard-a", "verify-sharded"])
def test_upload_overwrite_covers_all_protected_roles(tmp_path: Path, job: str, value: object) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    if job == "acceptance":
        path = root / ".github/workflows/acceptance.yml"
        data = cast(dict[str, dict[str, dict[str, object]]], yaml.safe_load(path.read_text(encoding="utf-8")))
    else:
        path, data = _workflow(root)
    step: dict[str, object] = {"uses": "actions/upload-artifact@v4", "with": {"overwrite": value}}
    if job == "verify-sharded":
        step["if"] = "${{ always() }}"
    steps = cast(list[dict[str, object]], data["jobs"][job]["steps"])
    steps.append(step)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert any("overwrite" in hit for hit in _check(root))
    step["with"] = {"overwrite": False}
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert _check(root) == []


def test_upload_overwrite_ignores_action_name_case(tmp_path: Path) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    path, data = _workflow(root)
    steps = cast(list[dict[str, object]], data["jobs"]["shard-a"]["steps"])
    steps.append({"uses": "Actions/Upload-Artifact@v4", "with": {"overwrite": True}})
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert any("overwrite" in hit for hit in _check(root))


@pytest.mark.parametrize("job", ["shard-a", "verify-sharded"])
def test_registered_event_missing_is_two(tmp_path: Path, job: str) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    path = root / "governance/rules/ci-jobs-cannot-die-quietly.toml"
    text = path.read_text(encoding="utf-8")
    text = text.replace(f', "{job}" = ["opened", "synchronize", "reopened"]', "")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ToolBroken, match="沒在卡"):
        _check(root)


@pytest.mark.parametrize("path", [str(Path("/tmp") / "other.toml"), str(Path("..") / "other.toml")])
def test_registry_cannot_leave_scan_root(tmp_path: Path, path: str) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    files = [p for p in root.rglob("*") if p.is_file()]
    settings = ci._card_settings(root, files)
    cast(dict[str, dict[str, object]], settings["finalizer_jobs"])["verify-sharded"]["registry"] = path
    with pytest.raises(ToolBroken, match="相對路徑"):
        ci._finalizer_settings_problems(settings, "card.toml")


def _rename(path: Path, pairs: list[tuple[str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    for old, new in pairs:
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def test_needs_collection_and_settings_are_not_hardcoded(tmp_path: Path) -> None:
    """分片名、job 層與步驟層 if、判決 env 與判決命令全換成別的值，照樣綠；兩種 if 刻意不同，接反就紅。"""
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    card = root / "governance/rules/ci-jobs-cannot-die-quietly.toml"
    names = [("shard-a", "part-x"), ("shard-b", "part-y")]
    env = [("AOSR_FINALIZER_NEEDS", "OTHER_NEEDS"), ("${{ toJSON(needs) }}", "${{ toJson(needs) }}"),
           ("--name verdict", "--name judge")]
    _rename(card, [*names, *env, ('job_if = "${{ always() }}"', 'job_if = "${{ always() && true }}"'),
                   ('step_if = "${{ always() }}"', 'step_if = "${{ always() || true }}"')])
    _rename(root / "governance" / "verify-shards.toml", names)
    path, _ = _workflow(root)
    _rename(path, [*names, *env])
    _, data = _workflow(root)
    finalizer = data["jobs"]["verify-sharded"]
    finalizer["if"] = "${{ always() && true }}"
    for step in cast(list[dict[str, object]], finalizer["steps"]):
        step["if"] = "${{ always() || true }}"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert _check(root) == []


@pytest.mark.parametrize("job_if,needs,passes", [("${{ always() }}", ["shard-b", "shard-a"], True),
                                               (None, ["shard-a", "shard-b"], False),
                                               ("${{ !cancelled() }}", ["shard-a", "shard-b"], False),
                                               ("${{ always() }}", None, False)])
def test_identity_exception_requires_registered_condition_and_needs(job_if: object, needs: object,
                                                                   passes: bool) -> None:
    from governance.checks.ci_required_gate import Finalizer, job_identity_problems

    finalizer = Finalizer({"job_if": "${{ always() }}"}, frozenset({"shard-a", "shard-b"}))
    job = {"if": job_if, "needs": needs}
    bad = job_identity_problems("workflow", "end", job, frozenset({"end"}), finalizer)
    assert (not bad) is passes


def test_finalizer_auxiliary_run_still_requires_wrapper(tmp_path: Path) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    path, data = _workflow(root)
    steps = cast(list[dict[str, object]], data["jobs"]["verify-sharded"]["steps"])
    steps.append({"if": "${{ always() }}", "run": "uv run pytest"})
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert any("開頭不是抄寫員" in hit for hit in _check(root))


def test_shard_step_condition_must_match_registered_whole_value(tmp_path: Path) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    path, data = _workflow(root)
    steps = cast(list[dict[str, object]], data["jobs"]["shard-a"]["steps"])
    steps[1]["if"] = "${{ always() }}"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert any("沒在卡上按 job 與步驟名登記" in hit for hit in _check(root))


def test_finalizer_settings_reject_unknown_field(tmp_path: Path) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    files = [p for p in root.rglob("*") if p.is_file()]
    settings = ci._card_settings(root, files)
    cast(dict[str, dict[str, object]], settings["finalizer_jobs"])["verify-sharded"]["extra"] = "x"
    with pytest.raises(ToolBroken, match="finalizer_jobs"):
        ci._finalizer_settings_problems(settings, "card.toml")


def test_single_shard_needs_may_be_a_string(tmp_path: Path) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    (root / "governance" / "verify-shards.toml").write_text('shards = ["shard-a"]\n', encoding="utf-8")
    path, data = _workflow(root)
    del data["jobs"]["shard-b"]
    data["jobs"]["verify-sharded"]["needs"] = "shard-a"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert _check(root) == []


def test_registry_outside_scanned_files_is_two(tmp_path: Path) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    registry = root / "governance" / "verify-shards.toml"
    files = [p for p in root.rglob("*") if p.is_file() and p != registry]
    with pytest.raises(ToolBroken, match="不在掃描檔案"):
        ci.check(root, files)


def test_comment_cannot_pose_as_verdict_step(tmp_path: Path) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    path, data = _workflow(root)
    steps = cast(list[dict[str, object]], data["jobs"]["verify-sharded"]["steps"])
    steps[1]["run"] = "uv sync --locked  # uv run python -m governance.status.record_step --name verdict"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert any("恰好一步" in hit for hit in _check(root))


@pytest.mark.parametrize("key,value", [("overwrite", "false"), ("overwrite", 0), ("overwrite", None),
                                       ("Overwrite", True), ("OVERWRITE", "true")])
def test_upload_overwrite_rejects_other_values_and_key_case(tmp_path: Path, key: str, value: object) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    path, data = _workflow(root)
    steps = cast(list[dict[str, object]], data["jobs"]["shard-a"]["steps"])
    steps.append({"uses": "actions/upload-artifact@v4", "with": {key: value}})
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert any("不是 false——證據不准被覆蓋" in hit for hit in _check(root))


def test_finalizer_table_missing_from_card_is_two(tmp_path: Path) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    card = root / "governance/rules/ci-jobs-cannot-die-quietly.toml"
    text = card.read_text(encoding="utf-8")
    start = text.index("[settings.finalizer_jobs.verify-sharded]")
    card.write_text(text[:start] + text[text.index("\n[", start + 1) + 1:], encoding="utf-8")
    with pytest.raises(ToolBroken, match="finalizer_jobs"):
        _check(root)


@pytest.mark.parametrize("change,job", [("missing", "shard-b"), ("twice", "shard-a"), ("twice", "verify-sharded")])
def test_protected_shard_and_finalizer_appear_exactly_once(tmp_path: Path, change: str, job: str) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    path, data = _workflow(root)
    if change == "missing":
        del data["jobs"][job]
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
    else:
        second = {"on": data["on"], "jobs": {job: data["jobs"][job]}}
        (path.parent / "second.yml").write_text(yaml.safe_dump(second), encoding="utf-8")
    assert any(repr(job) in hit and "必須恰好出現一次" in hit for hit in _check(root))


WRAPPED = ("uv run python -m governance.status.record_step --name verdict --out-dir governance/receipts/steps"
           " -- uv run python -m governance.checks.green_must_be_real_green --scan-root .")


@pytest.mark.parametrize("run", [f"uv sync --locked || {WRAPPED}", f"{WRAPPED} || uv sync --locked",
                                 f"{WRAPPED}\nuv sync --locked", f"{WRAPPED} & uv sync --locked",
                                 f"{WRAPPED} --unknown=word#tail || uv sync --locked",
                                 WRAPPED + r' \" || uv sync --locked #"', f"{WRAPPED} --help"])
def test_verdict_step_must_equal_registered_run(tmp_path: Path, run: str) -> None:
    root = _tree(tmp_path, "ci-jobs-cannot-die-quietly-green", "control")
    path, data = _workflow(root)
    steps = cast(list[dict[str, object]], data["jobs"]["verify-sharded"]["steps"])
    steps[1]["run"] = run
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert any("run 必須整串等於卡上登記的 verdict_run" in hit for hit in _check(root))
