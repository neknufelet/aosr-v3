"""收據生產者的測試：包裝器真的跑小子程序（寫進暫存目錄），合成器全餵假資料、不上網。

斷言刻意不鎖死數量（規矩卡 assertions-not-pinned-to-counts）：離開碼比的是
`governance/exit_codes.py` 登記的那三個名字，片段與收據比的是「有沒有這幾格、值對不對」。
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from governance.exit_codes import CLEAN, TOOL_BROKEN, VIOLATION, ToolBroken
from governance.status import build_receipt, collect, record_step
from governance.status.model import StepResult

PY = sys.executable
FAKE_RUN_ID = 424242
FAKE_ATTEMPT = 2


# ── 包裝器 record_step ────────────────────────────────────────────────────────


def wrap(tmp_path: Path, name: str, code: str) -> tuple[int, dict[str, object]]:
    """用包裝器跑一段小 Python，回（離開碼, 片段）。"""
    out_dir = tmp_path / "steps"
    rc = record_step.main(["--name", name, "--out-dir", str(out_dir), "--", PY, "-c", code])
    fragment = json.loads((out_dir / f"{name}.json").read_text(encoding="utf-8"))
    assert isinstance(fragment, dict)
    return rc, fragment


def test_wrapper_returns_the_child_exit_code_unchanged(tmp_path: Path) -> None:
    """0 就 0、1 就 1、2 就 2——抄寫員不裁判。"""
    for expected in (CLEAN, VIOLATION, TOOL_BROKEN):
        rc, fragment = wrap(tmp_path, f"code{expected}", f"import sys; sys.exit({expected})")
        assert rc == expected
        assert fragment["exit_code"] == expected
        assert fragment["signal"] is None


def test_wrapper_keeps_the_report_line_and_relays_output(
    tmp_path: Path, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    """判決收據那一行存進片段；子程序的 stdout／stderr 原樣轉出去。"""
    code = "import sys; print('\\x1b[31mHIT\\x1b[0m'); print('scan_root=. files=7 hits=0'); print('e', file=sys.stderr)"
    rc, fragment = wrap(tmp_path, "report", code)
    assert rc == CLEAN
    assert fragment["report"] == "scan_root=. files=7 hits=0"
    tail = fragment["stdout_tail"]
    assert isinstance(tail, list)
    assert "HIT" in "".join(str(line) for line in tail)  # 顏色碼去掉了
    assert not any("\x1b" in str(line) for line in tail)
    captured = capfdbinary.readouterr()
    assert b"\x1b[31mHIT" in captured.out  # log 上原樣，顏色碼還在
    assert b"e\n" in captured.err


def test_wrapper_records_null_report_when_the_child_prints_none(tmp_path: Path) -> None:
    """子程序沒印那一行就記 null，不編一行。"""
    _, fragment = wrap(tmp_path, "silent", "pass")
    assert fragment["report"] is None


def test_wrapper_maps_a_signal_death_to_a_nonzero_exit(tmp_path: Path) -> None:
    """被訊號殺掉：離開碼非零、片段記下訊號號碼。"""
    rc, fragment = wrap(tmp_path, "killed", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)")
    assert rc != CLEAN
    assert fragment["signal"] is not None
    assert fragment["exit_code"] == record_step.SIGNAL_EXIT_BASE + int(str(fragment["signal"]))


def test_wrapper_refuses_a_second_fragment_with_the_same_name(tmp_path: Path) -> None:
    """同名第二片是 workflow 寫錯：回 2，不覆蓋。"""
    rc, _ = wrap(tmp_path, "twice", "pass")
    assert rc == CLEAN
    out_dir = tmp_path / "steps"
    rc = record_step.main(["--name", "twice", "--out-dir", str(out_dir), "--", PY, "-c", "pass"])
    assert rc == TOOL_BROKEN


def test_wrapper_returns_two_when_the_child_cannot_be_spawned(tmp_path: Path) -> None:
    """子程序叫不動：回 2（工具自壞），而且不寫片段。"""
    out_dir = tmp_path / "steps"
    rc = record_step.main(
        ["--name", "absent", "--out-dir", str(out_dir), "--", "aosr-no-such-binary-xyz"]
    )
    assert rc == TOOL_BROKEN
    assert not (out_dir / "absent.json").exists()


def test_wrapper_rejects_bad_names_and_missing_child() -> None:
    """名字不是 ASCII、或 -- 後面沒東西，都是用法錯誤。"""
    with pytest.raises(SystemExit):
        record_step.parse_args(["--name", "中文", "--out-dir", "x", "--", PY])
    with pytest.raises(SystemExit):
        record_step.parse_args(["--name", "ok", "--out-dir", "x"])


# ── 合成器 build_receipt ──────────────────────────────────────────────────────


def fake_run_row(conclusion: str = "success") -> dict[str, object]:
    return {
        "id": FAKE_RUN_ID,
        "name": collect.VERIFY_WORKFLOW,
        "status": "completed",
        "conclusion": conclusion,
        "run_attempt": FAKE_ATTEMPT,
        "event": "pull_request",
        "head_branch": "fake-branch",
        "head_sha": "abc123def456",
        "html_url": "https://example.invalid/run",
        "run_started_at": "2026-09-10T01:00:00Z",
        "updated_at": "2026-09-10T01:03:00Z",
    }


def fake_jobs_row(conclusion: str = "success") -> dict[str, object]:
    step = {
        "name": "假的一步",
        "conclusion": conclusion,
        "started_at": "2026-09-10T01:00:10Z",
        "completed_at": "2026-09-10T01:00:20Z",
    }
    return {
        "jobs": [
            {
                "name": collect.VERIFY_WORKFLOW,
                "conclusion": conclusion,
                "started_at": "2026-09-10T01:00:00Z",
                "completed_at": "2026-09-10T01:02:00Z",
                "steps": [step],
            }
        ]
    }


class FakeShell:
    """假的外殼：照路徑回假 JSON，不 spawn 任何東西。"""

    def __init__(self, run_conclusion: str = "success", job_conclusion: str = "success") -> None:
        self.run_conclusion = run_conclusion
        self.job_conclusion = job_conclusion

    def out(self, argv: Sequence[str], what: str) -> str:
        path = argv[-1]
        if path.endswith("/jobs"):
            return json.dumps(fake_jobs_row(self.job_conclusion))
        return json.dumps(fake_run_row(self.run_conclusion))


def lay_artifact(tmp_path: Path, exit_codes: dict[str, int], junit: bool = True) -> Path:
    """在暫存目錄裡擺一個 gh run download 下來的樣子。"""
    folder = tmp_path / "artifacts" / f"{build_receipt.ARTIFACT_PREFIX}{FAKE_RUN_ID}-{FAKE_ATTEMPT}"
    steps = folder / build_receipt.STEPS_SUBDIR
    steps.mkdir(parents=True)
    for name, code in exit_codes.items():
        fragment = {
            "schema": 1,
            "name": name,
            "argv": ["uv", "run", "fake"],
            "started": "2026-09-10T01:00:10+00:00",
            "seconds": 1.5,
            "exit_code": code,
            "signal": None,
            "report": f"scan_root=. files=3 hits={1 if code else 0}",
            "stdout_tail": [],
            "stderr_tail": [],
        }
        (steps / f"{name}.json").write_text(json.dumps(fragment), encoding="utf-8")
    if junit:
        (folder / build_receipt.JUNIT_NAME).write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="pytest" tests="9" failures="0" '
            'errors="0" skipped="0" time="1.0"></testsuite></testsuites>',
            encoding="utf-8",
        )
    return tmp_path / "artifacts"


def test_receipt_is_written_with_cloud_authority_and_all_sections(tmp_path: Path) -> None:
    """合得成：一份 JSON，authority 是 cloud-run，四段（run／job／checks／pytest）都在。"""
    artifacts = lay_artifact(tmp_path, {"alpha": CLEAN, "beta": CLEAN})
    shell = FakeShell()
    run = build_receipt.read_run(shell, "fake/fake", FAKE_RUN_ID)
    job = build_receipt.read_verify_job(shell, "fake/fake", FAKE_RUN_ID)
    folder = build_receipt.artifact_dir(artifacts, run)
    receipt = build_receipt.compose(
        run, job, build_receipt.read_fragments(folder), build_receipt.read_junit(folder), {"GITHUB_RUN_ID": "7"}
    )
    assert receipt["authority"] == build_receipt.AUTHORITY
    assert set(receipt) >= {"run", "job", "checks", "pytest", "consistency", "written_by"}
    checks = receipt["checks"]
    assert isinstance(checks, list)
    assert {c["name"] for c in checks} == {"alpha", "beta"}
    assert receipt["pytest"] == {"tests": 9, "failures": 0, "errors": 0, "skipped": 0}
    assert receipt["consistency"] == {"ok": True, "notes": []}


def test_receipt_says_loudly_when_github_and_fragments_disagree(tmp_path: Path) -> None:
    """GitHub 判綠、片段說非零（或反過來）：consistency.ok 是 False，理由寫出來。"""
    fragments = build_receipt.read_fragments(
        build_receipt.artifact_dir(lay_artifact(tmp_path, {"alpha": VIOLATION}), build_receipt.read_run(FakeShell(), "f/f", FAKE_RUN_ID))
    )
    green_job = build_receipt.VerifyJob(conclusion="success", seconds=1, steps=())
    ok, notes = build_receipt.consistency(green_job, fragments)
    assert not ok
    assert any("吞" in n for n in notes)

    red_step = StepResult(name="ruff", conclusion="failure", seconds=1)
    red_job = build_receipt.VerifyJob(conclusion="failure", seconds=1, steps=(red_step,))
    clean = (build_receipt.Fragment(name="alpha", exit_code=CLEAN, signal=None, report=None, seconds=1.0),)
    ok, notes = build_receipt.consistency(red_job, clean)
    assert not ok
    assert any("ruff" in n for n in notes)


def test_missing_junit_is_null_not_invented(tmp_path: Path) -> None:
    """pytest 那一步沒跑到：pytest 欄是 null，不編四個 0。"""
    artifacts = lay_artifact(tmp_path, {"alpha": CLEAN}, junit=False)
    folder = build_receipt.artifact_dir(artifacts, build_receipt.read_run(FakeShell(), "f/f", FAKE_RUN_ID))
    assert build_receipt.read_junit(folder) is None


def test_missing_artifact_or_fragments_is_tool_broken(tmp_path: Path) -> None:
    """artifact 不在、或一片都沒有：ToolBroken（回 2），不產空收據。"""
    run = build_receipt.read_run(FakeShell(), "f/f", FAKE_RUN_ID)
    with pytest.raises(ToolBroken):
        build_receipt.artifact_dir(tmp_path / "nothing", run)
    empty = tmp_path / "artifacts" / f"{build_receipt.ARTIFACT_PREFIX}{FAKE_RUN_ID}-{FAKE_ATTEMPT}"
    empty.mkdir(parents=True)
    with pytest.raises(ToolBroken):
        build_receipt.read_fragments(empty)


def test_unfinished_run_cannot_get_a_receipt() -> None:
    """那一跑還沒跑完就不准先寫收據。"""

    class Running(FakeShell):
        def out(self, argv: Sequence[str], what: str) -> str:
            row = fake_run_row()
            row["status"] = "in_progress"
            return json.dumps(row)

    with pytest.raises(ToolBroken):
        build_receipt.read_run(Running(), "f/f", FAKE_RUN_ID)


def test_entry_point_maps_tool_broken_to_exit_code_two(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """合不成：離開碼 2，而且不留半份產出物。"""

    def boom(*args: object, **kwargs: object) -> Path:
        raise ToolBroken("假的：gh 不在")

    monkeypatch.setattr(build_receipt, "build", boom)
    out = tmp_path / "out"
    rc = build_receipt.main(["--run-id", str(FAKE_RUN_ID), "--artifacts", str(tmp_path), "--out", str(out)])
    assert rc == TOOL_BROKEN
    assert not out.exists()


def test_builder_refuses_to_write_into_the_repo_tree(tmp_path: Path) -> None:
    """產出目錄落在版控樹裡就回 2：收據不進主線。"""
    with pytest.raises(ToolBroken):
        build_receipt.build(tmp_path, FAKE_RUN_ID, tmp_path / "a", tmp_path / "out", 1, {})


def test_shell_is_the_only_layer_that_spawns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """合成器走的是狀態頁那一層的 Shell：gh 不在就 ToolBroken。"""

    def absent(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(collect.GH)

    monkeypatch.setattr(subprocess, "run", absent)
    with pytest.raises(ToolBroken):
        build_receipt.read_run(collect.Shell(cwd=tmp_path, timeout=1), "f/f", FAKE_RUN_ID)
