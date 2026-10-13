"""收據生產者的測試：包裝器真的跑小子程序（寫進暫存目錄），合成器全餵假資料、不上網。

斷言刻意不鎖死數量（規矩卡 assertions-not-pinned-to-counts）：離開碼比的是
`governance/exit_codes.py` 登記的那三個名字，片段與收據比的是「有沒有這幾格、值對不對」。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from governance.exit_codes import CLEAN, TOOL_BROKEN, VIOLATION, ToolBroken
from governance.status import build_receipt, collect, mirror_receipts, record_step
from governance.status.model import StepResult
from tests.conftest import SANDBOX_IDENTITY, GitSandbox

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


def test_wrapper_finds_the_report_line_anywhere_in_the_output(tmp_path: Path) -> None:
    """判決那一行印在前面、後面接一長串輸出，照樣找得到。

    原本只在最後幾行裡找（那個數字是猜的），輸出長一點那一格就變成 null——而 null 的
    意思是「這一步本來就不印那一行」，兩件事會混在一起。整段找就沒有那個猜的數字。
    """
    line = "scan_root=. files=7 hits=0"
    # 印完那一行之後再印一大串，遠比片段存的尾巴長（尾巴幾行由 --tail-lines 決定）。
    code = f"print({line!r});" + "[print('x' + str(i)) for i in range(500)]"
    _, fragment = wrap(tmp_path, "buried", code)
    assert fragment["report"] == line
    tail = fragment["stdout_tail"]
    assert isinstance(tail, list)
    # 尾巴裡本來就沒有那一行（它被沖走了），所以這一格證明的是「不是從尾巴撈到的」。
    assert line not in [str(row) for row in tail]


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
        if path.split("?")[0].endswith("/jobs"):
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
            "stderr_tail": ["FAIL(1) 有 1 筆違規"] if code else [],
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
    jobs = build_receipt.read_jobs(shell, "fake/fake", FAKE_RUN_ID)
    folder = build_receipt.artifact_dir(artifacts, run)
    junit = build_receipt.read_junit(folder)
    assert junit is not None
    receipt = build_receipt.compose(
        run, jobs, {"verify": build_receipt.read_fragments(folder)},
        [{"job": "verify", "tests": junit.tests, "failures": junit.failures,
          "errors": junit.errors, "skipped": junit.skipped, **junit.evidence}], {"GITHUB_RUN_ID": "7"}
    )
    assert receipt["authority"] == build_receipt.AUTHORITY
    assert set(receipt) >= {"run", "job", "checks", "pytest", "consistency", "written_by"}
    checks = receipt["checks"]
    assert isinstance(checks, list)
    assert {c["name"] for c in checks} == {"alpha", "beta"}
    block = collect.as_table(receipt["pytest"], "pytest")
    summary = collect.as_table(collect.as_rows(block["parts"], "parts")[0], "part")
    assert {k: summary[k] for k in ("tests", "failures", "errors", "skipped")} == {"tests": 9, "failures": 0, "errors": 0, "skipped": 0}
    evidence = summary
    assert evidence["junit"] == build_receipt.JUNIT_NAME
    assert isinstance(evidence["sha256"], str) and len(evidence["sha256"]) == len(hashlib.sha256(b"").hexdigest())
    assert receipt["job"] == {"conclusion": "success", "seconds": 120, "steps": [{"name": "假的一步", "conclusion": "success", "seconds": 10}]}
    assert receipt["schema"] == build_receipt.RECEIPT_SCHEMA
    assert all("stderr_tail" in c for c in checks)
    assert receipt["consistency"] == {"ok": True, "notes": []}


def test_receipt_says_loudly_when_github_and_fragments_disagree(tmp_path: Path) -> None:
    """GitHub 判綠、片段說非零（或反過來）：consistency.ok 是 False，理由寫出來。"""
    fragments = build_receipt.read_fragments(
        build_receipt.artifact_dir(lay_artifact(tmp_path, {"alpha": VIOLATION}), build_receipt.read_run(FakeShell(), "f/f", FAKE_RUN_ID))
    )
    green_job = build_receipt.VerifyJob(name="verify", conclusion="success", seconds=1, steps=())
    ok, notes = build_receipt.consistency(green_job, fragments)
    assert not ok
    assert any("吞" in n for n in notes)

    red_step = StepResult(job="verify", name="ruff", conclusion="failure", seconds=1)
    red_job = build_receipt.VerifyJob(name="verify", conclusion="failure", seconds=1, steps=(red_step,))
    clean = (build_receipt.Fragment(name="alpha", exit_code=CLEAN, signal=None, report=None, seconds=1.0, stderr_tail=()),)
    ok, notes = build_receipt.consistency(red_job, clean)
    assert not ok
    assert any("ruff" in n for n in notes)


def test_missing_junit_is_null_not_invented(tmp_path: Path) -> None:
    """單機 pytest 那一步沒跑到：讀取回 None，合成器另有題目核空 parts，不編四個 0。"""
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


# ── 鏡像 mirror_receipts ──────────────────────────────────────────────────────


def test_mirror_copies_receipts_and_records_provenance(git_sandbox: GitSandbox, tmp_path: Path) -> None:
    """暫存 repo 上造一條 status 分支放兩份收據，鏡過來要有那兩份，來源要記到提交與身分。"""
    root = git_sandbox.root
    (root / "README").write_text("main\n", encoding="utf-8")
    git_sandbox.git("add", "README")
    git_sandbox.git("commit", "-q", "-m", "main")
    git_sandbox.git("switch", "-q", "--orphan", "status")
    (root / "receipts").mkdir()
    for name in ("1-1.json", "2-1.json"):
        (root / "receipts" / name).write_text('{"schema": 1}', encoding="utf-8")
    git_sandbox.git("add", "receipts")
    git_sandbox.git("commit", "-q", "-m", "receipt: 假的")
    git_sandbox.git("switch", "-q", "main")

    out = tmp_path / "cloud"
    provenance = mirror_receipts.mirror(root, "status", out, timeout=30)
    assert {p.name for p in (out / mirror_receipts.BRANCH_DIR).glob("*.json")} == {"1-1.json", "2-1.json"}
    record = json.loads(provenance.read_text(encoding="utf-8"))
    assert record["ref"] == "status"
    assert set(record["files"]) == {"1-1.json", "2-1.json"}
    origin = record["files"]["1-1.json"]
    assert origin["author_email"] == SANDBOX_IDENTITY["GIT_AUTHOR_EMAIL"]
    assert origin["committer_email"] == SANDBOX_IDENTITY["GIT_COMMITTER_EMAIL"]
    assert len(origin["commit"]) == len(record["commit"])


def test_mirror_refuses_missing_ref_and_empty_branch(git_sandbox: GitSandbox, tmp_path: Path) -> None:
    """ref 不在、或那條分支上一份收據都沒有：ToolBroken（回 2），而且不留半份鏡像。"""
    root = git_sandbox.root
    (root / "README").write_text("main\n", encoding="utf-8")
    git_sandbox.git("add", "README")
    git_sandbox.git("commit", "-q", "-m", "main")
    out = tmp_path / "cloud"
    with pytest.raises(ToolBroken):
        mirror_receipts.mirror(root, "no-such-ref", out, timeout=30)
    with pytest.raises(ToolBroken):
        mirror_receipts.mirror(root, "main", out, timeout=30)
    assert not out.exists()


def test_mirror_entry_point_maps_tool_broken_to_two(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def boom(*args: object, **kwargs: object) -> Path:
        raise ToolBroken("假的：ref 不在")

    monkeypatch.setattr(mirror_receipts, "mirror", boom)
    assert mirror_receipts.main(["--out", str(tmp_path / "x")]) == TOOL_BROKEN


def _receipt_commit(sandbox: GitSandbox, folder: Path, email: str, parent: str | None = None) -> str:
    """沙箱內明定兩種提交身分；fixture 的身分環境固定，故用原始 commit 物件建歷史。"""
    tree = sandbox.git("write-tree").stdout.strip()
    parents = f"parent {parent}\n" if parent else ""
    stamp = "946684800 +0000"
    body = f"tree {tree}\n{parents}author Writer <{email}> {stamp}\ncommitter Writer <{email}> {stamp}\n\nreceipt\n"
    commit_file = folder / "commit.txt"
    commit_file.write_text(body, encoding="utf-8")
    commit = sandbox.git("hash-object", "-t", "commit", "-w", str(commit_file)).stdout.strip()
    sandbox.git("update-ref", "refs/heads/status", commit)
    return commit


def test_mirror_attributes_each_receipt_to_its_latest_commit(git_sandbox: GitSandbox, tmp_path: Path) -> None:
    """既有考卷每份收據只被一筆提交動過，抓不到取錯那一筆；這題改寫同一份、換身分。"""
    root = git_sandbox.root
    git_sandbox.git("switch", "-q", "--orphan", "status")
    target = root / mirror_receipts.BRANCH_DIR / "1-1.json"
    target.parent.mkdir()
    target.write_text('{"schema": 1}', encoding="utf-8")
    git_sandbox.git("add", "receipts")
    first = _receipt_commit(git_sandbox, tmp_path, "41898282+github-actions[bot]@users.noreply.github.com")
    target.write_text('{"schema": 2}', encoding="utf-8")
    git_sandbox.git("add", "receipts")
    email = "other@example.invalid"
    latest = _receipt_commit(git_sandbox, tmp_path, email, parent=first)

    provenance = mirror_receipts.mirror(root, "status", tmp_path / "cloud", timeout=30)
    origin = json.loads(provenance.read_text(encoding="utf-8"))["files"][target.name]
    assert origin == {"commit": latest, "author_email": email, "committer_email": email}


def test_mirror_refuses_merge_commits_touching_receipts(git_sandbox: GitSandbox, tmp_path: Path) -> None:
    """合併守門拿掉就會收下合併；守門放太晚就會破壞先前好的鏡像。"""
    root = git_sandbox.root
    git_sandbox.git("switch", "-q", "--orphan", "status")
    target = root / mirror_receipts.BRANCH_DIR / "1-1.json"
    target.parent.mkdir()
    target.write_text('{"schema": 1}', encoding="utf-8")
    git_sandbox.git("add", "receipts")
    git_sandbox.git("commit", "-q", "-m", "good receipt")
    base = git_sandbox.git("rev-parse", "HEAD").stdout.strip()
    out = tmp_path / "cloud"
    provenance = mirror_receipts.mirror(root, "status", out, timeout=30)
    before = {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()}

    # 兩條線各有一筆提交；合併提交的樹另改收據，與兩邊父提交都不同。
    git_sandbox.git("commit", "-q", "--allow-empty", "-m", "status side")
    status_parent = git_sandbox.git("rev-parse", "HEAD").stdout.strip()
    git_sandbox.git("switch", "-q", "-c", "receipt-side", base)
    git_sandbox.git("commit", "-q", "--allow-empty", "-m", "other side")
    side_parent = git_sandbox.git("rev-parse", "HEAD").stdout.strip()
    target.write_text('{"schema": 2}', encoding="utf-8")
    git_sandbox.git("add", "receipts")
    tree = git_sandbox.git("write-tree").stdout.strip()
    merge = git_sandbox.git("commit-tree", tree, "-p", status_parent, "-p", side_parent, "-m", "merge receipt").stdout.strip()
    git_sandbox.git("update-ref", "refs/heads/status", merge)

    with pytest.raises(ToolBroken, match="合併提交"):
        mirror_receipts.mirror(root, "status", out, timeout=30)
    assert provenance.is_file()
    assert {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()} == before


SHARD_SAMPLE = Path(__file__).resolve().parents[1] / "governance/fixtures/green-must-be-real-green-green/shard-topology"


class JobsShell(FakeShell):
    """回完整 job 清單；不執行外部工具。"""

    def __init__(self, jobs: list[dict[str, object]]) -> None:
        super().__init__()
        self.jobs = jobs

    def out(self, argv: Sequence[str], what: str) -> str:
        if argv[-1].split("?")[0].endswith("/jobs"):
            return json.dumps({"jobs": self.jobs})
        return super().out(argv, what)


def receipt_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path, list[dict[str, object]]]:
    """單機包加兩片樣本；登記簿與輸出均在暫存目錄。"""
    root = tmp_path / "source"
    registry = root / "governance/verify-shards.toml"
    registry.parent.mkdir(parents=True)
    shutil.copyfile(SHARD_SAMPLE / "governance/verify-shards.toml", registry)
    artifacts = lay_artifact(tmp_path, {"alpha": CLEAN})
    jobs = [collect.as_table(row, "job") for row in collect.as_rows(fake_jobs_row()["jobs"], "jobs")]
    for name in ("shard-a", "shard-b"):
        source = SHARD_SAMPLE / "governance/receipts/parts" / f"receipts-123-2-{name}"
        shutil.copytree(source, artifacts / f"receipts-{FAKE_RUN_ID}-{FAKE_ATTEMPT}-{name}")
        job = dict(jobs[0])
        job["name"] = name
        jobs.append(job)
    shell = JobsShell(jobs)
    monkeypatch.setattr(build_receipt, "Shell", lambda **kwargs: shell)
    return root, artifacts, tmp_path / "out", jobs


def test_three_jobs_include_all_fragments_and_junit_from_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """濾掉分片、漏一片或把 junit 路徑寫死，都會漏掉本題的證據。"""
    root, artifacts, out, jobs = receipt_sandbox(tmp_path, monkeypatch)
    registry = root / "governance/verify-shards.toml"
    registry.write_text(registry.read_text().replace('dir = "governance/receipts/shard"', 'dir = "governance/receipts/custom"').replace('junit = "junit.xml"', 'junit = "result.xml"'))
    for name in ("shard-a", "shard-b"):
        folder = artifacts / f"receipts-{FAKE_RUN_ID}-{FAKE_ATTEMPT}-{name}"
        (folder / "shard").rename(folder / "custom")
        (folder / "custom" / "junit.xml").rename(folder / "custom" / "result.xml")
    target = build_receipt.build(root, FAKE_RUN_ID, artifacts, out, 1, {"GITHUB_REPOSITORY": "fake/fake"})
    body = json.loads(target.read_text())
    assert body["schema"] == 3
    assert {job["name"] for job in body["jobs"]} == {job["name"] for job in jobs}
    assert body["job"]["conclusion"] == "success"
    expected = {(job["name"], json.loads(path.read_text())["name"]) for job in jobs
                for path in (artifacts / (f"receipts-{FAKE_RUN_ID}-{FAKE_ATTEMPT}" + ("" if job["name"] == "verify" else f'-{job["name"]}')) / "steps").glob("*.json")}
    assert {(check["job"], check["name"]) for check in body["checks"]} == expected
    parts = {part["job"]: part for part in body["pytest"]["parts"]}
    assert set(parts) == {job["name"] for job in jobs}
    for name, part in parts.items():
        folder = artifacts / (f"receipts-{FAKE_RUN_ID}-{FAKE_ATTEMPT}" + ("" if name == "verify" else f"-{name}"))
        raw = (folder / part["junit"]).read_bytes()
        assert part["sha256"] == hashlib.sha256(raw).hexdigest()
        assert part["bytes"] == len(raw)
        assert all(type(part[key]) is int for key in ("tests", "failures", "errors", "skipped"))


def test_check_job_comes_from_package_even_if_fragment_claims_another(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """片段 job 寫錯或舊片段沒有 job，包的歸屬仍以 API 名為準。"""
    root, artifacts, out, _ = receipt_sandbox(tmp_path, monkeypatch)
    fragment = artifacts / f"receipts-{FAKE_RUN_ID}-{FAKE_ATTEMPT}-shard-a" / "steps" / "style_guard.json"
    row = json.loads(fragment.read_text())
    row["job"] = "impostor"
    fragment.write_text(json.dumps(row))
    body = json.loads(build_receipt.build(root, FAKE_RUN_ID, artifacts, out, 1, {"GITHUB_REPOSITORY": "fake/fake"}).read_text())
    checks = {(row["job"], row["name"]) for row in body["checks"]}
    assert ("shard-a", "style_guard") in checks
    assert ("impostor", "style_guard") not in checks
    assert ("verify", "alpha") in checks


@pytest.mark.parametrize("missing", ["package", "junit"])
def test_shard_missing_package_or_junit_returns_two_without_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    """缺包或分片缺 junit，不能合成半份收據或 null。"""
    root, artifacts, out, _ = receipt_sandbox(tmp_path, monkeypatch)
    folder = artifacts / f"receipts-{FAKE_RUN_ID}-{FAKE_ATTEMPT}-shard-b"
    if missing == "package":
        shutil.rmtree(folder)
    else:
        (folder / "shard" / "junit.xml").unlink()
    monkeypatch.setattr(build_receipt, "repo_root", lambda: root)
    monkeypatch.setenv("GITHUB_REPOSITORY", "fake/fake")
    assert build_receipt.main(["--run-id", str(FAKE_RUN_ID), "--artifacts", str(artifacts), "--out", str(out)]) == TOOL_BROKEN
    assert not out.exists()


def test_single_job_without_junit_has_empty_parts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """單機尚未跑到 pytest，parts 如實為空，不編數字。"""
    root, artifacts, out, jobs = receipt_sandbox(tmp_path, monkeypatch)
    jobs[:] = [jobs[0]]
    (artifacts / f"receipts-{FAKE_RUN_ID}-{FAKE_ATTEMPT}" / "pytest.junit.xml").unlink()
    body = json.loads(build_receipt.build(root, FAKE_RUN_ID, artifacts, out, 1, {"GITHUB_REPOSITORY": "fake/fake"}).read_text())
    assert body["pytest"] == {"parts": []}
    assert {job["name"] for job in body["jobs"]} == {"verify"}


def test_job_api_is_read_past_a_full_page() -> None:
    """第二頁的 job 也要進收據，第一頁滿了不能提早收工。"""
    first = [collect.as_table(collect.as_rows(fake_jobs_row()["jobs"], "jobs")[0], "job")
             for _ in range(100)]
    for index, row in enumerate(first):
        row["name"] = f"worker-{index}"
    final = collect.as_table(collect.as_rows(fake_jobs_row()["jobs"], "jobs")[0], "job")

    class PagedJobs(FakeShell):
        def out(self, argv: Sequence[str], what: str) -> str:
            return json.dumps({"jobs": first if argv[-1].endswith("&page=1") else [final]})

    jobs = build_receipt.read_jobs(PagedJobs(), "fake/fake", FAKE_RUN_ID)
    assert {job.name for job in jobs} == {str(row["name"]) for row in [*first, final]}
