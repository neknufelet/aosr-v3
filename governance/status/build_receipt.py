"""把雲端 `verify` 那一跑合成一份機器收據。

跑法（`status.yml` 的 receipt job，接在 verify 跑完之後）：

    uv run python -m governance.status.build_receipt --run-id <verify 的 run id> \\
        --artifacts <gh run download 下來的目錄> --out <一個暫存目錄>

**兩邊資料，各管一半。**

* GitHub 記的：那一跑的 commit、分支、事件、結論、第幾次嘗試，以及 verify 那個 job 每一步是綠
  是紅。這是**紅綠的權威**——由 GitHub 的機器記，沒有任何一步是自報的。
* 片段（`record_step` 每一步寫的那一片）：每支檢查真正的離開碼（0／1／2 分得開）、判決收據
  那一行、最後幾行輸出。這是**細節**。

兩邊要交叉比對：GitHub 說 job 綠而有片段說非零、或 GitHub 說 job 紅而每一片都說 0，
都寫進收據的 `consistency` 欄——那正是 v2 事故 `verifier-trusts-self-reported-fields`
（驗證器信自報欄位）要擋的形狀，只是這回被驗的是包裝器自己。

**離開碼只有兩種。** 0 合成了、2 合不成（GitHub 回不了話、artifact 不在、片段一片都沒有、
形狀看不懂）。沒有 1：這支程式不下判決。合不成就不產檔——一份空收據看起來像「這一跑很乾淨」。

產出：`<out>/receipts/<run id>-<第幾次嘗試>.json`。`authority` 欄一律 `cloud-run`
（決策紙 `docs/decisions/cloud-run-is-the-only-authority.md`：這一份就是那一跑的收據，
本機任何人寫的都不會走這條路）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from governance.exit_codes import CLEAN, TOOL_BROKEN, ToolBroken, note, repo_root
from governance.status.build_status import assert_outside_repo
from governance.status.collect import (
    GH,
    VERIFY_WORKFLOW,
    Shell,
    as_rows,
    as_table,
    elapsed_seconds,
    field_int,
    field_text,
    parse_json,
    read_slug,
)
from governance.status.model import StepResult

# schema 2（2026-09-10）：checks[] 多 stderr_tail（去掉終端機控制碼的最後幾行），pytest 多 evidence（junit 的雜湊與大小）。
# 舊收據（schema 1）不改：規矩卡 receipt-authority-is-the-cloud-run 的第④條——新規矩不准回頭把舊收據判成無效。
RECEIPT_SCHEMA = 2
AUTHORITY = "cloud-run"
RECEIPTS_DIR = "receipts"
# artifact 的名字：verify.yml 上傳時取 `receipts-<run id>-<第幾次嘗試>`，這裡照同一個形狀找。
ARTIFACT_PREFIX = "receipts-"
STEPS_SUBDIR = "steps"
JUNIT_NAME = "pytest.junit.xml"


class Runner(Protocol):
    """會跑外部指令的那一層的形狀。真的是 `collect.Shell`；測試餵一個只會回假 JSON 的。"""

    def out(self, argv: Sequence[str], what: str) -> str: ...


def ask(shell: Runner, path: str, what: str) -> object:
    """打一支 GitHub API（跟 `collect.api` 同一條路，只是收 Runner 而不是具體的 Shell）。"""
    return parse_json(shell.out([GH, "api", path], what), f"gh api {path}")


@dataclass(frozen=True)
class RunMeta:
    """GitHub 記的那一跑。"""

    run_id: int
    attempt: int
    event: str
    head_branch: str
    head_sha: str
    conclusion: str
    url: str
    started: str
    completed: str


@dataclass(frozen=True)
class VerifyJob:
    """那一跑裡 verify 這個 job：結論加每一步。"""

    conclusion: str
    seconds: int
    steps: tuple[StepResult, ...]


@dataclass(frozen=True)
class Fragment:
    """`record_step` 寫的一片（只讀這幾格，多的欄位不管）。"""

    name: str
    exit_code: int
    signal: int | None
    report: str | None
    seconds: float
    stderr_tail: tuple[str, ...]


@dataclass(frozen=True)
class PytestSummary:
    """junit 收據上的四個數，加上證據（那一份 junit 的雜湊與大小——數字是從它算的）。"""

    tests: int
    failures: int
    errors: int
    skipped: int
    evidence: dict[str, object]


def read_run(shell: Runner, slug: str, run_id: int) -> RunMeta:
    """問 GitHub 那一跑的基本資料。"""
    row = as_table(ask(shell, f"repos/{slug}/actions/runs/{run_id}", "問那一跑"), "那一跑")
    if field_text(row, "name") != VERIFY_WORKFLOW:
        raise ToolBroken(f"run {run_id} 不是 {VERIFY_WORKFLOW} 那一跑（name={field_text(row, 'name')!r}）")
    if field_text(row, "status") != "completed":
        raise ToolBroken(f"run {run_id} 還沒跑完（status={field_text(row, 'status')!r}），收據不能先寫")
    return RunMeta(
        run_id=field_int(row, "id"),
        attempt=field_int(row, "run_attempt"),
        event=field_text(row, "event"),
        head_branch=field_text(row, "head_branch"),
        head_sha=field_text(row, "head_sha"),
        conclusion=field_text(row, "conclusion"),
        url=field_text(row, "html_url"),
        started=field_text(row, "run_started_at"),
        completed=field_text(row, "updated_at"),
    )


def read_verify_job(shell: Runner, slug: str, run_id: int) -> VerifyJob:
    """那一跑裡 verify 這個 job 的結論與每一步（每一步就是一支檢查）。"""
    rows = as_rows(
        as_table(ask(shell, f"repos/{slug}/actions/runs/{run_id}/jobs", "問那一跑的 job"), "job").get(
            "jobs"
        ),
        "job 清單",
    )
    for item in rows:
        job = as_table(item, "一個 job")
        if field_text(job, "name") != VERIFY_WORKFLOW:
            continue
        steps = tuple(
            StepResult(
                name=field_text(step, "name"),
                conclusion=field_text(step, "conclusion"),
                seconds=elapsed_seconds(field_text(step, "started_at"), field_text(step, "completed_at")),
            )
            for step in (as_table(entry, "一步") for entry in as_rows(job.get("steps", []), "job 的步驟"))
        )
        return VerifyJob(
            conclusion=field_text(job, "conclusion"),
            seconds=elapsed_seconds(field_text(job, "started_at"), field_text(job, "completed_at")),
            steps=steps,
        )
    raise ToolBroken(f"run {run_id} 裡沒有叫 {VERIFY_WORKFLOW} 的 job")


def artifact_dir(artifacts: Path, run: RunMeta) -> Path:
    """`gh run download --pattern 'receipts-*'` 會替每個 artifact 開一個子目錄，挑這一次嘗試的那個。"""
    target = artifacts / f"{ARTIFACT_PREFIX}{run.run_id}-{run.attempt}"
    if not target.is_dir():
        found = sorted(p.name for p in artifacts.iterdir()) if artifacts.is_dir() else []
        raise ToolBroken(
            f"找不到 artifact 目錄 {target.name}（{artifacts} 底下有：{found}）"
            "——那一跑沒上傳片段（被 timeout 殺掉、或 verify.yml 少了上傳那一步），收據合不成"
        )
    return target


def field_optional_int(row: Mapping[str, object], key: str) -> int | None:
    """可以是 null 的整數欄。"""
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolBroken(f"片段的 {key} 不是整數也不是 null：{value!r}")
    return value


def read_fragment(path: Path) -> Fragment:
    """讀一片。形狀不對就 ToolBroken。"""
    row = as_table(parse_json(path.read_text(encoding="utf-8"), f"片段 {path.name}"), f"片段 {path.name}")
    report = row.get("report")
    if report is not None and not isinstance(report, str):
        raise ToolBroken(f"片段 {path.name} 的 report 不是字串也不是 null：{report!r}")
    seconds = row.get("seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ToolBroken(f"片段 {path.name} 的 seconds 不是數字：{seconds!r}")
    tail = row.get("stderr_tail", [])
    if not isinstance(tail, list) or not all(isinstance(line, str) for line in tail):
        raise ToolBroken(f"片段 {path.name} 的 stderr_tail 不是字串清單：{tail!r}"[:300])
    return Fragment(
        name=field_text(row, "name"),
        exit_code=field_int(row, "exit_code"),
        signal=field_optional_int(row, "signal"),
        report=report,
        seconds=float(seconds),
        stderr_tail=tuple(tail),
    )


def read_fragments(folder: Path) -> tuple[Fragment, ...]:
    """讀 `steps/` 底下每一片。一片都沒有就 ToolBroken——那一跑連第一步都沒記到。"""
    steps = folder / STEPS_SUBDIR
    paths = sorted(steps.glob("*.json")) if steps.is_dir() else []
    if not paths:
        raise ToolBroken(f"{steps} 底下一片收據片段都沒有——沒有片段就沒有離開碼，收據合不成")
    return tuple(read_fragment(p) for p in paths)


def read_junit(folder: Path) -> PytestSummary | None:
    """junit 收據上的四個數。檔不在（pytest 那一步沒跑到）就 None，不編。"""
    path = folder / JUNIT_NAME
    if not path.is_file():
        return None
    raw = path.read_bytes()
    try:
        root = ET.fromstring(raw.decode("utf-8"))
    except ET.ParseError as exc:
        raise ToolBroken(f"junit 收據 {path} 解不開：{exc}") from exc
    # pytest 的 junit 是 <testsuites> 包一個或多個 <testsuite>；iter 會連根自己是 testsuite 的也算進來。
    suites = list(root.iter("testsuite"))
    if not suites:
        raise ToolBroken(f"junit 收據 {path} 裡一個 testsuite 都沒有")

    def total(key: str) -> int:
        try:
            return sum(int(s.get(key, "0")) for s in suites)
        except ValueError as exc:
            raise ToolBroken(f"junit 收據的 {key} 不是整數：{exc}") from exc

    return PytestSummary(
        tests=total("tests"),
        failures=total("failures"),
        errors=total("errors"),
        skipped=total("skipped"),
        evidence={"junit": JUNIT_NAME, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)},
    )


def consistency(job: VerifyJob, fragments: Sequence[Fragment]) -> tuple[bool, tuple[str, ...]]:
    """GitHub 記的紅綠跟片段說的離開碼對不對得上。對不上就寫出來，不吞。"""
    notes: list[str] = []
    red_fragments = [f.name for f in fragments if f.exit_code != CLEAN]
    red_steps = [s.name for s in job.steps if s.conclusion == "failure"]
    if job.conclusion == "success" and red_fragments:
        notes.append(f"GitHub 判 job 綠，但這幾片說非零：{red_fragments}——有一層把離開碼吞掉了")
    if job.conclusion == "failure" and not red_fragments:
        notes.append(
            f"GitHub 判 job 紅（紅在：{red_steps}），但每一片都說 0"
            "——紅在沒包片段的那一步（checkout、setup、上傳片段那幾步），或片段沒記到"
        )
    if job.conclusion not in {"success", "failure"}:
        notes.append(f"job 結論是 {job.conclusion!r}，不是綠也不是紅——這一跑沒有正常跑完")
    return (not notes), tuple(notes)


def compose(
    run: RunMeta,
    job: VerifyJob,
    fragments: Sequence[Fragment],
    pytest: PytestSummary | None,
    env: Mapping[str, str],
) -> dict[str, object]:
    """把收據排成要寫下去的形狀。每一格都是量到的。"""
    ok, notes = consistency(job, fragments)
    return {
        "schema": RECEIPT_SCHEMA,
        "authority": AUTHORITY,
        "run": asdict(run),
        "job": {
            "conclusion": job.conclusion,
            "seconds": job.seconds,
            "steps": [asdict(s) for s in job.steps],
        },
        "checks": [asdict(f) for f in fragments],
        "pytest": asdict(pytest) if pytest else None,
        "consistency": {"ok": ok, "notes": list(notes)},
        "written_by": {
            "run_id": env.get("GITHUB_RUN_ID", "").strip() or None,
            "attempt": env.get("GITHUB_RUN_ATTEMPT", "").strip() or None,
        },
    }


def build(root: Path, run_id: int, artifacts: Path, out: Path, timeout: int, env: Mapping[str, str]) -> Path:
    """問、讀、合、寫。回寫下去的那個檔。"""
    assert_outside_repo(root, out)
    shell = Shell(cwd=root, timeout=timeout)
    slug = read_slug(shell, env)
    run = read_run(shell, slug, run_id)
    job = read_verify_job(shell, slug, run_id)
    folder = artifact_dir(artifacts, run)
    receipt = compose(run, job, read_fragments(folder), read_junit(folder), env)
    target = out / RECEIPTS_DIR / f"{run.run_id}-{run.attempt}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return target


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """命令列參數。逾時秒數在這裡宣告，不寫成程式裡的常數。"""
    parser = argparse.ArgumentParser(description="把雲端 verify 那一跑合成一份機器收據（合不成回 2）")
    parser.add_argument("--run-id", type=int, required=True, help="verify 那一跑的 run id")
    parser.add_argument("--artifacts", required=True, help="gh run download 下來的目錄")
    parser.add_argument("--out", required=True, help="產出目錄（不准指到版控樹裡）")
    parser.add_argument("--timeout", type=int, default=60, help="每一個外部指令的看門狗秒數")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """入口。ToolBroken 一律翻成離開碼 2，而且不留下半份產出物。"""
    args = parse_args(argv)
    try:
        target = build(repo_root(), args.run_id, Path(args.artifacts), Path(args.out), args.timeout, os.environ)
    except ToolBroken as exc:
        note(f"收據合不成，離開碼 2（工具自壞）：{exc}")
        return TOOL_BROKEN
    note(f"寫好了：{target}")
    return CLEAN


if __name__ == "__main__":
    sys.exit(main())
