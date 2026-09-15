"""FEniCS 凍結答案的出身、題目雜湊與禁止只換數字重錄規矩。"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from governance.cloud_receipts import card_settings
from governance.exit_codes import ToolBroken, note, run
from governance.loader import setting_strings, setting_text


CHECK_REL = "governance/checks/fenics_answers_carry_provenance.py"


@dataclass(frozen=True)
class Rules:
    patterns: tuple[str, ...]
    schema_field: str
    schema_prefix: str
    provenance_field: str
    cases_field: str
    required_fields: tuple[str, ...]
    image_digest: str
    rerun_fields: tuple[str, ...]
    problem_file_field: str
    problem_sha256_field: str
    image_digest_field: str
    range_base_env: str
    range_head_env: str
    fixture_range_file: str
    fixture_base_cases_file: str


@dataclass(frozen=True)
class CommitRange:
    work_tree: Path
    base: str
    head: str
    label: str


def read_rules(settings: Mapping[str, object]) -> Rules:
    """卡上所有欄位名、分類樣式、digest 與範圍介面。"""
    return Rules(
        patterns=tuple(setting_strings(settings, "answer_file_patterns")),
        schema_field=setting_text(settings, "schema_field"),
        schema_prefix=setting_text(settings, "answer_schema_prefix"),
        provenance_field=setting_text(settings, "provenance_field"),
        cases_field=setting_text(settings, "cases_field"),
        required_fields=tuple(setting_strings(settings, "required_provenance_fields")),
        image_digest=setting_text(settings, "registered_image_digest"),
        rerun_fields=tuple(setting_strings(settings, "rerun_identity_fields")),
        problem_file_field=setting_text(settings, "problem_file_field"),
        problem_sha256_field=setting_text(settings, "problem_sha256_field"),
        image_digest_field=setting_text(settings, "image_digest_field"),
        range_base_env=setting_text(settings, "range_base_env"),
        range_head_env=setting_text(settings, "range_head_env"),
        fixture_range_file=setting_text(settings, "fixture_range_file"),
        fixture_base_cases_file=setting_text(settings, "fixture_base_cases_file"),
    )


def _blueprint_json(scan_root: Path, files: list[Path]) -> list[Path]:
    return sorted(
        path
        for path in files
        if path.parent == scan_root / "blueprint" and path.suffix == ".json"
    )


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """所有 blueprint JSON、自己的卡與暫存提交範圍宣告。"""
    picked = _blueprint_json(scan_root, files)
    rules_dir = scan_root / "governance" / "rules"
    governance_dir = scan_root / "governance"
    picked.extend(
        path
        for path in files
        if (
            path.parent == rules_dir
            and path.name == "fenics-answers-carry-provenance.toml"
        )
        or (
            path.parent == governance_dir
            and path.name.startswith("fixture-fenics-answer-")
        )
    )
    return sorted(set(picked))


def _load_json(path: Path, scan_root: Path) -> object:
    rel = path.relative_to(scan_root).as_posix()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ToolBroken(f"{rel} JSON 讀不懂（{exc}）——剖不開就不出結論") from exc


def _matches(path: Path, data: object, rules: Rules) -> bool:
    by_name = any(fnmatch.fnmatchcase(path.name, pattern) for pattern in rules.patterns)
    schema = data.get(rules.schema_field) if isinstance(data, dict) else None
    by_schema = isinstance(schema, str) and schema.startswith(rules.schema_prefix)
    return by_name or by_schema


def _answer_data(
    scan_root: Path, files: list[Path], rules: Rules
) -> dict[Path, dict[str, object]]:
    answers: dict[Path, dict[str, object]] = {}
    for path in _blueprint_json(scan_root, files):
        data = _load_json(path, scan_root)
        if not _matches(path, data, rules):
            continue
        if not isinstance(data, dict):
            raise ToolBroken(f"{path.relative_to(scan_root)} 的答案頂層不是一張表")
        answers[path] = {str(key): value for key, value in data.items()}
    if not answers:
        raise ToolBroken("blueprint/*.json 裡沒有任何 FEniCS 答案檔——掃描面沒有對象")
    return answers


def _nonempty(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping | list | tuple):
        return bool(value)
    return value is not None


def _static_hits(
    scan_root: Path, path: Path, data: dict[str, object], rules: Rules
) -> list[str]:
    rel = path.relative_to(scan_root).as_posix()
    bad: list[str] = []
    schema = data.get(rules.schema_field)
    if not isinstance(schema, str) or not schema.startswith(rules.schema_prefix):
        bad.append(f"{rel} 檔名命中登記樣式，但 {rules.schema_field} 沒有命中登記前綴")
    provenance = data.get(rules.provenance_field)
    if not isinstance(provenance, dict):
        return [*bad, f"{rel} 的 {rules.provenance_field} 不是一張非空出身表"]
    missing = [field for field in rules.required_fields if not _nonempty(provenance.get(field))]
    if missing:
        bad.append(f"{rel} 的 {rules.provenance_field} 缺值或空白：{missing}")
    if provenance.get(rules.image_digest_field) != rules.image_digest:
        bad.append(f"{rel} 的映像 digest 不等於卡上登記值")
    bad.extend(_problem_hash_hits(scan_root, rel, provenance, rules))
    return bad


def _problem_hash_hits(
    scan_root: Path, rel: str, provenance: dict[object, object], rules: Rules
) -> list[str]:
    raw_path = provenance.get(rules.problem_file_field)
    claimed = provenance.get(rules.problem_sha256_field)
    if not isinstance(raw_path, str) or not raw_path.strip():
        return []
    problem_path = (scan_root / raw_path).resolve()
    if not problem_path.is_relative_to(scan_root) or not problem_path.is_file():
        return [f"{rel} 指的題目檔 {raw_path!r} 不存在於掃描根內"]
    try:
        actual = hashlib.sha256(problem_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ToolBroken(f"題目檔 {raw_path!r} 讀不開：{exc}") from exc
    if claimed != actual:
        return [f"{rel} 的 {rules.problem_sha256_field} 與機器重算題目 SHA-256 不同"]
    return []


def _run_git(argv: list[str], cwd: Path, what: str, env: dict[str, str] | None = None) -> str:
    try:
        proc = subprocess.run(["git", *argv], cwd=cwd, env=env, capture_output=True, text=True)
    except (FileNotFoundError, OSError) as exc:
        raise ToolBroken(f"叫不動 git（{what}）：{exc}") from exc
    if proc.returncode != 0:
        raise ToolBroken(f"git {' '.join(argv)} 回 {proc.returncode}（{what}）：{proc.stderr.strip()[:300]}")
    return proc.stdout


def _rev_exists(work_tree: Path, rev: str) -> bool:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"],
            cwd=work_tree,
            capture_output=True,
        )
    except (FileNotFoundError, OSError) as exc:
        raise ToolBroken(f"叫不動 git（確認 {rev}）：{exc}") from exc
    if proc.returncode not in (0, 1):
        raise ToolBroken(f"git 無法確認提交 {rev}（回 {proc.returncode}）")
    return proc.returncode == 0


def _toplevel(scan_root: Path) -> Path | None:
    """這棵樹是哪個 git 工作樹的一部分？不是 git 工作樹就回 None。"""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=scan_root,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError) as exc:
        raise ToolBroken(f"叫不動 git（問這棵樹是不是 git 工作樹的根）：{exc}") from exc
    if proc.returncode not in (0, 128):
        raise ToolBroken(
            f"git rev-parse --show-toplevel 回 {proc.returncode}"
            f"（問這棵樹是不是 git 工作樹的根）：{proc.stderr.strip()[:300]}"
        )
    top = proc.stdout.strip()
    return Path(top).resolve() if proc.returncode == 0 and top else None


def _real_range(scan_root: Path, rules: Rules) -> CommitRange:
    top = _run_git(["rev-parse", "--show-toplevel"], scan_root, "確認工作樹").strip()
    if Path(top).resolve() != scan_root:
        raise ToolBroken(f"{scan_root} 不是 git 工作樹根，且沒有樣本範圍宣告")
    base = os.environ.get(rules.range_base_env, "").strip()
    head = os.environ.get(rules.range_head_env, "").strip() or "HEAD"
    if os.environ.get("GITHUB_EVENT_NAME") == "pull_request" and not base:
        raise ToolBroken(f"pull_request 缺 {rules.range_base_env}，拿不到 base..head")
    if not _rev_exists(scan_root, head):
        raise ToolBroken(f"解不出 head={head}")
    if base:
        if not _rev_exists(scan_root, base):
            raise ToolBroken(f"解不出 base={base}")
        return CommitRange(scan_root, base, head, f"{base}..{head}")
    parent = f"{head}~1"
    if _rev_exists(scan_root, parent):
        return CommitRange(scan_root, parent, head, f"{parent}..{head}")
    return CommitRange(scan_root, "", head, f"{head}（根提交）")


def _fixture_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "FEniCS fixture",
            "GIT_AUTHOR_EMAIL": "fixture@aosr.invalid",
            "GIT_COMMITTER_NAME": "FEniCS fixture",
            "GIT_COMMITTER_EMAIL": "fixture@aosr.invalid",
            "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+00:00",
        }
    )
    return env


def _write_fixture_answer(
    source: Path, target: Path, mode: str, base_cases_path: Path, rules: Rules
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "same":
        shutil.copy2(source, target)
        return
    data = json.loads(source.read_text(encoding="utf-8"))
    data[rules.cases_field] = json.loads(base_cases_path.read_text(encoding="utf-8"))
    target.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")


def _fixture_range(
    scan_root: Path, answers: dict[Path, dict[str, object]], rules: Rules
) -> tuple[CommitRange, Path]:
    declaration = scan_root / rules.fixture_range_file
    try:
        mode = declaration.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"樣本範圍宣告讀不開：{exc}") from exc
    if mode not in ("same", "base-cases"):
        raise ToolBroken(f"樣本範圍宣告只認 same 或 base-cases，實際是 {mode!r}")
    if mode == "base-cases" and len(answers) != 1:
        raise ToolBroken("base-cases 樣本必須恰好有一份答案檔")
    temp_root = Path(tempfile.mkdtemp(prefix="aosr-fenics-range-"))
    work = temp_root / "repo"
    work.mkdir()
    env = _fixture_env()
    _run_git(["-c", "init.defaultBranch=main", "init", "--quiet"], work, "建暫存 repo", env)
    base_cases = scan_root / rules.fixture_base_cases_file
    for source in answers:
        _write_fixture_answer(source, work / source.relative_to(scan_root), mode, base_cases, rules)
    _run_git(["add", "."], work, "加入 base 答案", env)
    _run_git(["commit", "--quiet", "--no-verify", "-m", "base"], work, "提交 base", env)
    base = _run_git(["rev-parse", "HEAD"], work, "讀 base", env).strip()
    for source in answers:
        target = work / source.relative_to(scan_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    _run_git(["add", "."], work, "加入 head 答案", env)
    _run_git(
        ["commit", "--quiet", "--no-verify", "--allow-empty", "-m", "head"],
        work,
        "提交 head",
        env,
    )
    head = _run_git(["rev-parse", "HEAD"], work, "讀 head", env).strip()
    return CommitRange(work, base, head, f"{base[:9]}..{head[:9]}（樣本）"), temp_root


def _read_commit_json(rng: CommitRange, rev: str, rel: str) -> dict[str, object]:
    raw = _run_git(["show", f"{rev}:{rel}"], rng.work_tree, f"讀 {rev}:{rel}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolBroken(f"提交範圍裡的 {rel} JSON 讀不懂：{exc}") from exc
    if not isinstance(data, dict):
        raise ToolBroken(f"提交範圍裡的 {rel} 頂層不是一張表")
    return {str(key): value for key, value in data.items()}


def _range_hits(rng: CommitRange, rules: Rules) -> list[str]:
    if not rng.base:
        return []
    changed = _run_git(
        ["diff", "--name-status", rng.base, rng.head, "--", "blueprint"],
        rng.work_tree,
        f"讀 {rng.label} 差異",
    )
    bad: list[str] = []
    for line in changed.splitlines():
        parts = line.split("\t")
        status = parts[0]
        if status == "A" or status == "D":
            continue
        old_rel, new_rel = (parts[1], parts[2]) if status.startswith("R") else (parts[1], parts[1])
        # 只讀 .json：blueprint/ 底下也住產生器與獨立檢查程式（.py），改到它們不是改答案，
        # 拿去剖 JSON 只會把整跑判成 2（2026-09-15 PR #320 實際撞到）。
        if Path(new_rel).suffix != ".json":
            continue
        head_data = _read_commit_json(rng, rng.head, new_rel)
        if not _matches(Path(new_rel), head_data, rules):
            continue
        base_data = _read_commit_json(rng, rng.base, old_rel)
        if base_data.get(rules.cases_field) == head_data.get(rules.cases_field):
            continue
        base_prov = base_data.get(rules.provenance_field)
        head_prov = head_data.get(rules.provenance_field)
        if not isinstance(base_prov, dict) or not isinstance(head_prov, dict):
            continue
        if all(base_prov.get(field) == head_prov.get(field) for field in rules.rerun_fields):
            bad.append(f"{new_rel} 的 {rules.cases_field} 已變，但重錄身分三格全都沒變")
    return bad


def check(scan_root: Path, files: list[Path]) -> list[str]:
    _card, settings = card_settings(scan_root, files, CHECK_REL)
    rules = read_rules(settings)
    declarations = [
        rel
        for rel in (rules.fixture_range_file, rules.fixture_base_cases_file)
        if (scan_root / rel).exists()
    ]
    if _toplevel(scan_root) == scan_root and declarations:
        raise ToolBroken(
            f"這是真的 git 工作樹的根，卻放著樣本用的宣告檔 {declarations}"
            "——宣告檔只准活在樣本迷你樹；出現在真樹等於把規則 5 的真正 PR 範圍比對關掉"
        )
    answers = _answer_data(scan_root, files, rules)
    bad = [
        hit
        for path, data in answers.items()
        for hit in _static_hits(scan_root, path, data, rules)
    ]
    declaration = scan_root / rules.fixture_range_file
    temp_root: Path | None = None
    try:
        if declaration.is_file():
            rng, temp_root = _fixture_range(scan_root, answers, rules)
        else:
            rng = _real_range(scan_root, rules)
        bad.extend(_range_hits(rng, rules))
        note(f"range={rng.label} answers={len(answers)}")
    finally:
        if temp_root is not None:
            shutil.rmtree(temp_root)
    return bad


if __name__ == "__main__":
    sys.exit(run(check, description="FEniCS 凍結答案要帶出身且不准只換數字", targets=targets))
