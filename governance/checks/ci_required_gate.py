"""必要檢查的名稱、觸發與步驟條件；判準由 ci-jobs 卡傳入。"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import cast

from governance.exit_codes import ToolBroken
from governance.loader import setting_text
from governance.checks.ci_command_segments import (
    SEGMENT_SEP_RE, _command_lines, _command_text, _is_wrapped, _split_outside_quotes,
)

FINALIZER_KEY = "finalizer_jobs"
FINALIZER_FIELDS = ("registry", "job_if", "step_if", "verdict_env_name", "verdict_env_value")


@dataclass(frozen=True)
class Finalizer:
    """卡的整串設定與登記簿的分片集合；沒有內建判準值。"""
    settings: dict[str, str]
    shards: frozenset[str]


def finalizer_entries(settings: dict[str, object]) -> dict[str, dict[str, str]]:
    """設定形狀已在主檔驗證，這裡只供型別與共用取值。"""
    return cast(dict[str, dict[str, str]], settings[FINALIZER_KEY])


def finalizer_registry_paths(settings: dict[str, object]) -> list[str]:
    """掃描面列舉使用卡的路徑，空登記不多讀任何檔。"""
    return sorted({entry["registry"] for entry in finalizer_entries(settings).values()})


def read_finalizers(scan_root: Path, files: list[Path], settings: dict[str, object]) -> dict[str, Finalizer]:
    """本步只讀 shards；不存在、不可讀、不可解或形狀不對均回 2。"""
    finalizers: dict[str, Finalizer] = {}
    for job_id, entry in finalizer_entries(settings).items():
        rel = entry["registry"]
        path = scan_root / rel
        if path not in files:
            raise ToolBroken(f"收尾 {job_id!r} 的登記簿 {rel} 不在掃描檔案裡")
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ToolBroken(f"收尾 {job_id!r} 的登記簿 {rel} 讀不開：{exc}") from exc
        shards = data.get("shards")
        if not isinstance(shards, list) or not shards or not all(isinstance(s, str) and s.strip() for s in shards):
            raise ToolBroken(f"登記簿 {rel} 的 shards 必須是非空的分片 id 清單")
        if len(set(shards)) != len(shards) or set(shards).intersection(finalizer_entries(settings)):
            raise ToolBroken(f"登記簿 {rel} 的 shards 不准重複或包含收尾 id")
        finalizers[job_id] = Finalizer(entry, frozenset(shards))
    return finalizers


def _needs_match(job: dict[str, object], finalizer: Finalizer) -> bool:
    """只接受字面 id 或字面 id 清單，不把運算式當依賴集合。"""
    needs = job.get("needs")
    if isinstance(needs, str):
        return frozenset({needs}) == finalizer.shards
    return isinstance(needs, list) and all(isinstance(n, str) for n in needs) and frozenset(needs) == finalizer.shards


def job_identity_problems(
    where: str, job_id: str, job: dict[str, object], required: frozenset[str], finalizer: Finalizer | None = None,
) -> list[str]:
    """GitHub 以 job name（未寫才用 id）比必要檢查名。"""
    raw_name = job.get("name")
    actual = raw_name if isinstance(raw_name, str) else job_id
    bad: list[str] = []
    if job_id in required and "name" in job and raw_name != job_id:
        bad.append(f"{where} 是必要檢查，name: {raw_name!r} 與 job id {job_id!r} 不同——GitHub 按 name 比必要檢查")
    if actual in required and job_id != actual:
        bad.append(f"{where} 的 name: {actual!r} 冒用必要檢查名，跳過的冒名 job 也會回報 Success")
    if job_id in required and "needs" in job and finalizer is None:
        bad.append(f"{where} 是必要檢查卻有 needs:——前置 job 被跳過會連帶跳過它並回報 Success")
    if finalizer is not None and not _needs_match(job, finalizer):
        bad.append(f"{where} 的收尾 needs 集合必須等於登記簿分片 {sorted(finalizer.shards)}")
    if finalizer is not None and job.get("if") != finalizer.settings["job_if"]:
        bad.append(f"{where} 的收尾 job 層 if 必須整串等於卡上登記的 {finalizer.settings['job_if']!r}，缺席也紅")
    if isinstance(raw_name, str) and "${{" in raw_name:
        # 運算式的值要到雲端才算得出來，這裡判不出它會不會等於必要檢查名，一律不准。
        bad.append(f"{where} 的 name: {raw_name!r} 是運算式——算出來可能冒用必要檢查名，這裡判不出，不准")
    if job_id in required and "strategy" in job:
        bad.append(f"{where} 是必要檢查卻有 strategy:——matrix 會把檢查名改成帶括號的另一個名字，必要檢查就對不上它")
    return bad


def step_if_problems(
    where: str, job_id: str, label: str, step: dict[str, object], allowed: object, finalizer: Finalizer | None = None,
) -> list[str]:
    """必要工作只有卡上指名的步驟與 if 值可帶條件。"""
    if finalizer is not None or "if" not in step:
        return []
    if not isinstance(allowed, dict):
        raise ToolBroken("allowed_required_step_ifs 不是表")
    per_job = allowed.get(job_id)
    if not isinstance(per_job, dict):
        raise ToolBroken(f"必要 job {job_id!r} 沒在 allowed_required_step_ifs 登記")
    if per_job.get(label) == step["if"]:
        return []
    return [f"{where} 是必要檢查的步驟，if: {step['if']!r} 沒在卡上按 job 與步驟名登記——步驟可被跳過"]


def required_trigger_problems(
    rel: str, data: dict[str, object], jobs: dict[str, object],
    required: frozenset[str], settings: dict[str, object],
) -> list[str]:
    """必要 job 的 pull_request 事件必須明寫 types 且不得過濾。"""
    registered = settings["required_pull_request_events"]
    filters = settings["forbidden_pull_request_filters"]
    if not isinstance(registered, dict) or not isinstance(filters, list):
        raise ToolBroken("必要檢查的事件設定不是表與名單")
    present = sorted(required.intersection(jobs))
    for job in present:
        if job not in registered:
            raise ToolBroken(f"{rel} 的必要 job {job!r} 沒在卡的 required_pull_request_events 登記")
    raw_data = cast(Mapping[object, object], data)
    triggers = raw_data.get("on", raw_data.get(True))
    if not isinstance(triggers, dict):
        return [f"{rel} 的必要檢查所在 workflow 的 on 要寫成表、pull_request 要明寫 types"] if present else []
    pull = triggers.get("pull_request")
    bad: list[str] = []
    for job in present:
        if not isinstance(pull, dict):
            bad.append(f"{rel} 的必要 job {job!r}：on 要寫成表、pull_request 要明寫 types")
            continue
        for key in filters:
            if isinstance(key, str) and key in pull:
                bad.append(f"{rel} 的必要 job {job!r} 有 on.pull_request.{key} 過濾，檢查會停在 Pending")
        types = pull.get("types")
        if not isinstance(types, list) or not all(isinstance(item, str) for item in types):
            bad.append(f"{rel} 的必要 job {job!r}：pull_request 要明寫 types")
            continue
        expected = registered[job]
        if not isinstance(expected, list):
            raise ToolBroken(f"required_pull_request_events.{job} 不是事件名單")
        missing = [item for item in expected if item not in types]
        if missing:
            bad.append(f"{rel} 的必要 job {job!r} 少了 pull_request.types 事件 {missing}")
    return bad


def protected_ids_unique(
    workflows: list[tuple[str, dict[str, object], dict[str, object]]], protected: frozenset[str],
) -> list[str]:
    """每個受保護 id 在全部 workflow 恰好一份，缺席與重複均紅。"""
    bad: list[str] = []
    for job_id in sorted(protected):
        locations = [rel for rel, _, jobs in workflows if job_id in jobs]
        if len(locations) != 1:
            bad.append(f"受保護 id {job_id!r} 在全部 workflow 必須恰好出現一次，實際所在：{locations}")
    return bad


def _wrapped_step(step: dict[str, object], wrapper: str) -> bool:
    """沿用⑤切段器，只認真的命令開頭；註解與引號都不能冒充。"""
    body = step.get("run")
    if not isinstance(body, str):
        return False
    return any(
        _is_wrapped(command, wrapper)
        for line in _command_lines(body)
        for command in _split_outside_quotes(_command_text(line), SEGMENT_SEP_RE)
    )


def finalizer_problems(
    rel: str, jobs: dict[str, object], finalizers: dict[str, Finalizer], settings: dict[str, object],
) -> list[str]:
    """登記只豁免 needs 與條件的禁令；整串、判決環境與工作集合仍嚴格守住。"""
    bad: list[str] = []
    for job_id, finalizer in finalizers.items():
        job = jobs.get(job_id)
        if not isinstance(job, dict):
            continue  # 缺席由唯一性判；非表由主檔回 2。
        where = f"{rel} 的收尾 job {job_id!r}"
        entry = finalizer.settings
        if set(jobs) != finalizer.shards | {job_id}:
            bad.append(f"{where} 所在 workflow 只准有收尾和分片，實際工作：{sorted(jobs)}")
        steps = job.get("steps")
        if not isinstance(steps, list):
            continue
        wrapped: list[dict[str, object]] = []
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            if step.get("if") != entry["step_if"]:
                bad.append(f"{where} 第 {index} 步的步驟 if 必須整串等於卡上登記的 {entry['step_if']!r}")
            if _wrapped_step(step, setting_text(settings, "wrapper_command")):
                wrapped.append(step)
        if len(wrapped) != 1:
            bad.append(f"{where} 必須恰好一步包抄寫員，作為判決步驟")
        else:
            env = wrapped[0].get("env")
            if not isinstance(env, dict) or env.get(entry["verdict_env_name"]) != entry["verdict_env_value"]:
                bad.append(f"{where} 的判決 env.{entry['verdict_env_name']} 必須整串等於 {entry['verdict_env_value']!r}")
    return bad


def shard_job_problems(rel: str, jobs: dict[str, object], finalizers: dict[str, Finalizer]) -> list[str]:
    """分片的身分禁令由 job_identity_problems 共用；這裡守 job 層 if。"""
    shards = frozenset(s for finalizer in finalizers.values() for s in finalizer.shards)
    return [
        f"{rel} 的分片 job {job_id!r} 不准有 job 層 if"
        for job_id in sorted(shards.intersection(jobs))
        if isinstance(jobs[job_id], dict) and "if" in cast(dict[str, object], jobs[job_id])
    ]


def upload_overwrite_problems(rel: str, jobs: dict[str, object], protected: frozenset[str]) -> list[str]:
    """受保護工作上傳不准覆蓋已有證據；未寫與 false 均可。"""
    bad: list[str] = []
    for job_id in sorted(protected.intersection(jobs)):
        job = jobs[job_id]
        if not isinstance(job, dict) or not isinstance(job.get("steps"), list):
            continue
        for index, step in enumerate(job["steps"], start=1):
            if not isinstance(step, dict):
                continue
            uses, options = step.get("uses"), step.get("with")
            if isinstance(uses, str) and uses.startswith("actions/upload-artifact@") and isinstance(options, dict):
                if options.get("overwrite") is True:
                    bad.append(f"{rel} 的受保護 job {job_id!r} 第 {index} 步不准 overwrite: true——證據不准被覆蓋")
    return bad
