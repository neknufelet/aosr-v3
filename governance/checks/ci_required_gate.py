"""必要檢查的名稱、觸發與步驟條件；判準由 ci-jobs 卡傳入。"""
from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from governance.exit_codes import ToolBroken


def job_identity_problems(
    where: str, job_id: str, job: dict[str, object], required: frozenset[str]
) -> list[str]:
    """GitHub 以 job name（未寫才用 id）比必要檢查名。"""
    raw_name = job.get("name")
    actual = raw_name if isinstance(raw_name, str) else job_id
    bad: list[str] = []
    if job_id in required and "name" in job and raw_name != job_id:
        bad.append(f"{where} 是必要檢查，name: {raw_name!r} 與 job id {job_id!r} 不同——GitHub 按 name 比必要檢查")
    if actual in required and job_id != actual:
        bad.append(f"{where} 的 name: {actual!r} 冒用必要檢查名，跳過的冒名 job 也會回報 Success")
    if job_id in required and "needs" in job:
        bad.append(f"{where} 是必要檢查卻有 needs:——前置 job 被跳過會連帶跳過它並回報 Success")
    if isinstance(raw_name, str) and "${{" in raw_name:
        # 運算式的值要到雲端才算得出來，這裡判不出它會不會等於必要檢查名，一律不准。
        bad.append(f"{where} 的 name: {raw_name!r} 是運算式——算出來可能冒用必要檢查名，這裡判不出，不准")
    if job_id in required and "strategy" in job:
        bad.append(f"{where} 是必要檢查卻有 strategy:——matrix 會把檢查名改成帶括號的另一個名字，必要檢查就對不上它")
    return bad


def step_if_problems(
    where: str, job_id: str, label: str, step: dict[str, object], allowed: object
) -> list[str]:
    """必要工作只有卡上指名的步驟與 if 值可帶條件。"""
    if "if" not in step:
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
