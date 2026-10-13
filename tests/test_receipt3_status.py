"""逐 job 狀態頁證明；只餵假 API 回應。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from governance.status import collect, render
from governance.status.model import CloudRun
from tests.test_build_status import fake_hub, fake_job_row


def test_all_jobs_keep_duplicate_step_names_and_run_span(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """把步驟按名稱覆蓋或只收 verify，分片那個紅就不見。"""
    red = fake_job_row("shard-a", "取碼")
    collect.as_table(collect.as_rows(red["steps"], "steps")[0], "step")
    red["steps"] = [{"name": "取碼", "conclusion": "failure", "started_at": "", "completed_at": ""}]
    red["started_at"] = "2026-09-10T01:00:00Z"
    red["completed_at"] = "2026-09-10T01:01:00Z"
    green = fake_job_row("shard-b", "取碼")
    green["started_at"] = "2026-09-10T01:00:30Z"
    green["completed_at"] = "2026-09-10T01:02:00Z"
    final = fake_job_row("verify", "收尾判決")
    final["started_at"] = "2026-09-10T01:02:10Z"
    final["completed_at"] = "2026-09-10T01:03:00Z"
    hub, _ = fake_hub(monkeypatch, tmp_path, [json.dumps({"jobs": [red, green, final]})], 100)
    seconds, steps = collect.read_run_steps(hub, 555001)
    assert {(step.job, step.name, step.conclusion) for step in steps} == {
        ("shard-a", "取碼", "failure"), ("shard-b", "取碼", "success"), ("verify", "收尾判決", "success")}
    assert seconds == collect.elapsed_seconds(str(red["started_at"]), str(final["completed_at"]))
    cloud = CloudRun(run_id=555001, conclusion="failure", status="completed", started="", finished="", head_sha="", url="", job_seconds=seconds, steps=steps)
    page = render.cloud_details(cloud)
    assert "shard-a" in page and "shard-b" in page and "收尾判決" in page
    assert "failure" in page or "紅" in page or "失敗" in page
