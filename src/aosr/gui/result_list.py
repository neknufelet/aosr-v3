"""結果清單只讀身分與版本，不重評結果。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from aosr.reporting.result import quality_targets_fingerprint


class ResultSummary(BaseModel):
    """畫面可直接顯示的一列；沒有可空欄位。"""

    model_config = ConfigDict(frozen=True)
    run_id: str
    scheme_id: str
    finished_text: str
    duration_text: str
    engine_text: str
    registry_text: str
    result_url: str


def summarize_result(path: Path, engine_commit: str, registry_fingerprint: str) -> ResultSummary:
    """讀一份結果的摘要；壞檔仍留一列給使用者找得到。"""
    run_id = path.stem
    base = {"run_id": run_id, "result_url": f"/results/{run_id}"}
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or result["schema_version"] != "aosr.scheme_result.v2":
            raise ValueError("結果檔格式是舊版")
        scheme_id = result["scheme"]["scheme_id"]
        duration = float(result["timings"]["total_s"])
        commit = result["engine_commit"]
        fingerprint = result["quality_targets_fingerprint"]
        if not all(isinstance(value, str) for value in (scheme_id, commit, fingerprint)):
            raise ValueError("結果檔欄位不符合現行格式")
        finished = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except (KeyError, TypeError, ValueError, OSError, UnicodeDecodeError):
        return ResultSummary(**base, scheme_id="（結果檔讀不出方案代號）",
                             finished_text="讀不出", duration_text="讀不出",
                             engine_text="讀不出", registry_text="讀不出")
    version = ("跟目前引擎同版" if commit == engine_commit else
               "跟目前引擎不同版：不能跟現在算的結果直接比較")
    registry = ("品質登記簿跟現在相同" if fingerprint == registry_fingerprint
                else "品質登記簿已換：打開會被拒收，要重算")
    return ResultSummary(**base, scheme_id=scheme_id, finished_text=finished,
                         duration_text=f"{duration:.3f} 秒",
                         engine_text=f"{commit[:7]}：{version}", registry_text=registry)


class ResultList:
    """每個網頁伺服器各自保留摘要；檔案變動就重讀。"""

    def __init__(self, engine_commit: str, quality_targets_path: Path) -> None:
        self.engine_commit = engine_commit
        self.quality_targets_path = quality_targets_path
        self.registry_fingerprint = ""
        self._cache: dict[Path, tuple[tuple[int, int], ResultSummary]] = {}

    def list(self, paths: list[Path]) -> list[ResultSummary]:
        current = quality_targets_fingerprint(self.quality_targets_path)
        if current != self.registry_fingerprint:
            self.registry_fingerprint = current
            self._cache.clear()
        found: list[ResultSummary] = []
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                continue
            key = (stat.st_mtime_ns, stat.st_size)
            cached = self._cache.get(path)
            if cached is None or cached[0] != key:
                cached = (key, summarize_result(path, self.engine_commit,
                                                self.registry_fingerprint))
                self._cache[path] = cached
            found.append(cached[1])
        return found
