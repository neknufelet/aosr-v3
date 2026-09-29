"""結果清單只讀身分與版本，不重評結果。"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from aosr.reporting.calculation_fingerprint import calculation_fingerprint
from aosr.reporting.result import quality_targets_fingerprint


class ResultSummary(BaseModel):
    """畫面可直接顯示的一列；沒有可空欄位。"""

    model_config = ConfigDict(frozen=True)
    run_id: str
    scheme_id: str
    finished_text: str
    duration_text: str
    calculation_text: str
    registry_text: str
    result_url: str


def summarize_result(path: Path, current_fingerprint: str,
                     registry_fingerprint: str) -> ResultSummary:
    """讀一份結果的摘要；壞檔仍留一列給使用者找得到。"""
    run_id = path.stem
    base = {"run_id": run_id, "result_url": f"/results/{run_id}"}
    scheme_id: str | None = None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("結果檔不是物件")
        scheme = result.get("scheme")
        found_id = scheme.get("scheme_id") if isinstance(scheme, dict) else None
        if not isinstance(found_id, str):
            raise ValueError("結果檔讀不出方案代號")
        scheme_id = found_id
        if result.get("schema_version") != "aosr.scheme_result.v3":
            return ResultSummary(**base, scheme_id=scheme_id,
                                 finished_text="讀不出", duration_text="讀不出",
                                 calculation_text="舊格式（v2），要重算", registry_text="讀不出")
        duration = float(result["timings"]["total_s"])
        commit = result["engine_commit"]
        calculation = result["calculation_fingerprint"]
        fingerprint = result["quality_targets_fingerprint"]
        if not all(isinstance(value, str) for value in (scheme_id, commit, fingerprint)) \
                or not isinstance(calculation, str) \
                or re.fullmatch(r"calc-v1:[0-9a-f]{64}", calculation) is None:
            raise ValueError("結果檔欄位不符合現行格式")
        finished = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except (KeyError, TypeError, ValueError, OSError, UnicodeDecodeError):
        return ResultSummary(**base, scheme_id=scheme_id or "（結果檔讀不出方案代號）",
                             finished_text="讀不出", duration_text="讀不出",
                             calculation_text="讀不出", registry_text="讀不出")
    version = ("計算指紋跟現在相同" if calculation == current_fingerprint else
               "計算指紋跟現在不同：程式或設定改過，不能跟現在算的結果直接比較")
    registry = ("品質登記簿跟現在相同" if fingerprint == registry_fingerprint
                else "品質登記簿已換：打開會被拒收，要重算")
    return ResultSummary(**base, scheme_id=scheme_id, finished_text=finished,
                         duration_text=f"{duration:.3f} 秒",
                         calculation_text=f"{version}（引擎提交 {commit[:7]}）",
                         registry_text=registry)


class ResultList:
    """每個網頁伺服器各自保留摘要；檔案變動就重讀。"""

    def __init__(self, capabilities_path: Path, quality_targets_path: Path) -> None:
        self.capabilities_path = capabilities_path
        self.quality_targets_path = quality_targets_path
        self.registry_fingerprint = ""
        self.current_fingerprint = ""
        self._cache: dict[Path, tuple[tuple[int, int], ResultSummary]] = {}

    def list(self, paths: list[Path]) -> list[ResultSummary]:
        # 每次請求都現量；伺服器開著時程式改了，不能沿用上次的「現在」標籤。
        calculation = calculation_fingerprint(capabilities_path=self.capabilities_path)
        current = quality_targets_fingerprint(self.quality_targets_path)
        if current != self.registry_fingerprint or calculation != self.current_fingerprint:
            self.registry_fingerprint = current
            self.current_fingerprint = calculation
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
                cached = (key, summarize_result(path, self.current_fingerprint,
                                                self.registry_fingerprint))
                self._cache[path] = cached
            found.append(cached[1])
        return found
