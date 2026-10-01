"""結果清單只讀身分與版本，不重評結果。"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from aosr.gui.jobs import ResultStatus
from aosr.gui.labels import RESULT_RUN_LABELS
from aosr.reporting.calculation_fingerprint import calculation_fingerprint, short_fingerprint
from aosr.reporting.evaluation import quality_targets_fingerprint


class ResultSummary(BaseModel):
    """畫面可直接顯示的一列；沒有可空欄位。

    calculation_text 是「計算版本」那一格的白話；calculation_detail 是滑鼠停在那一格才出現的
    技術細節（計算指紋、程式提交），不上主畫面。
    """

    model_config = ConfigDict(frozen=True)
    run_id: str
    scheme_id: str
    finished_text: str
    duration_text: str
    calculation_text: str
    calculation_detail: str
    registry_text: str
    result_url: str
    run_status: str = "none"
    status_text: str = RESULT_RUN_LABELS["none"][0]


def _format_detail(version: object) -> str:
    """格式認不得那一列的滑鼠說明：格式欄缺了就說缺，有值就附上（太長截斷）。"""
    if version is None:
        return "結果檔沒有格式欄（schema_version）"
    return f"結果檔格式欄（schema_version）：{str(version)[:60]}"


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
        version = result.get("schema_version")
        if version == "aosr.scheme_result.v2":
            return ResultSummary(**base, scheme_id=scheme_id,
                                 finished_text=datetime.fromtimestamp(path.stat().st_mtime).strftime(
                                     "%Y-%m-%d %H:%M"), duration_text="舊格式不顯示",
                                 calculation_text="舊格式（程式更新前算的），要重算", registry_text="舊格式不顯示",
                                 calculation_detail=f"結果檔格式：{version}")
        if version != "aosr.scheme_result.v3":
            # 沒有版本欄、版本認不得或比現在新：不冒充成 v2，代號照樣保住（有正常完成結果的方案不准同名改）。
            # 欄位名與它的值是技術細節，只放在滑鼠停留的說明裡。
            return ResultSummary(**base, scheme_id=scheme_id, finished_text="讀不出",
                                 duration_text="讀不出", registry_text="讀不出",
                                 calculation_text="格式認不得，要重算",
                                 calculation_detail=_format_detail(version))
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
                             calculation_text="讀不出", registry_text="讀不出",
                             calculation_detail="結果檔打不開，或欄位不符合現行格式")
    # 格子裡只講白話（跟現在的程式同不同）；計算指紋與程式提交代號收進滑鼠停留的說明。
    version = ("跟現在的程式相同" if calculation == current_fingerprint else
               "跟現在的程式不同，要重算")
    # 欄名已經是「品質登記簿」，格子裡不再重說一次。
    registry = ("跟現在相同" if fingerprint == registry_fingerprint
                else "已換：打開會被拒收，要重算")
    return ResultSummary(**base, scheme_id=scheme_id, finished_text=finished,
                         duration_text=f"{duration:.3f} 秒", calculation_text=version,
                         calculation_detail=(f"計算指紋前 12 碼 {short_fingerprint(calculation)}"
                                             f"（現在 {short_fingerprint(current_fingerprint)}）；"
                                             f"程式提交 {commit[:7]}"),
                         registry_text=registry)


class ResultList:
    """每個網頁伺服器各自保留摘要；檔案變動就重讀。"""

    def __init__(self, capabilities_path: Path, quality_targets_path: Path) -> None:
        self.capabilities_path = capabilities_path
        self.quality_targets_path = quality_targets_path
        self.registry_fingerprint = ""
        self.current_fingerprint = ""
        self._cache: dict[Path, tuple[tuple[int, int], ResultSummary]] = {}

    def list(self, paths: list[Path],
             result_status: Callable[[str], ResultStatus]) -> list[ResultSummary]:
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
            # 計算可能已經結束但產物沒動；狀態不能沿用結果檔的快取。
            status = result_status(path.stem)
            found.append(cached[1].model_copy(update={
                "run_status": status.status, "status_text": status.status_text}))
        return found
