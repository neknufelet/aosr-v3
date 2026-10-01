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
from aosr.config.quality_targets import load_quality_targets
from aosr.reporting.calculation_fingerprint import short_fingerprint
from aosr.reporting.result import RESULT_SCHEMA_VERSION


class ResultSummary(BaseModel):
    """畫面可直接顯示的一列；沒有可空欄位。

    calculation_text 是「物理」那一格的白話；calculation_detail 是滑鼠停在那一格才出現的
    技術細節（物理身分、程式提交），不上主畫面。
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


def _format_summary(base: dict[str, str], scheme_id: str, path: Path,
                    version: object) -> ResultSummary | None:
    """版本用整數比；比現在舊才標舊格式，新版或無法辨認另給白話。"""
    if version == RESULT_SCHEMA_VERSION:
        return None
    pattern = r"aosr\.scheme_result\.v([0-9]+)"
    found = re.fullmatch(pattern, version) if isinstance(version, str) else None
    current = re.fullmatch(pattern, RESULT_SCHEMA_VERSION)
    if found is not None and current is not None and int(found[1]) < int(current[1]):
        return ResultSummary(**base, scheme_id=scheme_id,
                             finished_text=datetime.fromtimestamp(path.stat().st_mtime).strftime(
                                 "%Y-%m-%d %H:%M"), duration_text="舊格式不顯示",
                             calculation_text="舊格式（程式更新前算的），要重算", registry_text="舊格式不顯示",
                             calculation_detail=f"結果檔格式：{version}")
    # 沒有版本欄、版本認不得或比現在新：保住代號（有正常完成結果的方案不准同名改）。
    # 欄位名與它的值是技術細節，只放在滑鼠停留的說明裡。
    return ResultSummary(**base, scheme_id=scheme_id, finished_text="讀不出",
                         duration_text="讀不出", registry_text="讀不出",
                         calculation_text="格式認不得，要重算", calculation_detail=_format_detail(version))


def summarize_result(path: Path, current_physics_identity: str,
                     settings_fingerprints: dict[str, str]) -> ResultSummary:
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
        outdated = _format_summary(base, scheme_id, path, result.get("schema_version"))
        if outdated is not None:
            return outdated
        duration = float(result["timings"]["total_s"])
        commit = result["engine_commit"]
        physics = result["physics_identity"]
        program = result["program_fingerprint"]
        settings = result["purpose_settings"]["fingerprint"]
        purpose = result["scheme"]["purpose"]
        if not all(isinstance(value, str) for value in (commit, physics, program, settings, purpose)):
            raise ValueError("結果檔欄位不符合現行格式")
        if (re.fullmatch(r"phys-v1:[0-9a-f]{64}", physics) is None
                or re.fullmatch(r"calc-v1:[0-9a-f]{64}", program) is None
                or re.fullmatch(r"[0-9a-f]{64}", settings) is None):
            raise ValueError("結果檔欄位不符合現行格式")
        finished = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except (KeyError, TypeError, ValueError, OSError, UnicodeDecodeError):
        return ResultSummary(**base, scheme_id=scheme_id or "（結果檔讀不出方案代號）",
                             finished_text="讀不出", duration_text="讀不出",
                             calculation_text="讀不出", registry_text="讀不出",
                             calculation_detail="結果檔打不開，或欄位不符合現行格式")
    # 格子裡只講白話；物理身分與程式提交代號收進滑鼠停留的說明。
    version = "跟現在相同" if physics == current_physics_identity else "物理改過，要重算"
    # 欄名已經是「評分設定」，格子裡不再重說一次。
    registry = ("跟現在相同" if settings == settings_fingerprints.get(purpose)
                else "改過：打開時自動重新排名或重量，不用重算")
    return ResultSummary(**base, scheme_id=scheme_id, finished_text=finished,
                         duration_text=f"{duration:.3f} 秒", calculation_text=version,
                         calculation_detail=(f"物理身分前 12 碼 {short_fingerprint(physics)}"
                                             f"（現在 {short_fingerprint(current_physics_identity)}）；"
                                             f"程式提交 {commit[:7]}"), registry_text=registry)


class ResultList:
    """每個網頁伺服器各自保留摘要；檔案變動就重讀。"""

    def __init__(self, startup_physics_identity: str, quality_targets_path: Path) -> None:
        self.startup_physics_identity = startup_physics_identity
        self.quality_targets_path = quality_targets_path
        self.settings_fingerprints: dict[str, str] = {}
        self._cache: dict[Path, tuple[tuple[int, int], ResultSummary]] = {}

    def list(self, paths: list[Path],
             result_status: Callable[[str], ResultStatus]) -> list[ResultSummary]:
        # 原寫法：每次請求都現量；伺服器開著時程式改了，不能沿用上次的「現在」標籤。
        # 現在沿用伺服器啟動的物理身分；程式變動由伺服器過期檢查擋住，評分快照仍每次讀。
        current = {purpose.name: purpose.fingerprint
                   for purpose in load_quality_targets(self.quality_targets_path).purposes}
        if current != self.settings_fingerprints:
            self.settings_fingerprints = current
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
                cached = (key, summarize_result(path, self.startup_physics_identity,
                                                self.settings_fingerprints))
                self._cache[path] = cached
            # 計算可能已經結束但產物沒動；狀態不能沿用結果檔的快取。
            status = result_status(path.stem)
            found.append(cached[1].model_copy(update={
                "run_status": status.status, "status_text": status.status_text}))
        return found
