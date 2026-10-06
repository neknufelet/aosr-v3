"""網頁模態工作控制：伺服器只讀小文件，網格與診斷全部交給子行程。"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from aosr.gui.jobs import JobManager
from aosr.gui.labels import RUN_EXIT_TEXT
from aosr.gui.modal_view import REFERENCE_ROOM, REFERENCE_SECONDS, build_modal_view
from aosr.reporting.modal_diagnosis import modal_identity
from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState
from aosr.reporting.modal_lookup import (
    key_from_scheme, placement_digest, placement_matches, placement_path, read_placement, save_diagnosis,
)
from aosr.reporting.scheme import Scheme


def scheme_snapshot(path: Path) -> Scheme:
    """結果已在原頁核對；診斷輪詢只取快照，不再重評整份物理結果。"""
    document: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "scheme" not in document:
        raise ValueError("結果沒有方案快照")
    return Scheme.model_validate(document["scheme"])


class ModalJobs:
    """一把房間鑰匙同時只有一份工作；不同擺位接回後再查自己的擺位。"""

    def __init__(self, data_dir: Path, runner: tuple[str, ...], engine_commit: str, capabilities: Path) -> None:
        self.cache_dir = data_dir / "modal-cache"
        self.identity = modal_identity()
        self.manager = JobManager(data_dir / "modal-jobs", runner, engine_commit, capabilities,
            reference_seconds=REFERENCE_SECONDS, reference_label=REFERENCE_ROOM,
            result_url_template="/api/modal-jobs/{run_id}")

    def _room_label(self, scheme: Scheme) -> str:
        key = key_from_scheme(scheme)
        return key.digest if key is not None else hashlib.sha256(scheme.scene.model_dump_json().encode()).hexdigest()

    def _start(self, scheme: Scheme, *, cache_only: bool, result_id: str) -> dict[str, object]:
        extra = ("--cache-dir", str(self.cache_dir), *(("--cache-only",) if cache_only else ()))
        fields: dict[str, object] = {"placement_digest": placement_digest(scheme), "cache_only": cache_only,
                                     "source_result_id": result_id, "display_label": scheme.scheme_id}
        if cache_only:
            fields.update(running_label="查快取中", reference_seconds=15,
                reference_label=f"{REFERENCE_ROOM}，命中房間快取共 14 對實測約 14.5 秒；"
                    "沒有房間快取就直接回未計算；同一間房若正在別處計算，要等它算完")
        return self.manager.start_snapshot(scheme.model_dump_json(), self._room_label(scheme),
            extra_args=extra, job_fields=fields)

    def _response(self, scheme: Scheme, job: dict[str, object]) -> dict[str, object]:
        if job["status"] == "running":
            return {"state": "checking_cache" if job.get("cache_only") else "computing",
                    "state_text": "查快取中" if job.get("cache_only") else "計算低頻模態診斷中",
                    "job": job}
        if job["status"] == "done":
            diagnosis = ModalDiagnosis.model_validate_json(Path(str(job["result_path"])).read_text(encoding="utf-8"))
            if diagnosis.state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED:
                if not placement_matches(diagnosis, scheme, identity=self.identity):
                    raise ValueError("模態工作輸出的房間、身分或擺位與方案不符")
                # 真命令列已發布；替身或升級前工作只帶輸出檔時，同樣驗對這組方案才發布。
                path = placement_path(scheme, cache_dir=self.cache_dir, identity=self.identity)
                if path is not None and not path.exists():
                    save_diagnosis(diagnosis, path)
        else:
            if job["status"] == "stopped":
                diagnosis = ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED,
                    reason_text="已停止，沒有算完；要診斷請再按計算")
            else:
                stderr = Path(str(job["stderr_path"]))
                original = stderr.read_text(errors="replace") if stderr.is_file() else ""
                code = job.get("exit_code")
                if isinstance(code, int):
                    head = (f"模態工作結束{RUN_EXIT_TEXT.format(code=code)}但沒有診斷文件" if code == 0
                            else f"模態工作非正常結束{RUN_EXIT_TEXT.format(code=code)}")
                else:
                    head = str(job.get("process_note") or "模態工作未完成，沒有診斷文件")
                diagnosis = ModalDiagnosis(state=ModalDiagnosisState.FAILED,
                    reason_text=head + ("；錯誤輸出原文：\n" + original if original.strip() else ""))
        return {"view": build_modal_view(diagnosis, scheme), "job": job}

    def lookup(self, scheme: Scheme, *, result_id: str, calculate: bool = False,
               job_id: str | None = None) -> dict[str, object]:
        """開頁先讀擺位診斷，沒有才起只查；輪詢指定工作，避免未計算無限重開。"""
        with self.manager._lock:
            saved = read_placement(scheme, cache_dir=self.cache_dir, identity=self.identity)
            if saved is not None:
                return {"view": build_modal_view(saved, scheme)}
            if job_id is not None:
                if re.fullmatch(r"[0-9a-f]{32}", job_id) is None:
                    raise ValueError("模態工作代號無效")
                job = self.manager.get(job_id)
                if job["scheme_id"] != self._room_label(scheme):
                    raise ValueError("模態工作不是這個房間")
                if job.get("placement_digest") == placement_digest(scheme) or job["status"] == "running":
                    return self._response(scheme, job)
            running = self.manager.list_recent()["running"]
            if isinstance(running, list):
                for active in running:
                    if isinstance(active, dict) and active.get("scheme_id") == self._room_label(scheme) \
                            and not (calculate and active.get("cache_only")):
                        return self._response(scheme, active)
            return self._response(scheme, self._start(scheme, cache_only=not calculate, result_id=result_id))
