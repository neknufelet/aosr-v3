"""梯度搜尋資料夾裡各檔的名字與 ``status.json`` 的格式（#1093 從 ``run`` 搬出來）。

這一份不拉 JAX：網頁讀進度、放停止記號都從這裡取，網頁行程不 import 搜尋本體（拉了 JAX，哪天誤叫一個陣列運算就會在
網頁行程裡佔走顯卡記憶體）。``run`` 與 ``endpoint_run`` 照舊從這裡取同一組名字，舊的 ``run.STATUS_FILE`` 等寫法不變。
"""
from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

PROJECT_FILE: Final = "project.json"
LEDGER_FILE: Final = "ledger.jsonl"
TEMPLATES_FILE: Final = "templates.jsonl"
STATUS_FILE: Final = "status.json"
RESULT_FILE: Final = "result.json"
STOP_FILE: Final = "stop"
ENDPOINT_FILE: Final = "endpoint.json"
EXPORTED_SCHEME_FILE: Final = "scheme-endpoint.json"

State = Literal["running", "finished", "stopped", "failed", "interrupted"]


class GradientStatus(BaseModel):
    """``status.json``：狀態、這一次子行程開始與最後寫的時間、庚二帳本記的累計求值次數、做完幾輪、帳本最後一列的值、
    迴圈收尾的停止原因（收尾才有）、換樣板帳本記的累計求值次數與做完幾輪（另兩欄，沒有換樣板帳本是 0）、
    子行程編號、失敗時的錯誤原文。庚三之前寫的狀態檔沒有那兩欄，讀成 0（報告照印；下一次子行程寫的時候補上）。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    state: State
    started_at: str
    updated_at: str
    evaluations: int = Field(ge=0, strict=True)
    rounds_done: int = Field(ge=0, strict=True)
    last_value: float | None
    stop_reason: str | None
    template_evaluations: int = Field(default=0, ge=0, strict=True)
    template_rounds_done: int = Field(default=0, ge=0, strict=True)
    pid: int = Field(ge=1, strict=True)
    message: str | None = None
