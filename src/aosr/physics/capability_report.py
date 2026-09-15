"""三個入口共用的 capability 節：把「這次落在能力表哪一條」印成一行人話。

**為什麼要印範圍與輸出欄。** 只印 `status=validated` 會把整條組合的邊界蓋掉：
剛性入口的頻率軸到 300 Hz、晚期能量印的欄位比驗過的多，卻都掛著同一個
`validated`，讀的人分不出「這一跑裡的哪一段、哪幾欄」真的被量過。所以印出來的
那一行必須帶著那一條自己宣告的 `frequency_hz` 範圍與 `outputs` 清單——範圍與欄位
是表上寫的，不是這一跑算了什麼。

**只有一個印法。** 三個入口都走 :func:`capability_line`：各印各的會漂成三種格式，
而這種「哪一格蓋到哪裡」的行，格式一漂就沒人看得出差別。
"""
from __future__ import annotations

from aosr.config.capabilities import Capability


def format_frequency_range(frequency_hz: tuple[float, float]) -> str:
    """把兩個端點寫成 `[下限, 上限]`；用 `:g` 免得 226.27 這種值被四捨五入掉。"""
    lower, upper = frequency_hz
    return f"[{lower:g}, {upper:g}]"


def capability_line_from_record(
    entry: str,
    room: str,
    materials: str,
    record: Capability,
) -> str:
    """從表上那一條本人印出那一行。"""
    return capability_line(
        entry,
        room,
        materials,
        frequency_hz=record.frequency_hz,
        outputs=record.outputs,
        status=record.status,
        evidence=record.evidence,
    )


def capability_line(
    entry: str,
    room: str,
    materials: str,
    *,
    frequency_hz: tuple[float, float] | None = None,
    outputs: tuple[str, ...] = (),
    status: str | None = None,
    evidence: tuple[str, ...] = (),
) -> str:
    """回傳一行人話；`status` 是 `None` 代表這一跑沒查表。

    沒查表時狀態欄寫 `unchecked`、範圍與輸出欄寫 `none`——那不是第四種狀態，
    是「這一跑沒有憑據」，跟 `status=None` 的 `ReportCapability` 同一個意思。
    """
    if status is None or frequency_hz is None:
        return (
            f"capability entry={entry} room={room} materials={materials} "
            "frequency_hz=none outputs=none status=unchecked evidence=none"
        )
    joined = ",".join(evidence) if evidence else "none"
    return (
        f"capability entry={entry} room={room} materials={materials} "
        f"frequency_hz={format_frequency_range(frequency_hz)} "
        f"outputs={','.join(outputs)} "
        f"status={status} evidence={joined}"
    )
