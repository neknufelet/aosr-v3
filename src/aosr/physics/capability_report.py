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


def describe_missing_half(
    status: str | None,
    frequency_hz: tuple[float, float] | None,
) -> str:
    """報錯時說清楚是哪一格漏了；兩格都給不可能走到這裡。

    `frequency_hz` 收到的是還沒窄化的可選型別，所以這裡自己判一次；不然呼叫端
    傳進來的 `tuple | None` 過不了型別警衛（那是這一跑真的可能沒值的格子）。
    """
    if status is None:
        shown = "none" if frequency_hz is None else format_frequency_range(frequency_hz)
        return f"frequency_hz={shown}，少了 status"
    return f"status={status}，少了 frequency_hz"


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

    `status` 與 `frequency_hz` 兩格要嘛都是 `None`（沒查表），要嘛都有值：只給
    一格等於「只查了一半」，以前會安靜印成 `unchecked`，讓 `status="validated"`
    這種半套的憑據被降級吞掉；現在直接報錯。有值時 `outputs` 不准是空的——
    表上每一條都有輸出欄，空的代表那一條沒說它蓋到哪幾欄。
    """
    if (status is None) != (frequency_hz is None):
        raise ValueError(
            f"capability entry={entry} room={room} materials={materials}："
            "status 與 frequency_hz 要嘛都給、要嘛都不給（兩個都是 None＝這一跑沒查表）；"
            f"這一跑只給了{describe_missing_half(status, frequency_hz)}"
        )
    if status is None:
        return (
            f"capability entry={entry} room={room} materials={materials} "
            "frequency_hz=none outputs=none status=unchecked evidence=none"
        )
    if frequency_hz is None:
        raise ValueError(
            f"capability entry={entry} room={room} materials={materials}："
            f"status={status}，但 frequency_hz 是空的；要嘛兩格都給、要嘛都不給"
        )
    if not outputs:
        raise ValueError(
            f"capability entry={entry} room={room} materials={materials}："
            f"status={status} 有頻率範圍 frequency_hz={format_frequency_range(frequency_hz)}，"
            "但 outputs 是空的；表上每一條都有輸出欄，空的代表這一條沒說它蓋到哪幾欄"
        )
    joined = ",".join(evidence) if evidence else "none"
    return (
        f"capability entry={entry} room={room} materials={materials} "
        f"frequency_hz={format_frequency_range(frequency_hz)} "
        f"outputs={','.join(outputs)} "
        f"status={status} evidence={joined}"
    )
