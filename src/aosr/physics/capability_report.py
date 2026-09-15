"""三個入口共用的 capability 節：把「這次落在能力表哪一條」印成一行人話。

**為什麼要印範圍與輸出欄。** 只印 `status=validated` 會把整條組合的邊界蓋掉：
剛性入口的頻率軸到 300 Hz、晚期能量印的欄位比驗過的多，卻都掛著同一個
`validated`，讀的人分不出「這一跑裡的哪一段、哪幾欄」真的被量過。所以印出來的
那一行必須帶著那一條自己宣告的 `frequency_hz` 範圍與 `outputs` 清單——範圍與欄位
是表上寫的，不是這一跑算了什麼。

**只收表上那一條本人，或沒查表。** 這一支不再一格一格收 `status`／
`frequency_hz`／`outputs`／`evidence`：那條路會讓只給一半的憑據有機可乘——給了
`status="validated"` 而 `evidence` 空，就印得出「沒證據卻說驗過」；範圍與狀態都
沒給、卻給了輸出欄或證據，給了的那幾格會被安靜丟掉、印成 `unchecked`。表上
那一條是凍結的 :class:`~aosr.config.capabilities.Capability`，建構時就驗過
「validated 必有 evidence、outputs 不空、頻率範圍正數遞增、unsupported 不掛
證據」。所以這一格只收本人或 `None`（`None`＝這一跑沒查表），規則只住載入器
一個地方，上述兩條路就造不出來。

**只有一個印法。** 三個入口都走 :func:`capability_line`：各印各的會漂成三種格式，
而這種「哪一格蓋到哪裡」的行，格式一漂就沒人看得出差別。
"""
from __future__ import annotations

from aosr.config.capabilities import Capability


def format_frequency_range(frequency_hz: tuple[float, float]) -> str:
    """把兩個端點寫成 `[下限, 上限]`；用 `:g` 免得 226.27 這種值被四捨五入掉。"""
    lower, upper = frequency_hz
    return f"[{lower:g}, {upper:g}]"


def capability_line(
    entry: str,
    room: str,
    materials: str,
    record: Capability | None,
) -> str:
    """回傳一行人話；`record` 是 `None` 代表這一跑沒查表。

    沒查表時狀態欄寫 `unchecked`、範圍與輸出欄寫 `none`——那不是第四種狀態，
    是「這一跑沒有憑據」，跟 `ReportCapability.record is None` 同一個意思。
    有 record 時照它本人印，範圍與輸出欄因此一定帶著；半套的憑據進不來，
    因為那種 `Capability` 在建構時就 :class:`~pydantic.ValidationError` 了。
    """
    if record is None:
        return (
            f"capability entry={entry} room={room} materials={materials} "
            "frequency_hz=none outputs=none status=unchecked evidence=none"
        )
    joined = ",".join(record.evidence) if record.evidence else "none"
    return (
        f"capability entry={entry} room={room} materials={materials} "
        f"frequency_hz={format_frequency_range(record.frequency_hz)} "
        f"outputs={','.join(record.outputs)} "
        f"status={record.status} evidence={joined}"
    )
