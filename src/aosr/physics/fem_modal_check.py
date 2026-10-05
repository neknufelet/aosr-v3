"""移位圓聯集的保證高度與原始個數對照；不判過關、不補算。

最近 requested 個 k 根的開圓內不可能另有未返回根（收斂且最近根
選取完整時）；ω=ck、c>0 使圓心與半徑都乘 c。半徑須由全部原始
返回根算，包含負頻鏡像 −conj(ω)、靜態、純衰減及超出頻率上限的根；
它們都佔名額。邊界同距離的重根可能只返回部分，故保證用嚴格小於。
這是離散矩陣頻譜的幾何保證，不是連續物理頻譜的離散化精度契約。
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

from aosr.physics.modal_convention import ModalKind, modal_quantities

if TYPE_CHECKING:
    from aosr.physics.fem_modal import FemModalSpectrum, FemMode, ModalShift


class DecayOrigin(Enum):
    """非振盪根的延拓來源；未知不能靠排序冒充零支或其他過阻尼支。"""

    UNCONFIRMED = "unconfirmed"
    ZERO_MODE_CONTINUATION = "zero_mode_continuation"
    OVERDAMPED = "overdamped"


@dataclass(frozen=True)
class ModalCountBand:
    """頻段採 (下界, 上界]，共振不含零支；估計差值保留正負、不作裁判。"""

    lower_hz: float
    upper_hz: float
    found_resonances: int
    weyl_estimate: float
    found_minus_weyl: float
    rigid_reference_count: int | None
    found_minus_rigid: int | None


@dataclass(frozen=True)
class FemModalCheck:
    """高度 rad/s、最低 T60 秒與解種類個數；未給延拓證據就記來源未確認。"""

    guaranteed_decay_rate_rad_s: float
    guaranteed_min_t60_s: float
    count_bands: tuple[ModalCountBand, ...]
    static_count: int
    zero_mode_continuation_count: int
    overdamped_count: int
    unconfirmed_decay_count: int
    returned_above_limit_count: int
    weyl_terms: str = "Neumann: volume*k^3/(6*pi^2) + surface*k^2/(16*pi); no edge term"


def guaranteed_decay_height(shifts: Sequence[ModalShift], frequency_max_hz: float) -> float:
    """min_x max_i sqrt(max(0,R_i²−(x−s_i)²))，x∈[0,2π fmax]。

    每個平方截面是同曲率凹拋物線；差為直線。上包絡在兩交點間
    屬於其中一條凹拋物線（或零），最小值必在區間端點。列舉所有
    成對交點與全域兩端即可，含未覆蓋空隙與互含圓；不用網格採樣。
    """
    if not math.isfinite(frequency_max_hz) or frequency_max_hz <= 0:
        raise ValueError("頻率上限必須為正有限數")
    disks = [(2 * math.pi * shift.frequency_hz, shift.radius_rad_s) for shift in shifts]
    if any(not math.isfinite(s) or not math.isfinite(r) or r < 0 for s, r in disks):
        raise ValueError("移位圓心必須有限、半徑必須非負有限")
    cap = 2 * math.pi * frequency_max_hz
    candidates = [0.0, cap]
    for i, (s, r) in enumerate(disks):
        for t, q in disks[i + 1:]:
            if s != t:
                x = (s + t) / 2 + (q - r) * (q + r) / (2 * (s - t))
                if 0 < x < cap:
                    candidates.append(x)
    # 乘 (R−距離)(R+距離) 避免在圓周附近平方相減。
    return math.sqrt(min(max((max(0.0, (r - abs(x - s)) * (r + abs(x - s)))
                             for s, r in disks), default=0.0) for x in candidates))


def count_bands(rows: Sequence[FemMode], edges_hz: Sequence[float],
                weyl_cumulative: Sequence[float],
                rigid_reference_frequencies_hz: Sequence[float] | None = None) -> tuple[ModalCountBand, ...]:
    """只列原始數字；Weyl 由求解器同一份網格公式供給，不加入邊項。"""
    edges = tuple(float(edge) for edge in edges_hz)
    if (len(edges) < 2 or edges[0] != 0 or len(weyl_cumulative) != len(edges)
            or any(not math.isfinite(x) or x < 0 for x in edges)
            or any(a >= b for a, b in zip(edges, edges[1:]))):
        raise ValueError("頻段界線必須從零嚴格遞增，累積估計逐一對應")
    if rigid_reference_frequencies_hz is not None and any(
            not math.isfinite(x) or x < 0 for x in rigid_reference_frequencies_hz):
        raise ValueError("剛性參考頻率必須非負有限")
    bands = []
    for i, (lower, upper) in enumerate(zip(edges, edges[1:])):
        found = int(sum(row.kind is ModalKind.RESONANCE and lower < row.frequency_hz <= upper for row in rows))
        estimate = weyl_cumulative[i + 1] - weyl_cumulative[i]
        rigid = (None if rigid_reference_frequencies_hz is None else
                 int(sum(lower < f <= upper for f in rigid_reference_frequencies_hz)))
        bands.append(ModalCountBand(lower, upper, found, estimate, found - estimate,
                                    rigid, None if rigid is None else found - rigid))
    return tuple(bands)


def build_check(rows: Sequence[FemMode], shifts: Sequence[ModalShift], cap: float,
                bands: tuple[ModalCountBand, ...], returned_above_limit: int) -> FemModalCheck:
    """只作幾何與計數，不求新根。高度為零時回報最低 T60 無限（無正高度保證）。"""
    height = guaranteed_decay_height(shifts, cap)
    decays = [row for row in rows if row.kind is ModalKind.NONOSCILLATING_DECAY]
    t60 = modal_quantities(complex(0, height), zero_rad_s=0.0).t60_s if height else math.inf
    assert t60 is not None
    return FemModalCheck(height, t60, bands,
                         sum(row.kind is ModalKind.STATIC for row in rows),
                         sum(row.decay_origin is DecayOrigin.ZERO_MODE_CONTINUATION for row in decays),
                         sum(row.decay_origin is DecayOrigin.OVERDAMPED for row in decays),
                         sum(row.decay_origin is DecayOrigin.UNCONFIRMED for row in decays),
                         returned_above_limit)


def mark_decay_origins(spectrum: FemModalSpectrum,
                       continuation_indices: Mapping[int, tuple[int, int, int]]) -> FemModalSpectrum:
    """呼叫端用延拓證據完成一對一配對後，傳 solution 列號→剛性起點指標。

    本函式不從 ω 排序猜來源、不作半解析配對或新求解。未提供的純衰減
    列維持未確認；零支重複、非法列號及非純衰減列明確拒收；非零剛性指標的正負
    鏡像可在過阻尼後成為兩個不同純虛根，因此允許共用非零指標。
    """
    if spectrum.check is None:
        raise ValueError("頻譜缺少自檢結果")
    existing = {i: row.continuation_index for i, row in enumerate(spectrum.solutions)
                if row.continuation_index is not None and i not in continuation_indices}
    labels = (*existing.values(), *continuation_indices.values())
    if labels.count((0, 0, 0)) > 1 or any(len(index) != 3 or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in index) for index in labels):
        raise ValueError("延拓指標須為三個非負整數，零支只能一列")
    rows = list(spectrum.solutions)
    for row_number, index in continuation_indices.items():
        if (isinstance(row_number, bool) or not isinstance(row_number, int)
                or not 0 <= row_number < len(rows)
                or rows[row_number].kind is not ModalKind.NONOSCILLATING_DECAY):
            raise ValueError("來源證據必須指向已找到的非振盪衰減列")
        origin = DecayOrigin.ZERO_MODE_CONTINUATION if index == (0, 0, 0) else DecayOrigin.OVERDAMPED
        rows[row_number] = replace(rows[row_number], decay_origin=origin, continuation_index=index)
    check = replace(spectrum.check,
                    zero_mode_continuation_count=sum(r.decay_origin is DecayOrigin.ZERO_MODE_CONTINUATION for r in rows),
                    overdamped_count=sum(r.decay_origin is DecayOrigin.OVERDAMPED for r in rows),
                    unconfirmed_decay_count=sum(r.decay_origin is DecayOrigin.UNCONFIRMED for r in rows))
    return replace(spectrum, solutions=tuple(rows), check=check)


def mark_zero_mode_continuation(spectrum: FemModalSpectrum, solution_index: int) -> FemModalSpectrum:
    """呼叫端證實唯一剛性常數零支的非零延拓根後，分類全部純衰減解。

    適用單一連通房間、剛性零特徵空間只有常數的問題。solution_index
    必須由延拓與一對一根／形狀配對提供，不能按衰減排序猜。唯一零支
    已辨認，其他非零純衰減根就來自非零剛性模態（含正負鏡像）；按排除法
    標過阻尼，未確認的具體非零指標仍為 None。若零支未解或未找到，
    不得調用本函式，保留 UNCONFIRMED；本函式不另算任何根。
    """
    marked = mark_decay_origins(spectrum, {solution_index: (0, 0, 0)})
    assert marked.check is not None
    rows = tuple(replace(row, decay_origin=DecayOrigin.OVERDAMPED)
                 if row.kind is ModalKind.NONOSCILLATING_DECAY
                 and row.decay_origin is DecayOrigin.UNCONFIRMED else row for row in marked.solutions)
    check = replace(marked.check, unconfirmed_decay_count=0,
                    overdamped_count=sum(row.decay_origin is DecayOrigin.OVERDAMPED for row in rows))
    return replace(marked, solutions=rows, check=check)
