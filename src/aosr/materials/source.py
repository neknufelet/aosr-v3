"""材料的出身資訊——刻意放在可微分的那個 pytree **外面**。

``MaterialResponse`` 是求解器在執行期扛著跑的東西，愈小愈好；「這份阻抗是從哪來的」
（解析式、TMM 算的、還是量到的吸收係數）是帳，不是運算元。兩件事混在同一個容器裡，
每一次 ``tree_map`` 都要拖著一串字串走。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from aosr.materials.response import MaterialResponse

SourceKind = Literal["analytic", "computed_tmm", "measured_alpha"]
"""一份材料回應是怎麼來的。"""

SOURCE_KINDS: frozenset[str] = frozenset(get_args(SourceKind))


@dataclass(frozen=True)
class MaterialEntry:
    """註冊表裡的一筆：回應本身，加上它的出身。

    ``frozen=True`` 是行為不是裝飾：一筆已經登記的出身被就地改掉，帳就無聲地變了
    （改的人不會留下任何痕跡）。要換就換一筆新的、走註冊表的覆寫那條路。
    """

    response: MaterialResponse
    source_kind: SourceKind
    source_ref: str | None = None


def is_aosr_material_id(material_id: str) -> bool:
    """這個材料 id 有沒有照 AOSR 的命名慣例（軟性的，不是強制）。"""
    return material_id.startswith("aosr:")
