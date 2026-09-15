"""只用標準庫獨立核對 ART 答案的吸收率、擴散場量級與精度契約。

吸收率走法向入射 ``R = (Z - rho_c) / (Z + rho_c)``、``alpha = 1 - |R|**2``；
面積平均依長方體六面牆的真實面積加權。Sabine 擴散場估計沿用凍結原稿記下的源項正規化：
donor ``lib.physics.art_kernel`` 的原對應段固定 ``power=1`` 並以總面積均勻注入；連同
``lib.config.source_reference`` 的定義給出 ``16*pi/R_room`` 的觀測尺度。

精度契約界線由呼叫端從唯一登記簿傳入，這支獨立檢查不持有副本。
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Final, Mapping

SABINE_CHARACTERIZATION_REL: Final[float] = 1e-3

_WALL_ORDER: Final[tuple[str, ...]] = ("floor", "ceiling", "x0", "xL", "y0", "yL")


@dataclass(frozen=True)
class Room:
    """長方體房間三軸尺寸（m）。"""

    lx: float
    ly: float
    lz: float


def _float32(value: float) -> float:
    """以 IEEE-754 binary32 round-to-nearest-even 收窄一格。"""
    return float(struct.unpack(">f", struct.pack(">f", value))[0])


def _float32_bits(value: float) -> int:
    """有限、非負 float 的 binary32 位元，供相鄰格距離核對。"""
    rounded = _float32(value)
    if not math.isfinite(rounded) or rounded < 0.0:
        raise ValueError(f"只接受有限非負值，收到 {value!r}")
    return int(struct.unpack(">I", struct.pack(">f", rounded))[0])


def float32_cells_apart(left: float, right: float) -> int:
    """兩個有限非負實數各自收窄成 float32 後，相隔幾個可表示值。"""
    return abs(_float32_bits(left) - _float32_bits(right))


def _complex_divide_float32(numerator: complex, denominator: complex) -> complex:
    """用 Smith 除法逐運算收窄，獨立模擬 complex64 的穩定除法。"""
    a = _float32(numerator.real)
    b = _float32(numerator.imag)
    c = _float32(denominator.real)
    d = _float32(denominator.imag)
    if abs(c) >= abs(d):
        ratio = _float32(d / c)
        scale = _float32(1.0 / _float32(c + _float32(d * ratio)))
        real = _float32(_float32(a + _float32(b * ratio)) * scale)
        imag = _float32(_float32(b - _float32(a * ratio)) * scale)
    else:
        ratio = _float32(c / d)
        scale = _float32(1.0 / _float32(d + _float32(c * ratio)))
        real = _float32(_float32(b + _float32(a * ratio)) * scale)
        imag = _float32(_float32(-a + _float32(b * ratio)) * scale)
    return complex(real, imag)


def normal_incidence_absorption_float32(impedance: complex, rho_c: float) -> float:
    """獨立算 ``1-|R|**2``，並在 donor 的 complex64／float32 運算邊界收窄。

    ``R=(Z-rho_c)/(Z+rho_c)``。標準庫沒有 complex64，故以 :mod:`struct` 明確做 binary32
    收窄；不同 CPU 的複數除法／hypot 最末位可差一格，考卷允許一個 float32 相鄰格。
    """
    real = _float32(impedance.real)
    imag = _float32(impedance.imag)
    medium = _float32(rho_c)
    reflection = _complex_divide_float32(
        complex(_float32(real - medium), imag),
        complex(_float32(real + medium), imag),
    )
    magnitude = _float32(math.hypot(reflection.real, reflection.imag))
    reflected_power = _float32(magnitude * magnitude)
    return _float32(1.0 - reflected_power)


def normal_incidence_absorption_without_square(impedance: complex, rho_c: float) -> float:
    """故意錯的控制算式 ``1-|R|``；只供考卷證明平方遺漏會被抓到。"""
    reflection = (impedance - rho_c) / (impedance + rho_c)
    return 1.0 - abs(reflection)


def wall_areas(room: Room) -> dict[str, float]:
    """六面牆名到面積（m2），順序與 donor 的 canonical wall order 相同。"""
    return {
        "floor": room.lx * room.ly,
        "ceiling": room.lx * room.ly,
        "x0": room.ly * room.lz,
        "xL": room.ly * room.lz,
        "y0": room.lx * room.lz,
        "yL": room.lx * room.lz,
    }


def area_weighted_alpha(room: Room, alpha_by_wall: Mapping[str, float]) -> float:
    """按六面實際面積算 ``sum(area*alpha)/sum(area)``，缺牆或多牆都拒絕。"""
    areas = wall_areas(room)
    if set(alpha_by_wall) != set(_WALL_ORDER):
        raise ValueError(f"牆集合不等於六面規範牆：{sorted(alpha_by_wall)!r}")
    total_area = sum(areas[wall] for wall in _WALL_ORDER)
    absorbed_area = sum(areas[wall] * alpha_by_wall[wall] for wall in _WALL_ORDER)
    return absorbed_area / total_area


def sabine_diffuse_energy(room: Room, alpha_by_wall: Mapping[str, float]) -> float:
    """以 donor 的單位功率正規化回傳 Sabine 擴散場能量 ``16*pi/R_room``。

    源項錨點為 ``art_kernel.py:407`` 的 ``power=1``、`:408` 的總面積均勻 emittance；
    `:386` 與 ``source_reference.py:16,20`` 對應這裡的 ``16*pi``。Sabine room constant
    為 ``R_room=A/(1-alpha_bar)``，其中 ``A=sum(S_i*alpha_i)``。
    """
    areas = wall_areas(room)
    alpha_bar = area_weighted_alpha(room, alpha_by_wall)
    absorbed_area = sum(areas[wall] * alpha_by_wall[wall] for wall in _WALL_ORDER)
    room_constant = absorbed_area / (1.0 - alpha_bar)
    return 16.0 * math.pi / room_constant


def art_contract_accepts(
    candidate: float,
    donor_reference: float,
    tolerance_rel: float,
) -> bool:
    """依呼叫端給定的相對容差判決 v3 精確解與上一代能量。"""
    if not math.isfinite(candidate) or not math.isfinite(donor_reference):
        return False
    return abs(candidate - donor_reference) <= tolerance_rel * abs(donor_reference)
