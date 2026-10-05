"""阻尼模態唯一慣例：時間 e^(+jωt)，特徵值 λ=k=ω/c（rad/m）。

p(t)∝e^(j Re(ω)t)e^(-Im(ω)t)，因此 Im ω>0 是衰減。
頻率 Re ω/(2π) 為 Hz；壓力包絡掉 10^-3（能量掉 10^-6，即 60 dB）
所需時間 T60=3 ln(10)/Im ω，單位 s。Q=Re ω/(2 Im ω) 無量綱，
是振盪頻率與兩倍包絡衰減率之比；零衰減振盪的 T60、Q 均為無限。
純虛數解有 T60，依公式的算術 Q=0，但不是房間共振；靜態解兩者皆未定義。
後兩者不進模態表。
ω=0 時阻抗項 jkCt 消失，Neumann 的常數壓力使 K 奇異，任何阻抗都
保留此靜態根。這不等於從同一零起點分岔的非零衰減根。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


CLASSIFICATION_TOL = 64 * math.ulp(1.0)
TIME_CONVENTION = "exp(+j*omega*t)"
EIGENVALUE_VARIABLE = "k=omega/c"


class ModalKind(Enum):
    """只收非負頻率、被動衰減支；其他支明確拒絕。"""

    RESONANCE = "oscillating_resonance"
    NONOSCILLATING_DECAY = "nonoscillating_decay"
    STATIC = "static"


@dataclass(frozen=True)
class ModalQuantities:
    """頻率 Hz、衰減秒數、無量綱 Q；None 表示該解不定義此量。"""

    kind: ModalKind
    frequency_hz: float
    t60_s: float | None
    q: float | None


def angular_frequency(wave_number: complex, sound_speed_m_s: float) -> complex:
    """λ=k（rad/m）換成 ω（rad/s），聲速須由呼叫端給 m/s。"""
    if not math.isfinite(sound_speed_m_s) or sound_speed_m_s <= 0:
        raise ValueError("聲速必須為正有限數")
    return sound_speed_m_s * wave_number


def classify_omega(omega: complex, *, zero_rad_s: float | None = None) -> ModalKind:
    """分類 ω（rad/s）。

    zero_rad_s：多小的實部、虛部、模長算零（rad/s，絕對值），由呼叫端依整個頻譜的尺度與殘差給——
    數值求解（有限元素）的雜訊量級隨設定不同（第 0 步：有阻尼時靜態根約 1e-12；剛性時靜態根裂成
    兩份、約 4e-5，無阻尼模態虛部雜訊最大約 1.7e-9），拿 ω 自己的大小去量會把它們誤判成增長或共振。
    不給時只容許相對浮點舍入，適用零支實部精確為零的半解析真值。
    """
    if not math.isfinite(omega.real) or not math.isfinite(omega.imag):
        raise ValueError("ω 必須有限")
    if zero_rad_s is not None and (not math.isfinite(zero_rad_s) or zero_rad_s < 0):
        raise ValueError("zero_rad_s 必須為非負有限數")
    zero = CLASSIFICATION_TOL * max(1.0, abs(omega)) if zero_rad_s is None else zero_rad_s
    if omega.real < -zero or omega.imag < -zero:
        raise ValueError("只分類非負頻率的被動支；負頻率或增長解另行處理")
    if abs(omega) <= zero:
        return ModalKind.STATIC
    if abs(omega.real) <= zero:
        return ModalKind.NONOSCILLATING_DECAY
    return ModalKind.RESONANCE


def modal_quantities(omega: complex, *, zero_rad_s: float | None = None) -> ModalQuantities:
    """按上述定義換算，分類後才決定哪些量有意義；zero_rad_s 同 classify_omega。"""
    kind = classify_omega(omega, zero_rad_s=zero_rad_s)
    if kind is ModalKind.STATIC:
        return ModalQuantities(kind, 0.0, None, None)
    decay = max(0.0, omega.imag)
    t60 = 3 * math.log(10) / decay if decay else math.inf
    if kind is ModalKind.NONOSCILLATING_DECAY:
        return ModalQuantities(kind, 0.0, t60, 0.0)
    quality = omega.real / (2 * decay) if decay else math.inf
    return ModalQuantities(kind, omega.real / (2 * math.pi), t60, quality)
