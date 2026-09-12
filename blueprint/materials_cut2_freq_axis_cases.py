"""票 #134 第二刀：``freq_axis`` 那一支的 case 清單。

這一支是**上一代 ``lib/config/freq_axis.py`` 的公開行為**寫成的題目：五個模組層級的常數
（值、容器種類、陣列的 dtype 與 shape 都要凍）加上唯一那支公開函式 ``interp_to_bands``。

``interp_to_bands`` 有四件呼叫端看得到的事，每一件各自有題：**預設的來源軸是
``FREQS_HZ``**、**預設的目標是 ``CHORAS_BANDS``**、**插值在 log10(Hz) 空間做**（線性空間
會給另一個數）、**兩端往外是平的**（``np.interp`` 的夾擠，不是外插）。回傳值一定是
``list[float]``——餵整數進去也一樣，這一格由 codec 的容器與型別記號看得見。
"""
from __future__ import annotations

from typing import Final

from blueprint.materials_cut2_steps import (
    CaseEntry,
    call,
    flt,
    get,
    read,
    ref,
    seq,
    signature,
)

# 上一代那張 68 點的網格有多長，這裡不寫死一個數字：``interp_to_bands`` 要的是「跟來源軸
# 一樣長的一串值」，所以餵一條斜坡，長度由凍結常數 FREQS_HZ 的實際長度決定——由
# :func:`ramp` 在產生答案與考卷兩邊各算一次（兩邊算出來不一樣本身就是紅）。
GRID_POINTS: Final[int] = 68


def ramp(count: int) -> dict[str, object]:
    """一條斜坡（0.0、1.0、2.0 …）：值本身不重要，重要的是它跟來源軸對得上。"""
    return seq(*[flt(float(index)) for index in range(count)])


FREQ_AXIS_CASES: Final[list[CaseEntry]] = [
    {
        "id": "freq_axis.interp_to_bands.defaults",
        "steps": [call("freq_axis.interp_to_bands", "out", values=ramp(GRID_POINTS))],
        "report": ["out"],
    },
    {
        # 預設的來源軸**就是** FREQS_HZ：明著把常數餵進去，結果要跟上面那一筆一模一樣。
        "id": "freq_axis.interp_to_bands.default_source_is_freqs_hz",
        "steps": [
            get("freq_axis.FREQS_HZ", "src"),
            call(
                "freq_axis.interp_to_bands",
                "out",
                values=ramp(GRID_POINTS),
                src_freqs_hz=ref("src"),
            ),
        ],
        "report": ["out"],
    },
    {
        # 預設的目標**就是** CHORAS_BANDS，同上。
        "id": "freq_axis.interp_to_bands.default_target_is_choras_bands",
        "steps": [
            get("freq_axis.CHORAS_BANDS", "tgt"),
            call(
                "freq_axis.interp_to_bands",
                "out",
                values=ramp(GRID_POINTS),
                target_bands=ref("tgt"),
            ),
        ],
        "report": ["out"],
    },
    {
        "id": "freq_axis.interp_to_bands.custom_source_and_target",
        "steps": [
            call(
                "freq_axis.interp_to_bands",
                "out",
                values=seq(flt(1.0), flt(2.0), flt(4.0)),
                src_freqs_hz=seq(flt(100.0), flt(200.0), flt(400.0)),
                target_bands=seq(flt(150.0), flt(300.0)),
            )
        ],
        "report": ["out"],
    },
    {
        # log10 空間的正中間：100 Hz 與 10 kHz 之間的 1 kHz。線性空間會給 0.0909…，
        # log 空間給 0.5——這一格就是「它在 log 空間插值」的證據。
        "id": "freq_axis.interp_to_bands.interpolates_in_log_space",
        "steps": [
            call(
                "freq_axis.interp_to_bands",
                "out",
                values=seq(flt(0.0), flt(1.0)),
                src_freqs_hz=seq(flt(100.0), flt(10000.0)),
                target_bands=seq(flt(1000.0)),
            )
        ],
        "report": ["out"],
    },
    {
        # 兩端往外是**平的**（夾到端點值），不是外插。
        "id": "freq_axis.interp_to_bands.outside_the_source_range_is_flat",
        "steps": [
            call(
                "freq_axis.interp_to_bands",
                "out",
                values=seq(flt(1.0), flt(2.0)),
                src_freqs_hz=seq(flt(100.0), flt(200.0)),
                target_bands=seq(flt(10.0), flt(1000.0)),
            )
        ],
        "report": ["out"],
    },
    {
        # 餵整數，回來的必須是 float（codec 的 int／float 是兩種記號）。
        "id": "freq_axis.interp_to_bands.integer_input_returns_floats",
        "steps": [
            call(
                "freq_axis.interp_to_bands",
                "out",
                values=seq(1, 2),
                src_freqs_hz=seq(flt(100.0), flt(200.0)),
                target_bands=seq(flt(150.0)),
            )
        ],
        "report": ["out"],
    },
    {
        # 目標一個都沒有 → 空的一串（容器種類也記，list 變 tuple 是行為變了）。
        "id": "freq_axis.interp_to_bands.empty_target_bands_give_an_empty_list",
        "steps": [
            call(
                "freq_axis.interp_to_bands",
                "out",
                values=seq(flt(1.0), flt(2.0)),
                src_freqs_hz=seq(flt(100.0), flt(200.0)),
                target_bands=seq(),
            )
        ],
        "report": ["out"],
    },
    {
        # 值跟來源軸不一樣長 → 炸（``np.interp`` 的守門，訊息逐字比）。
        "id": "freq_axis.interp_to_bands.length_mismatch_raises",
        "steps": [
            call(
                "freq_axis.interp_to_bands",
                "out",
                values=seq(flt(1.0), flt(2.0), flt(3.0)),
                src_freqs_hz=seq(flt(100.0), flt(200.0)),
                target_bands=seq(flt(150.0)),
            )
        ],
        "report": ["out"],
    },
    {
        "id": "freq_axis.interp_to_bands.signature",
        "steps": [signature("freq_axis.interp_to_bands", "sig")],
        "report": ["sig"],
    },
    {
        # 模組層級那個已經建好的軸：它的四格（陣列、解析度記號、慣例、點數與兩端）。
        "id": "freq_axis.FREQ_AXIS.fields_and_derived_values",
        "steps": [
            get("freq_axis.FREQ_AXIS", "axis"),
            read("axis", "freqs_hz", "freqs_hz"),
            read("axis", "resolution", "resolution"),
            read("axis", "convention", "convention"),
            read("axis", "n_freq", "n_freq"),
            read("axis", "f_min", "f_min"),
            read("axis", "f_max", "f_max"),
        ],
        "report": ["freqs_hz", "resolution", "convention", "n_freq", "f_min", "f_max"],
    },
    {
        # 那個軸是從 FREQS_HZ 建的：同一個常數自己建一次，兩邊的陣列要一樣。
        "id": "freq_axis.FREQ_AXIS.is_built_from_freqs_hz",
        "steps": [
            get("freq_axis.FREQS_HZ", "src"),
            call("response.FrequencyAxis.from_hz", "axis", freqs_hz=ref("src")),
            read("axis", "freqs_hz", "freqs_hz"),
            read("axis", "resolution", "resolution"),
        ],
        "report": ["freqs_hz", "resolution"],
    },
    {
        # 身分用的那條 tuple 是 float32 量出來的，跟 FREQS_HZ 本身**不是**逐位元同一組值：
        # 這一筆把兩邊各自的值都記下來，差異才看得見（x64 開關那一格由 x64 探針另外守）。
        "id": "freq_axis.FREQS_HZ_IDENTITY.differs_from_the_raw_array_values",
        "steps": [
            get("freq_axis.FREQS_HZ_IDENTITY", "identity"),
            get("freq_axis.FREQS_HZ", "raw"),
            call("response.FrequencyAxis.from_hz", "axis", freqs_hz=ref("raw")),
            read("axis", "freqs_hz", "freqs_hz"),
        ],
        "report": ["identity", "freqs_hz"],
    },
]
