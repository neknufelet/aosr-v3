"""原文骨架與逐格錄答案的共用裁判；不重新序列化、不遮字串中的數字。

分類照決策紙第 6 條：只有輸入、四則與 sqrt 的浮點留骨架；math／cmath 的超越函式
及其衍生量也要逐格比。頻率軸依紙的鍵例外照舊逐位；未知浮點保守當機器相關。
角度經 atan2／asin；路徑能量經 cmath.exp，解析指向性另經 numpy.power／expm1。
篩檢留存能量經複數取模與次方，沒有正確捨入保證，保守逐格比①級。
命令列 sin／cos 替身、Paris 積分（leggauss 特徵值）、艾林 log、交接 log2 也都用①級。
網頁曲線承接交接總能量，評分與聆聽區經 numpy.log2 等陣列函式，用③級。
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator

from tests.engine._cross_cpu_tolerance import GEOMETRIC_REL, SCORING_ABS, SCORING_REL, Cell, assert_within, cell

JsonPath = tuple[str | int, ...]
Answer = tuple[str, tuple[str, ...]]
_TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|(?P<number>-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)')
_SPATIAL = frozenset({"source_m", "receiver_m", "delay_s", "distance_m", "direction_vector",
                      "frequencies_hz"})


def machine_float(path: JsonPath, family: str) -> bool:
    """欄位傳播分類；只有已核定的輸入／基本算術欄逐字，其他（含未知）交容差。"""
    names = {part for part in path if isinstance(part, str)}
    if family == "gui":
        if names & {"timings", "frequency_hz", "center_frequency_hz", "limit"}:
            return False
        if path[:2] == ("overlay", "frequency_hz"):
            return False
        if path[0] == "frequency_plot_data" and path[2] == 0:
            return False
        return True
    if names & _SPATIAL:
        return False
    if family == "path":
        # 參考房六面散射相同，_scattering_at 直接回輸入值，不用反射加權結果。
        return "scattering_coefficient" not in names
    if family.startswith("window:"):
        return not bool(names & {"window_s", "direct_delay_s", "next_uncomputed_earliest_relative_s",
                                 "next_order_earliest_delay_s", "round_trip_delay_s", "scattering_coefficient"})
    assert family == "cli", family
    if path[0] == "path_table":
        return True
    if path[0] == "capability":
        return "frequency_hz" not in names
    if path[0] == "top":
        return not bool(names & {"crossover_lower_hz", "crossover_upper_hz", "room_volume_m3"})
    if path[0] == "points":
        return "frequency_hz" not in names
    if path[0] == "bands":
        return not bool(names & {"center_frequency_hz", "t20_s", "t30_s"})
    return True


def float_leaves(value: object, path: JsonPath = ()) -> Iterator[tuple[JsonPath, float]]:
    """依原 JSON 的鍵順序列浮點；整數與字串不進容差清單。"""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from float_leaves(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from float_leaves(item, (*path, index))
    elif isinstance(value, float):
        yield path, value


def snapshot(text: str, family: str) -> tuple[str, list[tuple[tuple[str | int, ...], float]]]:
    """用解析樹定位、原文跨度替換；浮點變整數也會改骨架，不會被當同一格。"""
    leaves = float_leaves(json.loads(text))
    spans = [match for match in _TOKEN.finditer(text)
             if match.group("number") and any(char in match.group("number") for char in ".eE")]
    parts: list[str] = []
    values: list[tuple[JsonPath, float]] = []
    start = 0
    for match, (path, value) in zip(spans, leaves, strict=True):
        if machine_float(path, family):
            parts.extend((text[start:match.start()], "<float>"))
            start = match.end()
            values.append((path, value))
    parts.append(text[start:])
    return hashlib.sha256("".join(parts).encode()).hexdigest(), values


def answer_cells(values: list[tuple[JsonPath, float]], family: str, expected: Answer) -> list[Cell]:
    """①絕對 0；命令列干涉取同列直達＋反射的答案 × ①；③絕對 5e-12。"""
    answers = dict(zip((path for path, _ in values), expected[1], strict=True))
    rel, absolute = (SCORING_REL, SCORING_ABS) if family == "gui" else (GEOMETRIC_REL, 0.0)
    cells = []
    for path, value in values:
        scale = absolute
        if family == "cli" and path[-1] == "interference_energy":
            scale = rel * (float.fromhex(answers[(*path[:-1], "direct_energy")])
                           + float.fromhex(answers[(*path[:-1], "reflected_energy")]))
        cells.append(cell(f"{family}." + ".".join(map(str, path)), value, answers[path], rel=rel, abs_=scale))
    return cells


def assert_json_control(text: str, family: str, expected: Answer) -> None:
    """先逐字比骨架，再逐格比 hex；絕對尺只取存下來的答案，不取算出值。"""
    skeleton, values = snapshot(text, family)
    assert skeleton == expected[0], f"{family}：原文骨架不同"
    assert_within(answer_cells(values, family, expected))
