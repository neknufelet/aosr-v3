"""票 #187 第二段的考卷：v3 自己算參考房（一階到三階），跟凍結答案逐位比。

**這一支是引擎考卷**（住 ``tests/engine/``，規矩卡 ``green-must-be-real-green`` 的引擎籃），
它 import 新家的 ``aosr``（``aosr.geometry.shoebox`` 與 ``aosr.physics.room_paths``）。
比對的標準答案是唯讀的 ``blueprint/reference_room_answers*.json``——考卷可以 import
``blueprint``，``src/`` 不行（規矩卡 ``layers-import-downward-only`` 只准 ``src/aosr``
往下引，藍圖在它外面）。

**逐位相等不是 isclose。** 每條路徑的 ``image/dist/delay`` 的 ``hex`` 字串逐字相等
（``float.hex()`` 的十六進位是精確表示，跨平台不漂）。數量不鎖死成一個整數：identity 用
集合相等（跟獨立幾何 ``blueprint.reference_room_geometry`` 的 ``enumerate_identities``
比），條數用兩邊現算的 ``len`` 互比（規矩卡 ``assertions-not-pinned-to-counts``）。

**三份答案檔參數化。** order 1 用 ``reference_room_answers.json``、order 2/3 用
``reference_room_answers_order{2,3}.json``；每一份帶自己的 ``max_order``。反彈點另外跟
獨立幾何的 ``expand_bounces`` 逐位比（牆名、``t`` hex、反彈點三軸 hex），反彈數等於
``order_of(identity)``、``t`` 在 (0,1)、反彈點落在牆矩形內。

**不碰真環境。** 只寫 ``tmp_path``（規矩卡 ``tests-isolated-from-real-env``）：改壞的
答案檔、範例輸入檔全部在 ``tmp_path`` 生成，不寫進 repo。
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from aosr.geometry.shoebox import Point, Room, order_of
from aosr.physics.room_paths import (
    RoomInput,
    RoomPath,
    compare_paths,
    image_source_paths,
    load_room_input,
    main,
)

# 三份答案檔位置（唯讀）：repo 根往上兩層是 tests/engine，答案是 blueprint 底下。
# (max_order, 答案檔路徑) —— order 1 那份叫 reference_room_answers.json，其餘帶 order 尾綴。
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_ANSWER_FILES: tuple[tuple[int, Path], ...] = (
    (1, _REPO_ROOT / "blueprint" / "reference_room_answers.json"),
    (2, _REPO_ROOT / "blueprint" / "reference_room_answers_order2.json"),
    (3, _REPO_ROOT / "blueprint" / "reference_room_answers_order3.json"),
)


def _frozen(max_order: int) -> dict[str, object]:
    """某一階那份答案檔整包（唯讀）。"""
    path = next(p for o, p in _ANSWER_FILES if o == max_order)
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise AssertionError("答案檔不是一層表")
    return data


def _frozen_parameters(max_order: int) -> dict[str, object]:
    """某一階答案檔的 parameters（凍結、唯讀）。"""
    params = _frozen(max_order)["parameters"]
    if not isinstance(params, dict):
        raise AssertionError("parameters 不是一層表")
    return params


def _frozen_answer_paths(max_order: int) -> list[dict[str, object]]:
    """某一階答案檔的 paths（唯讀）。"""
    raw = _frozen(max_order)["paths"]
    if not isinstance(raw, list):
        raise AssertionError("paths 不是一串東西")
    return [entry for entry in raw if isinstance(entry, dict)]


# 輸入檔只吃這些欄位（答案檔 parameters 還帶 units/convention/grid_shapes 這種 donor 自己的
# 中介資料，v3 的 CLI 讀的是幾何/聲學欄位加 max_order）。
_INPUT_KEYS: tuple[str, ...] = (
    "room",
    "source_xyz_m",
    "receiver_xyz_m",
    "sound_speed_m_s",
    "max_order",
)


def _input_params(max_order: int) -> dict[str, object]:
    """從某一階凍結 parameters 摘出輸入檔欄位（回一份新的 dict，不污染凍結那份）。"""
    params = _frozen_parameters(max_order)
    return {key: params[key] for key in _INPUT_KEYS}


def _write_input(tmp_path: Path, params: dict[str, object]) -> Path:
    """把一份輸入欄位寫成範例輸入檔到 tmp_path，回路徑。"""
    path = tmp_path / "room.json"
    path.write_text(json.dumps(params, sort_keys=True), encoding="utf-8")
    return path


def _room_input(tmp_path: Path, max_order: int, **overrides: object) -> RoomInput:
    """用某一階答案檔輸入欄位寫輸入檔再餵 loader（可逐格覆寫頂層欄位）。"""
    params = _input_params(max_order)
    for key, value in overrides.items():
        params[key] = value
    return load_room_input(_write_input(tmp_path, params))


def _v3_paths(inp: RoomInput) -> list[RoomPath]:
    """用一份輸入跑 v3，回路徑。"""
    return image_source_paths(inp.room, inp.source, inp.receiver, inp.sound_speed, inp.max_order)


def _answer_identity(entry: dict[str, object]) -> tuple[int, int, int, int, int, int]:
    """答案檔一條路徑的 identity 六元組。"""
    raw = entry["identity"]
    assert isinstance(raw, list)
    return (raw[0], raw[1], raw[2], raw[3], raw[4], raw[5])


def _answer_hex_str(entry: dict[str, object], key: str) -> str:
    """答案檔 ``dist_m``／``delay_s`` 那一格的 hex 字串。"""
    cell = entry[key]
    assert isinstance(cell, dict)
    hexed = cell["hex"]
    assert isinstance(hexed, str)
    return hexed


def _answer_image_hex(entry: dict[str, object]) -> tuple[str, str, str]:
    """答案檔 ``image_xyz`` 三軸的 hex 字串。"""
    cell = entry["image_xyz"]
    assert isinstance(cell, dict)
    out: list[str] = []
    for axis in ("x", "y", "z"):
        one = cell[axis]
        assert isinstance(one, dict)
        hexed = one["hex"]
        assert isinstance(hexed, str)
        out.append(hexed)
    return (out[0], out[1], out[2])


# 凍結房間／聲源／接收點（票 #175 的合約，獨立幾何那一層不吃答案檔、只吃呼叫端餵的尺寸）。
_ROOM_DIMS: tuple[float, float, float] = (6.0, 4.0, 3.0)
_SRC_XYZ: tuple[float, float, float] = (1.5, 1.0, 1.2)
_RECV_XYZ: tuple[float, float, float] = (4.0, 3.0, 1.5)


# ── 逐位比對（三份答案檔參數化）────────────────────────────────────────────────


@pytest.mark.parametrize("max_order", [1, 2, 3])
def test_paths_match_frozen_hex_exactly(tmp_path: Path, max_order: int) -> None:
    """用答案檔 parameters 跑 v3，每一條的 image/dist/delay hex 逐字相等。"""
    v3 = _v3_paths(_room_input(tmp_path, max_order))
    answers = _frozen_answer_paths(max_order)
    assert len(v3) == len(answers), "v3 條數跟答案檔不同"

    for ours, theirs in zip(v3, answers):
        assert ours.dist_m.hex() == _answer_hex_str(theirs, "dist_m"), "dist_m.hex 不同"
        assert ours.delay_s.hex() == _answer_hex_str(theirs, "delay_s"), "delay_s.hex 不同"
        assert tuple(v.hex() for v in ours.image) == _answer_image_hex(theirs), "鏡像 hex 不同"
        assert ours.identity == _answer_identity(theirs), "identity 不同"
        assert ours.index == theirs["index"], "index 不等於答案檔"
        assert ours.order == theirs["order"], "order 不等於答案檔"


@pytest.mark.parametrize("max_order", [1, 2, 3])
def test_identity_set_matches_independent_enumeration(tmp_path: Path, max_order: int) -> None:
    """v3 的 identity 集合等於獨立幾何 ``enumerate_identities`` 的集合。"""
    from blueprint import reference_room_geometry as geom

    v3 = _v3_paths(_room_input(tmp_path, max_order))
    ours = {p.identity for p in v3}
    theirs = {ident for ident in geom.enumerate_identities(max_order)}
    assert ours == theirs, f"identity 集合不等：差 {ours ^ theirs!r}"
    # 沒有重複（集合大小等於條數）。
    assert len(v3) == len(ours), "identity 有重複"


@pytest.mark.parametrize("max_order", [1, 2, 3])
def test_order_and_index_are_independently_consistent(tmp_path: Path, max_order: int) -> None:
    """index 等於位置、order 等於 ``order_of(identity)``、反彈數等於 order。"""
    v3 = _v3_paths(_room_input(tmp_path, max_order))
    for position, p in enumerate(v3):
        assert p.index == position, f"index {p.index} 不等於位置 {position}"
        assert p.order == order_of(p.identity), f"order {p.order} ≠ order_of(identity)"
        assert len(p.bounces) == p.order, f"反彈數 {len(p.bounces)} ≠ order {p.order}"


@pytest.mark.parametrize("max_order", [1, 2, 3])
def test_bounces_in_wall_t_open_interval(tmp_path: Path, max_order: int) -> None:
    """每一條的反彈點全落在牆內、``t`` 在 (0,1)；直達路徑沒有反彈。"""
    v3 = _v3_paths(_room_input(tmp_path, max_order))
    for p in v3:
        if p.order == 0:
            assert p.bounces == (), "直達路徑不該有反彈"
            continue
        for bounce in p.bounces:
            assert bounce.in_wall is True, f"{p.identity} 的 {bounce.wall} 不在牆上"
            assert 0.0 < bounce.t < 1.0, f"{p.identity} 的 t={bounce.t!r} 不在 (0,1)"


@pytest.mark.parametrize("max_order", [2, 3])
def test_bounces_match_independent_geometry(tmp_path: Path, max_order: int) -> None:
    """逐次反彈跟獨立幾何 ``expand_bounces`` 逐位比（牆名、t hex、反彈點三軸 hex）。"""
    from blueprint import reference_room_geometry as geom

    v3 = _v3_paths(_room_input(tmp_path, max_order))
    room = geom.Room(lx=_ROOM_DIMS[0], ly=_ROOM_DIMS[1], lz=_ROOM_DIMS[2], c=343.0)

    for p in v3:
        ref = geom.expand_bounces(room, p.identity, _SRC_XYZ, _RECV_XYZ)
        assert len(ref) == len(p.bounces), f"{p.identity} 反彈數不同"
        for ours, theirs in zip(p.bounces, ref):
            assert ours.wall == theirs.wall, f"{p.identity} 牆名不同"
            assert ours.t.hex() == theirs.t.hex(), f"{p.identity} t hex 不同"
            for axis_index, axis_label in enumerate(("x", "y", "z")):
                assert (
                    ours.point[axis_index].hex() == theirs.point[axis_index].hex()
                ), f"{p.identity} 反彈點 {axis_label} hex 不同"


@pytest.mark.parametrize("max_order", [1, 2, 3])
def test_moving_receiver_changes_at_least_one_dist(tmp_path: Path, max_order: int) -> None:
    """接收點移一格，至少一條距離 hex 變。"""
    inp = _room_input(tmp_path, max_order)
    baseline = [p.dist_m.hex() for p in _v3_paths(inp)]

    moved = RoomInput(
        room=inp.room,
        source=inp.source,
        receiver=Point(inp.receiver.x + 1.0, inp.receiver.y, inp.receiver.z),
        sound_speed=inp.sound_speed,
        max_order=inp.max_order,
    )
    changed = [p.dist_m.hex() for p in _v3_paths(moved)]

    differs = [b for b, c in zip(baseline, changed) if b != c]
    assert differs, "接收點移一格後沒有一條 dist hex 變"


# ── 比對要咬得住（少一條、identity 改一位、order 錯、牆名序列）─────────────────


@pytest.mark.parametrize("max_order", [1, 2, 3])
def test_compare_all_same_for_frozen(tmp_path: Path, max_order: int) -> None:
    """凍結參數整份比對，回空差異清單。"""
    v3 = _v3_paths(_room_input(tmp_path, max_order))
    answers = _frozen_answer_paths(max_order)
    results = compare_paths(v3, answers)
    assert all(not r.diffs for r in results), f"有差異：{[r for r in results if r.diffs]}"


@pytest.mark.parametrize("max_order", [1, 2, 3])
def test_compare_reports_a_missing_path(tmp_path: Path, max_order: int) -> None:
    """把 v3 少一條餵給比對，要報條數不同，並指名少了哪一條（牆名序列）。"""
    v3 = _v3_paths(_room_input(tmp_path, max_order))
    answers = _frozen_answer_paths(max_order)

    truncated = compare_paths(v3[:-1], answers)
    missing = [r for r in truncated if r.diffs]
    assert missing, "少一條沒被抓到"
    assert any("少了這一條" in d for r in missing for d in r.diffs), "沒指名少了哪一條"


@pytest.mark.parametrize("max_order", [1, 2, 3])
def test_compare_reports_identity_flipped(tmp_path: Path, max_order: int) -> None:
    """identity 改一位，比對要報 identity 不同。"""
    v3 = _v3_paths(_room_input(tmp_path, max_order))
    answers = _frozen_answer_paths(max_order)

    victim = v3[1] if len(v3) > 1 else v3[0]
    flipped = list(victim.identity)
    flipped[1] = -flipped[1]
    flipped_identity = (flipped[0], flipped[1], flipped[2], flipped[3], flipped[4], flipped[5])
    tampered = replace(victim, identity=flipped_identity)
    tampered_list = [tampered if p is victim else p for p in v3]

    results = compare_paths(tampered_list, answers)
    assert any(r.diffs for r in results), "identity 改一位沒被抓到"


@pytest.mark.parametrize("max_order", [1, 2, 3])
def test_compare_reports_bad_order(tmp_path: Path, max_order: int) -> None:
    """order 錯，比對要報 order 不同。"""
    v3 = _v3_paths(_room_input(tmp_path, max_order))
    answers = _frozen_answer_paths(max_order)

    victim = v3[1] if len(v3) > 1 else v3[0]
    tampered = replace(victim, order=victim.order + 1)
    tampered_list = [tampered if p is victim else p for p in v3]

    results = compare_paths(tampered_list, answers)
    assert any("order 不同" in d for r in results for d in r.diffs), "order 錯沒被抓到"


@pytest.mark.parametrize("max_order", [2, 3])
def test_compare_wall_seq_derived_from_identity(tmp_path: Path, max_order: int) -> None:
    """牆名序列由 identity 推：改 identity 使序列變另一串，比對要報牆名序列不同。"""
    v3 = _v3_paths(_room_input(tmp_path, max_order))
    answers = _frozen_answer_paths(max_order)

    # 拿一條反彈路徑，把它某一軸的 sign 翻掉，使推出來的牆名序列跟著變。
    victim = next(p for p in v3 if p.order >= 1)
    flipped = list(victim.identity)
    # 翻 n 的位置：把某一軸的 n 加 1，會讓該軸的牆次換掉。
    flipped[0] = flipped[0] + 1
    flipped_identity = (flipped[0], flipped[1], flipped[2], flipped[3], flipped[4], flipped[5])
    tampered = replace(victim, identity=flipped_identity)
    tampered_list = [tampered if p is victim else p for p in v3]

    results = compare_paths(tampered_list, answers)
    assert any("牆名序列不同" in d for r in results for d in r.diffs), "牆名序列改掉沒被抓到"


# ── max_order 邊界（ValueError，不是 NotImplementedError）────────────────────


@pytest.mark.parametrize("bad_order", [0, 4, -1])
def test_image_source_paths_rejects_bad_max_order(tmp_path: Path, bad_order: int) -> None:
    """max_order 0、4、-1 是 ValueError（4 以上是上一代上限，不是還沒寫）。"""
    inp = _room_input(tmp_path, 1)
    with pytest.raises(ValueError, match="max_order"):
        image_source_paths(inp.room, inp.source, inp.receiver, inp.sound_speed, bad_order)


# ── 退化組態（反彈點打在牆的邊上 → ValueError，不靜靜少算）─────────────────────
#
# 找碴（票 #187 的洞）：一間 2×2×2 的房、聲源 (0.5,0.5,1)、接收點 (1.5,1.5,1)、
# max_order 2，會有一條路徑的反彈點恰好打在 x0 與 y0 交界的角上（(0,0,1)）；舊版
# ``expand_bounces`` 用閉區間 [0,L] 判 ``in_wall``，把這點記成 True、還靜靜少算一次反彈。
# 這一題守的是「退化要炸，不能靜靜少算」。
_EDGE_ROOM = Room(Lx=2.0, Ly=2.0, Lz=2.0)
_EDGE_SRC = Point(x=0.5, y=0.5, z=1.0)
_EDGE_RECV = Point(x=1.5, y=1.5, z=1.0)


def test_degenerate_bounce_raises_value_error() -> None:
    """反彈點打在牆的邊上（角）→ image_source_paths 丟 ValueError，附 identity 與那一點。"""
    with pytest.raises(ValueError, match="反彈點打在牆的邊上，是退化組態，上一代也明說不支援"):
        image_source_paths(_EDGE_ROOM, _EDGE_SRC, _EDGE_RECV, 343.0, max_order=2)


def test_degenerate_bounce_cli_exit_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """命令列碰到同一間退化房 → 離開碼 2，並印出退化那一句。"""
    params = {
        "room": {"Lx_m": 2.0, "Ly_m": 2.0, "Lz_m": 2.0},
        "source_xyz_m": {"x": 0.5, "y": 0.5, "z": 1.0},
        "receiver_xyz_m": {"x": 1.5, "y": 1.5, "z": 1.0},
        "sound_speed_m_s": 343.0,
        "max_order": 2,
    }
    input_path = _write_input(tmp_path, params)
    exit_code = main([str(input_path)])
    assert exit_code == 2
    assert "反彈點打在牆的邊上，是退化組態，上一代也明說不支援" in capsys.readouterr().out


# ── 載入器 ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("max_order", [1, 2, 3])
def test_loader_reads_max_order_and_passes_down(tmp_path: Path, max_order: int) -> None:
    """max_order 從輸入檔讀進來，餵給 image_source_paths。"""
    inp = _room_input(tmp_path, max_order)
    assert inp.max_order == max_order
    assert len(_v3_paths(inp)) == len(_frozen_answer_paths(max_order))


def test_loader_rejects_missing_field(tmp_path: Path) -> None:
    """缺欄位（sound_speed_m_s）要丟 ValueError，訊息講「缺欄位」不是「不是一個數」。"""
    params = _input_params(1)
    params.pop("sound_speed_m_s")
    with pytest.raises(ValueError, match="缺欄位"):
        load_room_input(_write_input(tmp_path, params))


def test_loader_rejects_missing_xyz_axis(tmp_path: Path) -> None:
    """缺 source_xyz_m.z 要講「缺欄位」，不是「不是一個數」。"""
    params = _input_params(1)
    source = params["source_xyz_m"]
    if isinstance(source, dict):
        source.pop("z")
    with pytest.raises(ValueError, match="缺欄位"):
        load_room_input(_write_input(tmp_path, params))


def test_loader_rejects_negative(tmp_path: Path) -> None:
    """負數（room.Lz_m）要丟 ValueError 指名哪一格。"""
    params = _input_params(1)
    room = params["room"]
    if isinstance(room, dict):
        room["Lz_m"] = -3.0
    with pytest.raises(ValueError, match="Lz_m"):
        load_room_input(_write_input(tmp_path, params))


def test_loader_rejects_zero_sound_speed(tmp_path: Path) -> None:
    """聲速 0 要丟 ValueError 指名哪一格。"""
    params = _input_params(1)
    params["sound_speed_m_s"] = 0.0
    with pytest.raises(ValueError, match="sound_speed_m_s"):
        load_room_input(_write_input(tmp_path, params))


def test_loader_rejects_zero_room_side(tmp_path: Path) -> None:
    """房間三邊其中一邊 0 要丟 ValueError 指名哪一格。"""
    params = _input_params(1)
    room = params["room"]
    if isinstance(room, dict):
        room["Lx_m"] = 0.0
    with pytest.raises(ValueError, match="Lx_m"):
        load_room_input(_write_input(tmp_path, params))


def test_loader_rejects_extra_field(tmp_path: Path) -> None:
    """多餘欄位（寫錯字）要丟 ValueError，並列出多的欄位名。"""
    params = _input_params(1)
    params["sound_speed_ms"] = 343.0  # 寫錯字：多打一個 s 掉（正確是 sound_speed_m_s）
    with pytest.raises(ValueError, match="sound_speed_ms"):
        load_room_input(_write_input(tmp_path, params))


def test_loader_rejects_missing_path(tmp_path: Path) -> None:
    """路徑不存在要炸（OSError 的那一類）。"""
    with pytest.raises((ValueError, OSError, FileNotFoundError)):
        load_room_input(tmp_path / "does_not_exist.json")


# ── 命令列 ──────────────────────────────────────────────────────────────────


def _write_answer(tmp_path: Path, max_order: int, mutate: bool, name: str) -> Path:
    """把某一階答案檔整份抄到 tmp_path（選擇性弄壞一格 hex）。"""
    root = _frozen(max_order)
    if mutate:
        paths = root["paths"]
        if isinstance(paths, list) and paths:
            first = paths[0]
            if isinstance(first, dict):
                raw = first["dist_m"]["hex"]
                bumped = math.nextafter(float.fromhex(raw), math.inf)
                first["dist_m"]["hex"] = bumped.hex()
    path = tmp_path / name
    path.write_text(json.dumps(root, sort_keys=True), encoding="utf-8")
    return path


@pytest.mark.parametrize("max_order", [1, 2, 3])
def test_cli_compare_frozen_exit_zero(tmp_path: Path, max_order: int) -> None:
    """--compare 對凍結參數離開碼 0。"""
    answer_path = next(p for o, p in _ANSWER_FILES if o == max_order)
    input_path = _write_input(tmp_path, _input_params(max_order))
    exit_code = main([str(input_path), "--compare", str(answer_path)])
    assert exit_code == 0


@pytest.mark.parametrize("max_order", [1, 2, 3])
def test_cli_compare_tampered_exit_one(tmp_path: Path, max_order: int) -> None:
    """--compare 對一份改過 hex 的答案檔副本離開碼 1。"""
    input_path = _write_input(tmp_path, _input_params(max_order))
    tampered = _write_answer(tmp_path, max_order, mutate=True, name="tampered.json")
    exit_code = main([str(input_path), "--compare", str(tampered)])
    assert exit_code == 1


def test_cli_missing_input_exit_two(tmp_path: Path) -> None:
    """讀不到輸入檔離開碼 2。"""
    exit_code = main([str(tmp_path / "nope.json")])
    assert exit_code == 2


def test_cli_zero_sound_speed_exit_two(tmp_path: Path) -> None:
    """聲速 0 的輸入檔 → 離開碼 2（ValueError → main 裡回 2）。"""
    params = _input_params(1)
    params["sound_speed_m_s"] = 0.0
    input_path = _write_input(tmp_path, params)
    exit_code = main([str(input_path)])
    assert exit_code == 2


def test_cli_bad_max_order_exit_two(tmp_path: Path) -> None:
    """max_order 0 的輸入檔 → 離開碼 2。"""
    params = _input_params(1)
    params["max_order"] = 0
    input_path = _write_input(tmp_path, params)
    exit_code = main([str(input_path)])
    assert exit_code == 2


# ── 裝進環境（不靠 pytest 的 pythonpath） ─────────────────────────────────────


def test_importable_without_pytest_pythonpath(tmp_path: Path) -> None:
    """新家 ``aosr`` 不靠 pytest 的 ``pythonpath = ["src"]`` 也 import 得到。

    起一支 Python 直譯器子程序（``sys.executable`` 是直譯器不是版控工具，
    ``tests-isolated-from-real-env`` 只咬 git／gh 這種版控工具），``cwd`` 指到
    ``tmp_path``（不寫真 repo），**複製 ``os.environ`` 但刪掉 ``PYTHONPATH``**——這樣
    子程序解析 ``import aosr`` 就只靠「editable 裝進環境」那一條路（``[tool.uv]
    package = true``），pytest 在設定裡開的那條 ``pythonpath`` 傳不進去。印出來的
    ``aosr.__file__`` 必須落在 repo 的 ``src/aosr/`` 底下（不是 site-packages 的複本）。

    這一題守的是「裝進環境」這個功能：把 ``[tool.uv] package`` 翻回 false、``uv sync``
    一次，這題就紅（``import aosr`` 直接 ModuleNotFoundError）。
    """
    src_aosr = _REPO_ROOT / "src" / "aosr"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-c", "import aosr, aosr.physics.room_paths; print(aosr.__file__)"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, f"子程序 import 失敗：{completed.stderr}"
    printed = completed.stdout.strip()
    installed = Path(printed).resolve()
    assert installed.is_relative_to(src_aosr.resolve()), (
        f"aosr.__file__ = {printed!r} 不在 repo 的 src/aosr/ 底下（{src_aosr}）"
    )
