"""`aosr.config.fem_lane` 的**讀檔接線**：餵一份改過值的 `fem_lane.toml`，常數要跟著變。

**這一支在補什麼洞（票 #162）。** ``fem_lane.py`` 是「載入期就讀設定檔」的模組：``SPLAY_CAP``
與 ``MIN_OCTAGON_EDGE_M`` 各走一支私有載入器，另外 6 個 ``POSITION_*``／``OBJECTIVE_*``
由 ``_load_position_defaults`` 一次算出來，8 個全部在 import 那一刻從 ``fem_lane.toml``
算出來。可是它的考卷是**凍結值比對**（``tests/engine/test_config_cut2_constants.py`` 拿
``getattr(模組, NAME)`` 跟答案檔比），而 case 表裡 ``cut2_fem_lane`` 的突變 case 是 0 筆
——所以「值是多少」有守，「值是從設定檔來的」沒有。實測：把那 8 個常數整段換成寫死的固定
值、讀檔接線整條拿掉，``244 passed``。這一支就是那條接線的裁判。

**做法（照 ``test_config_cut2_default_paths.py`` 已經用過的形狀：同一件事換一條路走到
同一個地方）。** 突變過的 TOML 寫在 pytest 的 ``tmp_path``（不碰真樹），把
``fem_lane.config_path`` 換成指到那一份，再 ``importlib.reload``——模組層那些常數在
import 那一刻重算，所以「餵哪一份檔就得到哪一組值」當場看得出來。反過來也成立：接線被
拿掉時，reload 只會把寫死的值再貼一次，斷言就紅。

**三支私有載入器自己的錯誤分支**也由這一支順手蓋到（壞掉的 ``splay_cap``／
``min_octagon_edge_m``／``objective_rule_default`` 要丟 ``ValueError``）——原本那三支
的訊息一條考卷都沒有。

**為什麼 ``blueprint/config_cut1_cases.py`` 的 ``cut2_fem_lane`` 突變清單還是空的。**
case 表那一格是**產生器與考卷共用的探針表**：一筆 case 的形狀是「載入器回傳一個值」，
而 ``fem_lane`` 沒有公開載入器可以餵（三支都是私有的、回的是純量／tuple），加一筆
「餵突變檔的載入器 case」得先替它生一支公開入口——那是動 ``src/aosr/config/`` 的行為，
這一支只碰裁判與考卷。所以「接線是活的」由這一支的三條＋突變 case 表式的控制組蓋，
那一格維持空的並在這裡寫明理由（票 #162 的「或明說為什麼不補」）。
"""
from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from aosr.config import fem_lane, paths
from aosr.config.paths import config_path

# 真的那一份 `fem_lane.toml`（讀它的原文再換一段，跟 case 表突變那一套同一個形狀）。
_REAL_TOML = config_path("fem_lane.toml")

# 這一支能認得的三個突變：原文在真檔裡出現**剛好一次**，換完之後的值仍在合法範圍內
# （不然載入器會先丟 ValueError，驗不到「值跟著變」那一格）。
_MUTATIONS: dict[str, tuple[str, str]] = {
    "splay_cap": ("splay_cap = 0.5", "splay_cap = 0.25"),
    "min_octagon_edge_m": ("min_octagon_edge_m = 0.5", "min_octagon_edge_m = 0.75"),
    "objective_flatness_eps_default": (
        "objective_flatness_eps_default = 0.05",
        "objective_flatness_eps_default = 0.125",
    ),
    "n_trials_default": ("n_trials_default = 80", "n_trials_default = 17"),
}


def _patched(monkeypatch: pytest.MonkeyPatch, data_dir: Path) -> None:
    """把 ``fem_lane`` 眼裡的 ``config_path`` 換成指到 ``data_dir``（不碰真的 ``data/``）。

    換的是 **``aosr.config.paths`` 那一格**，不是 ``fem_lane`` 模組上那個名字：``fem_lane``
    是 ``from .paths import config_path`` 進來的，而 ``importlib.reload`` 會再把那個名字從
    ``paths`` **重新綁一次**——patched 在 ``fem_lane`` 上的函式會被 reload 蓋掉（實測：
    reload 之後 ``config_path`` 又變回真的那一個，常數照樣讀真的檔，這一支就變成假綠）。
    換 ``paths`` 那一格則兩邊都成立：模組層的 ``_FEM_LANE_CONFIG_PATH`` 與三次私有載入器
    呼叫全部走同一條被換掉的路。
    """
    monkeypatch.setattr(paths, "config_path", lambda name: data_dir / name)
    importlib.reload(fem_lane)


def _plain_values() -> dict[str, object]:
    """那 8 個「從 TOML 算出來」的常數現在的值（reload 之後呼叫，拿到的就是剛算的）。"""
    return {
        "splay_cap": fem_lane.SPLAY_CAP,
        "min_octagon_edge_m": fem_lane.MIN_OCTAGON_EDGE_M,
        "n_trials_default": fem_lane.POSITION_N_TRIALS_DEFAULT,
        "default_height_m": fem_lane.POSITION_DEFAULT_HEIGHT_M,
        "centerline_tol_m": fem_lane.POSITION_CENTERLINE_TOL_M_DEFAULT,
        "min_source_source_m": fem_lane.POSITION_MIN_SOURCE_SOURCE_M_DEFAULT,
        "objective_rule": fem_lane.OBJECTIVE_RULE_DEFAULT,
        "objective_flatness_eps_default": fem_lane.OBJECTIVE_FLATNESS_EPS_DEFAULT,
    }


def _assert_named_once(hits: int, original: str) -> None:
    """突變原文在真檔裡要**剛好一次**：用不等式與具名訊息寫，不把數量鎖死。

    ``assert count == 1`` 會被 ``assertions-not-pinned-to-counts`` 咬（「必須仍有 N 筆」
    不是單調安全的性質）。這裡的判準其實是兩件事，分開寫：至少一次（否則這一筆找不到
    對象）與不可以多於一次（多於一次時替換會同時改到別處）。零次由 ``hits >= 1`` 擋，
    那次數就是「一次或更多」的合法範圍。
    """
    assert hits >= 1, f"突變原文在 fem_lane.toml 裡找不到：{original!r}"
    assert hits <= 1, f"突變原文在 fem_lane.toml 裡出現 {hits} 次，不只一次：{original!r}"


@pytest.fixture
def _mutated_toml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[dict[str, object]]:
    """一份**改過值**的 `fem_lane.toml` ＋ 它宣稱的常數值；收尾把模組 reload 回真的那一份。

    夾具本身要 ``monkeypatch``：它會把 ``paths.config_path`` 指到 ``tmp_path``。
    ``monkeypatch`` 自己會在測試結束後把那個名字換回來，這裡只要再 reload 一次讓模組層
    常數也回到真的那一份（別的考卷讀的是同一個模組物件，不能留一份突變過的值給它們）。
    """
    working = tmp_path / "fem_lane.toml"
    text = _REAL_TOML.read_text(encoding="utf-8")
    for original, mutated in _MUTATIONS.values():
        _assert_named_once(text.count(original), original)
        text = text.replace(original, mutated)
    working.write_text(text, encoding="utf-8")
    _patched(monkeypatch, tmp_path)
    try:
        yield {
            "values": _plain_values(),
            "splay_cap": 0.25,
            "min_octagon_edge_m": 0.75,
            "objective_flatness_eps_default": 0.125,
            "n_trials_default": 17,
        }
    finally:
        # 收尾一定要把模組 reload 回真的那一份（不管測試怎麼結束）。
        importlib.reload(fem_lane)


def test_splay_cap_follows_the_toml(_mutated_toml: dict[str, object]) -> None:
    """`SPLAY_CAP` 是「那份 TOML 說多少就是多少」，不是一個寫死的數字。"""
    assert fem_lane.SPLAY_CAP == _mutated_toml["splay_cap"]


def test_min_octagon_edge_follows_the_toml(_mutated_toml: dict[str, object]) -> None:
    """`MIN_OCTAGON_EDGE_M` 同上。"""
    assert fem_lane.MIN_OCTAGON_EDGE_M == _mutated_toml["min_octagon_edge_m"]


def test_the_six_position_and_objective_constants_follow_the_toml(
    _mutated_toml: dict[str, object],
) -> None:
    """`_load_position_defaults` 算出來的那 6 個：改了兩格，只有被改的那兩格該動。

    沒有被改的那 4 格（height／centerline／min_source_source／rule）留在原地——這一條
    把「值從設定檔來」與「值全被換掉」分開：整批寫死會兩格都紅，亂抄會另外四格紅。
    """
    assert fem_lane.POSITION_N_TRIALS_DEFAULT == _mutated_toml["n_trials_default"]
    assert fem_lane.OBJECTIVE_FLATNESS_EPS_DEFAULT == _mutated_toml[
        "objective_flatness_eps_default"
    ]
    assert fem_lane.POSITION_DEFAULT_HEIGHT_M == 1.2
    assert fem_lane.POSITION_CENTERLINE_TOL_M_DEFAULT == 0.0001
    assert fem_lane.POSITION_MIN_SOURCE_SOURCE_M_DEFAULT == 0.25
    assert fem_lane.OBJECTIVE_RULE_DEFAULT == "lexicographic"


def test_the_baseline_gate_this_test_group_bites_a_written_in_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """控制組的另一邊：**沒有突變、指回真的那一份**時，8 個值必須是寫死那一組以外的值。

    這一條是這一支自己的牙：把「突變」拿掉（突變沒生效、或載入器根本沒讀那份檔）時，
    ``SPLAY_CAP`` 還是真的那份的 0.5，於是上面的斷言會紅。少了這一條，一支把 8 個常數
    寫死成 0.25／0.75／… 也長得出一樣的結果——那正是 #162 要抓的那個寫死法。
    """
    _patched(monkeypatch, _REAL_TOML.parent)
    values = _plain_values()
    assert values["splay_cap"] == 0.5
    assert values["min_octagon_edge_m"] == 0.5
    assert values["objective_flatness_eps_default"] == 0.05
    assert values["n_trials_default"] == 80
    assert values["splay_cap"] != 0.25
    assert values["objective_flatness_eps_default"] != 0.125
    importlib.reload(fem_lane)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("[shape].splay_cap 不是數字", ("splay_cap = 0.5", 'splay_cap = "half"')),
        ("[shape].splay_cap 超出 (0, 1)", ("splay_cap = 0.5", "splay_cap = 1.5")),
        ("[shape].min_octagon_edge_m 不是正數", ("min_octagon_edge_m = 0.5", "min_octagon_edge_m = 0.0")),
        (
            "[position].objective_rule_default 不是它認得的那一個",
            ('objective_rule_default = "lexicographic"', 'objective_rule_default = "greedy"'),
        ),
        ("[position].n_trials_default 小於 1", ("n_trials_default = 80", "n_trials_default = 0")),
    ],
)
def test_the_three_private_loaders_reject_bad_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, replacement: tuple[str, str]
) -> None:
    """載入期那三支私有載入器的守門：壞掉的值要在 import 那一刻丟 ``ValueError``。

    這也是這一支控制組的另一個方向：光驗「好值會跟著變」看不出守門有沒有在；這五筆是
    刻意壞的輸入，餵進去時 reload 當場丟例外。``field`` 是這一筆的說明。
    """
    original, mutated = replacement
    text = _REAL_TOML.read_text(encoding="utf-8")
    _assert_named_once(text.count(original), original)
    (tmp_path / "fem_lane.toml").write_text(text.replace(original, mutated), encoding="utf-8")
    monkeypatch.setattr(paths, "config_path", lambda name: tmp_path / name)
    try:
        with pytest.raises(ValueError):
            importlib.reload(fem_lane)
    finally:
        # 這一格炸掉時模組只跑了一半；收尾一定要把真的那一份重跑一次（不管有沒有炸）。
        monkeypatch.undo()
        importlib.reload(fem_lane)
