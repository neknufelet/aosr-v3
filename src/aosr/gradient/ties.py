"""綁定表（#1206，左右對稱的計算那半邊）：成對的兩片 surface 從開跑到結果檔，厚度與空腔逐位相等。

每一片保留自己的 θ 格，另加一張表記「哪兩片成對」；表是空的時，搜尋的每一步都不經過這裡，跟加這張表之前逐位相同。
存法固定：一對裡兩個代號照字串排、對與對也照字串排（``A=B`` 與 ``B=A`` 是同一張表、同一個資料夾名）。
成對的兩片算一個單位，代表是**面序在前**的那一片（設計清單的順序；候選代號、換樣板代號都用它）。

存檔缺這一欄一律是「不綁」：預設值只准是空表，「預設打開」是網頁算新計畫那一層的事，不住在任何存檔模型裡。
這支檔不載入 JAX（陣列運算只用傳進來的陣列自己的方法與 numpy），網頁行程也讀得了。
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Annotated

from pydantic import AfterValidator

from aosr.gradient.boundary import patch_key

if TYPE_CHECKING:
    from aosr.gradient.material_gradient import SurfaceDesign

TiePairs = tuple[tuple[str, str], ...]


def checked_ties(value: Iterable[Sequence[str]]) -> TiePairs:
    """成對的片代號：每一對恰兩片、代號認得、不准自己配自己、同一對不准給兩次、一片只能在一對裡。回固定存法。"""
    seen: dict[str, tuple[str, str]] = {}
    for item in value:
        names = sorted(str(name) for name in item)
        if len(names) != 2:
            raise ValueError(f"綁定要是兩片一對，收到 {list(item)}")
        pair = (names[0], names[1])
        for name in pair:
            patch_key(name)
        if pair[0] == pair[1]:
            raise ValueError(f"綁定 {pair[0]}={pair[1]}：兩片相同，不能自己配自己")
        for name in pair:
            other = seen.get(name)
            if other == pair:
                raise ValueError(f"綁定 {pair[0]}={pair[1]} 給了兩次")
            if other is not None:
                raise ValueError(f"片 {name} 同時在兩對綁定裡（{other[0]}={other[1]}、{pair[0]}={pair[1]}）：一片只能跟一片成對")
            seen[name] = pair
    return tuple(sorted(set(seen.values())))


Ties = Annotated[TiePairs, AfterValidator(checked_ties)]


def parse_ties(values: Sequence[str]) -> TiePairs:
    """可重複的 ``--tie 片=片``。格式錯、自己配自己、重複、一片在兩對裡都拒收。"""
    pairs = []
    for item in values:
        names = item.split("=")
        if len(names) != 2 or not all(name.strip() for name in names):
            raise ValueError(f"--tie 要寫成 片=片（成對兩片的面片代號，例如 y0@a=yL@a），收到 {item!r}")
        pairs.append((names[0].strip(), names[1].strip()))
    return checked_ties(pairs)


def check_ties_against_start(ties: Iterable[Sequence[str]], designs: Sequence[SurfaceDesign],
                             allowed: Mapping[str, Sequence[str]]) -> None:
    """開跑前把綁定表跟起跑配置對一次，點名拒收，不悄悄拿掉綁定。``allowed`` 是展開到每一片的准用清單。

    兩片都要在這一場的面片清單裡、都要是交給搜尋的片（固定面不是）；成對兩片的家族、起跑樣板、θ 上下界、
    深度上限（家族與個案取嚴之後的總深度，以及個案上限本身）、裝法、准用清單要完全相同。
    """
    by_id = {design.surface_id: design for design in designs}
    for first, second in checked_ties(ties):
        label = f"綁定 {first}={second}"
        for name in (first, second):
            if name not in by_id:
                raise ValueError(f"{label}：片 {name} 不在這一場的面片清單 {list(by_id)}")
            if by_id[name].fixed_material_id is not None:
                raise ValueError(f"{label}：{name} 是固定面，不是交給搜尋的片，不能綁")
        left, right = by_id[first], by_id[second]
        compared: tuple[tuple[str, object, object], ...] = (
            ("家族", left.family_id, right.family_id), ("起跑樣板", left.preset_id, right.preset_id),
            ("θ 上下界", (left.lower, left.upper), (right.lower, right.upper)),
            ("深度上限", (left.max_total_depth_m, left.max_depth_m), (right.max_total_depth_m, right.max_depth_m)),
            ("裝法", left.mount, right.mount), ("准用清單", allowed.get(first), allowed.get(second)))
        for words, mine, theirs in compared:
            if mine != theirs:
                raise ValueError(f"{label}：{words}不同（{first} 是 {mine!r}，{second} 是 {theirs!r}）；"
                                 "成對兩片要完全相同，不會自動拿掉綁定")


def check_ties_in_surfaces(ties: Iterable[Sequence[str]], names: Sequence[str], what: str) -> None:
    """帳本標頭用：綁定表的每一片都要在那本帳的面裡。"""
    missing = sorted({name for pair in ties for name in pair if name not in names})
    if missing:
        raise ValueError(f"綁定表的片 {missing} 不在{what}的面裡")


def tie_followers(ties: Iterable[Sequence[str]], names: Sequence[str]) -> dict[str, str]:
    """成對裡面序在後的那一片 → 面序在前的那一片（``names`` 是面序）；表是空的回空表。"""
    order = {name: index for index, name in enumerate(names)}
    check_ties_in_surfaces(ties, names, "這一場")
    found: dict[str, str] = {}
    for pair in ties:
        leader, follower = sorted(pair, key=lambda name: order[name])
        found[follower] = leader
    return found
