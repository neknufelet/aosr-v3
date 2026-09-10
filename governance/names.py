"""名字綁定解析：一份 ``.py`` 裡的某個名字節點，回指哪一個 ``(模組, 名字)``。

**為什麼要有這個檔。** 三張卡各自在比名字，各自有同一族的洞：

* ``type-guard`` 第②層——``from typing import Any as X`` 之後寫 ``def f() -> X``。
  只比字面的版本 hits=0（PR #46 實測），已在那個 PR 修掉，解析的程式碼當時住在那支檢查裡。
* ``style-guard`` 第①條——``p = print``、``import builtins`` 之後 ``builtins.print(...)``、
  ``from builtins import print as w`` 之後 ``w(...)``，三種實測都 hits=0（issue #47）。
* ``uv-single-entrypoint`` 第②條——``import sys as s`` 之後 ``s.path.insert(...)``、
  ``from sys import path`` 之後 ``path.insert(...)``、``p = sys.path`` 之後 ``p.insert(...)``，
  三種實測也都 hits=0（同一張 issue）。

同一族的洞散在三支程式裡就會各修一次、各漏一種。這個檔把「一個名字回指哪裡」抽成一份，
三張卡都走它；哪一種寫法追不到，是這一份的事，不是三支程式各自的事。

**回答的問題只有一個。** 給一個運算式節點，回一個 :class:`Origin`
——``(模組, 那個模組裡的名字)``；``name`` 是空字串就代表「這個名字就是那個模組本身」。
``sys`` 是 ``Origin("sys", "")``、``sys.path`` 是 ``Origin("sys", "path")``、
``print`` 是 ``Origin("builtins", "print")``。追不到就回 ``None``——**追不到不等於沒事**，
呼叫端要自己決定那時候怎麼辦（三張卡的做法都是保留原本的字面判準當地板，見下）。

**追得到的四種綁定**

1. ``import x``／``import x.y``——本地名字是 ``x``（點號寫法綁的是最上層那一段），
   指向模組本身。
2. ``import x as y``／``import x.y as z``——本地名字是 ``y``／``z``，指向那個完整模組名。
3. ``from x import a``／``from x import a as b``——本地名字指向 ``(x, a)``。
   相對 import（``from . import a``）的模組名前面帶點，照實記，不猜它解析到哪個套件。
4. **模組層的賦值別名**，接力幾手都追：``p = print``、``MyAny = Any``、``q = p``、
   ``MyAny: TypeAlias = Any``。收到不再長大為止，沒有寫死的層數上限。

**``from x import *`` 視為不透明。** 那一行之後，哪些名字被綁進來、綁成什麼，靜態上看不出來。
這裡不猜：那些名字照樣回 ``None``，星號 import 本身另外記在 :attr:`Names.stars` 上，
由呼叫端決定要不要當違規（``type-guard`` 對 typing 那一族就是直接判紅——量不到就不准當乾淨）。

**``assume``：不必 import 就成立的名字。** ``print`` 是內建、``Any`` 在一份檔裡叫這個名字
只可能是那件事、``sys`` 這個名字寫出來就是那個模組。呼叫端把這種名字用 ``assume`` 交進來，
解析時當成已經綁好。真的有 import 的話 import 贏（``from mymod import Any`` 之後 ``Any``
回指的是 ``mymod``，不是 typing）——那才是這份檔的實況。

**刻意不做作用域分析。** 賦值別名是整份檔一起收的（``ast.walk``），函式裡寫的
``p = print`` 也會讓 ``p`` 在這份檔裡回指 print。理論上會誤咬（某支函式的區域變數剛好
同名），實務上方向是對的：這三條規矩要擋的是「換個名字繞過去」，漏抓比誤咬糟得多，
而真的撞名的時候那個 diff 在 PR 上看得見。要做作用域就得把整套名字查詢做成一棵符號表，
那是另一件事，今天沒有對象。

**追不到的兩種，照實記在三張卡的卡面。** 跨檔 re-export（自家模組 A 寫
``from typing import Any``，檔 B 再從 A 把它 import 進來）——這裡只讀一份檔自己的 import，
跨檔追不到。執行期才決定的名字（``globals()["X"] = Any``、``getattr(sys, "path")``）
——AST 看不到。兩種都要多寫一段明顯在繞的程式碼，那個 diff 在 PR 上看得見。
"""
from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

# 內建那個模組的名字。``print`` 這種不必 import 就在的名字出自這裡。
BUILTINS = "builtins"

# ``from x import *`` 的那個星號。
STAR_IMPORT = "*"

# 點號。兩個用途：拆 ``import os.path`` 的模組名，以及相對 import 的層級前綴
# （``from .. import x`` 的模組名記成兩個點加模組名，照實記、不猜它解析到哪個套件）。
DOT = "."


@dataclass(frozen=True)
class Origin:
    """一個名字的出處：``(模組, 那個模組裡的名字)``。

    ``name`` 是空字串就代表「這個名字就是那個模組本身」（``import sys`` 的 ``sys``）。
    """

    module: str
    """模組全名。相對 import 的話前面帶點（``"."``／``".."``），照實記。"""

    name: str
    """那個模組裡的名字。空字串＝這個名字指的是模組自己。屬性接下去會變成點號串
    （``import os.path as p`` 之後 ``p.join`` 是 ``Origin("os.path", "join")``；
    ``import os`` 之後 ``os.path.join`` 是 ``Origin("os", "path.join")``）。"""


@dataclass(frozen=True)
class StarImport:
    """一行 ``from <模組> import *``：出現在第幾行、倒的是哪個模組。"""

    lineno: int
    module: str


def _import_bindings(node: ast.Import) -> Iterator[tuple[str, Origin]]:
    """``import x``／``import x.y as z`` 綁出來的本地名字。

    沒有 ``as`` 的點號寫法綁的是最上層那一段（``import os.path`` 之後本地名字是 ``os``），
    有 ``as`` 的綁的是完整模組名。
    """
    for alias in node.names:
        if alias.asname:
            yield alias.asname, Origin(alias.name, "")
        else:
            top = alias.name.split(DOT)[0]
            yield top, Origin(top, "")


def _from_module(node: ast.ImportFrom) -> str:
    """``from`` 後面那個模組名。相對 import 前面帶點，不猜它解析到哪個套件。"""
    return DOT * node.level + (node.module or "")


def _simple_assignments(tree: ast.Module) -> Iterator[tuple[str, ast.expr]]:
    """一份檔裡「單一名字 ＝ 一個運算式」的每一筆（帶標註的賦值也算）。

    刻意走整棵樹、不只模組層：函式裡寫的 ``p = print`` 一樣是一個別名（見模組說明的
    「刻意不做作用域分析」）。
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        else:
            continue
        if node.value is None:
            continue
        for target in targets:
            yield target, node.value


@dataclass(frozen=True)
class Names:
    """一份檔解析出來的名字表。用 :func:`resolve` 生。"""

    bound: Mapping[str, Origin]
    """這份檔裡每一個追得到的本地名字，各自回指哪裡。"""

    stars: tuple[StarImport, ...]
    """這份檔裡每一行 ``from <模組> import *``。那些名字追不到，由呼叫端決定怎麼辦。"""

    def origin(self, node: ast.AST) -> Origin | None:
        """這個運算式節點回指哪一個 ``(模組, 名字)``。追不到回 ``None``。

        裸名字查名字表；屬性寫法先解析底座，底座是模組就接上屬性名，底座本身已經是
        某個模組裡的名字就接成點號串。別的形狀（呼叫、下標、字面值）一律追不到。
        """
        if isinstance(node, ast.Name):
            return self.bound.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self.origin(node.value)
            if base is None:
                return None
            joined = f"{base.name}{DOT}{node.attr}" if base.name else node.attr
            return Origin(base.module, joined)
        return None

    def denotes(self, node: ast.AST, *wanted: Origin) -> bool:
        """這個節點回指的是不是那幾個東西之一。"""
        found = self.origin(node)
        return found is not None and found in wanted

    def local_names(self, *wanted: Origin) -> frozenset[str]:
        """這份檔裡回指那幾個東西的每一個本地名字（反查，給只有字串沒有節點的場合用）。"""
        return frozenset(local for local, found in self.bound.items() if found in wanted)

    def local_names_called(self, name: str) -> frozenset[str]:
        """這份檔裡「回指的那一格就叫這個名字」的每一個本地名字，不管出自哪個模組。

        比 :meth:`local_names` 寬：``import 隨便一個模組 as m`` 之後的 ``m.Any`` 也算。
        ``type-guard`` 第②層要的是這一種——``Any`` 這個名字在標註位置出現，
        不必先知道它出自哪個模組。
        """
        return frozenset(local for local, found in self.bound.items() if found.name == name)


def _seed(tree: ast.Module, assume: Mapping[str, Origin]) -> tuple[dict[str, Origin], list[StarImport]]:
    """先收 import：``assume`` 打底，真的 import 覆蓋它（有 import 就以這份檔的實況為準）。"""
    bound = dict(assume)
    stars: list[StarImport] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bound.update(_import_bindings(node))
        elif isinstance(node, ast.ImportFrom):
            module = _from_module(node)
            for alias in node.names:
                if alias.name == STAR_IMPORT:
                    stars.append(StarImport(node.lineno, module))
                else:
                    bound[alias.asname or alias.name] = Origin(module, alias.name)
    return bound, stars


def resolve(tree: ast.Module, assume: Mapping[str, Origin] | None = None) -> Names:
    """解析一份檔的名字綁定。

    ``assume`` 是「不必 import 就成立的名字」（``print`` 是內建、``sys`` 寫出來就是那個模組、
    ``Any`` 在一份檔裡叫這個名字只可能是那件事）。真的有 import 的話 import 贏。

    賦值別名收到不再長大為止，所以接力幾手都追得到；已經綁好的名字不被賦值蓋掉
    ——``x = print`` 之後又 ``x = 5`` 的那個 ``x`` 照樣算 print，這一條刻意偏向紅。
    """
    bound, stars = _seed(tree, assume or {})
    # 這一份 Names 拿的就是上面那個 dict（同一個物件），所以下面每收一個別名，
    # 它的 origin() 立刻看得到——別名接力靠的就是這件事。
    names = Names(bound, tuple(stars))
    growing = True
    while growing:
        growing = False
        for target, value in _simple_assignments(tree):
            if target in bound:
                continue
            found = names.origin(value)
            if found is None:
                continue
            bound[target] = found
            growing = True
    return names
