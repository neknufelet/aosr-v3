#!/usr/bin/env python3
"""每張卡宣告的掃描面，必須等於那支檢查實際列舉出來的檔案集合。

一條規矩，兩個方向：**宣告了沒掃到是洞，掃了沒宣告是越權。**

怎麼量：

1. 讀掃描根底下每一張規矩卡的 ``scope``，用列舉集合（``git ls-files``）展開成
   「宣告集合」。展開的語法只有一份定義，在 :mod:`governance.loader`
   （``.``、目錄前綴、單檔、glob；前面加 ``!`` 就是扣掉）。
2. 跑那張卡宣告的檢查、帶 ``--list-files``，拿它自己印出來的清單當「實際集合」。
   那個旗標是共用外殼的列舉模式：只跑那支檢查的 ``targets()``、不下判斷，所以
   不會跑動態探針、不會遞迴，一次就是一次 ``git ls-files``。
3. 兩邊做集合差。少了的是洞，多了的是越權，各印成一筆違規。

**掃描面不是檔案的卡怎麼辦。** ``commit-author-allowlisted`` 判的是提交 metadata
（author／committer），不是路徑；它的 ``scope`` 寫的是名單檔。那種卡在卡上明寫
``scope_kind = "commits"``，這支檢查對它只驗兩件事：宣告在不在、宣告的每一條在版控裡
真的有對象。集合不比——比了只會逼人把提交那半塞成假的檔案清單。

**什麼會紅**

* 卡沒宣告 ``scope``（或不是非空的字串 list）——量不到掃描面，等於沒有宣告。
* ``scope_kind`` 寫了但不在列舉裡（打錯字的宣告等於沒有宣告）。
* ``scope_kind = "files"``（預設）的卡，宣告集合與實際集合不相等。
* 卡指的檢查模組在版控裡不存在，或它在列舉模式下印不出清單——量不到就不准當乾淨。
* ``scope_kind = "commits"`` 的卡，某一條宣告在列舉集合裡一個對象都沒有。

**什麼不會紅（刻意的，留在卡面不要當成漏掉）**

* 宣告了一個今天還沒有檔案的前綴（例如還沒建的目錄）不算違規：那個前綴今天沒有對象，
  展開出來是空集合，不是洞。明天有檔進去，兩邊會一起長。
* 「語法檢查必須真的解析、不准用正則猜」那半沒做——「用正則猜」不是機器可判定的性質，
  降成寫法慣例（見卡面）。
* 這條抓的是**漂移**：掃描器改了、忘了同步卡上的宣告。抓不到「一開始就宣告得跟掃描器
  一樣窄」的共謀式收窄——那要人看 PR。理由與代價寫在卡面。

**沒掃到東西就回 2**

一張卡都讀不到、卡解不開、檢查的列舉輸出看不懂，一律 raise :class:`ToolBroken`
讓外殼回 2。這一跑沒量到東西，「沒問題」這句話就不算數。
"""
from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

from governance.exit_codes import CLEAN, LIST_BEGIN, LIST_END, ToolBroken, run
from governance.loader import (
    DEFAULT_SCOPE_KIND,
    RULES_DIR,
    SCOPE_KINDS,
    SCOPE_NEGATE,
    expand_scope,
    scope_matches,
)

# 跑一次列舉模式的看門狗秒數。登記在 thresholds-live-only-in-registry 的例外裡：
# 調窄只會讓這支檢查回 2（這一跑不算數），調寬變不出綠。
LIST_TIMEOUT = 300

COMMITS_KIND = "commits"


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """這支檢查真的會讀的檔：掃描根底下每一張規矩卡。

    別人的檢查程式不算：這支檢查不讀它們的內容，是**跑**它們（列舉模式），
    拿它們自己印出來的清單當證據。
    """
    return sorted(f for f in files if f.parent == scan_root / RULES_DIR and f.suffix == ".toml")


def _load(path: Path, rel: str) -> dict[str, object]:
    """讀一張卡。刻意自己 tomllib 解，不走載入器的 ``load_card``。

    載入器對整張卡的必填欄位一次驗完、一有問題就 raise，那是第一張卡
    （``rule-card-required-fields``）的工作；這裡要的是「就算這張卡別的欄位壞掉，
    ``scope`` 這一欄的死活還是要判得出來」——不然「沒宣告 scope」這種卡會被跳過，
    變成量不到卻回綠。
    """
    try:
        data = tomllib.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ToolBroken(f"{rel} 讀不開或解不開（{exc}）——我沒看懂就不出結論") from exc
    return data


def _scope(data: dict[str, object]) -> list[str] | None:
    """卡上宣告的掃描面。沒宣告、或形狀不是非空的字串 list，就回 None。"""
    scope = data.get("scope")
    if not isinstance(scope, list) or not scope:
        return None
    if not all(isinstance(entry, str) and entry.strip() for entry in scope):
        return None
    return [entry.strip() for entry in scope]


def _probe_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    # 不要在被量的樹裡留 __pycache__：那些是別的檢查的掃描面上的雜物。
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _module(check: str) -> str:
    """卡上寫的檢查路徑換成模組名（把副檔名去掉、斜線換成點），給 ``python -m`` 用。"""
    return check.removesuffix(".py").replace("/", ".")


def _parse_list(stdout: str, where: str) -> list[str]:
    """從列舉模式的輸出裡挖出那份清單。界線之間一行一個路徑。"""
    lines = stdout.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(LIST_BEGIN + "=")]
    ends = [i for i, line in enumerate(lines) if line.strip() == LIST_END]
    if not starts or not ends or ends[-1] < starts[-1]:
        raise ToolBroken(
            f"{where} 的列舉模式沒印出 {LIST_BEGIN}=<個數> ／ 路徑 ／ {LIST_END} 那一段"
            "——拿不到它實際列舉的清單，就沒有人能證明它掃的跟卡上宣告的是同一組檔"
        )
    begin, end = starts[-1], ends[-1]
    said = lines[begin].split("=", 1)[1].strip()
    names = [line.strip() for line in lines[begin + 1 : end] if line.strip()]
    if said != str(len(names)):
        raise ToolBroken(
            f"{where} 的列舉模式自己說有 {said} 個，界線之間卻是 {len(names)} 行"
            "——這份清單我沒看懂，不出結論"
        )
    return names


def _actual(scan_root: Path, card_rel: str, check: str) -> list[str] | str:
    """跑那支檢查的列舉模式，拿它實際列舉的清單。跑不起來就回一句人話（那是一筆違規）。"""
    where = f"{card_rel} 宣告的 {check}"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", _module(check), "--scan-root", str(scan_root), "--list-files"],
            cwd=scan_root,
            env=_probe_env(),
            capture_output=True,
            text=True,
            timeout=LIST_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolBroken(f"{where} 的列舉模式跑超過 {LIST_TIMEOUT} 秒") from exc
    if proc.returncode != CLEAN:
        tail = " ／ ".join((proc.stdout + proc.stderr).strip().splitlines()[-3:])
        return (
            f"{where} 在列舉模式回 {proc.returncode}，量不到它的掃描面"
            f"——沒有實際列舉的清單就比不了集合，這張卡的宣告等於沒人驗過：{tail}"
        )
    return _parse_list(proc.stdout, where)


def _commits_problems(card_rel: str, scope: list[str], names: list[str]) -> list[str]:
    """``scope_kind = "commits"``：只驗宣告的每一條真的有對象，不比集合。"""
    bad: list[str] = []
    for entry in scope:
        if entry.startswith(SCOPE_NEGATE):
            continue
        if not any(scope_matches(entry, name) for name in names):
            bad.append(
                f"{card_rel} 宣告 scope_kind={COMMITS_KIND!r}，但 scope 裡的 {entry!r}"
                "在列舉集合裡一個對象都沒有"
                "——掃描面不是檔案的卡，至少要說得出它讀的那份名單住在哪"
            )
    return bad


def _card_hits(scan_root: Path, path: Path, names: list[str]) -> list[str]:
    card_rel = path.relative_to(scan_root).as_posix()
    data = _load(path, card_rel)

    scope = _scope(data)
    if scope is None:
        return [
            f"{card_rel} 沒宣告 scope（要掃哪些路徑），或它不是非空的字串 list"
            "——沒有宣告就沒有東西可以跟實際列舉的集合比對，這張卡的掃描面沒有人在看"
        ]

    kind = data.get("scope_kind", DEFAULT_SCOPE_KIND)
    if kind not in SCOPE_KINDS:
        return [
            f"{card_rel} 的 scope_kind={kind!r} 不在列舉裡，只認 {list(SCOPE_KINDS)}"
            "——打錯字的宣告等於沒有宣告，這張卡會被當成不用比集合而整條漏掉"
        ]
    if kind == COMMITS_KIND:
        return _commits_problems(card_rel, scope, names)

    check = data.get("check")
    if not isinstance(check, str) or not check.strip():
        return [f"{card_rel} 沒宣告 check（哪一支檢查），量不到它實際列舉了什麼"]
    if check not in names:
        return [
            f"{card_rel} 宣告的檢查 {check} 在版控的列舉集合裡不存在"
            "——跑不起來就量不到實際列舉的集合，這張卡的宣告沒有人驗得了"
        ]

    actual = _actual(scan_root, card_rel, check)
    if isinstance(actual, str):
        return [actual]

    declared = expand_scope(scope, names)
    scanned = set(actual)
    bad: list[str] = []
    missing = sorted(declared - scanned)
    if missing:
        bad.append(
            f"{card_rel} 宣告 scope={scope} 蓋到 {len(declared)} 個檔，"
            f"但 {check} 實際列舉的集合裡少了這些：{missing}"
            "——宣告了沒掃到就是洞：卡上寫得像有在守，那些檔實際上沒有人看"
        )
    extra = sorted(scanned - declared)
    if extra:
        bad.append(
            f"{card_rel} 宣告 scope={scope}，但 {check} 實際還掃了這些沒宣告的檔：{extra}"
            "——掃了沒宣告就是越權：卡面看不出它會咬到那裡，改到那裡的人不知道有這道閘"
        )
    return bad


def check(scan_root: Path, files: list[Path]) -> list[str]:
    cards = targets(scan_root, files)
    if not cards:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下一張版控裡的規矩卡都沒有"
            "——這一跑沒量到任何一張卡的掃描面，「沒問題」這句話不算數"
        )

    names = sorted(f.relative_to(scan_root).as_posix() for f in files)
    bad: list[str] = []
    for path in cards:
        bad += _card_hits(scan_root, path, names)
    return bad


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="每張卡宣告的掃描面要等於那支檢查實際列舉出來的檔案集合",
            targets=targets,
        )
    )
