#!/usr/bin/env python3
"""檔案放哪裡走白名單，路徑名字一律純 ASCII。

掃描面是 ``git ls-files --cached --others --exclude-standard``（版控裡的，加上沒被忽略的
未追蹤檔）。兩關：

**第一關 分層白名單（只看白名單宣告的那幾層）**

白名單是資料，寫在這張卡自己的 ``[[allowlist]]``（``dir`` ／ ``files`` ／ ``dirs``），
不寫死在這支程式裡：換一棵樹就換一份清單，要改寬清單就得改卡、走 PR、留紀錄。
每一層列舉那一格底下「直接的」項目名——一個名字出現時是檔就要在 ``files`` 上、是目錄就要在
``dirs`` 上，兩邊都沒有就是違規。隱藏項目（開頭是點）一樣算。

對到的 v2 事故是 ``accumulating-disk-state-has-no-owner``：根層被搬走 15 個
``.engine-*``／``.val-*``／``.slim-*`` 證據目錄（1074 個檔、236 MB），慢慢長出來、沒有單一
commit 造成，所以本機 hook 抓不到；那筆事故自己的對策寫的就是「CI 斷言根目錄項目 ⊆
allowlist」。**漏的是深度**：這一關只看宣告的那一格（根層、docs 根），已核准目錄裡面堆什麼
它看不到，同一個形狀換成 ``blueprint/.engine-foo/`` 就照過——要另一張卡。

**第二關 路徑必須純 ASCII**

決策紙 ``docs/decisions/ascii-filenames.md`` 的機器版。整個列舉集合的每一個路徑都要是純
ASCII；非 ASCII 只准出現在卡宣告的必紅樣本樹的樣本目錄底下（那是刻意的壞樣本，由後設測試
單獨餵給檢查，不是被放過）。被放過幾個、放在哪，印一行 ``NOTE:`` 說清楚。
列舉時 git 帶 ``-c core.quotePath=false``（見 ``governance/exit_codes.py``），不然中文名字會
被印成八進位跳脫碼、那串東西自己是純 ASCII，這一關會假綠。名字還是被 git 加了引號的（含
控制字元、引號、反斜線那種）一律判違規。

**什麼情況回 2（工具自壞，這一跑不算數）**

讀不到宣告本檢查的卡、卡裡沒有 ``[[allowlist]]``、白名單形狀壞掉、宣告要管的某一層在這棵樹
裡一個項目都沒有（等於那一層沒掃到）、扣掉樣本樹之後一個路徑都不剩。一律不准回 0。
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

from governance.exit_codes import ToolBroken, note, run
from governance.loader import (
    ALLOWLIST_FIELD,
    RULES_DIR,
    AllowlistLevel,
    allowlist_levels,
    allowlist_problems,
)

# 這支檢查自己。宣告白名單的那張卡，就是 check 欄指到這裡的那一張。
SELF_CHECK = "governance/checks/file_placement_allowlist.py"


def _tracked_rules(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描根的 ``governance/rules/*.toml``——只認列舉集合裡的，不自己走檔案系統。"""
    return sorted(f for f in files if f.parent == scan_root / RULES_DIR and f.suffix == ".toml")


def _read_toml(path: Path, rel: str) -> dict[str, object]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ToolBroken(f"讀不到 {rel}：{exc}") from exc
    try:
        return tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"{rel} 解不開（{exc}）——我沒看懂就不出結論") from exc


def _levels(scan_root: Path, files: list[Path]) -> tuple[AllowlistLevel, ...]:
    """從卡裡讀白名單。找不到、或形狀壞掉，一律回 2。"""
    mine: list[tuple[str, dict[str, object]]] = []
    for path in _tracked_rules(scan_root, files):
        rel = str(path.relative_to(scan_root))
        data = _read_toml(path, rel)
        if data.get("check") == SELF_CHECK:
            mine.append((rel, data))
    if not mine:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下沒有任何一張卡把 check 指到 {SELF_CHECK}"
            "——白名單寫在卡裡，讀不到卡就沒有尺，這一跑不算數"
        )
    if len(mine) > 1:
        raise ToolBroken(f"有 {len(mine)} 張卡都宣告了 {SELF_CHECK}（{[r for r, _ in mine]}），我不知道該用哪一份白名單")

    rel, data = mine[0]
    problems = allowlist_problems(data)
    if problems:
        raise ToolBroken(f"{rel} 的白名單形狀壞掉：" + "；".join(problems))
    levels = allowlist_levels(data)
    if not levels:
        raise ToolBroken(f"{rel} 沒有 [[{ALLOWLIST_FIELD}]]——這張卡沒給白名單，沒有清單可比對")
    return levels


def _fixture_case_prefixes(scan_root: Path, files: list[Path]) -> list[str]:
    """卡宣告的必紅樣本樹底下的樣本目錄。非 ASCII 只准住在這裡面。

    只放過 ``<negative_fixture>/<樣本目錄>/...``，樣本樹自己那一層的檔案照樣要管。
    """
    out: list[str] = []
    for path in _tracked_rules(scan_root, files):
        data = _read_toml(path, str(path.relative_to(scan_root)))
        neg = data.get("negative_fixture")
        if isinstance(neg, str) and neg.strip():
            out.append(neg.strip("/") + "/")
    return sorted(set(out))


def _in_fixture_case(rel: str, prefixes: list[str]) -> bool:
    for prefix in prefixes:
        if rel.startswith(prefix) and "/" in rel[len(prefix) :]:
            return True
    return False


def _level_hits(level: AllowlistLevel, rels: list[str]) -> list[str]:
    """一層白名單：那一格底下的直接項目名，檔要在 files 上、目錄要在 dirs 上。"""
    seen_files: dict[str, str] = {}
    seen_dirs: dict[str, str] = {}
    prefix = level.prefix
    for rel in rels:
        if prefix and not rel.startswith(prefix):
            continue
        rest = rel[len(prefix) :]
        if not rest:
            continue
        head, sep, _tail = rest.partition("/")
        if sep:
            seen_dirs.setdefault(head, rel)
        else:
            seen_files.setdefault(head, rel)

    if not seen_files and not seen_dirs:
        raise ToolBroken(
            f"白名單宣告要管 {level.dir!r} 那一層，但列舉集合裡它底下一個項目都沒有"
            "——那一層根本沒掃到，「沒問題」這句話不算數"
        )

    hits: list[str] = []
    for name, sample in sorted(seen_files.items()):
        if name not in level.files:
            hits.append(
                f"{level.dir} 這一層多了一個不在白名單上的檔 {name!r}（例：{sample}）"
                f"——這一層准出現的檔名只有 {list(level.files)}；要它進來就改卡的 [[{ALLOWLIST_FIELD}]]"
            )
    for name, sample in sorted(seen_dirs.items()):
        if name not in level.dirs:
            hits.append(
                f"{level.dir} 這一層多了一個不在白名單上的目錄 {name!r}（例：{sample}）"
                f"——這一層准出現的目錄名只有 {list(level.dirs)}；要它進來就改卡的 [[{ALLOWLIST_FIELD}]]"
            )
    return hits


def _ascii_hits(rels: list[str], prefixes: list[str]) -> list[str]:
    """路徑必須純 ASCII。非 ASCII 只准出現在卡宣告的必紅樣本的樣本目錄底下。"""
    hits: list[str] = []
    scanned = 0
    skipped: list[str] = []
    for rel in rels:
        if _in_fixture_case(rel, prefixes):
            skipped.append(rel)
            continue
        scanned += 1
        if rel.startswith('"'):
            hits.append(
                f"路徑被 git 加了引號逃脫（{rel}）——名字裡有引號、反斜線或控制字元，"
                "機器要碰的名字不准長這樣"
            )
        elif not rel.isascii():
            bad = sorted({ch for ch in rel if not ch.isascii()})
            hits.append(
                f"路徑不是純 ASCII：{rel}（非 ASCII 字元 {bad}）"
                "——檔名、目錄名一律英文（決策紙 docs/decisions/ascii-filenames.md）；"
                "內文中文沒問題，名字不行"
            )
    if scanned == 0:
        raise ToolBroken(
            "扣掉必紅樣本樹之後一個路徑都不剩——純 ASCII 那一關沒掃到東西，這一跑不算數"
        )
    if skipped:
        note(
            f"純 ASCII 那一關放過樣本目錄底下 {len(skipped)} 個路徑"
            f"（樣本樹 {prefixes}，刻意的壞樣本由後設測試單獨餵）"
        )
    return hits


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面：列舉集合裡的每一個路徑。

    這張卡判的是「名字」與「住在哪一層」，不是內容，所以每一個路徑都真的被判過一次
    （第②條的純 ASCII 對全集生效）。白名單與樣本樹前綴從卡上讀，那些卡也在這一組裡面。
    """
    return sorted(files)


def check(scan_root: Path, files: list[Path]) -> list[str]:
    rels = sorted(str(f.relative_to(scan_root).as_posix()) for f in targets(scan_root, files))
    levels = _levels(scan_root, files)
    hits: list[str] = []
    for level in levels:
        hits += _level_hits(level, rels)
    hits += _ascii_hits(rels, _fixture_case_prefixes(scan_root, files))
    return hits


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="檔案放哪裡走白名單（分層），而且路徑名字一律純 ASCII",
            targets=targets,
        )
    )
