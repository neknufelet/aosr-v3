#!/usr/bin/env python3
"""入口檔的規矩節產生器（決策紙：規矩節生成，其餘手寫，上限一個固定行數）。

決策紙 `docs/decisions/rules-section-generated-rest-handwritten.md` 的機器版。那張紙說：
規矩節由卡片生成、用標記包住、CI 重生比對、手改即紅；標記外手寫（座標、一行驗證指令、
狀態與決策入口、七條習慣）；全檔硬上限寫成登記簿常數——登記簿就是卡，所以上限住在
`governance/rules/entry-files-rendered-from-registry.toml` 的 `[settings]`，不寫在這裡。

**兩個入口檔的關係。** 卡上 `[settings] entry_files` 是一張有序清單：第一個是原稿
（手寫段住在那裡），其餘每一份都是它的逐字副本。重生就是「把原稿標記之間換成新的規矩節，
再把整份原稿複製到每一個副本」。這樣走的理由寫在卡的註解裡：v2 兩份入口檔內容不同，
其中一份只有一家工具看得到，於是「開工前讀哪裡」在兩份檔裡各自漂
（v2 事故 enforcer-declared-but-never-installed 與 governance-file-has-no-guard）。
逐字相同就沒有漂的空間，而且要寫的只有一個地方。

**判決只有一個家。** 這個模組交出 :func:`problems` 與 :func:`targets`，規矩卡的檢查程式
`governance/checks/entry_files_rendered_from_registry.py` 直接把它們餵進共用外殼；
這支自己的 ``--check`` 也是同一個呼叫。所以「重生比對」這件事沒有第二份實作可以漂。

用法（`uv` 是 Python 的環境與執行工具，`uv run` 是這個 repo 唯一的入口）：

* `uv run python -m governance.render_entry`——重生（會改檔）。
* `uv run python -m governance.render_entry --check`——只比對，不同就回 1。

標記不見了不會自己補：那代表原稿被改壞了，回 2（這一跑不算數），不猜。
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

from governance.exit_codes import (
    CLEAN,
    TOOL_BROKEN,
    ToolBroken,
    enumerate_files,
    note,
    resolve_scan_root,
    run,
)
from governance.loader import RULES_DIR, setting_int, setting_strings

# 這支產生器在卡裡的名字。門檻只從「宣告了這支檢查」的那張卡讀。
CHECK_REL = "governance/checks/entry_files_rendered_from_registry.py"

# 規矩節的界線。標記本身是 markdown 註解，在算出來的頁面上看不見，但在原稿裡看得見
# ——看得見才有人知道那一段不是手寫的。
MARKER_BEGIN = "<!-- rules:begin generated from governance/rules - do not edit -->"
MARKER_END = "<!-- rules:end -->"

# 規矩節的固定開頭（生成的一部分，不是手寫段）。
SECTION_HEADING = "## 規矩"
SECTION_LEAD = (
    "一條規矩一張卡，卡住在 `governance/rules/`，每張卡自帶檢查程式與必紅樣本。"
    "下面一行就是一張卡的人話，括號裡是它擋不擋合併。"
)

# 卡的 `[settings]` 只認這兩格。打錯字的門檻等於沒有門檻，所以多一格、少一格都回 2。
SETTINGS_KEYS = ("entry_files", "max_lines")

# 渲染一張卡要讀到的三格。缺一格就回 2：讀不出人話就渲染不出規矩節，不猜。
CARD_ID = "id"
CARD_HUMAN = "human"
CARD_MOUNTPOINT = "mountpoint"
CARD_BLOCKS = "blocks_merge"

GATE_LABELS = {True: "擋合併", False: "只會叫"}


def _read_text(path: Path, rel: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"讀不到 {rel}：{exc}") from exc


def _load_toml(path: Path, rel: str) -> dict[str, object]:
    try:
        return tomllib.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ToolBroken(f"讀不開 {rel}：{exc}——我沒看懂就不出結論") from exc


def _rule_files(scan_root: Path, files: list[Path]) -> list[Path]:
    return sorted(f for f in files if f.parent == scan_root / RULES_DIR and f.suffix == ".toml")


def _settings_problems(settings: dict[str, object], rel: str) -> None:
    bad: list[str] = []
    extra = sorted(k for k in settings if k not in SETTINGS_KEYS)
    if extra:
        bad.append(f"多了不認識的鍵 {extra}，只認 {list(SETTINGS_KEYS)}（打錯字的門檻等於沒門檻）")
    entry = settings.get("entry_files")
    if not isinstance(entry, list) or not all(isinstance(x, str) and x.strip() for x in entry):
        bad.append(f"entry_files 必須是字串 list（第一個是原稿，其餘是它的副本），實際是 {entry!r}")
    elif len(entry) < 2:
        bad.append(
            f"entry_files 只列了 {len(entry)} 份——這張卡守的就是「兩份入口檔逐字相同」，"
            "只有一份的話那半條規矩沒有對象"
        )
    else:
        bad += _entry_path_problems([str(x) for x in entry])
    cap = settings.get("max_lines")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        bad.append(f"max_lines 必須是正整數（入口檔的行數上限），實際是 {cap!r}")
    if bad:
        raise ToolBroken(f"{rel} 的 [settings] 形狀不對：" + "；".join(bad))


def _entry_path_problems(entry: list[str]) -> list[str]:
    bad: list[str] = []
    for rel in entry:
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            bad.append(f"entry_files 裡的 {rel!r} 是絕對路徑或夾了 ..——入口檔只准寫成相對掃描根的路徑")
    if len(set(entry)) != len(entry):
        bad.append(f"entry_files 有重複的路徑：{entry!r}")
    return bad


def card_settings(scan_root: Path, files: list[Path]) -> dict[str, object]:
    """從掃描根自己的 `governance/rules/` 讀這支產生器的門檻與入口檔清單。

    找不到、找到多張、或形狀不對，一律 raise :class:`ToolBroken`——沒有門檻就不出結論。
    """
    mine: list[tuple[Path, dict[str, object]]] = []
    for path in _rule_files(scan_root, files):
        data = _load_toml(path, path.relative_to(scan_root).as_posix())
        if data.get("check") == CHECK_REL:
            mine.append((path, data))
    if len(mine) != 1:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下宣告 check={CHECK_REL} 的卡有 {len(mine)} 張，要剛好 1 張"
            "——門檻只寫在卡上，讀不到門檻這一跑就不算數"
        )
    path, data = mine[0]
    rel = path.relative_to(scan_root).as_posix()
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ToolBroken(f"{rel} 沒有 [settings] 表（入口檔清單與行數上限都寫在那裡）")
    _settings_problems(settings, rel)
    return settings


def _card_row(data: dict[str, object], rel: str) -> tuple[str, str, bool]:
    """一張卡渲染成一行要的三格：卡 id、人話、擋不擋合併。"""
    cid = data.get(CARD_ID)
    human = data.get(CARD_HUMAN)
    mount = data.get(CARD_MOUNTPOINT)
    if not isinstance(cid, str) or not cid.strip():
        raise ToolBroken(f"{rel} 讀不出 {CARD_ID}——渲染不出規矩節就不出結論")
    if not isinstance(human, str) or not human.strip():
        raise ToolBroken(f"{rel} 讀不出 {CARD_HUMAN}（人話一句）——渲染不出規矩節就不出結論")
    if not isinstance(mount, dict) or not isinstance(mount.get(CARD_BLOCKS), bool):
        raise ToolBroken(f"{rel} 的 [{CARD_MOUNTPOINT}] 讀不出 {CARD_BLOCKS}（擋不擋合併）")
    return cid.strip(), " ".join(human.split()), bool(mount[CARD_BLOCKS])


def card_rows(scan_root: Path, files: list[Path]) -> list[tuple[str, str, bool]]:
    """掃描根底下每一張卡的 (卡 id, 人話, 擋不擋合併)，照卡 id 排序。

    刻意自己讀 toml、不走載入器的 ``load_card``：載入器會把整張卡的必填欄位一次驗完
    （那是規矩卡 rule-card-required-fields 的工作），而這裡只要渲染得出那三格。
    """
    rows = [
        _card_row(_load_toml(p, p.relative_to(scan_root).as_posix()), p.relative_to(scan_root).as_posix())
        for p in _rule_files(scan_root, files)
    ]
    if not rows:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下一張卡都沒有——渲染出空的規矩節等於宣稱這個 repo 沒有規矩"
        )
    return sorted(rows)


def render_section(rows: list[tuple[str, str, bool]]) -> str:
    """把卡渲染成規矩節（含前後兩個標記）。"""
    lines = [MARKER_BEGIN, "", SECTION_HEADING, "", SECTION_LEAD, ""]
    for cid, human, blocks in rows:
        lines.append(f"- **{cid}**（{GATE_LABELS[blocks]}）：{human}")
    lines.append("")
    lines.append(MARKER_END)
    return "\n".join(lines)


def marker_problems(text: str, rel: str) -> list[str]:
    """標記的死活。缺一個、多一個、前後顛倒，都是「這一份的規矩節界線量不出來」。"""
    lines = text.splitlines()
    begins = [i for i, line in enumerate(lines) if line.strip() == MARKER_BEGIN]
    ends = [i for i, line in enumerate(lines) if line.strip() == MARKER_END]
    bad: list[str] = []
    if len(begins) != 1:
        bad.append(f"{rel} 裡開始標記出現 {len(begins)} 次，要剛好一次——界線量不出來就不知道哪一段是生成的")
    if len(ends) != 1:
        bad.append(f"{rel} 裡結束標記出現 {len(ends)} 次，要剛好一次——界線量不出來就不知道哪一段是生成的")
    if len(begins) == 1 and len(ends) == 1 and ends[0] < begins[0]:
        bad.append(f"{rel} 的結束標記排在開始標記前面——那樣包不住任何東西")
    return bad


def _marker_span(text: str) -> tuple[int, int]:
    """標記那兩行的位置。呼叫前必須先過 :func:`marker_problems`。"""
    lines = text.splitlines()
    begin = next(i for i, line in enumerate(lines) if line.strip() == MARKER_BEGIN)
    end = next(i for i, line in enumerate(lines) if line.strip() == MARKER_END)
    return begin, end


def rules_block(text: str) -> str:
    """這一份檔裡標記之間（含標記本身）的那一段。"""
    begin, end = _marker_span(text)
    return "\n".join(text.splitlines()[begin : end + 1])


def splice(text: str, section: str) -> str:
    """把標記之間換成新的規矩節，標記外的內容原樣保留。"""
    lines = text.splitlines()
    begin, end = _marker_span(text)
    return "\n".join([*lines[:begin], *section.splitlines(), *lines[end + 1 :]]) + "\n"


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """真的會讀／會判的檔：卡上宣告的每一份入口檔，加上掃描根底下每一張卡。

    卡在裡面是因為規矩節就是從它們渲染出來的——它們是這一段內容的來源，不是旁證。
    """
    entry = setting_strings(card_settings(scan_root, files), "entry_files")
    present = set(files)
    picked = [*_rule_files(scan_root, files)]
    picked += [scan_root / rel for rel in entry if scan_root / rel in present]
    return sorted(set(picked))


def _entry_problems(scan_root: Path, rel: str, section: str, cap: int) -> tuple[list[str], str]:
    """一份入口檔的判決，加上它的內容（不在就回空字串）。"""
    path = scan_root / rel
    if not path.is_file():
        return (
            [
                f"{rel} 不在——卡上宣告的入口檔一份都不准缺（少一份就等於有一家工具讀不到規矩，"
                "那正是 v2 兩份入口檔各自漂的起點）"
            ],
            "",
        )
    text = _read_text(path, rel)
    bad: list[str] = []
    count = len(text.splitlines())
    if count > cap:
        bad.append(
            f"{rel} 有 {count} 行，超過卡上登記的上限 {cap} 行"
            "——入口檔每次載入都在花注意力，長大就是變薄；歷史往決策紙與知識區沉，不往這裡堆"
        )
    marker_bad = marker_problems(text, rel)
    if marker_bad:
        return [*bad, *marker_bad], text
    if rules_block(text) != section:
        bad.append(
            f"{rel} 標記之間的規矩節跟重生的結果不一樣"
            "——那一段是產物不是原稿（手改了，或加了卡沒重生）；"
            "跑 `uv run python -m governance.render_entry` 重生"
        )
    return bad, text


def problems(scan_root: Path, files: list[Path]) -> list[str]:
    """規矩卡 entry-files-rendered-from-registry 的判決。空 list 就是乾淨。"""
    settings = card_settings(scan_root, files)
    entry = setting_strings(settings, "entry_files")
    cap = setting_int(settings, "max_lines")
    section = render_section(card_rows(scan_root, files))

    bad: list[str] = []
    texts: dict[str, str] = {}
    for rel in entry:
        one, text = _entry_problems(scan_root, rel, section, cap)
        bad += one
        if text:
            texts[rel] = text
    source, *copies = entry
    for rel in copies:
        if source in texts and rel in texts and texts[rel] != texts[source]:
            bad.append(
                f"{rel} 跟 {source} 內容不一樣——這兩份必須逐字相同（一份是另一份的產物）；"
                "手寫段只改原稿那一份，再重生"
            )
    return bad


def _write_entry_files(scan_root: Path, entry: list[str], section: str) -> list[str]:
    """重生：原稿換掉規矩節，副本整份照抄。回傳這一跑真的改了哪幾份。"""
    source, *copies = entry
    src = scan_root / source
    if not src.is_file():
        raise ToolBroken(f"原稿 {source} 不在——手寫段住在它裡面，這支程式不會替你編一份出來")
    text = _read_text(src, source)
    bad = marker_problems(text, source)
    if bad:
        raise ToolBroken("；".join(bad) + "——標記不對就不重生：這支程式不會自己補標記，也不會猜界線在哪")
    fresh = splice(text, section)
    changed: list[str] = []
    if fresh != text:
        src.write_text(fresh, encoding="utf-8")
        changed.append(source)
    for rel in copies:
        dst = scan_root / rel
        if not dst.is_file() or _read_text(dst, rel) != fresh:
            dst.write_text(fresh, encoding="utf-8")
            changed.append(rel)
    return changed


def regenerate(raw_root: str) -> int:
    """重生模式。0 就是「寫完了」，2 就是「這一跑不算數」。"""
    try:
        root = resolve_scan_root(raw_root)
        files = enumerate_files(root)
        entry = setting_strings(card_settings(root, files), "entry_files")
        changed = _write_entry_files(root, entry, render_section(card_rows(root, files)))
    except ToolBroken as exc:
        note(f"FAIL(2) 工具自壞：{exc}")
        return TOOL_BROKEN
    if changed:
        note(f"重生了：{'、'.join(changed)}")
    else:
        note("入口檔的規矩節已經跟卡一致，沒有東西要改")
    return CLEAN


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="入口檔的規矩節產生器（重生／比對）")
    parser.add_argument("--scan-root", default=".", help="要重生或比對的樹根（必須在 repo 裡）")
    parser.add_argument(
        "--check",
        action="store_true",
        help="只比對不改檔：標記之間的內容不等於重生結果就回 1（規矩卡走的就是這一條）",
    )
    args = parser.parse_args(argv)
    if args.check:
        return run(
            problems,
            ["--scan-root", args.scan_root],
            description="入口檔的規矩節必須等於重生結果",
            targets=targets,
        )
    return regenerate(args.scan_root)


if __name__ == "__main__":
    sys.exit(main())
