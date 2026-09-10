#!/usr/bin/env python3
"""入口檔的規矩節產生器（決策紙：規矩節生成，其餘手寫，上限一個固定行數）。

決策紙 `docs/decisions/rules-section-generated-rest-handwritten.md` 的機器版。那張紙說：
規矩節由卡片生成、用標記包住、CI 重生比對、手改即紅；標記外手寫（座標、一行驗證指令、
狀態與決策入口、七條習慣）；全檔硬上限寫成登記簿常數——登記簿就是卡，所以上限住在
`governance/rules/entry-files-rendered-from-registry.toml` 的 `[settings]`，不寫在這裡。

**兩個入口檔的關係（2026-09-10 老闆改過一次）。** 卡上 `[settings] entry_files` 是一張有序
清單：第一個是原稿（手寫段與規矩節住在那裡），其餘每一份是**指路牌**——內容固定，就是卡上
登記的 `pointer_text`，一個字都不准差。重生就是「把原稿標記之間換成新的規矩節，再把指路牌
那幾份寫成登記的那段文字」。
舊做法是「其餘每一份都是原稿的逐字副本」，這一版改掉：兩份長文各載入一次是雙倍的注意力，
而且第二份的存在只是為了讓另一家工具找得到入口。改成指路牌之後，內容只有一個家，
另一家工具讀到的是一行「去讀原稿」。要防的事沒變——v2 兩份入口檔內容不同、其中一份只有
一家工具看得到，於是「開工前讀哪裡」在兩份檔裡各自漂（事故
enforcer-declared-but-never-installed 與 governance-file-has-no-guard）——只是防法從
「逐字相同」換成「指路牌逐字等於登記的那一段」。

**一張卡渲染成幾寬。** 只取人話的第一句（切點是句號，見 :func:`first_sentence`）。卡的
`human` 欄是規格的家，越寫越長是對的；入口檔那一段是產物，多寬由這支產生器決定，不是由
卡的作者寫多長決定。這一句就是規矩卡 `derived-content-rendered-not-handwritten` 併進來的
那半精神（衍生內容不手寫、形狀由生成器定），也是入口檔過得了「單行字元上限」那道閘的方式
——那顆牙住在 `doc-frontmatter-and-dates`，咬 docs 三類與這兩份入口檔的每一行。

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
from governance.loader import RULES_DIR, setting_int, setting_strings, setting_text

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
    "下面一行是一張卡人話的第一句（整段住在卡上），括號裡是它擋不擋合併。"
)

# 卡的 `[settings]` 只認這四格。打錯字的門檻等於沒有門檻，所以多一格、少一格都回 2。
SETTINGS_KEYS = ("entry_files", "max_lines", "placeholder_markers", "pointer_text")
MARKERS_KEY = "placeholder_markers"
POINTER_KEY = "pointer_text"

# 渲染一張卡要讀到的三格。缺一格就回 2：讀不出人話就渲染不出規矩節，不猜。
CARD_ID = "id"
CARD_HUMAN = "human"
CARD_MOUNTPOINT = "mountpoint"
CARD_BLOCKS = "blocks_merge"

GATE_LABELS = {True: "擋合併", False: "只會叫"}

# 一張卡只渲染人話的**第一句**，切點就是這個句號。
#
# 為什麼不整段渲染：卡的 human 欄是規格的家，越寫越長是對的（今天最長那一張一千多字），
# 可是整段渲染出來就是入口檔裡一行一千多字元的一行。那正是 v2 事故
# governance-file-has-no-guard 記的形狀之一（frontmatter 版本史一行 1,027 字元）。
# 「單行字元上限」那顆牙住在 doc-frontmatter-and-dates，它咬的是 docs 三類與這兩份入口檔；
# 要讓入口檔過得了那道閘，做法有兩種：規矩節那一段放行，或者渲染時每張卡只取第一句。
# 選後者——這正是「衍生內容不手寫、由生成器決定形狀」：入口檔那一段是產物，它多寬由這支
# 產生器決定，不是由卡的作者寫多長決定。要看整段就去讀卡，卡的路徑就印在規矩節的開頭。
SENTENCE_END = "。"


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
        bad.append(f"entry_files 必須是字串 list（第一個是原稿，其餘是指路牌），實際是 {entry!r}")
    elif len(entry) < 2:
        bad.append(
            f"entry_files 只列了 {len(entry)} 份——這張卡守的其中一條是「指路牌那幾份的內容"
            "等於卡上登記的那一段」，只有原稿一份的話那條規矩沒有對象"
        )
    else:
        bad += _entry_path_problems([str(x) for x in entry])
    cap = settings.get("max_lines")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        bad.append(f"max_lines 必須是正整數（入口檔的行數上限），實際是 {cap!r}")
    pointer = settings.get(POINTER_KEY)
    if not isinstance(pointer, str) or not pointer.strip():
        bad.append(f"{POINTER_KEY} 必須是非空字串（指路牌那幾份入口檔的固定內容），實際是 {pointer!r}")
    elif not pointer.endswith("\n"):
        bad.append(
            f"{POINTER_KEY} 必須以換行結尾——它是整份檔的內容，"
            "檔尾少一個換行也算「多一字少一字」，那樣這條規矩會變成一個誰都滿足不了的要求"
        )
    elif MARKER_BEGIN in pointer or MARKER_END in pointer:
        bad.append(
            f"{POINTER_KEY} 裡出現規矩節的標記——指路牌不放規矩節（那一段只住在原稿裡），"
            "標記寫進去只會讓「哪一段是產物」量不出來"
        )
    markers = settings.get(MARKERS_KEY)
    if not isinstance(markers, list) or not markers:
        bad.append(f"{MARKERS_KEY} 必須是非空的字串 list（生成段不准出現的佔位字樣），實際是 {markers!r}")
    elif not all(isinstance(x, str) and x.strip() for x in markers):
        bad.append(f"{MARKERS_KEY} 的每一格都要是非空字串，實際是 {markers!r}")
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


def first_sentence(human: str) -> str:
    """一張卡的人話取到第一個句號為止（含句號）；整段沒有句號就整段。

    空白先正規化成一個半形空格，所以卡上折行寫的人話渲染出來是一行。
    理由見 :data:`SENTENCE_END` 上面那一段。
    """
    flat = " ".join(human.split())
    head, sep, _rest = flat.partition(SENTENCE_END)
    return head + sep if sep else flat


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
    return cid.strip(), first_sentence(human), bool(mount[CARD_BLOCKS])


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
    """把卡渲染成規矩節（含前後兩個標記）。

    刻意不在標記與標題、標題與引言之間留空行：入口檔的行數上限住在卡上，那幾行空白也在
    上限裡面，而 markdown 那三個區塊不靠空行就分得開（清單前面那一行空白留著，
    清單要接在段落後面得先斷開）。
    """
    lines = [MARKER_BEGIN, SECTION_HEADING, SECTION_LEAD, ""]
    for cid, human, blocks in rows:
        lines.append(f"- **{cid}**（{GATE_LABELS[blocks]}）：{human}")
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


def placeholder_hits(where: str, block: str, markers: list[str]) -> list[str]:
    """生成段裡的佔位字樣。逐字、分大小寫的子字串比對，名單只從卡上讀。

    只餵標記之間那一段：標記外面是手寫段，人在自己的草稿裡寫 TODO 不是這條規矩的事。
    """
    found = [m for m in markers if m in block]
    if not found:
        return []
    return [
        f"{where}出現卡上登記的佔位字樣 {found}"
        "——生成段是產物，產物還沒填完就不該進版控；"
        "這一條不會被「等於重生結果」蓋住：佔位寫在某張卡的 human 裡，重生出來的那一段"
        "本來就帶著它，逐字比對照樣回綠"
    ]


def _missing_hit(rel: str) -> list[str]:
    return [
        f"{rel} 不在——卡上宣告的入口檔一份都不准缺（少一份就等於有一家工具讀不到入口，"
        "那正是 v2 兩份入口檔各自漂的起點）"
    ]


def _line_cap_hits(rel: str, text: str, cap: int) -> list[str]:
    count = len(text.splitlines())
    if count <= cap:
        return []
    return [
        f"{rel} 有 {count} 行，超過卡上登記的上限 {cap} 行"
        "——入口檔每次載入都在花注意力，長大就是變薄；歷史往決策紙與知識區沉，不往這裡堆"
    ]


def _pointer_problems(scan_root: Path, rel: str, pointer: str, cap: int) -> list[str]:
    """一份指路牌的判決：在、逐字等於卡上登記的那一段、不超過行數上限。

    不驗標記、不驗規矩節：指路牌不放內容，那一段只住在原稿裡。
    """
    path = scan_root / rel
    if not path.is_file():
        return _missing_hit(rel)
    text = _read_text(path, rel)
    bad = _line_cap_hits(rel, text, cap)
    if text != pointer:
        bad.append(
            f"{rel} 的內容不等於卡上登記的指路牌文本（`[settings] {POINTER_KEY}`），"
            "多一字少一字都算——這一份不是原稿，它只負責把人指去原稿；"
            "要改它就改卡上那一段，再跑 `uv run python -m governance.render_entry` 重生"
        )
    return bad


def _source_problems(
    scan_root: Path, rel: str, section: str, cap: int, markers: list[str]
) -> list[str]:
    """原稿那一份的判決：在、標記剛好一對、規矩節等於重生結果、沒有佔位、不超過行數上限。"""
    path = scan_root / rel
    if not path.is_file():
        return _missing_hit(rel)
    text = _read_text(path, rel)
    bad: list[str] = _line_cap_hits(rel, text, cap)
    marker_bad = marker_problems(text, rel)
    if marker_bad:
        return [*bad, *marker_bad]
    block = rules_block(text)
    if block != section:
        bad.append(
            f"{rel} 標記之間的規矩節跟重生的結果不一樣"
            "——那一段是產物不是原稿（手改了，或加了卡沒重生）；"
            "跑 `uv run python -m governance.render_entry` 重生"
        )
    bad += placeholder_hits(f"{rel} 標記之間的生成段裡", block, markers)
    return bad


def problems(scan_root: Path, files: list[Path]) -> list[str]:
    """規矩卡 entry-files-rendered-from-registry 的判決。空 list 就是乾淨。"""
    settings = card_settings(scan_root, files)
    entry = setting_strings(settings, "entry_files")
    cap = setting_int(settings, "max_lines")
    markers = setting_strings(settings, MARKERS_KEY)
    pointer = setting_text(settings, POINTER_KEY)
    section = render_section(card_rows(scan_root, files))

    # 先咬來源：重生結果裡就有佔位，代表某張卡的人話自己還沒填完。這一筆跟入口檔在不在、
    # 有沒有被手改無關，所以單獨記一筆（不然「加了卡還沒重生」的時候這顆牙會看不到它）。
    # 指路牌那一段也是產物，同一顆牙一起量。
    bad: list[str] = placeholder_hits(
        "重生結果（來源是 governance/rules/ 底下的卡）裡", section, markers
    )
    bad += placeholder_hits(f"卡上登記的 {POINTER_KEY} 裡", pointer, markers)
    source, *pointers = entry
    bad += _source_problems(scan_root, source, section, cap, markers)
    for rel in pointers:
        bad += _pointer_problems(scan_root, rel, pointer, cap)
    return bad


def _write_entry_files(
    scan_root: Path, entry: list[str], section: str, pointer: str
) -> list[str]:
    """重生：原稿換掉規矩節，指路牌寫成卡上登記的那一段。回傳這一跑真的改了哪幾份。"""
    source, *pointers = entry
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
    for rel in pointers:
        dst = scan_root / rel
        if not dst.is_file() or _read_text(dst, rel) != pointer:
            dst.write_text(pointer, encoding="utf-8")
            changed.append(rel)
    return changed


def regenerate(raw_root: str) -> int:
    """重生模式。0 就是「寫完了」，2 就是「這一跑不算數」。"""
    try:
        root = resolve_scan_root(raw_root)
        files = enumerate_files(root)
        settings = card_settings(root, files)
        changed = _write_entry_files(
            root,
            setting_strings(settings, "entry_files"),
            render_section(card_rows(root, files)),
            setting_text(settings, POINTER_KEY),
        )
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
