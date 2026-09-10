#!/usr/bin/env python3
"""每次開工會載入的檔，全文不准出現模型名、模型家的 CLI 名、用量字樣。

規矩卡 ``no-model-names-in-entry-files`` 的檢查程式。掃描面是卡宣告的入口檔，加上卡宣告的
目錄（今天是決策紙住的那個目錄）底下的每一個檔——不挑副檔名，「全文都掃」就是全部都掃。

判準一句：把每一行的放行字串塗掉、casefold、空白折成一個半形空白，然後拿卡上三張清單
（``model_words``／``cli_words``／``quota_words``）逐條比對，命中即紅。

比對規則（都寫在卡的人話欄，這裡只記為什麼）：

* **不分大小寫**——兩邊都 casefold。事故現場那幾個名字每一種大小寫寫法都出現過，
  只比小寫等於留一條「改個寫法就過」的路。
* **只咬整個詞**——純英數底線的詞用 ``(?<![a-z_])…(?![a-z_])`` 錨住。數字刻意不算在
  「還在同一個詞裡面」：名字後面接版本數字是同一個名字，要咬。底線算在裡面，是為了不咬
  程式與卡裡的識別字（這張卡自己的鍵名 ``quota_words`` 就是一個）。``.`` 與 ``-``
  兩邊算斷開，所以名字被寫進網址、檔名、複合詞裡照樣咬——躲不掉的那幾段走具名放行。
* **帶空白或中文的片語**——直接找子字串（中文沒有詞邊界可錨），比對前先把半形／全形空白
  折成一個半形空白，所以片語只寫一種寫法就好。

放行是「塗掉逐字（連大小寫）相同的那一段」，不是「這一行放過」：塗完同一行剩下的字照樣比對。
放行清單這一跑用到幾條、哪幾條沒用到，跑完印一行——讓清單長大這件事看得見。

字典只住在卡的 ``[settings]``，這支檢查沒有任何預設值：讀不到那一段、宣告這支檢查的卡不是
剛好一張、掃描面上一個檔都沒有、某一份檔不是合法 UTF-8，一律回 2（工具自壞），不回 0。

血債 ``dispatch-roster-inside-session-loaded-file``（v2-audit/lessons.json，legacy_id L08）：
引一句原文（root_cause）「路由隨模型可用性與心證變動最快卻寫進每 session 檔與決策索引，
一換就過期說謊」。事故當下入口檔的分工節躺著幾個代號，而同一份檔另有一節明寫「本節不出現
模型名」——宣稱不是機器。必紅樣本 case-v2-dispatch-roster 就是那個形狀。

已知的縫（照 blueprint/cards-38.json 的 critic_v2 記，不遮）：這是黑名單，不在字典裡的裸代號
咬不到（翻成角色白名單是另一張卡的事，理由寫在規矩卡的註解裡）；提交本文掃不到（那筆事故有
一筆證據就在某個提交的 body 裡）；只咬字不咬結構——入口檔在不在、規矩節是不是產物由
entry-files-rendered-from-registry 守，決策紙的欄位與段落由 decision-paper-structure 守，
三張卡看同一批檔的不同格，刻意不重疊。
"""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

from governance.exit_codes import ToolBroken, note, run
from governance.loader import EXEMPTION_KEYS, RULES_DIR, setting_strings, setting_tables

# 這支檢查在卡裡的名字。字典只從「宣告了這支檢查」的那張卡讀。
CHECK_REL = "governance/checks/no_model_names_in_entry_files.py"

# 字典的三張清單。分開不是為了好看，是為了錯誤訊息說得出「你踩到的是哪一類」。
WORD_KEYS = ("model_words", "cli_words", "quota_words")
KIND_LABELS = {
    "model_words": "模型名",
    "cli_words": "模型家的 CLI 名",
    "quota_words": "用量字樣",
}
LIST_KEYS = ("entry_files", "scan_dirs", *WORD_KEYS)
ALLOW_KEY = "allow"
# 一筆放行三格：放行哪一段字面字串，加上放行條目共用的兩格（reason ＋ expires，形狀定義在
# governance/loader.py 的 EXEMPTION_KEYS，由規矩卡 exemptions-need-expiry 統一）。
ALLOW_ENTRY_KEYS = ("text", *EXEMPTION_KEYS)
SETTINGS_KEYS = (*LIST_KEYS, ALLOW_KEY)

# 純英數底線的詞：用得起詞邊界的那一種。數字刻意在字元集裡（`gpt4` 是一個詞），
# 但錨點的字元集沒有數字（`gpt4` 裡的 `gpt` 要咬）。
ASCII_WORD = re.compile(r"[a-z0-9_]+")
# 空白折成一個半形空白：半形空白、tab、全形空白。
SPACES = re.compile(r"[ \t　]+")
# 放行塗掉那一段之後填進去的字。刻意是一段不可能命中字典的字。
MASK = "[allowed]"


def _card_settings(scan_root: Path, files: list[Path]) -> dict[str, object]:
    """從掃描根自己的 ``governance/rules/`` 讀這支檢查的字典與掃描面宣告。

    找不到、找到多張、或形狀不對，一律 raise ToolBroken——沒有字典就不出結論。
    """
    rules_dir = scan_root / RULES_DIR
    mine: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(f for f in files if f.parent == rules_dir and f.suffix == ".toml"):
        try:
            data = tomllib.loads(path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ToolBroken(f"讀不開 {path.relative_to(scan_root)}：{exc}") from exc
        if data.get("check") == CHECK_REL:
            mine.append((path, data))
    if len(mine) != 1:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下宣告 check={CHECK_REL} 的卡有 {len(mine)} 張，要剛好 1 張"
            "——字典只寫在卡上，讀不到字典這一跑就不算數"
        )
    path, data = mine[0]
    rel = path.relative_to(scan_root)
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ToolBroken(f"{rel} 沒有 [settings] 表（字典與掃描面宣告都寫在那裡）")
    _assert_settings(settings, str(rel))
    return settings


def _assert_settings(settings: dict[str, object], rel: str) -> None:
    """字典的形狀。打錯字的字典等於沒有字典，所以多一個鍵、少一個鍵、型別不對，一律回 2。"""
    bad: list[str] = []
    extra = [k for k in settings if k not in SETTINGS_KEYS]
    if extra:
        bad.append(f"多了不認識的鍵 {sorted(extra)}，只認 {list(SETTINGS_KEYS)}（打錯字的字典等於沒字典）")
    for key in LIST_KEYS:
        value = settings.get(key)
        if not isinstance(value, list) or not all(isinstance(x, str) and x.strip() for x in value):
            bad.append(f"{key} 必須是非空字串的 list，實際是 {value!r}")
        elif not value:
            bad.append(f"{key} 不准是空 list——空的名單等於這一條沒在管")
    bad += _allow_problems(settings.get(ALLOW_KEY, []))
    if bad:
        raise ToolBroken(f"{rel} 的 [settings] 形狀不對：" + "；".join(bad))


def _allow_problems(allow: object) -> list[str]:
    """放行清單的形狀：一條一段字面字串，加理由與到期日，不准有不認識的鍵。"""
    if not isinstance(allow, list):
        return [f"{ALLOW_KEY} 寫了就必須是表的 list（一條一段放行的字面字串），實際是 {allow!r}"]
    bad: list[str] = []
    for index, entry in enumerate(allow):
        where = f"{ALLOW_KEY}[{index}]"
        if not isinstance(entry, dict):
            bad.append(f"{where} 必須是一張表（至少 {list(ALLOW_ENTRY_KEYS)} 三格），實際是 {entry!r}")
            continue
        for key in ALLOW_ENTRY_KEYS:
            value = entry.get(key)
            if not isinstance(value, str) or not value.strip():
                bad.append(
                    f"{where} 的 {key} 必須是非空字串"
                    f"——放行要指得出放過哪一段字、說得出為什麼、也要有失效的那一天（實際是 {value!r}）"
                )
        unknown = [k for k in entry if k not in ALLOW_ENTRY_KEYS]
        if unknown:
            bad.append(f"{where} 多了不認識的鍵 {sorted(unknown)}，只認 {list(ALLOW_ENTRY_KEYS)}")
    return bad


def _term_pattern(term: str) -> re.Pattern[str]:
    """一條字典條目編成一個樣式。純英數底線的詞錨住詞邊界，其餘（片語、中文）找子字串。"""
    folded = term.casefold()
    body = re.escape(folded)
    if ASCII_WORD.fullmatch(folded):
        return re.compile(rf"(?<![a-z_]){body}(?![a-z_])")
    return re.compile(body)


def _patterns(settings: dict[str, object]) -> list[tuple[str, str, re.Pattern[str]]]:
    """字典編成 ``(哪一張清單, 那個詞, 樣式)``。"""
    out: list[tuple[str, str, re.Pattern[str]]] = []
    for key in WORD_KEYS:
        for term in setting_strings(settings, key):
            out.append((key, term, _term_pattern(term)))
    return out


def _mask_allowed(line: str, allowed: list[str], used: set[str]) -> str:
    """把放行的字面字串（逐字、連大小寫都要一樣）從這一行塗掉，剩下的照樣比對。"""
    for text in allowed:
        if text in line:
            used.add(text)
            line = line.replace(text, MASK)
    return line


def _read_text(path: Path, rel: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(
            f"讀不到 {rel} 的內容（{exc}）——這一份是每次開工都會載入的檔，看不懂就不准說它乾淨"
        ) from exc


def _file_hits(
    rel: str,
    text: str,
    patterns: list[tuple[str, str, re.Pattern[str]]],
    allowed: list[str],
    used: set[str],
) -> list[str]:
    """一份檔的每一行比一次字典。"""
    hits: list[str] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = SPACES.sub(" ", _mask_allowed(raw, allowed, used).casefold())
        for key, term, pattern in patterns:
            if pattern.search(line):
                hits.append(
                    f"{rel}:{number} 出現{KIND_LABELS[key]}「{term}」（卡上 {key} 那張清單）"
                    "——每次開工都會載入的檔全文不准出現這些字：規矩綁角色，用哪一家、哪個代號"
                    "寫在那張票裡與派工工具裡。v2 的路由寫進每 session 檔與決策索引，"
                    "三個月改寫八次，一換就過期說謊"
                )
    return hits


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面：卡宣告的入口檔，加上卡宣告的目錄底下的每一個檔（不挑副檔名）。

    卡宣告的入口檔不存在時這裡不出聲：那一格是 entry-files-rendered-from-registry 在守，
    兩張卡刻意不重疊（兩張卡守同一件事會互相遮蔽）。掃描面整個空掉才回 2，見 :func:`check`。
    """
    settings = _card_settings(scan_root, files)
    entry = set(setting_strings(settings, "entry_files"))
    prefixes = [d.rstrip("/") + "/" for d in setting_strings(settings, "scan_dirs")]
    picked: list[Path] = []
    for path in files:
        rel = path.relative_to(scan_root).as_posix()
        if rel in entry or any(rel.startswith(prefix) for prefix in prefixes):
            picked.append(path)
    return sorted(picked)


def check(scan_root: Path, files: list[Path]) -> list[str]:
    settings = _card_settings(scan_root, files)
    patterns = _patterns(settings)
    allowed = [str(entry["text"]) for entry in setting_tables(settings, ALLOW_KEY)]
    picked = targets(scan_root, files)
    if not picked:
        raise ToolBroken(
            f"{scan_root} 底下一份卡宣告的入口檔與決策紙都沒有——掃描面是空的，"
            "「沒有模型名」這句話不算數"
        )

    used: set[str] = set()
    bad: list[str] = []
    for path in picked:
        rel = path.relative_to(scan_root).as_posix()
        if not path.is_file():
            # git 認得、檔案系統上不在（剛被刪掉還沒 commit）。不猜內容，跳過。
            continue
        bad += _file_hits(rel, _read_text(path, rel), patterns, allowed, used)

    unused = [text for text in allowed if text not in used]
    note(
        f"掃了 {len(picked)} 份每次開工會載入的檔，字典 {len(patterns)} 條；"
        f"具名放行 {len(allowed)} 條，這一跑用到 {len(used)} 條"
        + (f"，沒用到的：{unused}（該回頭看還需不需要）" if unused else "")
    )
    return bad


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="每次開工會載入的檔全文不准出現模型名、模型家的 CLI 名、用量字樣",
            targets=targets,
        )
    )
