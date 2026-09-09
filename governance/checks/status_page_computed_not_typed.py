#!/usr/bin/env python3
"""進度／待辦／狀態一律現算，不准手寫進版控。

決策紙 ``docs/decisions/backlog-in-github-issues.md``（待辦全走 GitHub issue、repo 不放手寫
待辦檔）與 ``docs/decisions/status-page-not-in-main.md``（狀態頁由機器算、不進主線）的機器版。
掃描面是 ``git ls-files``（含未追蹤但進得了版控的檔）的全集，扣掉卡上登記的前綴。

四條規則，門檻與名單全部只寫在卡的 ``[settings]``，讀不到就回 2（工具自壞），不回 0：

1. **檔名**——小寫的 basename 命中 ``name_patterns``（``next*``／``todo*``／``status*.md``／
   ``progress*``／``checkpoint*`` 這類）就是手寫的進度／待辦／狀態檔。
   ``name_exempt_prefixes`` 底下不看這一條：決策紙照題目命名，``docs/decisions/``
   底下本來就有 ``backlog-in-github-issues.md``、``status-page-not-in-main.md``，
   它們是在講規矩不是在記進度；那一類的種類與數量由 docs-three-classes 那組守。
2. **frontmatter**——md 開頭 ``---`` 區塊裡 ``type``／``kind``／``category`` 的**值**落在
   ``frontmatter_kinds``（todo／progress／status／backlog…）就紅。比對值不比對鍵名：
   決策紙每一份都有 ``status:`` 這個鍵，那個不算。這一條堵的是「換個無辜檔名就繞過去」。
3. **內容·勾選框**——一份 md 裡 ``- [ ]``／``- [x]`` 超過 ``max_checkboxes`` 個，就是一份
   待辦清單，不管檔名叫什麼。
4. **內容·段落標題**——標題正規化後等於 ``heading_labels`` 裡的字，或以它開頭而且不長於
   ``heading_max_chars``，就是「已完成／待辦／進度」那種手抄現況的段落。要求「短」是為了
   不誤咬正常標題：決策紙的 H1「待辦全走 GitHub issue，repo 不放手寫待辦檔」是在講規矩。

``[[settings.allow]]`` 是明文放行，一條一個檔，**必須帶 ``max_lines``**：放行只免掉第 1 條，
內容那兩條照管，而且行數超過上限一樣紅。找碴席指出這張卡最大的縫是「白名單直接豁免了違規
物件本身」——``handoff.md`` 跟 v2 的 ``docs/next.md`` 是同一種會持續長大的檔——所以放行帶
上限（``handoff.md`` 自己寫「這份超過 60 行就該砍」，v2 事故的對策也是行數上限）。

血債 ``status-file-grows-into-history-ledger``（v2-audit/lessons.json，legacy_id L38）：
事故現場就是手寫的 ``docs/next.md``，劃掉的完成項留 37 條、同一份檔同時寫「已 commit」與
「未 commit」，守衛只守 100 行照樣過。必紅樣本 ``case-v2-next-md`` 就是那個現場。

已知的縫（找碴席 critic_v2 指出，這裡照抄不遮）：檔名那一條是黑名單，換成 ``plan.md``、
中文檔名、或把清單塞進 json（不掃內容）都不命中；內容那兩條只認 md 的勾選框與段落標題，
管不到「抄進來的那句話還對不對」（那是 derived-content-rendered-not-handwritten 的事）。
放行名單的每一條從今天起要有 ``reason`` 與 ``expires``（到期日），過期即紅——那道閘在規矩卡
exemptions-need-expiry，形狀在 governance/loader.py。
"""
from __future__ import annotations

import fnmatch
import sys
import tomllib
from pathlib import Path

from governance.exit_codes import ToolBroken, run
from governance.loader import EXEMPTION_KEYS, RULES_DIR

# 這支檢查在卡裡的名字。門檻只從「宣告了這支檢查」的那張卡讀。
CHECK_REL = "governance/checks/status_page_computed_not_typed.py"

MARKDOWN_SUFFIXES = (".md",)

# 標題正規化：兩頭削掉這些字元再比。刻意連數字與 - 一起削，
# 「## 進度 2026-09-09」正規化後就是「進度」。
HEADING_STRIP = " \t#*_`：:（）()[]【】「」、，。.,!！?？/\\-—0123456789"

# 門檻的形狀。打錯字的門檻等於沒有門檻，所以多一個鍵、少一個鍵、型別不對，一律回 2。
LIST_KEYS = (
    "name_patterns",
    "name_exempt_prefixes",
    "scan_exempt_prefixes",
    "frontmatter_keys",
    "frontmatter_kinds",
    "heading_labels",
)
NONEMPTY_LIST_KEYS = ("name_patterns", "frontmatter_keys", "frontmatter_kinds", "heading_labels")
INT_KEYS = ("max_checkboxes", "heading_max_chars")
ALLOW_KEY = "allow"
# 一筆放行四格：放行哪個檔、它的行數上限，加上放行條目共用的兩格（reason ＋ expires，
# 形狀定義在 governance/loader.py 的 EXEMPTION_KEYS，由規矩卡 exemptions-need-expiry 統一）。
ALLOW_ENTRY_KEYS = ("path", "max_lines", *EXEMPTION_KEYS)
SETTINGS_KEYS = (*LIST_KEYS, *INT_KEYS, ALLOW_KEY)


def _card_settings(scan_root: Path, files: list[Path]) -> dict[str, object]:
    """從掃描根自己的 ``governance/rules/`` 讀這支檢查的門檻。

    找不到、找到多張、或門檻的形狀不對，一律 raise ToolBroken——沒有門檻就不出結論。
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
            "——門檻只寫在卡上，讀不到門檻這一跑就不算數"
        )
    path, data = mine[0]
    rel = path.relative_to(scan_root)
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ToolBroken(f"{rel} 沒有 [settings] 表（門檻與名單都寫在那裡）")
    _settings_problems(settings, str(rel))
    return settings


def _settings_problems(settings: dict[str, object], rel: str) -> None:
    bad: list[str] = []
    extra = [k for k in settings if k not in SETTINGS_KEYS]
    if extra:
        bad.append(f"多了不認識的鍵 {sorted(extra)}，只認 {list(SETTINGS_KEYS)}（打錯字的門檻等於沒門檻）")
    for key in LIST_KEYS:
        value = settings.get(key)
        if not isinstance(value, list) or not all(isinstance(x, str) and x.strip() for x in value):
            bad.append(f"{key} 必須是字串 list，實際是 {value!r}")
        elif key in NONEMPTY_LIST_KEYS and not value:
            bad.append(f"{key} 不准是空 list——空的名單等於這一條沒在管")
    for key in INT_KEYS:
        value = settings.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            bad.append(f"{key} 必須是 0 或正整數，實際是 {value!r}")
    allow = settings.get(ALLOW_KEY, [])
    if not isinstance(allow, list):
        bad.append(f"{ALLOW_KEY} 寫了就必須是表的 list（一條一個放行的檔），實際是 {allow!r}")
    else:
        for i, entry in enumerate(allow):
            if not isinstance(entry, dict):
                bad.append(f"{ALLOW_KEY}[{i}] 必須是一張表（path ＋ max_lines），實際是 {entry!r}")
                continue
            if not isinstance(entry.get("path"), str) or not str(entry.get("path")).strip():
                bad.append(f"{ALLOW_KEY}[{i}] 的 path 必須是非空字串，實際是 {entry.get('path')!r}")
            cap = entry.get("max_lines")
            if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
                bad.append(
                    f"{ALLOW_KEY}[{i}] 的 max_lines 必須是正整數"
                    f"——放行不准沒有行數上限，不然等於無條件豁免（實際是 {cap!r}）"
                )
            for key in EXEMPTION_KEYS:
                value = entry.get(key)
                if not isinstance(value, str) or not value.strip():
                    bad.append(
                        f"{ALLOW_KEY}[{i}] 的 {key} 必須是非空字串"
                        f"——放行要說得出為什麼、也要有失效的那一天（實際是 {value!r}）"
                    )
            keys = [k for k in entry if k not in ALLOW_ENTRY_KEYS]
            if keys:
                bad.append(f"{ALLOW_KEY}[{i}] 多了不認識的鍵 {sorted(keys)}，只認 {list(ALLOW_ENTRY_KEYS)}")
    if bad:
        raise ToolBroken(f"{rel} 的 [settings] 形狀不對：" + "；".join(bad))


def _read_text(path: Path, rel: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"讀不到 {rel}：{exc}") from exc


def _frontmatter(text: str) -> dict[str, str]:
    """md 開頭的 ``---`` 區塊，攤成 {鍵: 值}。刻意只認一層 ``鍵: 值``，不引入 yaml 依賴。"""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    out: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() in ("---", "..."):
            break
        if ":" not in line or line.startswith((" ", "\t", "-", "#")):
            continue
        key, _, value = line.partition(":")
        out[key.strip().casefold()] = value.strip().strip("\"'").strip()
    return out


def _checkbox_count(text: str) -> int:
    n = 0
    for line in text.splitlines():
        body = line.lstrip(" \t")
        if body[:1] in ("-", "*", "+") and body[1:].lstrip(" \t")[:3] in ("[ ]", "[x]", "[X]"):
            n += 1
    return n


def _ledger_headings(text: str, labels: list[str], max_chars: int) -> list[str]:
    """挑出「已完成／待辦／進度」那種進度段落標題。"""
    folded = [label.casefold() for label in labels]
    hits: list[str] = []
    for line in text.splitlines():
        if not line.startswith("#"):
            continue
        norm = line.strip(HEADING_STRIP).strip()
        if not norm:
            continue
        low = norm.casefold()
        for label in folded:
            if low == label or (low.startswith(label) and len(norm) <= max_chars):
                hits.append(norm)
                break
    return hits


def _name_hit(rel: str, patterns: list[str], exempt: list[str]) -> str:
    """檔名那一條。回傳命中的樣式，沒命中回空字串。"""
    if any(rel.startswith(prefix) for prefix in exempt):
        return ""
    name = Path(rel).name.casefold()
    for pattern in patterns:
        if fnmatch.fnmatch(name, pattern.casefold()):
            return pattern
    return ""


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面：列舉集合扣掉卡上登記的前綴。

    每一個都真的被判過名字（是不是手寫的進度／待辦檔），md 另外被讀內容。
    名單與前綴從卡上讀，所以規矩卡也在這一組裡面（它們是 .toml，本來就沒被前綴扣掉）。
    """
    settings = _card_settings(scan_root, files)
    scan_exempt = [str(p) for p in settings["scan_exempt_prefixes"]]  # type: ignore[union-attr]  # expires=2026-12-08 reason=卡的 settings 是 tomllib 讀出來的動態表，型別標註看不出這個值是 list；到期時重審
    return sorted(
        f
        for f in files
        if not any(f.relative_to(scan_root).as_posix().startswith(prefix) for prefix in scan_exempt)
    )


def check(scan_root: Path, files: list[Path]) -> list[str]:
    settings = _card_settings(scan_root, files)
    name_patterns = list(settings["name_patterns"])  # type: ignore[arg-type]  # expires=2026-12-08 reason=卡的 settings 是 tomllib 讀出來的動態表，型別標註看不出這個值是 list；到期時重審
    name_exempt = list(settings["name_exempt_prefixes"])  # type: ignore[arg-type]  # expires=2026-12-08 reason=卡的 settings 是 tomllib 讀出來的動態表，型別標註看不出這個值是 list；到期時重審
    scan_exempt = list(settings["scan_exempt_prefixes"])  # type: ignore[arg-type]  # expires=2026-12-08 reason=卡的 settings 是 tomllib 讀出來的動態表，型別標註看不出這個值是 list；到期時重審
    fm_keys = [k.casefold() for k in settings["frontmatter_keys"]]  # type: ignore[union-attr]  # expires=2026-12-08 reason=卡的 settings 是 tomllib 讀出來的動態表，型別標註看不出這個值是 list；到期時重審
    fm_kinds = [k.casefold() for k in settings["frontmatter_kinds"]]  # type: ignore[union-attr]  # expires=2026-12-08 reason=卡的 settings 是 tomllib 讀出來的動態表，型別標註看不出這個值是 list；到期時重審
    max_checkboxes = int(settings["max_checkboxes"])  # type: ignore[call-overload]  # expires=2026-12-08 reason=卡的 settings 是 tomllib 讀出來的動態表，型別標註看不出這個值是 list；到期時重審
    labels = list(settings["heading_labels"])  # type: ignore[arg-type]  # expires=2026-12-08 reason=卡的 settings 是 tomllib 讀出來的動態表，型別標註看不出這個值是 list；到期時重審
    heading_max = int(settings["heading_max_chars"])  # type: ignore[call-overload]  # expires=2026-12-08 reason=卡的 settings 是 tomllib 讀出來的動態表，型別標註看不出這個值是 list；到期時重審
    allow = {
        str(entry["path"]): int(entry["max_lines"])
        for entry in settings.get(ALLOW_KEY, [])  # type: ignore[union-attr]  # expires=2026-12-08 reason=卡的 settings 是 tomllib 讀出來的動態表，型別標註看不出這個值是 list；到期時重審
    }

    bad: list[str] = []
    for path in targets(scan_root, files):
        rel = path.relative_to(scan_root).as_posix()
        if not path.is_file():
            # git 認得、檔案系統上不在（剛被刪掉還沒 commit）。不猜內容，跳過。
            continue

        pattern = _name_hit(rel, name_patterns, name_exempt)
        if pattern and rel not in allow:
            bad.append(
                f"{rel} 是手寫的進度／待辦／狀態檔（檔名命中卡上登記的 {pattern!r}）"
                "——這種東西一律由機器現算，待辦走 GitHub issue，不進版控"
                "（docs/decisions/backlog-in-github-issues.md）"
            )

        if rel in allow:
            lines = len(_read_text(path, rel).splitlines())
            if lines > allow[rel]:
                bad.append(
                    f"{rel} 在放行名單上，但它 {lines} 行，超過卡上登記的上限 {allow[rel]} 行"
                    "——放行帶上限才不是無條件豁免；v2 的 docs/next.md 就是這樣長成歷史帳本的"
                )

        if path.suffix.casefold() not in MARKDOWN_SUFFIXES:
            continue
        text = _read_text(path, rel)

        front = _frontmatter(text)
        for key in fm_keys:
            value = front.get(key, "").casefold()
            if value and value in fm_kinds:
                bad.append(
                    f"{rel} 的 frontmatter 把自己標成 {key}: {front[key]!r}"
                    "——這是一份待辦／進度檔，換個無辜的檔名不算過關"
                )

        count = _checkbox_count(text)
        if count > max_checkboxes:
            bad.append(
                f"{rel} 裡有 {count} 個待辦勾選框，超過卡上登記的 {max_checkboxes} 個"
                "——這是一份手寫待辦清單，待辦走 GitHub issue"
            )

        headings = _ledger_headings(text, labels, heading_max)
        if headings:
            bad.append(
                f"{rel} 有進度／待辦段落標題 {headings}"
                "——手抄的現況會漂掉（v2 的 next.md 同一份檔同時寫「已併回」與「未併回」），要現算"
            )
    return bad


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="進度／待辦／狀態一律現算，不准手寫進版控",
            targets=targets,
        )
    )
