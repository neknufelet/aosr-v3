#!/usr/bin/env python3
"""決策紙一題一檔：格式齊、狀態只認三個值、取代關係雙向、鏈不成環、同題只一份生效。

決策紙 docs/decisions/one-decision-one-paper.md 的機器版。掃描面是決策紙那一層的 md
（哪一層寫在卡上）加上所有規矩卡（必填欄位與段落名住在卡上，這支檢查要打開每一張卡去找
自己那一張）。名單全部只寫在卡的 ``[settings]``，讀不到就回 2（工具自壞），不回 0。

七條：

1. **frontmatter 八格必填**——``title``／``date_created``／``date_modified``／``status``／
   ``kind``／``supersedes``／``superseded_by``／``summary``。取代那兩格准填空字串（沒有取代
   關係就是空的），其餘空的等於沒填。整份沒有 frontmatter 區塊也算這一條。
2. **正文段落**——問題、選項、決定、為什麼、代價，加上拍板。標題正規化後**逐字相等**，
   不認前綴：寫成「問題與背景」不算有「問題」那一段，不然一個大標題可以蓋掉五段。
3. **status 只認三個值**——``accepted``／``superseded``／``proposed``。自創或打錯字的狀態
   等於沒有狀態，第 7 條那一關就會漏掉它。
4. **標了 superseded 就要說被誰取代**——``superseded_by`` 非空，而且指到的檔真的在。
5. **取代關係雙向對稱**——某張的 ``supersedes`` 指到誰，那一張的 ``superseded_by`` 就要指
   回來；反過來也一樣。單向的鏈是 v2 那份索引三條並列的形狀：看得出有人重決過一次，
   卻看不出現在算數的是哪一份。
6. **鏈不成環**——沿 ``superseded_by`` 一路走，不准繞回走過的檔。成環就永遠走不到鏈尾，
   「現在算數的是哪一份」問不出答案。
7. **同一題只准一份 accepted**——「同題」用機器判得出來的鍵認：取代關係把檔連成一群
   （``supersedes`` 與 ``superseded_by`` 都算一條邊），同一群裡算同一題。**刻意不靠標題
   或關鍵字判同題**——那不可機器判定（找碴席第二輪的原話），靠檔名判也不行（換個檔名
   就繞過去）。

取代關係的值寫成同一層裡那份決策紙的**檔名**（不帶目錄）。刻意不寫成路徑：路徑形狀的
token 另有規矩卡 refs-and-links-resolve 在管，同一件事不要兩張卡各判一次。

日期只比 frontmatter 那兩格的先後（``date_modified`` 不准早於 ``date_created``），
刻意不比 git 的提交日——squash merge 之下作者寫檔當天不可能預知合併日。

血債：沒有。兩筆相關事故為什麼不算血債，寫在卡的 ``related_lessons_why``；這張卡刻意
沒管的六件事在那張卡的檔尾。

**沒掃到東西就回 2**

讀不到卡的 ``[settings]``、``[settings]`` 形狀壞掉、掃描面上一份決策紙都沒有、某份 md
讀不開，一律 raise :class:`ToolBroken`。這一跑沒量到東西，「沒問題」這句話就不算數。
"""
from __future__ import annotations

import sys
import tomllib
from datetime import date
from pathlib import Path

from governance.exit_codes import ToolBroken, note, run
from governance.loader import RULES_DIR, setting_strings, setting_text

# 這支檢查在卡裡的名字。名單只從「宣告了這支檢查」的那張卡讀。
CHECK_REL = "governance/checks/decision_paper_structure.py"

MARKDOWN_SUFFIX = ".md"
TOML_SUFFIX = ".toml"
FRONTMATTER_FENCE = "---"
FRONTMATTER_END = ("---", "...")
SECTION_MARK = "##"
# 段落標題正規化：兩頭削掉這些字元再比。
HEADING_STRIP = " \t#*_`：:（）()[]【】「」、，。.,！!？?/\\-—"

# 卡上必須有的名單。打錯字的名單等於沒有名單，所以多一個鍵、少一個鍵、型別不對一律回 2。
LIST_KEYS = ("required_fields", "may_be_empty", "required_sections", "status_values")
TEXT_KEYS = ("decisions_prefix", "status_accepted", "status_superseded")
SETTINGS_KEYS = (*LIST_KEYS, *TEXT_KEYS)

FIELD_SUPERSEDES = "supersedes"
FIELD_SUPERSEDED_BY = "superseded_by"
FIELD_STATUS = "status"
FIELD_CREATED = "date_created"
FIELD_MODIFIED = "date_modified"


def _card_files(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面的一組：所有規矩卡（名單只寫在卡上，這支檢查要打開每一張去找自己那一張）。"""
    rules_dir = scan_root / RULES_DIR
    return sorted(f for f in files if f.parent == rules_dir and f.suffix == TOML_SUFFIX)


def _card_settings(scan_root: Path, files: list[Path]) -> dict[str, object]:
    """從掃描根自己的 ``governance/rules/`` 讀這支檢查的名單。

    找不到、找到多張、或名單的形狀不對，一律 raise ToolBroken——沒有名單就不出結論。
    """
    mine: list[tuple[Path, dict[str, object]]] = []
    for path in _card_files(scan_root, files):
        try:
            data = tomllib.loads(path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ToolBroken(f"讀不開 {path.relative_to(scan_root)}：{exc}") from exc
        if data.get("check") == CHECK_REL:
            mine.append((path, data))
    if len(mine) != 1:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下宣告 check={CHECK_REL} 的卡有 {len(mine)} 張，要剛好 1 張"
            "——名單只寫在卡上，讀不到名單這一跑就不算數"
        )
    path, data = mine[0]
    rel = path.relative_to(scan_root)
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ToolBroken(f"{rel} 沒有 [settings] 表（必填欄位與段落名都寫在那裡）")
    _assert_settings(settings, str(rel))
    return settings


def _assert_settings(settings: dict[str, object], rel: str) -> None:
    """驗名單的形狀。壞掉就 raise ToolBroken——打錯字的名單等於沒有名單。"""
    bad: list[str] = []
    extra = [k for k in settings if k not in SETTINGS_KEYS]
    if extra:
        bad.append(f"多了不認識的鍵 {sorted(extra)}，只認 {list(SETTINGS_KEYS)}（打錯字的名單等於沒名單）")
    for key in LIST_KEYS:
        value = settings.get(key)
        if not isinstance(value, list) or not all(isinstance(x, str) and x.strip() for x in value):
            bad.append(f"{key} 必須是字串 list，實際是 {value!r}")
        elif key != "may_be_empty" and not value:
            bad.append(f"{key} 不准是空 list——空的名單等於這一條沒在管")
    for key in TEXT_KEYS:
        value = settings.get(key)
        if not isinstance(value, str) or not value.strip():
            bad.append(f"{key} 必須是非空字串，實際是 {value!r}")
    if not bad:
        values = setting_strings(settings, "status_values")
        for key in ("status_accepted", "status_superseded"):
            if setting_text(settings, key) not in values:
                bad.append(f"{key}={setting_text(settings, key)!r} 不在 status_values {values} 裡")
        unknown = [f for f in setting_strings(settings, "may_be_empty")
                   if f not in setting_strings(settings, "required_fields")]
        if unknown:
            bad.append(f"may_be_empty 裡的 {unknown} 不在 required_fields 裡——放寬了一個不存在的欄位")
    if bad:
        raise ToolBroken(f"{rel} 的 [settings] 形狀不對：" + "；".join(bad))


def _frontmatter(text: str) -> dict[str, str] | None:
    """md 開頭的 ``---`` 區塊攤成 {鍵: 值}。整份沒有那個區塊就回 None。

    刻意只認一層 ``鍵: 值``，不引 yaml 依賴（理由與代價寫在卡面）。
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != FRONTMATTER_FENCE:
        return None
    out: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() in FRONTMATTER_END:
            return out
        if ":" not in line or line.startswith((" ", "\t", "-", "#")):
            continue
        key, _, value = line.partition(":")
        out[key.strip().casefold()] = value.strip().strip("\"'").strip()
    return None  # 有開頭沒有結尾的區塊當成沒有 frontmatter


def _sections(text: str) -> set[str]:
    """正文裡的段落標題（正規化後）。"""
    found: set[str] = set()
    for line in text.splitlines():
        if line.startswith(SECTION_MARK):
            norm = line.strip(HEADING_STRIP).strip()
            if norm:
                found.add(norm)
    return found


def _read_text(path: Path, rel: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"讀不到 {rel}：{exc}") from exc


def _papers(scan_root: Path, files: list[Path], prefix: str) -> list[Path]:
    """卡上登記的那一層底下的決策紙（那一層，不含子目錄）。"""
    base = scan_root / prefix.rstrip("/")
    return sorted(
        f for f in files if f.parent == base and f.suffix.casefold() == MARKDOWN_SUFFIX
    )


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """這支檢查真的會讀／會判的檔：決策紙那一層的 md ＋ 所有規矩卡。"""
    settings = _card_settings(scan_root, files)
    prefix = setting_text(settings, "decisions_prefix")
    return sorted({*_papers(scan_root, files, prefix), *_card_files(scan_root, files)})


def _field_problems(name: str, front: dict[str, str] | None, settings: dict[str, object]) -> list[str]:
    """第 1 條：frontmatter 必填欄位。整份沒有那個區塊也算這一條。"""
    if front is None:
        return [
            f"{name} 沒有 frontmatter 區塊（開頭那一段 {FRONTMATTER_FENCE} 夾起來的鍵值）"
            f"——必填的 {setting_strings(settings, 'required_fields')} 一格都讀不到"
        ]
    bad: list[str] = []
    may_be_empty = setting_strings(settings, "may_be_empty")
    for field in setting_strings(settings, "required_fields"):
        if field not in front:
            bad.append(f"{name} 的 frontmatter 缺 {field} 那一格（必填欄位登記在卡上）")
        elif not front[field].strip() and field not in may_be_empty:
            bad.append(
                f"{name} 的 frontmatter 裡 {field} 是空的——只有 {may_be_empty} 准留空"
                "（那兩格空的意思是「沒有取代關係」）"
            )
    return bad


def _section_problems(name: str, text: str, settings: dict[str, object]) -> list[str]:
    """第 2 條：正文段落。標題正規化後逐字相等，不認前綴。"""
    found = _sections(text)
    missing = [s for s in setting_strings(settings, "required_sections") if s not in found]
    if not missing:
        return []
    return [
        f"{name} 的正文缺段落 {missing}（標題寫成 `{SECTION_MARK} X`，正規化後要逐字相等）"
        f"——實際有的是 {sorted(found)}"
    ]


def _status_problems(name: str, front: dict[str, str], settings: dict[str, object]) -> list[str]:
    """第 3、4 條：status 只認三個值；標了已被取代就要說被誰取代。"""
    bad: list[str] = []
    values = setting_strings(settings, "status_values")
    status = front.get(FIELD_STATUS, "").strip()
    if status and status not in values:
        bad.append(
            f"{name} 的 status={status!r} 不在卡上登記的 {values} 裡"
            "——自創或打錯字的狀態等於沒有狀態，「同題只准一份生效」那一關會漏掉它"
        )
    if status == setting_text(settings, "status_superseded") and not front.get(FIELD_SUPERSEDED_BY, "").strip():
        bad.append(
            f"{name} 標了 status={status!r} 卻沒填 {FIELD_SUPERSEDED_BY}"
            "——讀的人只知道「這份不算了」，找不到現在算數的是哪一份"
        )
    return bad


def _link_problems(name: str, front: dict[str, str], fronts: dict[str, dict[str, str]]) -> list[str]:
    """第 5 條：取代關係雙向對稱，而且指到的檔要真的在。"""
    bad: list[str] = []
    for here, back in ((FIELD_SUPERSEDES, FIELD_SUPERSEDED_BY), (FIELD_SUPERSEDED_BY, FIELD_SUPERSEDES)):
        target = front.get(here, "").strip()
        if not target:
            continue
        if target not in fronts:
            bad.append(
                f"{name} 的 {here}={target!r} 指到的決策紙不在——那一層底下沒有這個檔名"
                "（值寫成同一層裡的檔名，不帶目錄）"
            )
            continue
        pointed = fronts[target].get(back, "").strip()
        if pointed != name:
            bad.append(
                f"{name} 的 {here}={target!r}，但 {target} 的 {back} 是 {pointed!r}，沒指回來"
                "——取代關係要雙向都標，單向的鏈看得出有人重決過，看不出現在算數的是哪一份"
            )
    return bad


def _date_problems(name: str, front: dict[str, str]) -> list[str]:
    """第 3 條旁邊那一格：改檔日不准早於建檔日。刻意不比 git 的提交日。"""
    bad: list[str] = []
    parsed: dict[str, date] = {}
    for field in (FIELD_CREATED, FIELD_MODIFIED):
        raw = front.get(field, "").strip()
        if not raw:
            continue
        try:
            parsed[field] = date.fromisoformat(raw)
        except ValueError:
            bad.append(f"{name} 的 {field}={raw!r} 看不懂——日期寫成四位數年-兩位數月-兩位數日")
    if FIELD_CREATED in parsed and FIELD_MODIFIED in parsed and parsed[FIELD_MODIFIED] < parsed[FIELD_CREATED]:
        bad.append(
            f"{name} 的 {FIELD_MODIFIED}（{front[FIELD_MODIFIED]}）早於 "
            f"{FIELD_CREATED}（{front[FIELD_CREATED]}）"
            "——改的那天不可能早於建的那天，這兩格之後要拿來排哪份新哪份舊"
        )
    return bad


def _edges(fronts: dict[str, dict[str, str]]) -> list[tuple[str, str]]:
    """取代關係連成的邊（兩個欄位都算一條邊，指到不存在的檔不算）。"""
    out: list[tuple[str, str]] = []
    for name, front in fronts.items():
        for field in (FIELD_SUPERSEDES, FIELD_SUPERSEDED_BY):
            target = front.get(field, "").strip()
            if target in fronts:
                out.append((name, target))
    return out


def _cycle_problems(fronts: dict[str, dict[str, str]]) -> list[str]:
    """第 6 條：沿 superseded_by 一路走不准繞回走過的檔。"""
    seen: set[frozenset[str]] = set()
    bad: list[str] = []
    for start in sorted(fronts):
        walked: list[str] = []
        node = start
        while node in fronts and node not in walked:
            walked.append(node)
            node = fronts[node].get(FIELD_SUPERSEDED_BY, "").strip()
        if node not in walked:
            continue
        ring = walked[walked.index(node):]
        key = frozenset(ring)
        if key in seen:
            continue
        seen.add(key)
        bad.append(
            f"取代鏈成環：{' → '.join([*ring, node])}"
            f"——沿 {FIELD_SUPERSEDED_BY} 走永遠走不到鏈尾，「現在算數的是哪一份」問不出答案"
        )
    return bad


def _components(fronts: dict[str, dict[str, str]]) -> list[list[str]]:
    """把取代關係連在一起的檔分成一群一群（一群就是機器判得出來的「同一題」）。"""
    group: dict[str, str] = {name: name for name in fronts}

    def root(name: str) -> str:
        while group[name] != name:
            group[name] = group[group[name]]
            name = group[name]
        return name

    for left, right in _edges(fronts):
        group[root(left)] = root(right)
    buckets: dict[str, list[str]] = {}
    for name in sorted(fronts):
        buckets.setdefault(root(name), []).append(name)
    return [buckets[key] for key in sorted(buckets)]


def _one_in_force_problems(fronts: dict[str, dict[str, str]], settings: dict[str, object]) -> list[str]:
    """第 7 條：同一題（同一條取代鏈上的檔）只准一份 accepted。"""
    accepted = setting_text(settings, "status_accepted")
    bad: list[str] = []
    for group in _components(fronts):
        live = [n for n in group if fronts[n].get(FIELD_STATUS, "").strip() == accepted]
        if len(live) > 1:
            bad.append(
                f"同一題有 {len(live)} 份還在生效（status={accepted!r}）：{live}"
                f"——這一群檔靠取代關係連在一起（{group}），機器判得出來是同一題；"
                "同題有兩份以上還在生效，之後引哪一份都說得通，那就是 v2 決策索引三條並列的下場"
            )
    return bad


def check(scan_root: Path, files: list[Path]) -> list[str]:
    settings = _card_settings(scan_root, files)
    prefix = setting_text(settings, "decisions_prefix")
    papers = _papers(scan_root, files, prefix)
    if not papers:
        raise ToolBroken(
            f"{scan_root}/{prefix} 底下一份決策紙都沒有——這一跑沒有量到任何對象，"
            "「格式都對」這句話不算數"
        )

    bad: list[str] = []
    fronts: dict[str, dict[str, str]] = {}
    for path in papers:
        name = path.name
        text = _read_text(path, f"{prefix}{name}")
        front = _frontmatter(text)
        bad += _field_problems(name, front, settings)
        bad += _section_problems(name, text, settings)
        fronts[name] = front if front is not None else {}

    for name, front in fronts.items():
        bad += _status_problems(name, front, settings)
        bad += _link_problems(name, front, fronts)
        bad += _date_problems(name, front)
    bad += _cycle_problems(fronts)
    bad += _one_in_force_problems(fronts, settings)

    note(f"決策紙 {len(papers)} 份（{prefix}），取代鏈連成 {len(_components(fronts))} 群")
    return bad


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="決策紙一題一檔：格式齊、取代關係雙向、鏈不成環、同題只一份生效",
            targets=targets,
        )
    )
