#!/usr/bin/env python3
"""docs 的標頭要齊；份數逐類與總量都有上限；每一行有字元上限；三類裡面不准再分層。

決策紙 ``docs/decisions/docs-three-classes.md`` 的機器版（第二道柵欄「限數量」加上「標頭齊全」）。
掃描面是 docs 那一層的全部檔案、卡上 ``line_extra_paths`` 列的那兩份入口檔，加上所有規矩卡
（上限與名單只寫在卡上，這支檢查要打開每一張卡去找自己那一張）。
五條，門檻與名單全部只寫在卡的 ``[settings]``，讀不到就回 2、不回 0：

1. **標頭齊全**——卡上 ``frontmatter_classes`` 列的那幾類（今天是 design 與 cairn）底下的每一份
   ``.md``，frontmatter 必填 ``required_fields`` 那幾格，少一格或值是空的都算沒填；``kind``
   那一格的值必須等於它住的那個目錄名（目錄名與種類同名，所以不用另開對照表）；改檔日不准早於
   建檔日，日期讀不成日期也紅（**檔壞了不是尺壞了**，所以那是違規不是工具自壞）。
   **刻意不比 git 的提交日**：squash merge 之下作者寫檔那天不可能預知合併日（藍圖的
   ``defer_reason`` 記著實測過的那一筆），拿提交日當尺會變成作者滿足不了的要求。
2. **每一類的份數上限**——逐類數那一類底下的 ``.md``，超過卡上登記的上限就紅。數的是共用外殼
   用 ``git ls-files`` 列舉出來的集合，不是自己走檔案系統：v2 那筆事故裡有一次就是用 rglob
   把被忽略的檔算進去造成假紅（把 19 份 transcript 算進 main）。
3. **總量上限**——docs 底下全部的 ``.md`` 加起來也有一條線。逐類上限每一條都沒破也一樣紅：
   v2 那筆事故的病是「各類各自壓在自己那條線下面、加起來把設計埋掉」（1046 份裡 636 份是
   過程紀錄，有用的 76 份設計文件被淹掉）。卡上驗 ``total_cap`` 必須小於三類上限相加——
   等於相加的總量上限永遠不會先響，那就是一條只會回綠的規矩。
4. **單行字元上限**——docs 底下每一份 ``.md``，加上卡上 ``line_extra_paths`` 列的那兩份入口檔，
   每一行的字元數都不准超過 ``max_line_chars``。v2 的形狀是一份治理檔的 frontmatter 版本史
   寫成一行 1,027 字元（事故 ``governance-file-has-no-guard``）：那種一行沒有人讀得完，
   diff 也看不出改了哪裡。入口檔的規矩節本來是一張卡一行整段人話（實測 1143 字元），
   這一條立起來的同一個 PR 把產生器改成一張卡只渲染第一句——那一段是產物，多寬由生成器決定。
5. **三類裡面不准再分層**——docs 底下的檔只准住在「類別目錄／檔名」那一層，再深一層就紅。
   決策紙寫的是「三類，平行不分層」，而 ``file-placement-allowlist`` 的白名單只看得到 docs
   根層那一格（那張卡自己的人話就寫「已核准的目錄裡面長多少子目錄它看不到」）。

**跟別的卡怎麼分工**（兩張卡掃同一件事會互相遮蔽，那是 v2 事故 guard-teeth-shadow-each-other
的形狀，所以每一條都寫明誰在看哪一格，卡面也寫了一份）：
``decision-paper-structure`` 判決策紙那一層的每一格，這張卡只把決策紙算進份數、不看它的欄位；
``file-placement-allowlist`` 判 docs 根層准出現哪些目錄名與檔名（類別封閉也是它），沒登記的
第四類目錄由它咬，這支檢查只印一行 ``NOTE`` 說「這一類沒登記、我不判它」；
``entry-files-rendered-from-registry`` 判入口檔有幾行，這張卡的第④條判同兩份檔每一行多寬
——兩個門檻兩張卡，各一個家。2026-09-10 老闆拍板不另立 ``doc-size-cap``，它沒人守的那兩顆牙
（總量上限、單行字元上限）就是上面的第③④條。

**刻意沒管的**：決策紙第三道柵欄「算得出來的不存檔」。一份手抄的架構圖跟一份真的設計文件
在機器眼裡長得一模一樣，判不出來；卡面記著這一格靠人看 PR（見卡的 related_lessons_why）。

血債兩筆，都寫在卡上。``docs-volume-buries-the-design``（legacy_id L37）：那筆事故的
``enforcer`` 欄逐字寫著「CI：docs 各類 `git ls-files -- docs/<class>` 計數上限測試 ＋ 目錄
schema test」，第②條就是那一句；第③條是找碴席第二輪補的（見 blueprint/cards-38.json）。
事故當下 docs 有 1046 份（其中 636 份是過程紀錄），任何一個合理的每類上限都被穿破。事故自己
記的反例（139 份在 cap 150 之下不會響，靠人事後把上限砍到 24）照抄在卡面上，不遮：上限訂多鬆
才算鬆，機器判不了；總量那條線吃同一個縫。``governance-file-has-no-guard``（legacy_id L06）：
那筆事故的 ``what_happened`` 逐字寫著「frontmatter 版本史一行 1,027 字元」，第④條在當下會回紅。
"""
from __future__ import annotations

import sys
import tomllib
from datetime import date
from pathlib import Path

from governance.exit_codes import ToolBroken, note, run
from governance.loader import RULES_DIR, setting_int, setting_strings, setting_text

# 這支檢查在卡裡的名字。上限只從「宣告了這支檢查」的那張卡讀。
CHECK_REL = "governance/checks/doc_frontmatter_and_dates.py"
TOML_SUFFIX = ".toml"
FRONTMATTER_FENCE = "---"
FRONTMATTER_END = ("---", "...")

# 卡上必須有的東西。打錯字的門檻等於沒有門檻，所以多一個鍵、少一個鍵、型別不對，一律回 2。
CAPS_KEY = "class_caps"
# 兩個正整數門檻：docs 的總份數上限（第③條）、每一行的字元上限（第④條）。
INT_KEYS = ("total_cap", "max_line_chars")
LIST_KEYS = ("frontmatter_classes", "required_fields", "line_extra_paths")
TEXT_KEYS = (
    "docs_prefix",
    "doc_suffix",
    "kind_field",
    "date_created_field",
    "date_modified_field",
)
SETTINGS_KEYS = (*TEXT_KEYS, *LIST_KEYS, *INT_KEYS, CAPS_KEY)


def _card_files(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面的一組：所有規矩卡（上限只寫在卡上，這支檢查要打開每一張去找自己那一張）。"""
    rules_dir = scan_root / RULES_DIR
    return sorted(f for f in files if f.parent == rules_dir and f.suffix == TOML_SUFFIX)


def _load_toml(path: Path, rel: str) -> dict[str, object]:
    try:
        return tomllib.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ToolBroken(f"讀不開 {rel}：{exc}") from exc


def _caps_problems(node: object) -> list[str]:
    """``class_caps`` 的形狀：一類一格，鍵是目錄名、值是正整數的份數上限。"""
    if not isinstance(node, dict) or not node:
        return [f"{CAPS_KEY} 必須是非空的表（一類一格：目錄名 = 份數上限），實際是 {node!r}"]
    bad: list[str] = []
    for name, value in node.items():
        if not isinstance(name, str) or not name.strip() or "/" in name:
            bad.append(f"{CAPS_KEY} 的鍵 {name!r} 必須是一個目錄名（非空、不帶斜線）")
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            bad.append(f"{CAPS_KEY} 的 {name} 必須是正整數，實際是 {value!r}")
    return bad


def _settings_problems(settings: dict[str, object], rel: str) -> None:
    bad: list[str] = []
    extra = [k for k in settings if k not in SETTINGS_KEYS]
    if extra:
        bad.append(f"多了不認識的鍵 {sorted(extra)}，只認 {list(SETTINGS_KEYS)}（打錯字的門檻等於沒門檻）")
    for key in TEXT_KEYS:
        value = settings.get(key)
        if not isinstance(value, str) or not value.strip():
            bad.append(f"{key} 必須是非空字串，實際是 {value!r}")
    for key in LIST_KEYS:
        value = settings.get(key)
        if not isinstance(value, list) or not value:
            bad.append(f"{key} 必須是非空的字串 list，實際是 {value!r}")
        elif not all(isinstance(x, str) and x.strip() for x in value):
            bad.append(f"{key} 的每一格都要是非空字串，實際是 {value!r}")
    for key in INT_KEYS:
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            bad.append(f"{key} 必須是正整數（那是一個上限），實際是 {value!r}")
    bad += _caps_problems(settings.get(CAPS_KEY))
    if not bad:
        registered = _caps(settings)
        missing = [c for c in setting_strings(settings, "frontmatter_classes") if c not in registered]
        if missing:
            bad.append(
                f"frontmatter_classes 裡的 {missing} 沒有登記在 {CAPS_KEY} 裡"
                "——要驗欄位的類就必須同時登記份數上限，不然那一類有幾份沒人在看"
            )
        total = setting_int(settings, "total_cap")
        by_class = sum(registered.values())
        if total >= by_class:
            bad.append(
                f"total_cap（{total}）不小於三類上限相加（{by_class}）"
                "——那樣總量這一條永遠不會先響：三類全爆了它才爆，等於一條只會回綠的規矩。"
                "v2 的病正是「各類各自壓在自己那條線下面、加起來把設計埋掉」，"
                "所以總量的線必須比相加還緊"
            )
    if bad:
        raise ToolBroken(f"{rel} 的 [settings] 形狀不對：" + "；".join(bad))


def _card_settings(scan_root: Path, files: list[Path]) -> dict[str, object]:
    """從掃描根自己的 ``governance/rules/`` 讀這支檢查的上限與名單。

    找不到、找到多張、或形狀不對，一律 raise ToolBroken——沒有尺就不出結論。
    """
    mine: list[tuple[Path, dict[str, object]]] = []
    for path in _card_files(scan_root, files):
        data = _load_toml(path, path.relative_to(scan_root).as_posix())
        if data.get("check") == CHECK_REL:
            mine.append((path, data))
    if len(mine) != 1:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下宣告 check={CHECK_REL} 的卡有 {len(mine)} 張，要剛好 1 張"
            "——上限只寫在卡上，讀不到卡這一跑就不算數"
        )
    path, data = mine[0]
    rel = path.relative_to(scan_root).as_posix()
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ToolBroken(f"{rel} 沒有 [settings] 表（上限、欄位名與類別名都寫在那裡）")
    _settings_problems(settings, rel)
    return settings


def _caps(settings: dict[str, object]) -> dict[str, int]:
    """``class_caps`` 攤成 {類別目錄名: 份數上限}。形狀已由 :func:`_settings_problems` 驗過。"""
    raw = settings.get(CAPS_KEY)
    table = raw if isinstance(raw, dict) else {}
    return {str(name): int(value) for name, value in table.items() if isinstance(value, int)}


def _docs_rels(scan_root: Path, files: list[Path], prefix: str) -> list[str]:
    """docs 那一層底下的每一個路徑（相對掃描根、posix 寫法）。"""
    rels = sorted(f.relative_to(scan_root).as_posix() for f in files)
    return [rel for rel in rels if rel.startswith(prefix)]


def _read_text(path: Path, rel: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"讀不到 {rel}：{exc}") from exc


def _frontmatter(text: str) -> dict[str, str] | None:
    """md 開頭那個 ``---`` 區塊，攤成 {鍵: 值}；沒有那個區塊回 None。

    刻意只認一層 ``鍵: 值``，不引 yaml 依賴（形狀跟 decision_paper_structure 那份同一種方言）。
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


def _class_of(rel: str, prefix: str) -> tuple[str, list[str]]:
    """把 docs 底下的一個路徑切成（類別目錄名, 那一類底下剩下的路徑段）。

    檔直接住在 docs 那一層（不在任何一類底下）時，類別名回空字串。
    """
    rest = rel[len(prefix) :].split("/")
    if len(rest) < 2:
        return "", rest
    return rest[0], rest[1:]


def _parse_date(raw: str) -> date | None:
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _date_hits(rel: str, front: dict[str, str], settings: dict[str, object]) -> list[str]:
    """第①條的日期那半：讀得成日期，而且改檔日不早於建檔日。"""
    created_field = setting_text(settings, "date_created_field")
    modified_field = setting_text(settings, "date_modified_field")
    bad: list[str] = []
    parsed: dict[str, date] = {}
    for field in (created_field, modified_field):
        raw = front.get(field, "").strip()
        if not raw:
            continue  # 缺格／空格已經由必填那一條記過一筆，不重複記
        value = _parse_date(raw)
        if value is None:
            bad.append(
                f"{rel} 的 {field} 是 {raw!r}，讀不成一個日期（要寫成年-月-日，四碼年份補零）"
                "——檔壞了不是尺壞了，所以這是違規，不是這一跑不算數"
            )
        else:
            parsed[field] = value
    created = parsed.get(created_field)
    modified = parsed.get(modified_field)
    if created is not None and modified is not None and modified < created:
        bad.append(
            f"{rel} 的 {modified_field}（{modified}）早於 {created_field}（{created}）"
            "——改檔日不可能在建檔日之前。這一關刻意只比 frontmatter 那兩格，不比 git 的提交日："
            "squash merge 之下作者寫檔那天不可能預知合併日，拿提交日當尺是作者滿足不了的要求"
        )
    return bad


def _field_hits(
    rel: str, front: dict[str, str] | None, cls: str, settings: dict[str, object]
) -> list[str]:
    """第①條：標頭齊全、種類跟目錄同名、兩個日期的先後對。"""
    required = setting_strings(settings, "required_fields")
    if front is None:
        return [
            f"{rel} 沒有 frontmatter 區塊（開頭那一段 {FRONTMATTER_FENCE} 夾起來的鍵值）"
            f"——必填的 {required} 一格都讀不到"
        ]
    bad: list[str] = []
    for field in required:
        if field not in front:
            bad.append(f"{rel} 的 frontmatter 缺 {field} 那一格（必填欄位登記在卡上）")
        elif not front[field].strip():
            bad.append(f"{rel} 的 frontmatter 裡 {field} 是空的——空的等於沒填")
    kind_field = setting_text(settings, "kind_field")
    kind = front.get(kind_field, "").strip()
    if kind and kind != cls:
        bad.append(
            f"{rel} 的 {kind_field} 寫 {kind!r}，跟它住的那個類別目錄 {cls!r} 不一致"
            "——目錄名與種類同名，兩邊不同步之後誰都不知道這一份該照哪一類的慣例讀"
        )
    return bad + _date_hits(rel, front, settings)


def _count_hits(rels: list[str], prefix: str, suffix: str, caps: dict[str, int]) -> list[str]:
    """第②條：逐類數份數。數的是列舉集合，不是檔案系統（v2 用 rglob 數過假紅）。"""
    bad: list[str] = []
    for cls in sorted(caps):
        head = f"{prefix}{cls}/"
        found = [rel for rel in rels if rel.startswith(head) and rel.endswith(suffix)]
        if len(found) > caps[cls]:
            bad.append(
                f"{prefix}{cls} 這一類有 {len(found)} 份文件，超過卡上登記的上限 {caps[cls]} 份"
                "——上限的意思不是「可以長到這裡」，是「長到這裡就得先刪再加」；"
                "v2 那筆事故就是 docs 一路長到上千份，把真正有用的那幾十份設計文件埋掉"
            )
    return bad


def _total_hits(rels: list[str], prefix: str, suffix: str, total_cap: int) -> list[str]:
    """第③條：docs 底下全部的文件加起來的總量上限。

    為什麼逐類上限之外還要一條總量：v2 那筆事故的病是「各類各自壓在自己那條線下面、
    加起來把設計埋掉」（1046 份裡 636 份是過程紀錄，有用的 76 份設計文件被淹掉）。
    只有逐類上限的話，把量攤平在幾類之間就永遠不會響。
    """
    found = [rel for rel in rels if rel.endswith(suffix)]
    if len(found) <= total_cap:
        return []
    return [
        f"{prefix} 底下總共有 {len(found)} 份文件，超過卡上登記的總量上限 {total_cap} 份"
        "——逐類上限每一條都沒破也一樣紅：v2 的病是各類各自壓線通過、加起來把設計埋掉，"
        "總量這一條就是為那件事寫的"
    ]


def _line_targets(scan_root: Path, rels: list[str], suffix: str, extra: list[str]) -> list[str]:
    """第④條要量的檔：docs 底下每一份文件，加上卡上 ``line_extra_paths`` 列的那幾份。

    ``extra`` 裡列了、但這棵樹上沒有的，印一行 NOTE 就跳過——「那一份在不在」是別張卡
    （entry-files-rendered-from-registry）的事，這張卡只管它每一行多寬。
    """
    picked = [rel for rel in rels if rel.endswith(suffix)]
    missing: list[str] = []
    for rel in extra:
        if (scan_root / rel).is_file():
            picked.append(rel)
        else:
            missing.append(rel)
    if missing:
        note(
            f"卡上 line_extra_paths 列的 {missing} 在這棵樹上不存在，第④條沒量它們"
            "——那幾份在不在由 entry-files-rendered-from-registry 那張卡判"
        )
    return sorted(set(picked))


def _line_hits(scan_root: Path, targets: list[str], max_chars: int) -> list[str]:
    """第④條：每一行的字元數上限。

    v2 的形狀：一份治理檔的 frontmatter 版本史寫成一行 1,027 字元（事故
    governance-file-has-no-guard）。那種一行沒有人讀得完，diff 也看不出改了哪裡。
    """
    bad: list[str] = []
    for rel in targets:
        text = _read_text(scan_root / rel, rel)
        for number, line in enumerate(text.splitlines(), start=1):
            if len(line) > max_chars:
                bad.append(
                    f"{rel} 第 {number} 行有 {len(line)} 個字元，超過卡上登記的上限 {max_chars}"
                    "——一行讀不完的東西沒有人會讀，diff 也看不出改了哪裡；"
                    "拆行，或者把它變成生成的（產物多寬由生成器決定）"
                )
    return bad


def _depth_hits(rels: list[str], prefix: str, caps: dict[str, int]) -> list[str]:
    """第③條：三類裡面不准再開子目錄（決策紙寫的是「三類，平行不分層」）。"""
    bad: list[str] = []
    for rel in rels:
        cls, rest = _class_of(rel, prefix)
        if cls in caps and len(rest) > 1:
            bad.append(
                f"{rel} 住在 {prefix}{cls} 底下又開的一層子目錄裡（{rest[0]}）"
                "——決策紙定的是「三類，平行不分層」，一份文件只准住在「類別目錄／檔名」那一層。"
                "docs 根層的目錄白名單只看得到最上面那一層，再深一層沒有別人在看"
            )
    return bad


def _note_unjudged(rels: list[str], prefix: str, caps: dict[str, int]) -> None:
    """把「在掃描面上、但這張卡刻意不判」的那些印出來，別讓它們安靜地被當成乾淨。"""
    unknown: set[str] = set()
    loose: list[str] = []
    for rel in rels:
        cls, _rest = _class_of(rel, prefix)
        if not cls:
            loose.append(rel)
        elif cls not in caps:
            unknown.add(cls)
    if unknown:
        note(
            f"{prefix} 底下這幾個目錄沒有登記在卡上，這支檢查不判它們、也沒在數它們的份數："
            f"{sorted(unknown)}——沒登記的類別由 file-placement-allowlist 那張卡咬"
        )
    if loose:
        note(
            f"這幾個檔直接住在 {prefix} 那一層、不在任何一類底下，這支檢查不判它們：{loose}"
            "——那一層准出現哪些檔名由 file-placement-allowlist 那張卡咬"
        )


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """這支檢查真的會讀／會判的檔：docs 整層 ＋ 卡上 line_extra_paths 那幾份 ＋ 所有規矩卡。

    docs 底下的每一個路徑都真的被判過（份數與深度對全集生效，三類的 md 另外被讀標頭，
    每一份 md 逐行量寬度）。``line_extra_paths`` 那幾份是入口檔，第④條會逐行讀它們。
    """
    settings = _card_settings(scan_root, files)
    prefix = setting_text(settings, "docs_prefix")
    docs = [f for f in files if f.relative_to(scan_root).as_posix().startswith(prefix)]
    present = set(files)
    extra = [scan_root / rel for rel in setting_strings(settings, "line_extra_paths")]
    return sorted({*docs, *(p for p in extra if p in present), *_card_files(scan_root, files)})


def check(scan_root: Path, files: list[Path]) -> list[str]:
    settings = _card_settings(scan_root, files)
    prefix = setting_text(settings, "docs_prefix")
    suffix = setting_text(settings, "doc_suffix")
    rels = _docs_rels(scan_root, files, prefix)
    if not rels:
        raise ToolBroken(
            f"列舉集合裡 {prefix} 底下一個檔都沒有——這棵樹上這張卡沒有對象，"
            "「沒問題」這句話不算數"
        )
    caps = _caps(settings)
    classes = setting_strings(settings, "frontmatter_classes")
    _note_unjudged(rels, prefix, caps)
    bad = _count_hits(rels, prefix, suffix, caps)
    bad += _total_hits(rels, prefix, suffix, setting_int(settings, "total_cap"))
    bad += _line_hits(
        scan_root,
        _line_targets(scan_root, rels, suffix, setting_strings(settings, "line_extra_paths")),
        setting_int(settings, "max_line_chars"),
    )
    bad += _depth_hits(rels, prefix, caps)
    for rel in rels:
        cls, _rest = _class_of(rel, prefix)
        path = scan_root / rel
        if cls not in classes or not rel.endswith(suffix) or not path.is_file():
            # 不是要驗標頭的那幾類、不是文件、或 git 認得但檔案系統上不在（剛刪還沒 commit）。
            continue
        bad += _field_hits(rel, _frontmatter(_read_text(path, rel)), cls, settings)
    return bad


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="docs 的設計與知識文件標頭要齊全、每一類的份數有上限、三類裡面不准再分層",
            targets=targets,
        )
    )
