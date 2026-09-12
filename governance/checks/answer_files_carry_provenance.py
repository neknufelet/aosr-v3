"""規矩卡 answer-files-carry-provenance：從上一代抓來的答案檔要帶出身，donor 三格要等於登記的那一顆。

判準（全部只比字串，不解析、不上網、不碰物件庫）：

1. **分類器**：一份 ``blueprint/*.json`` 算不算答案檔，是「檔案名命中 [settings] 的
   ``answer_file_patterns``」聯集「頂層是 dict 而且有 ``donor`` 鍵」。頂層是 list 的 json
   不是答案檔，跳過不炸（blueprint 底下有 batch1-127、collapse-by-*、rules-436 那些頂層是 list）。
2. **冒牌**：有 ``donor`` 鍵但檔名不在樣式清單 → 紅（長得像答案檔卻沒登記）；檔名在樣式清單
   卻缺 ``donor`` → 紅，而且缺 donor 它根本不該被當答案檔放行。
3. **必填**：答案檔的 ``donor`` 要是 dict；``donor.tag`` 等於 ``donor_tag``、``donor.commit``
   等於 ``donor_commit``（四十位，字串相等，不解析——那顆 sha 不在本機物件庫）、``donor.clean``
   是 ``True``；``note`` 是非空字串。
4. **env**：必填，除非那份答案檔自己的 ``schema`` 在 ``env_optional_schemas``（今天只登記 3，
   config_cut1 的產生器不寫 env，另有票補）。

跟 identity-strings-generated 的分工：那張問「解析得到嗎」（``git cat-file -e``），這張問
「等於登記的那一顆嗎」（字串相等）。兩張都認領 hand-copied-identity-strings-drift，載體不同。

回 2（這一跑不算數）：讀不到卡的 settings、掃描面一份 json 都沒有、某份 json 剖不開。
"""
from __future__ import annotations

import fnmatch
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from governance.cloud_receipts import card_settings
from governance.exit_codes import ToolBroken, note, run
from governance.loader import setting_strings, setting_text

CHECK_REL = "governance/checks/answer_files_carry_provenance.py"


@dataclass(frozen=True)
class Rules:
    """卡上的判準：檔名樣式、donor 的 tag 與 commit、env 免填的 schema 名單。"""

    patterns: tuple[str, ...]
    donor_tag: str
    donor_commit: str
    env_optional_schemas: frozenset[int]


def read_rules(settings: Mapping[str, object]) -> Rules:
    optional = settings.get("env_optional_schemas")
    if not isinstance(optional, list):
        raise ToolBroken(
            f"卡上的 env_optional_schemas 必須是整數 list，實際是 {optional!r}——讀不到判準就不出結論"
        )
    schemas: set[int] = set()
    for item in optional:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ToolBroken(
                f"卡上的 env_optional_schemas 裡有非整數的一項（{item!r}）——名單形狀壞掉就不出結論"
            )
        schemas.add(item)
    return Rules(
        patterns=tuple(setting_strings(settings, "answer_file_patterns")),
        donor_tag=setting_text(settings, "donor_tag"),
        donor_commit=setting_text(settings, "donor_commit"),
        env_optional_schemas=frozenset(schemas),
    )


def _json_files(scan_root: Path, files: list[Path]) -> list[Path]:
    """blueprint 底下一層的 .json。掃描面只到這一層，不遞迴。"""
    return sorted(
        f
        for f in files
        if f.parent == scan_root / "blueprint" and f.suffix == ".json"
    )


def _name_matches(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def _top_level(path: Path, rel: str) -> object:
    """讀一份 json 的頂層值。剖不開就 ToolBroken（回 2），不當答案檔跳過。"""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"{rel} 讀不開（{exc}）——我沒看懂就不出結論") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolBroken(f"{rel} 不是合法 JSON（{exc}）——我沒看懂就不出結論") from exc


def _hits_for(f: Path, rel: str, name: str, rules: Rules) -> list[str]:
    data = _top_level(f, rel)
    matched = _name_matches(name, rules.patterns)
    if not isinstance(data, dict):
        # 頂層是 list：不是答案檔，跳過不炸。
        return []

    # 冒牌：有 donor 鍵但檔名不在樣式清單 → 長得像答案檔卻沒登記。
    if "donor" in data and not matched:
        return [
            f"{rel} 有 donor 鍵但檔名 {name!r} 不在樣式清單裡——長得像答案檔卻沒登記，"
            "多一份沒登記樣式的冒牌答案檔就是違規"
        ]
    if not matched:
        # 沒命中樣式、也沒 donor 鍵：不是答案檔，不管。
        return []

    donor = data.get("donor")
    # 在樣式清單裡卻缺 donor → 紅。
    if not isinstance(donor, dict):
        return [
            f"{rel} 檔名命中樣式清單，但沒有 donor 記號——答案檔要帶出身，缺 donor 不放行"
        ]

    bad: list[str] = []
    tag = donor.get("tag")
    if tag != rules.donor_tag:
        bad.append(f"{rel} 的 donor.tag={tag!r}，應為 {rules.donor_tag!r}——tag 打錯，出身對不上")
    commit = donor.get("commit")
    if commit != rules.donor_commit:
        bad.append(
            f"{rel} 的 donor.commit={commit!r}，應為登記的那顆 commit——"
            "七份答案檔必須來自同一顆上一代提交，這一格跟登記簿漂開就紅"
        )
    clean = donor.get("clean")
    if clean is not True:
        bad.append(f"{rel} 的 donor.clean={clean!r}，應為 True——自報的乾淨欄位不是 True 不放行")
    note_field = data.get("note")
    if not isinstance(note_field, str) or not note_field.strip():
        bad.append(f"{rel} 的 note 不是非空字串（{note_field!r}）——答案檔要記得出身來歷")

    if "env" not in data:
        schema = data.get("schema")
        if schema not in rules.env_optional_schemas:
            bad.append(
                f"{rel} 缺 env，而它的 schema={schema!r} 不在 env_optional_schemas"
                f"{rules.env_optional_schemas!r} 裡——env 必填，除非那份 schema 在免填名單上"
            )
    return bad


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """這支檢查真的會讀的檔：blueprint 底下一層的 .json，加上自己那張卡。"""
    card, _settings = card_settings(scan_root, files, CHECK_REL)
    return sorted({card, *_json_files(scan_root, files)})


def check(scan_root: Path, files: list[Path]) -> list[str]:
    card, settings = card_settings(scan_root, files, CHECK_REL)
    rules = read_rules(settings)
    jsons = _json_files(scan_root, files)
    if not jsons:
        raise ToolBroken("blueprint/ 底下一份 json 都沒有——沒有答案檔可對就不出結論")

    hits: list[str] = []
    recognized = 0
    for f in jsons:
        rel = f.relative_to(scan_root).as_posix()
        hits += _hits_for(f, rel, f.name, rules)
        # 認了多少份答案檔（冒牌也照算，它也是被分類器認出來的答案檔形狀）。
        if _name_matches(f.name, rules.patterns):
            recognized += 1
    note(f"掃了 {len(jsons)} 份 blueprint json，認出 {recognized} 份答案檔")
    return hits


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="答案檔的 donor 三格與 env 要等於卡上登記的那一顆；讀不到 settings 或某份 json 剖不開回 2",
            targets=targets,
        )
    )
