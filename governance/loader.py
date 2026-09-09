"""規矩卡的載入器。

讀 ``<scan_root>/governance/rules/*.toml``，每張卡驗必填欄位，格式壞掉就 raise
:class:`CardError`。門檻與掃描面都只寫在卡自己的 toml 裡，不寫死在檢查程式。

必填欄位（缺一不可）：

* ``id``——必須跟檔名一致
* ``human``——人話一句，講清楚什麼情況會紅
* ``scope``——這張卡要掃哪些路徑（非空 list）。寫法見 :data:`SCOPE_KINDS` 上面那一段：
  ``.``（整棵樹）、目錄前綴、單檔、glob（``**/*.py``），前面加 ``!`` 就是扣掉。
  規矩卡 ``scan-scope-has-no-holes`` 會把它展開，跟那支檢查實際列舉出來的檔案集合比對
* ``check``——檢查程式，必須指向 ``governance/checks/`` 底下真的存在的模組
* ``negative_fixture``——必紅樣本目錄，必須真的存在，底下至少一個樣本
* ``control_fixture``——控制樣本目錄，必須是 negative_fixture 底下的一個樣本
* ``declared_level``——``blocking`` 或 ``advisory``，二選一
* ``enforcer``——只認版控裡的機器（見 :data:`ENFORCERS`），填人／審查員／本機 hook 一律紅
* ``job``——GitHub Actions 的 job 名
* ``external_tools``——外部工具清單，**沒有也要明寫 ``[]``**，不准省略
* ``[mountpoint]``——掛載點四段寫滿：何時觸發／跑在哪台／結果去哪／卡不卡得住合併

選填欄位（沒寫不罰，寫了就要對）：

* ``scope_kind``——掃描面是哪一種東西：``files``（預設，一組檔案）或 ``commits``
  （提交 metadata，那種卡的掃描面不是路徑，集合比不了）。見 :data:`SCOPE_KINDS`
* ``blood_debt``——這張卡對到的 v2 事故 id（找碴確認「在事故當下會回紅」的才寫這裡）
* ``related_lessons``——有關聯但找碴判「不算血債」的事故 id（例如違規物件落在掃描面外）
* ``related_lessons_why``——``related_lessons`` 非空時必填：說明為什麼不算血債
* ``[junit]``——這張卡要判的 pytest junit 收據，三段都要寫滿：``path``（相對掃描根，不准
  絕對路徑、不准 ``..``）、``collected_floor``（收集數地板，正整數）、``floor_stale_ratio``
  （地板過期的倍數，大於 1 的數字——實跑收集數超過 ``collected_floor × floor_stale_ratio``
  就代表這個地板早該調了）。門檻只寫在卡自己的 toml 裡，檢查程式不准有預設值——沒有卡宣告
  ``[junit]``，那支檢查就該回 2 說「沒東西可判」
* ``[[allowlist]]``——分層白名單，一層一個表：``dir``（``"."`` 是掃描根）、``files``
  （那一層准出現的檔名）、``dirs``（准出現的目錄名），沒有也要明寫 ``[]``。清單是資料、
  放在卡裡，不寫死在檢查程式裡——改寬清單就要走 PR。
* ``tool_broken_fixture``——**該回 2 的**樣本目錄（每個子目錄一份，餵下去必須回「工具自壞」）。
  必紅樣本（``negative_fixture``）底下每一份都必須回 1，所以「這一跑不算數」那種樣本
  放不進去，只能另開一個目錄；寫了這欄，後設測試就多跑一回合（見
  ``tests/test_fixture_runner.py`` 的第 6 回）。必須真的存在、底下至少一份樣本，
  而且不准放在 ``negative_fixture`` 底下（不然同一份樣本會被要求同時回 1 又回 2）。
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

RULES_DIR = "governance/rules"
CHECKS_DIR = "governance/checks"

# 執行者白名單：只認版控裡的機器。人、審查員、健檢報告、本機 hook 都不算。
ENFORCERS = ("github-actions-job", "pytest-meta-test")

# 宣告等級：擋得住，還是只會叫。
LEVELS = ("blocking", "advisory")

# 掛載點四段：何時觸發／跑在哪台／結果去哪／卡不卡得住合併。
MOUNTPOINT_SEGMENTS = ("trigger", "runs_on", "result_to", "blocks_merge")

# 「跑在哪台」只認兩種：雲端那一跑（權威），或本機鏡像（只是提早知道）。
MOUNTPOINT_RUNS_ON = ("cloud-authority", "local-mirror")

# 選填欄位：血債（找碴確認會咬的 v2 事故）與關聯事故（有關但判不算血債的）。
OPTIONAL_LIST_FIELDS = ("blood_debt", "related_lessons")

# 掃描面是哪一種東西。``files``（預設）＝一組檔案，卡的 ``scope`` 展開得出檔案集合，
# 由 scan-scope-has-no-holes 拿去跟檢查實際列舉的集合比對；``commits``＝提交 metadata，
# 那種卡的掃描面根本不是路徑（commit-author-allowlisted 讀的是 author／committer），
# 集合比不了，只驗宣告在不在、宣告的名單檔存不存在。
SCOPE_KINDS = ("files", "commits")
DEFAULT_SCOPE_KIND = "files"

# scope 的寫法（一條一行，前面加 ``!`` 就是扣掉）：
#   ``.``                掃描根底下每一個檔
#   ``docs``／``docs/``   那個目錄底下每一個檔（遞迴）
#   ``pyproject.toml``   就那一個檔
#   ``**/*.py``          glob。``**/`` 吃掉零個或多個目錄段（所以根層的 .py 也算），
#                        ``*`` 不跨 ``/``，``?`` 一個字元
# 語法只有這一份定義，載入器與檢查程式共用（見 scope_matches／expand_scope）。
SCOPE_NEGATE = "!"
SCOPE_ROOT = "."
SCOPE_GLOB_CHARS = "*?"

# 選填的 [junit] 表：收據在哪（path）＋收集數地板（collected_floor）＋地板過期的倍數
# （floor_stale_ratio）。三段都要寫滿——少一段、多一段、型別不對，都算卡壞掉。
JUNIT_KEYS = ("path", "collected_floor", "floor_stale_ratio")
# 選填欄位 [[allowlist]]：分層白名單。一層三個鍵，缺一不可（沒有也要明寫空 list）。
ALLOWLIST_FIELD = "allowlist"
ALLOWLIST_KEYS = ("dir", "files", "dirs")

STRING_FIELDS = (
    "id",
    "human",
    "check",
    "negative_fixture",
    "control_fixture",
    "declared_level",
    "enforcer",
    "job",
)
REQUIRED_FIELDS = (*STRING_FIELDS, "scope", "external_tools", "mountpoint")


class CardError(Exception):
    """卡的格式壞掉。訊息裡列出這張卡所有的問題，不只第一個。"""


@dataclass(frozen=True)
class AllowlistLevel:
    """白名單的一層：``dir`` 那一格底下准出現哪些檔名（``files``）與目錄名（``dirs``）。"""

    dir: str
    files: tuple[str, ...]
    dirs: tuple[str, ...]

    @property
    def prefix(self) -> str:
        """接在路徑前面的前綴。``"."`` 是掃描根本身，前綴是空字串。"""
        return "" if self.dir == "." else self.dir.rstrip("/") + "/"


@dataclass(frozen=True)
class Card:
    id: str
    human: str
    scope: list[str]
    scope_kind: str
    check: str
    negative_fixture: str
    control_fixture: str
    declared_level: str
    enforcer: str
    job: str
    external_tools: list[str]
    mountpoint: dict[str, object]
    source: Path
    blood_debt: tuple[str, ...] = ()
    related_lessons: tuple[str, ...] = ()
    related_lessons_why: str = ""
    junit: dict[str, object] | None = None

    @property
    def junit_path(self) -> str:
        """卡宣告的 junit 收據路徑（相對掃描根）。沒宣告就是空字串。"""
        return str(self.junit["path"]) if self.junit else ""

    @property
    def junit_floor(self) -> int:
        """卡宣告的收集數地板。沒宣告就是 0（代表這張卡不管收據）。"""
        return int(self.junit["collected_floor"]) if self.junit else 0

    @property
    def junit_stale_ratio(self) -> float:
        """地板過期的倍數。實跑收集數超過「地板 × 這個倍數」就是地板早該調了。

        沒宣告 ``[junit]`` 就是 0（代表這張卡不管收據，也沒有地板可以過期）。
        """
        return float(self.junit["floor_stale_ratio"]) if self.junit else 0.0
    allowlist: tuple[AllowlistLevel, ...] = ()
    tool_broken_fixture: str = ""

    @property
    def check_module(self) -> str:
        """``governance/checks/rule_card_required_fields.py``
        -> ``governance.checks.rule_card_required_fields``（給 ``python -m`` 用）。"""
        return self.check.removesuffix(".py").replace("/", ".")

    def negative_cases(self, scan_root: Path) -> list[Path]:
        """必紅樣本目錄底下的每一個樣本。每一個都是一個掃描根。"""
        base = scan_root / self.negative_fixture
        return sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []

    def control_path(self, scan_root: Path) -> Path:
        return scan_root / self.control_fixture

    def tool_broken_cases(self, scan_root: Path) -> list[Path]:
        """該回 2 的樣本。沒宣告這欄就是空 list（大多數卡不需要）。"""
        if not self.tool_broken_fixture:
            return []
        base = scan_root / self.tool_broken_fixture
        return sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []

    @property
    def blocks_merge(self) -> bool:
        return bool(self.mountpoint.get("blocks_merge"))


def _field_problems(data: dict[str, object], stem: str) -> list[str]:
    bad: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in data:
            bad.append(f"缺必填欄位 {field}")
    for field in STRING_FIELDS:
        value = data.get(field)
        if field in data and (not isinstance(value, str) or not value.strip()):
            bad.append(f"欄位 {field} 必須是非空字串，實際是 {value!r}")
    if isinstance(data.get("id"), str) and data["id"] != stem:
        bad.append(f"id={data['id']!r} 跟檔名 {stem!r} 不一致")
    scope = data.get("scope")
    if "scope" in data and (
        not isinstance(scope, list) or not scope or not all(isinstance(s, str) and s.strip() for s in scope)
    ):
        bad.append(f"欄位 scope 必須是非空的字串 list（這張卡要掃哪些路徑），實際是 {scope!r}")
    elif isinstance(scope, list):
        bad += scope_problems([str(s) for s in scope])
    kind = data.get("scope_kind")
    if "scope_kind" in data and kind not in SCOPE_KINDS:
        bad.append(
            f"選填欄位 scope_kind={kind!r} 不在列舉裡，只認 {list(SCOPE_KINDS)}"
            f"（沒寫就是 {DEFAULT_SCOPE_KIND}：scope 展開得出一組檔案，"
            "而且必須等於那支檢查實際列舉出來的集合）"
        )
    tools = data.get("external_tools")
    if "external_tools" in data and (
        not isinstance(tools, list) or not all(isinstance(t, str) and t.strip() for t in tools)
    ):
        bad.append(f"欄位 external_tools 必須是字串 list（沒有外部工具就明寫 []），實際是 {tools!r}")
    level = data.get("declared_level")
    if "declared_level" in data and level not in LEVELS:
        bad.append(f"declared_level={level!r} 不在列舉裡，只認 {list(LEVELS)}")
    enforcer = data.get("enforcer")
    if "enforcer" in data and enforcer not in ENFORCERS:
        bad.append(
            f"enforcer={enforcer!r} 不是版控裡的機器；只認 {list(ENFORCERS)}"
            "（人、審查員、健檢報告、本機 hook、「開工時自己記得」都不算）"
        )
    for field in OPTIONAL_LIST_FIELDS:
        value = data.get(field)
        if field in data and (
            not isinstance(value, list) or not all(isinstance(s, str) and s.strip() for s in value)
        ):
            bad.append(f"選填欄位 {field} 寫了就必須是字串 list（v2 事故 id），實際是 {value!r}")
    related = data.get("related_lessons")
    why = data.get("related_lessons_why")
    if isinstance(related, list) and related and (not isinstance(why, str) or not why.strip()):
        bad.append(
            "related_lessons 非空時 related_lessons_why 必填——掛了事故卻不算血債，"
            f"要說明為什麼（實際是 {why!r}）"
        )
    broken = data.get("tool_broken_fixture")
    if "tool_broken_fixture" in data and (not isinstance(broken, str) or not broken.strip()):
        bad.append(f"選填欄位 tool_broken_fixture 寫了就必須是非空字串（該回 2 的樣本目錄），實際是 {broken!r}")
    check = data.get("check")
    if isinstance(check, str) and not check.startswith(CHECKS_DIR + "/"):
        bad.append(f"check={check!r} 必須指向 {CHECKS_DIR}/ 底下的模組")
    return bad


def scope_problems(scope: list[str]) -> list[str]:
    """驗 ``scope`` 每一條的形狀。回傳所有問題（空 list 就是寫得對）。

    只准相對路徑：絕對路徑與 ``..`` 會把掃描面帶出掃描根；全部都是扣掉（``!``）也不行
    ——那樣展開出來是空集合，等於沒有宣告掃描面。
    """
    bad: list[str] = []
    positives = 0
    for entry in scope:
        body = entry[len(SCOPE_NEGATE):] if entry.startswith(SCOPE_NEGATE) else entry
        if not entry.startswith(SCOPE_NEGATE):
            positives += 1
        where = f"scope 裡的 {entry!r}"
        if not body.strip():
            bad.append(f"{where} 只有一個 {SCOPE_NEGATE}，沒說要扣掉什麼")
            continue
        if body != body.strip():
            bad.append(f"{where} 前後有空白")
        if body.startswith("/"):
            bad.append(f"{where} 是絕對路徑——掃描面只准寫成相對掃描根的路徑")
        segments = body.strip("/").split("/")
        if ".." in segments:
            bad.append(f"{where} 夾了 ..——掃描面不准往掃描根外面指")
        if body.strip("/") != SCOPE_ROOT and SCOPE_ROOT in segments:
            bad.append(f"{where} 夾了 .——掃描根本身就寫 \".\"，不要夾在路徑中間")
    if scope and positives == 0:
        bad.append(
            f"scope 每一條都是扣掉（{SCOPE_NEGATE}）——展開出來是空集合，"
            "等於沒有宣告掃描面"
        )
    return bad


def _glob_regex(pattern: str) -> re.Pattern[str]:
    """把 scope 的 glob 譯成正則。``**/`` 吃零個或多個目錄段，``*`` 不跨 ``/``。"""
    out: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:[^/]+/)*")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + "$")


def scope_matches(entry: str, name: str) -> bool:
    """``scope`` 的一條（已經去掉 ``!``）有沒有蓋到這個路徑（相對掃描根、posix 寫法）。"""
    body = entry.strip("/")
    if body == SCOPE_ROOT:
        return True
    if any(ch in body for ch in SCOPE_GLOB_CHARS):
        return bool(_glob_regex(body).match(name))
    return name == body or name.startswith(body + "/")


def expand_scope(scope: list[str], names: list[str]) -> set[str]:
    """把卡宣告的 ``scope`` 展開成檔案集合：正的聯集，扣掉負的聯集。

    ``names`` 是列舉出來的路徑（相對掃描根、posix）。展開只認列舉集合裡的東西——
    宣告了一個版控裡沒有的目錄不會變出檔案來（也不算違規，那個前綴今天沒有對象）。
    """
    picked: set[str] = set()
    for entry in scope:
        if entry.startswith(SCOPE_NEGATE):
            continue
        picked |= {name for name in names if scope_matches(entry, name)}
    for entry in scope:
        if not entry.startswith(SCOPE_NEGATE):
            continue
        body = entry[len(SCOPE_NEGATE):]
        picked -= {name for name in picked if scope_matches(body, name)}
    return picked


def _junit_problems(data: dict[str, object]) -> list[str]:
    """選填的 ``[junit]`` 表：路徑、地板、地板過期倍數都要寫滿，而且路徑不准跑出掃描根。"""
    if "junit" not in data:
        return []
    junit = data["junit"]
    if not isinstance(junit, dict):
        return [
            "[junit] 必須是一個表（path ＋ collected_floor ＋ floor_stale_ratio），"
            f"實際是 {junit!r}"
        ]
    bad: list[str] = []
    missing = [k for k in JUNIT_KEYS if k not in junit]
    if missing:
        bad.append(
            f"[junit] 少了 {missing}——收據路徑 path、收集數地板 collected_floor、"
            "地板過期倍數 floor_stale_ratio 三段都要寫滿"
        )
    extra = [k for k in junit if k not in JUNIT_KEYS]
    if extra:
        bad.append(f"[junit] 多了不認識的段 {extra}，只認 {list(JUNIT_KEYS)}")
    path = junit.get("path")
    if "path" in junit:
        if not isinstance(path, str) or not path.strip():
            bad.append(f"[junit] path 必須是非空字串，實際是 {path!r}")
        elif Path(path).is_absolute() or ".." in Path(path).parts:
            bad.append(
                f"[junit] path={path!r} 必須是掃描根底下的相對路徑"
                "——絕對路徑或 .. 會把檢查帶出掃描根，檢查程式不准讀那裡"
            )
    floor = junit.get("collected_floor")
    if "collected_floor" in junit and (
        isinstance(floor, bool) or not isinstance(floor, int) or floor < 1
    ):
        bad.append(
            f"[junit] collected_floor 必須是 1 以上的整數（今天實跑的收集數），實際是 {floor!r}"
            "——地板寫 0 等於沒有地板"
        )
    ratio = junit.get("floor_stale_ratio")
    if "floor_stale_ratio" in junit and (
        isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or ratio <= 1
    ):
        bad.append(
            f"[junit] floor_stale_ratio 必須是大於 1 的數字，實際是 {ratio!r}"
            "——寫 1 或更小等於「實跑數只要沒掉就算地板過期」，那條會天天紅；"
            "寫 1 以下更是把地板變成天花板"
        )
    return bad


def _name_list_problems(entry: dict[str, object], key: str, where: str) -> list[str]:
    """白名單裡的一張名字清單：字串 list，每個都是單一層的名字（不准夾 ``/``）。"""
    value = entry.get(key)
    if key not in entry:
        return [f"{where} 缺 {key}（沒有也要明寫 {key} = []）"]
    if not isinstance(value, list) or not all(isinstance(n, str) and n.strip() for n in value):
        return [f"{where} 的 {key} 必須是字串 list，實際是 {value!r}"]
    bad: list[str] = []
    for name in value:
        if "/" in name:
            bad.append(f"{where} 的 {key} 裡 {name!r} 夾了 /——白名單一層只列那一層的名字，不列路徑")
        elif name != name.strip():
            bad.append(f"{where} 的 {key} 裡 {name!r} 前後有空白")
    if len(set(value)) != len(value):
        bad.append(f"{where} 的 {key} 有重複的名字：{value!r}")
    return bad


def allowlist_problems(data: dict[str, object]) -> list[str]:
    """驗 ``[[allowlist]]`` 的形狀。沒寫這個欄位就沒有問題（選填）。

    檢查程式與這裡共用同一個函式：白名單的形狀只有一份定義，兩邊不會各自解讀。
    """
    if ALLOWLIST_FIELD not in data:
        return []
    levels = data[ALLOWLIST_FIELD]
    if not isinstance(levels, list) or not levels or not all(isinstance(x, dict) for x in levels):
        return [f"選填欄位 {ALLOWLIST_FIELD} 寫了就必須是非空的 [[allowlist]] 表陣列，實際是 {levels!r}"]

    bad: list[str] = []
    seen: list[str] = []
    for index, entry in enumerate(levels):
        where = f"[[{ALLOWLIST_FIELD}]] 第 {index + 1} 層"
        extra = [k for k in entry if k not in ALLOWLIST_KEYS]
        if extra:
            bad.append(f"{where} 多了不認識的鍵 {sorted(extra)}，只認 {list(ALLOWLIST_KEYS)}")
        raw_dir = entry.get("dir")
        if "dir" not in entry:
            bad.append(f"{where} 缺 dir（那一層是哪個目錄，掃描根本身寫 \".\"）")
        elif not isinstance(raw_dir, str) or not raw_dir.strip():
            bad.append(f"{where} 的 dir 必須是非空字串，實際是 {raw_dir!r}")
        elif raw_dir != raw_dir.strip() or raw_dir.startswith("/") or raw_dir.endswith("/"):
            bad.append(f"{where} 的 dir={raw_dir!r} 前後有空白或斜線——要寫成相對掃描根的路徑")
        elif raw_dir != "." and (".." in raw_dir.split("/") or "." in raw_dir.split("/")):
            bad.append(f"{where} 的 dir={raw_dir!r} 夾了 . 或 ..——白名單不准往外指")
        else:
            if raw_dir in seen:
                bad.append(f"{where} 的 dir={raw_dir!r} 跟前面某一層重複——同一層只准列一次")
            seen.append(raw_dir)
        bad += _name_list_problems(entry, "files", where)
        bad += _name_list_problems(entry, "dirs", where)
        files = entry.get("files")
        dirs = entry.get("dirs")
        if isinstance(files, list) and isinstance(dirs, list):
            both = sorted({n for n in files if isinstance(n, str)} & {n for n in dirs if isinstance(n, str)})
            if both:
                bad.append(f"{where} 的 {both} 同時列在 files 與 dirs——同一個名字不會又是檔又是目錄")
    return bad


def allowlist_levels(data: dict[str, object]) -> tuple[AllowlistLevel, ...]:
    """把驗過的 ``[[allowlist]]`` 讀成一串 :class:`AllowlistLevel`。

    呼叫前必須先過 :func:`allowlist_problems`；形狀壞掉的時候這裡不負責報錯。
    """
    if ALLOWLIST_FIELD not in data:
        return ()
    out: list[AllowlistLevel] = []
    for entry in data[ALLOWLIST_FIELD]:  # type: ignore[union-attr]
        out.append(
            AllowlistLevel(
                dir=entry["dir"],
                files=tuple(entry["files"]),
                dirs=tuple(entry["dirs"]),
            )
        )
    return tuple(out)


def _mountpoint_problems(data: dict[str, object]) -> list[str]:
    mount = data.get("mountpoint")
    if "mountpoint" not in data:
        return []
    if not isinstance(mount, dict):
        return [f"mountpoint 必須是四段的表，實際是 {mount!r}"]
    bad: list[str] = []
    missing = [seg for seg in MOUNTPOINT_SEGMENTS if seg not in mount]
    if missing:
        bad.append(
            f"掛載點四段沒寫滿，缺 {missing}（要寫滿：何時觸發 trigger／跑在哪台 runs_on／"
            f"結果去哪 result_to／卡不卡得住合併 blocks_merge，只寫了 {len(mount)} 段）"
        )
    extra = [seg for seg in mount if seg not in MOUNTPOINT_SEGMENTS]
    if extra:
        bad.append(f"掛載點多了不認識的段 {extra}，只認 {list(MOUNTPOINT_SEGMENTS)}")
    if "runs_on" in mount and mount["runs_on"] not in MOUNTPOINT_RUNS_ON:
        bad.append(
            f"掛載點 runs_on={mount['runs_on']!r} 不在列舉裡，只認 {list(MOUNTPOINT_RUNS_ON)}"
        )
    if "blocks_merge" in mount and not isinstance(mount["blocks_merge"], bool):
        bad.append(f"掛載點 blocks_merge 必須是 true／false，實際是 {mount['blocks_merge']!r}")
    if data.get("declared_level") == "blocking" and mount.get("blocks_merge") is False:
        bad.append("declared_level 宣告 blocking，掛載點卻寫 blocks_merge = false——自稱擋得住，實際只會叫")
    if data.get("declared_level") == "advisory" and mount.get("blocks_merge") is True:
        bad.append("declared_level 宣告 advisory，掛載點卻寫 blocks_merge = true——兩欄互相打架")
    return bad


def _path_problems(data: dict[str, object], scan_root: Path) -> list[str]:
    bad: list[str] = []
    check = data.get("check")
    if isinstance(check, str) and check.startswith(CHECKS_DIR + "/") and not (scan_root / check).is_file():
        bad.append(f"check 指向的模組不存在：{check}")
    neg = data.get("negative_fixture")
    if isinstance(neg, str):
        neg_dir = scan_root / neg
        if not neg_dir.is_dir():
            bad.append(f"negative_fixture 指向的目錄不存在：{neg}")
        elif not any(p.is_dir() for p in neg_dir.iterdir()):
            bad.append(f"negative_fixture 目錄 {neg} 底下沒有任何樣本——等於沒有必紅樣本")
    ctrl = data.get("control_fixture")
    if isinstance(ctrl, str):
        ctrl_dir = scan_root / ctrl
        if not ctrl_dir.is_dir():
            bad.append(f"control_fixture 指向的目錄不存在：{ctrl}")
        elif isinstance(neg, str) and ctrl_dir.parent != (scan_root / neg):
            bad.append(f"control_fixture {ctrl} 不在 negative_fixture {neg} 底下")
    broken = data.get("tool_broken_fixture")
    if isinstance(broken, str) and broken.strip():
        broken_dir = scan_root / broken
        if not broken_dir.is_dir():
            bad.append(f"tool_broken_fixture 指向的目錄不存在：{broken}")
        elif not any(p.is_dir() for p in broken_dir.iterdir()):
            bad.append(f"tool_broken_fixture 目錄 {broken} 底下沒有任何樣本——等於沒有那一回合")
        if isinstance(neg, str) and (scan_root / neg) in (broken_dir, *broken_dir.parents):
            bad.append(
                f"tool_broken_fixture {broken} 放在 negative_fixture {neg} 底下"
                "——同一份樣本會被要求同時回 1 又回 2，要另開一個目錄"
            )
    return bad


def card_problems(path: Path, scan_root: Path) -> list[str]:
    """驗一張卡，回傳所有問題（空 list 就是這張卡填得對）。"""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [f"讀不到卡：{exc}"]
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        return [f"TOML 解析失敗：{exc}"]
    bad = _field_problems(data, path.stem)
    bad += _junit_problems(data)
    bad += _mountpoint_problems(data)
    bad += allowlist_problems(data)
    bad += _path_problems(data, scan_root)
    return bad


def load_card(path: Path, scan_root: Path) -> Card:
    """讀一張卡。格式壞掉就 raise :class:`CardError`。"""
    bad = card_problems(path, scan_root)
    if bad:
        raise CardError(f"{path.name}：" + "；".join(bad))
    data = tomllib.loads(path.read_bytes().decode("utf-8"))
    return Card(
        id=data["id"],
        human=data["human"],
        scope=list(data["scope"]),
        scope_kind=str(data.get("scope_kind", DEFAULT_SCOPE_KIND)),
        check=data["check"],
        negative_fixture=data["negative_fixture"],
        control_fixture=data["control_fixture"],
        declared_level=data["declared_level"],
        enforcer=data["enforcer"],
        job=data["job"],
        external_tools=list(data["external_tools"]),
        mountpoint=dict(data["mountpoint"]),
        source=path,
        blood_debt=tuple(data.get("blood_debt", ())),
        related_lessons=tuple(data.get("related_lessons", ())),
        related_lessons_why=data.get("related_lessons_why", ""),
        junit=dict(data["junit"]) if "junit" in data else None,
        allowlist=allowlist_levels(data),
        tool_broken_fixture=data.get("tool_broken_fixture", ""),
    )


def rule_files(scan_root: Path) -> list[Path]:
    return sorted((scan_root / RULES_DIR).glob("*.toml"))


def load_all_cards(scan_root: Path) -> list[Card]:
    """讀掃描根底下所有卡。任何一張壞掉就 raise。"""
    cards = [load_card(p, scan_root) for p in rule_files(scan_root)]
    return cards
