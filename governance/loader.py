"""規矩卡的載入器。

讀 ``<scan_root>/governance/rules/*.toml``，每張卡驗必填欄位，格式壞掉就 raise
:class:`CardError`。門檻與掃描面都只寫在卡自己的 toml 裡，不寫死在檢查程式。

必填欄位（缺一不可）：

* ``id``——必須跟檔名一致
* ``human``——人話一句，講清楚什麼情況會紅
* ``scope``——這張卡要掃哪些路徑（非空 list）
* ``check``——檢查程式，必須指向 ``governance/checks/`` 底下真的存在的模組
* ``negative_fixture``——必紅樣本目錄，必須真的存在，底下至少一個樣本
* ``control_fixture``——控制樣本目錄，必須是 negative_fixture 底下的一個樣本
* ``declared_level``——``blocking`` 或 ``advisory``，二選一
* ``enforcer``——只認版控裡的機器（見 :data:`ENFORCERS`），填人／審查員／本機 hook 一律紅
* ``job``——GitHub Actions 的 job 名
* ``external_tools``——外部工具清單，**沒有也要明寫 ``[]``**，不准省略
* ``[mountpoint]``——掛載點四段寫滿：何時觸發／跑在哪台／結果去哪／卡不卡得住合併

選填欄位（沒寫不罰，寫了就要對）：

* ``blood_debt``——這張卡對到的 v2 事故 id（找碴確認「在事故當下會回紅」的才寫這裡）
* ``related_lessons``——有關聯但找碴判「不算血債」的事故 id（例如違規物件落在掃描面外）
* ``related_lessons_why``——``related_lessons`` 非空時必填：說明為什麼不算血債
* ``[junit]``——這張卡要判的 pytest junit 收據：``path``（相對掃描根，不准絕對路徑、不准
  ``..``）＋ ``collected_floor``（收集數地板，正整數）。門檻只寫在卡自己的 toml 裡，
  檢查程式不准有預設值——沒有卡宣告 ``[junit]``，那支檢查就該回 2 說「沒東西可判」
"""
from __future__ import annotations

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

# 選填的 [junit] 表：收據在哪（path）＋收集數地板（collected_floor）。兩段都要寫滿。
JUNIT_KEYS = ("path", "collected_floor")

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
class Card:
    id: str
    human: str
    scope: list[str]
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
    def check_module(self) -> str:
        """``governance/checks/x.py`` -> ``governance.checks.x``（給 ``python -m`` 用）。"""
        return self.check.removesuffix(".py").replace("/", ".")

    def negative_cases(self, scan_root: Path) -> list[Path]:
        """必紅樣本目錄底下的每一個樣本。每一個都是一個掃描根。"""
        base = scan_root / self.negative_fixture
        return sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []

    def control_path(self, scan_root: Path) -> Path:
        return scan_root / self.control_fixture

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
    check = data.get("check")
    if isinstance(check, str) and not check.startswith(CHECKS_DIR + "/"):
        bad.append(f"check={check!r} 必須指向 {CHECKS_DIR}/ 底下的模組")
    return bad


def _junit_problems(data: dict[str, object]) -> list[str]:
    """選填的 ``[junit]`` 表：路徑與地板都要寫滿，而且路徑不准跑出掃描根。"""
    if "junit" not in data:
        return []
    junit = data["junit"]
    if not isinstance(junit, dict):
        return [f"[junit] 必須是一個表（path ＋ collected_floor），實際是 {junit!r}"]
    bad: list[str] = []
    missing = [k for k in JUNIT_KEYS if k not in junit]
    if missing:
        bad.append(f"[junit] 少了 {missing}——收據路徑 path 與收集數地板 collected_floor 都要寫滿")
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
    return bad


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
    )


def rule_files(scan_root: Path) -> list[Path]:
    return sorted((scan_root / RULES_DIR).glob("*.toml"))


def load_all_cards(scan_root: Path) -> list[Card]:
    """讀掃描根底下所有卡。任何一張壞掉就 raise。"""
    cards = [load_card(p, scan_root) for p in rule_files(scan_root)]
    return cards
