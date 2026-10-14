"""門檻不准硬編碼在別的地方——上一代那條守衛改寫成盯新家，住治理層那一籃。

**上一代那條長什麼樣（上一代那一棵樹的 config 考卷裡的 ``test_perceptual_thresholds_are_not_hardcoded_in_lib``）。**
它掃 ``(REPO_ROOT / "lib").rglob("*.py")``，禁四個抄進考卷的字面數字、豁免清單指名上一代
config 層的 art_lane 與 scoring 層的 perceptual。那條守衛在新家**是空的**：上一代那一層
不在這裡，路徑全部對不上。

**新家這一條掃整個新引擎、守的數字從設定檔讀（#158）。**

- 掃描根不寫死：讀 ``governance/rules/layers-import-downward-only.toml`` 的
  ``[settings].package_root``，列根底下全部 .py。那張卡順序表登記的每一層，只要資料夾存在
  就至少要掃到一支；根底下直屬的 .py 也要在掃到的集合裡。只掃一層（例如寫死回 config）會紅。
- 守的數字不抄：從 ``src/aosr/config/data/perceptual.toml`` 的三條鍵路徑讀
  （``GUARDED_KEY_PATHS``），每條都要解析得到、值是浮點數；設定檔改了數字，守衛跟著守新數字。
- 豁免一筆四格（檔、鍵路徑、那一行必須含的字、理由），對不到任何一行就紅：被守的數字改了、
  或那一行改掉了，過期的豁免當場清掉，不會變成整支檔放行。

**刻意沒管的（照實寫）。**

- ``profiles.*.rfz.window_ms`` 沒全收：material_rfz 的 20 毫秒跟頻率軸起點 20 Hz、有限元素
  剛性牆那幾行是同一個字面數字、不同的量，收了就誤咬。``early_reflection.required_attenuation_db``
  （正的 10）也沒收。
- 只比「整個數字 token」的浮點寫法；整數寫法（``-10``）、科學記號、算出來的數字抓不到。
- 抓的是漂移（守衛跟設定檔、跟那張卡的順序表對不上），不是同一支合併請求把錨點一起改窄。
- 這條守衛住在 ``tests/``（治理層那一籃），不 import ``aosr``：它讀的是原始碼文字。
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

# 這個 repo 的根（`<root>/tests/` 的上一層）。跟 cwd 無關。
REPO_ROOT = Path(__file__).resolve().parents[1]

# 新引擎住哪裡、分哪幾層：只登記在這張卡上，這裡讀它。
LAYERS_CARD = REPO_ROOT / "governance" / "rules" / "layers-import-downward-only.toml"

# 被守的數字住的那份設定檔。
PERCEPTUAL_TOML = REPO_ROOT / "src" / "aosr" / "config" / "data" / "perceptual.toml"

# 被守的數字從這三條鍵路徑讀；`*` 是那一層底下的每一個（profiles 底下每一種用途）。
GUARDED_KEY_PATHS = ("profiles.*.rfz.threshold_db", "early_reflection.window_ms", "jnd.c80_db")


@dataclass(frozen=True)
class Exemption:
    """一筆具名豁免：哪支檔、豁免的是哪一條鍵路徑的值、那一行必須含什麼字、為什麼。"""

    file: str
    key_path: str
    context: str
    reason: str


EXEMPTIONS = (
    Exemption("src/aosr/config/art_lane.py", "profiles.critical.rfz.threshold_db", "ART_WLS_T20_LO_DB",
              "T20 積分視窗的下限（art_lane 自己的結構常數），不是感知門檻"),
    Exemption("src/aosr/scoring/reflections.py", "profiles.monitoring.rfz.threshold_db", "* math.log10(",
              "能量比換成分貝的係數（10·log10），不是感知門檻"),
)


def _pattern(literal: str) -> re.Pattern[str]:
    """比**整個數字 token**，不是比子字串。

    裸 ``0.9`` 咬得到，``0.999``（Nyquist 邊緣的護欄）／``10.9``／``0.9e5`` 咬不到。上一代有一段
    註解記著這條：原本的 ``literal in text`` 子字串版誤咬過 ``auralize.py`` 的 ``nyquist_hz * 0.999``。
    """
    return re.compile(rf"(?<![\d.]){re.escape(literal)}(?![\d.eE])")


def _load(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _resolve(data: dict[str, object], key_path: str) -> list[object]:
    """照鍵路徑往下走；缺鍵或走到不是表的地方就紅，不准悄悄變成空的。"""
    nodes: list[object] = [data]
    for part in key_path.split("."):
        found: list[object] = []
        for node in nodes:
            assert isinstance(node, dict), f"{key_path}：走到 {part} 之前就不是表了"
            if part == "*":
                found.extend(node.values())
            else:
                assert part in node, f"{key_path}：缺 {part}"
                found.append(node[part])
        nodes = found
    assert nodes, f"{key_path}：一個值都沒解析到"
    return nodes


def _as_literal(key_path: str, value: object) -> str:
    assert isinstance(value, float), f"{key_path}：值 {value!r} 不是浮點數（整數寫法這條守衛比不到）"
    return repr(value)


@cache
def _guarded_literals() -> dict[str, str]:
    """被守的字面數字 → 它來自哪一條鍵路徑。"""
    data = _load(PERCEPTUAL_TOML)
    return {_as_literal(key_path, value): key_path
            for key_path in GUARDED_KEY_PATHS for value in _resolve(data, key_path)}


def _package_root() -> Path:
    settings = _load(LAYERS_CARD).get("settings")
    assert isinstance(settings, dict), f"{LAYERS_CARD.name} 沒有 [settings]"
    root = settings.get("package_root")
    assert isinstance(root, str) and root, f"{LAYERS_CARD.name} 的 package_root 不是非空字串"
    return REPO_ROOT / root


def _layers() -> set[str]:
    settings = _load(LAYERS_CARD).get("settings")
    assert isinstance(settings, dict), f"{LAYERS_CARD.name} 沒有 [settings]"
    groups = settings.get("layers")
    assert isinstance(groups, list) and groups, f"{LAYERS_CARD.name} 的 layers 不是非空陣列"
    names: set[str] = set()
    for group in groups:
        assert isinstance(group, list) and all(isinstance(name, str) for name in group), group
        names.update(group)
    return names


def _scanned_files() -> list[Path]:
    return sorted(_package_root().rglob("*.py"))


def _exempt_literal(exemption: Exemption) -> str:
    values = _resolve(_load(PERCEPTUAL_TOML), exemption.key_path)
    assert values[1:] == [], f"豁免的鍵路徑 {exemption.key_path} 要只解析得到一個值"
    return _as_literal(exemption.key_path, values[0])


@cache
def _scan() -> tuple[tuple[str, ...], frozenset[Exemption]]:
    """掃一遍：回（違規清單、真的有對到一行的豁免）。"""
    patterns = {literal: _pattern(literal) for literal in _guarded_literals()}
    exempt = {exemption: _exempt_literal(exemption) for exemption in EXEMPTIONS}
    offenders: list[str] = []
    used: set[Exemption] = set()
    for path in _scanned_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for literal, pattern in patterns.items():
                if not pattern.search(line):
                    continue
                hits = [e for e, value in exempt.items() if e.file == rel and value == literal and e.context in line]
                used.update(hits)
                if not hits:
                    offenders.append(f"{rel}:{number}: {literal}（{_guarded_literals()[literal]}）")
    return tuple(offenders), frozenset(used)


def test_guarded_values_come_from_the_toml_key_paths() -> None:
    """三條鍵路徑都解析得到、值都是浮點數；被守的集合不准是空的。"""
    data = _load(PERCEPTUAL_TOML)
    for key_path in GUARDED_KEY_PATHS:
        literals = {_as_literal(key_path, value) for value in _resolve(data, key_path)}
        assert literals <= set(_guarded_literals()), key_path


def test_every_registered_layer_that_exists_is_scanned() -> None:
    """順序表登記、資料夾存在的每一層都至少掃到一支；根底下直屬的 .py 也都要在。"""
    root = _package_root()
    scanned = {path.relative_to(root).parts[0] for path in _scanned_files()}
    expected = {layer for layer in _layers() if (root / layer).is_dir()}
    expected |= {path.name for path in root.glob("*.py")}
    # 被守的數字住的那一層一定要在期望裡：設定檔搬家、這條守衛還在盯舊根，就紅。
    assert PERCEPTUAL_TOML.relative_to(root).parts[0] in expected
    assert expected - scanned == set()


def test_no_perceptual_threshold_is_hardcoded_in_the_engine() -> None:
    """新引擎的每一支 .py 都不准把被守的數字寫死（具名豁免除外）。"""
    offenders, _ = _scan()
    assert list(offenders) == []


def test_every_exemption_still_matches_a_line() -> None:
    """每一筆豁免都要真的對到一行：檔對、那一行含上下文、那個值還在被守的集合裡。"""
    _, used = _scan()
    assert [exemption for exemption in EXEMPTIONS if exemption not in used] == []


def test_the_guard_would_actually_bite_a_hardcoded_threshold() -> None:
    """這一條守衛真的會咬：每一個被守的數字寫進一行，同一個判準都要抓到它。

    沒有這一條，「上面那一條綠」只證明「今天沒人寫」——分不出「乾淨」與「判準壞掉」。
    反向控制：多接一位數字、前面多一位數字（子字串那一版會誤咬的寫法），不准咬。
    """
    for literal in _guarded_literals():
        pattern = _pattern(literal)
        assert pattern.search(f"THRESHOLD = {literal}\n"), literal
        assert not pattern.search(f"GUARD = {literal}99\n"), literal
        assert not pattern.search(f"GUARD = 1{literal.lstrip('-')}\n"), literal
