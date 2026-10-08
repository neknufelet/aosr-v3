"""方案檢查不過時給網頁看的問題：中文欄名與白話；直達被擋每對一行，其餘同一句只說一次。

檢查本身不歸網頁層管（它是計算指紋的一部分，網頁層不准動），這裡只把它回的
（欄位路徑, 訊息）翻成方案輸入頁表單上的字；原本的路徑與訊息另外留著，給考卷與技術細節。
認不得的訊息照原文留著、認不得的路徑寫成看得懂的樣子，一條都不丟。
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from aosr.gui.labels import (
    LISTENING_POINTS, LOW_FREQUENCY_AXES, ROOM_LENGTHS, SOURCE_MODELS, SPEAKERS, WALLS)
from aosr.reporting.scheme import Scheme
from aosr.reporting.validation import SchemeProblem, SchemeValidationError, validated_scheme

Document = dict[str, object]

AXES = ("x", "y", "z")
# 整份方案的問題：表單上沒有對應的一格，只印訊息；跟有欄名的同一句合併時寫成這幾個字。
WHOLE_SCHEME = frozenset({"", "scheme"})
WHOLE_SCHEME_LABEL = "整份方案"
# 一條路徑對一格的：用表單上的字；表單上沒有的（聲速、密度等）用比較頁「改了哪裡」的字。
NAMED_FIELDS = {
    "scheme_id": "方案代號", "purpose": "方案用途", "schema_version": "方案格式版本",
    "furniture": "家具",
    "source_model": "聲源模型", "speakers": "喇叭", "receiver_set": "座位清單", "scene": "房間與材料",
    "receiver_set.points": "座位清單", "channel_group.feature_match_tolerance_hz": "峰谷配對容差",
    "scene.room_m": "房間長寬高", "scene.sound_speed_m_s": "聲速", "scene.density_kg_m3": "密度",
    "scene.impedance_pa_s_per_m_by_wall": "六面阻抗", "scene.scattering_by_wall": "六面散射",
    "scene.reflection_order_k": "反射階數", "scene.low_frequency_axis": "低頻軸",
    # 產品預設指向讓喇叭對準主位：這一格就是主位的座標。
    "scene.source_model.aim_m": "喇叭對準點（主位）",
}
WALL_FIELDS = {"impedance_pa_s_per_m_by_wall": "阻抗", "scattering_by_wall": "散射"}
SEAT_FIELDS = {"importance": "重要度", "direction_relative_to_primary": "相對方向",
               "role": "角色", "receiver_id": "代號"}
# 認不得的路徑：認得的段換中文、順序號寫第幾個，其餘照原樣，用「›」隔開。
PATH_WORDS = {"scene": "房間與材料", "room_m": "房間", "speakers": "喇叭", "receiver_set": "座位清單",
              "points": "座位", "channel_group": "聲道組", "channels": "聲道",
              "comparisons": "聲道比較", "position_m": "座標", "source_model": "聲源模型",
              "pairs": "喇叭與座位", **WALL_FIELDS}

OUTSIDE_ROOM = ("座標超出房間（x、y、z 都要在 0 到房間的長、寬、高之間，貼牆也算）；"
                "請核對這組座標和房間長寬高")
FINITE_NUMBER = "要填一般的數字（不收無限大或非數值）"
MISSING = "空著沒填"


class SchemeProblemsError(SchemeValidationError):
    """方案檢查不過，另帶原文件：翻中文名時要從文件查喇叭接哪個聲道、第幾個座位是誰。"""

    def __init__(self, problems: tuple[SchemeProblem, ...], document: object) -> None:
        super().__init__(problems)
        self.document = document


def checked_scheme(document: object) -> Scheme:
    """跟檢查同一支驗方案；不過就丟帶著原文件的錯，讓回應能寫出表單上的中文名。"""
    try:
        return validated_scheme(document)
    except SchemeValidationError as exc:
        raise SchemeProblemsError(exc.problems, document) from exc


def _as_document(document: object) -> Document:
    if isinstance(document, Scheme):
        return document.model_dump(mode="json")
    return document if isinstance(document, dict) else {}


def _child(value: object, key: str) -> object:
    return value.get(key) if isinstance(value, dict) else None


def _items(value: object) -> list[Document]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def speaker_name(document: Document, speaker_id: str) -> str:
    """喇叭顯示名：先找它接哪個聲道，用顯示名稱表的名字；表上沒有就寫「喇叭 代號」。"""
    roles = [item.get("role") for item in _items(_child(document.get("channel_group"), "channels"))
             if item.get("speaker_id") == speaker_id]
    for code in (*roles, speaker_id):
        if isinstance(code, str) and code in SPEAKERS:
            return SPEAKERS[code]
    return f"喇叭 {speaker_id}"


def seat_name(document: Document, receiver_id: str) -> str:
    """座位顯示名：顯示名稱表有這個代號就用表上的名字；沒有的主位叫主位，其他寫「座位 代號」。"""
    if receiver_id in LISTENING_POINTS:
        return LISTENING_POINTS[receiver_id]
    points = _items(_child(document.get("receiver_set"), "points"))
    if any(item.get("receiver_id") == receiver_id and item.get("role") == "primary" for item in points):
        return "主位"
    return f"座位 {receiver_id}"


def _seat_at(document: Document, index: str) -> str:
    points = _items(_child(document.get("receiver_set"), "points"))
    if not index.isdigit():
        return f"座位 {index}"
    code = points[int(index)].get("receiver_id") if int(index) < len(points) else None
    return seat_name(document, code) if isinstance(code, str) else f"第 {int(index) + 1} 個座位"


def _join(name: str, tail: str) -> str:
    """中文接中文不空格，碰到英數字才空一格（主位座標、主位 x 座標、座位 sofa 座標）。"""
    edges = (name[-1:], tail[:1])
    return f"{name} {tail}" if any(edge.isascii() and edge.isalnum() for edge in edges) else f"{name}{tail}"


def _readable(path: str) -> str:
    return " › ".join(f"第 {int(part) + 1} 個" if part.isdigit() else PATH_WORDS.get(part, part)
                      for part in path.split("."))


def _scene_field(rest: list[str]) -> str | None:
    if len(rest) == 2 and rest[0] == "room_m":
        return ROOM_LENGTHS.get(rest[1], f"房間 › {rest[1]}")
    if len(rest) == 2 and rest[0] in WALL_FIELDS:
        return f"{WALLS.get(rest[1], f'{rest[1]} 牆')}{WALL_FIELDS[rest[0]]}"
    return None


def _speaker_field(document: Document, rest: list[str]) -> str | None:
    if len(rest) == 1:
        return speaker_name(document, rest[0])
    if len(rest) == 2:
        name = speaker_name(document, rest[0])
        return f"{name} {rest[1]} 座標" if rest[1] in AXES else f"{name} › {rest[1]}"
    return None


def _seat_field(document: Document, rest: list[str]) -> str | None:
    if len(rest) < 2 or rest[0] != "points":
        return None
    seat, tail = _seat_at(document, rest[1]), rest[2:]
    if not tail:
        return seat
    if tail[0] == "position_m":
        axis = tail[1] if len(tail) == 2 else ""
        return _join(seat, f"{AXES[int(axis)]} 座標" if axis in ("0", "1", "2") else "座標")
    return _join(seat, SEAT_FIELDS[tail[0]]) if len(tail) == 1 and tail[0] in SEAT_FIELDS else None


def _pair_field(document: Document, rest: list[str]) -> str | None:
    """每一對喇叭與座位的問題：落在喇叭那邊的寫喇叭、座位那邊的寫座位，其他寫「喇叭 → 座位」。"""
    if len(rest) < 2:
        return None
    speaker, seat = speaker_name(document, rest[0]), seat_name(document, rest[1])
    field = rest[2] if len(rest) > 2 else ""
    if field.startswith("source"):
        return speaker
    return seat if field == "receiver_m" else f"{speaker} → {seat}"


def field_name(path: str, document: object) -> str | None:
    """問題落在表單哪一格的中文名；整份方案的問題沒有對應的一格，回 None。"""
    if path in WHOLE_SCHEME:
        return None
    if path in NAMED_FIELDS:
        return NAMED_FIELDS[path]
    scheme = _as_document(document)
    head, *rest = path.split(".")
    found: str | None = None
    if head == "scene":
        found = _scene_field(rest)
    elif head == "speakers":
        found = _speaker_field(scheme, rest)
    elif head == "receiver_set":
        found = _seat_field(scheme, rest)
    elif head == "pairs":
        found = _pair_field(scheme, rest)
    elif head == "channel_group":
        found = "聲道組"
    return found if found is not None else _readable(path)


@dataclass(frozen=True)
class Rule:
    """一種原文訊息：整句比對；message 寫白話，fields 有給就改寫落在哪幾格（原文點名的比路徑準）。"""

    pattern: re.Pattern[str]
    message: Callable[[re.Match[str]], str]
    fields: Callable[[re.Match[str], Document], tuple[str, ...]] | None = None


def _fixed(text: str) -> Callable[[re.Match[str]], str]:
    return lambda _match: text


def _quoted(text: str) -> list[str]:
    return re.findall(r"'([^']*)'", text)


def _choices(match: re.Match[str]) -> str:
    names = {**SOURCE_MODELS, **LOW_FREQUENCY_AXES}
    return "只能選" + "或".join(f"「{names.get(value, value)}」" for value in _quoted(match["choices"]))


def _rule(pattern: str, message: str | Callable[[re.Match[str]], str],
          fields: Callable[[re.Match[str], Document], tuple[str, ...]] | None = None) -> Rule:
    return Rule(re.compile(pattern, re.DOTALL), _fixed(message) if isinstance(message, str) else message,
                fields)


# 由上往下比，第一條對上的算數；「大於或等於」要排在「大於」前面。
RULES = (
    _rule(r"不符合擺位要求：喇叭 .+ 到座位 .+ 的直達路徑被家具 (?P<ids>.+) 擋住",
          lambda match: f"不符合擺位要求：直達路徑被家具 {match['ids']} 擋住"),
    _rule(r"喇叭 (?P<id>.+) 必須在房間閉區間內", OUTSIDE_ROOM,
          lambda match, document: (speaker_name(document, match["id"]),)),
    _rule(r"座位 (?P<id>.+) 必須在房間閉區間內", OUTSIDE_ROOM,
          lambda match, document: (seat_name(document, match["id"]),)),
    _rule(r"source_model\.aim_m 必須在房間的閉區間內", OUTSIDE_ROOM),
    _rule(r"source_model\.aim_m 不可與 source_m 重合",
          "跟主位放在同一點（產品預設指向讓喇叭對準主位，兩者不能重合）"),
    _rule(r"scheme_id 與 purpose 不可為空白", "不可空白（這兩格至少有一格是空白）",
          lambda _match, _document: ("方案代號", "方案用途")),
    _rule(r"方案必須剛好兩個聲道角色與一個比較對", "方案要剛好兩個聲道，再加一組這兩個聲道之間的比較",
          lambda _match, _document: ("聲道組",)),
    _rule(r"聲道組引用不存在的喇叭 (?P<id>.+)", lambda match: f"用到方案裡沒有的喇叭「{match['id']}」",
          lambda _match, _document: ("聲道組",)),
    _rule(r"未使用的喇叭 (?P<id>.+)", "這支喇叭沒有接到任何聲道",
          lambda match, document: (speaker_name(document, match["id"]),)),
    _rule(r"receiver_id 不可重複", "座位代號不可重複"),
    _rule(r"receiver_id 不可為空白", "座位代號不可空白"),
    _rule(r"接收點清單必須剛好有一個 primary", "要剛好一個主位"),
    _rule(r"接收點清單至少要有主位加一個 surrounding", "至少要有主位加一個周圍點"),
    _rule(r"surrounding 接收點必須帶相對主位方向", "周圍點要寫明它在主位的哪個方向"),
    _rule(r"方案各對報表的場景指紋不同",
          "每一組喇叭與座位拿到的房間設定對不上（程式內部的核對沒過，請告訴助理）"),
    _rule(r"必填", MISSING),
    _rule(r"Field required", "缺這一格"),
    _rule(r"Input should be a valid number.*", "要填數字"),
    _rule(r"Input should be a finite number", FINITE_NUMBER),
    _rule(r"Input should be a valid integer.*", "要填整數"),
    _rule(r"Input should be a valid string", "要填文字"),
    _rule(r"Input should be a valid dictionary.*|Input should be an object", "格式不對：這裡要一組欄位"),
    _rule(r"Input should be a valid (?:list|tuple|array).*", "格式不對：這裡要一串數"),
    _rule(r"Input should be greater than or equal to (?P<n>\S+)", lambda match: f"不可小於 {match['n']}"),
    _rule(r"Input should be greater than (?P<n>\S+)", lambda match: f"要大於 {match['n']}"),
    _rule(r"Input should be less than or equal to (?P<n>\S+)", lambda match: f"不可大於 {match['n']}"),
    _rule(r"Input should be less than (?P<n>\S+)", lambda match: f"要小於 {match['n']}"),
    _rule(r"Input should be (?P<choices>'.*')", _choices),
    _rule(r"String should have at least 1 character", "不可空白"),
    _rule(r"Extra inputs are not permitted|Unexpected keyword argument", "多了不認識的欄位"),
    _rule(r"Tuple should have at most (?P<n>\d+) items? after validation, not (?P<m>\d+)",
          lambda match: f"最多 {match['n']} 個數，現在有 {match['m']} 個"),
    _rule(r"Tuple should have at least (?P<n>\d+) items? after validation, not (?P<m>\d+)",
          lambda match: f"至少要 {match['n']} 個數，現在有 {match['m']} 個"),
    _rule(r"\S+ 必須是有限正數", "要大於 0"),
    _rule(r"\S+ 必須是有限數字", FINITE_NUMBER),
    _rule(r"\S+ 必須是 JSON 物件", "格式不對：這裡要一組欄位"),
    _rule(r"\S+ 多了不認識的牆名 (?P<unknown>\[.*?\])；.*",
          lambda match: f"多了不認識的牆名 {'、'.join(_quoted(match['unknown']))}"
                        f"（只收{'、'.join(WALLS.values())}這六面）"),
    _rule(r"\S+ 多了不認識的鍵 (?P<unknown>\[.*?\])；這一格只收 (?P<allowed>\[.*?\])。.*",
          lambda match: f"多了不認識的欄位 {'、'.join(_quoted(match['unknown']))}"
                        f"（只收 {'、'.join(_quoted(match['allowed']))}）"),
    _rule(r"\S+ 必須是正實數阻抗；.*", "要大於 0（這一版只收一個正的實數阻抗）"),
    _rule(r"impedance_pa_s_per_m_by_wall\.\S+：.*",
          "這一版只收一個正的實數阻抗，不收複數阻抗或逐頻阻抗"),
    _rule(r"scattering_by_wall\.\S+：散射係數只收一個實數", "散射係數只收一個數"),
    _rule(r"\S+ 必須落在 \[0,1\]（散射係數）", "要在 0 到 1 之間"),
)


def plain_problem(problem: SchemeProblem, document: object) -> tuple[tuple[str, ...], str]:
    """一條問題的（落在哪幾格, 白話訊息）；原文沒有對上任何一條就照原文。"""
    scheme = _as_document(document)
    field = field_name(problem.path, scheme)
    for rule in RULES:
        match = rule.pattern.fullmatch(problem.message)
        if match:
            fields = rule.fields(match, scheme) if rule.fields else (field,) if field else ()
            return fields, rule.message(match)
    return ((field,) if field else ()), problem.message


def plain_problems(problems: Sequence[SchemeProblem], document: object) -> list[dict[str, object]]:
    """給網頁的問題清單：直達被擋每對一行，其餘同一句合併欄名；原路徑與原文另外留著。

    每一條是 {text: 直接印的一行, message: 白話, fields: 表單中文欄名, paths: 原本的欄位路徑,
    details: 原本的「路徑：訊息」}；頁面只印 text，技術細節要看時才用 paths／details。
    """
    scheme = _as_document(document)
    groups: dict[tuple[str, str], list[tuple[tuple[str, ...], SchemeProblem]]] = {}
    for problem in problems:
        fields, message = plain_problem(problem, scheme)
        # 直達被擋按每一對列出，不能把相同家具擋住的不同聆聽點合成一行。
        pair = problem.path if message.startswith("不符合擺位要求：直達路徑被家具 ") else ""
        groups.setdefault((message, pair), []).append((fields, problem))
    return [_merged(message, members) for (message, _), members in groups.items()]


def _merged(message: str, members: list[tuple[tuple[str, ...], SchemeProblem]]) -> dict[str, object]:
    named = [field for fields, _ in members for field in fields]
    # 整份方案的問題跟有欄名的同一句合併：補寫「整份方案」，不讓它在那一行裡消失。
    if named and any(not fields for fields, _ in members):
        named.append(WHOLE_SCHEME_LABEL)
    fields = list(dict.fromkeys(named))
    return {"text": f"{'、'.join(fields)}：{message}" if fields else message, "message": message,
            "fields": fields, "paths": list(dict.fromkeys(problem.path for _, problem in members)),
            "details": list(dict.fromkeys(str(problem) for _, problem in members))}
