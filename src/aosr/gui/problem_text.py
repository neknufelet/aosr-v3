"""方案檢查不過時給網頁看的問題：中文欄名與白話；直達被擋每對一行，其餘同一句只說一次。

檢查本身不歸網頁層管（它是計算指紋的一部分，網頁層不准動），這裡只把它回的
（欄位路徑, 訊息）翻成方案輸入頁表單上的字；原本的路徑與訊息另外留著，給考卷與技術細節。
認不得的訊息照原文留著、認不得的路徑寫成看得懂的樣子，一條都不丟。
不適用的擺法分支只從畫面濾掉，原路徑與原句仍附在同件家具的技術明細。
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast, get_args

from pydantic import BaseModel
from aosr.materials.construction_models import AirLayer, Assembly, BoardLayer, ConstantImpedance, MeasuredImpedance, MeasuredImpedancePoint, PorousLayer, SurfaceMaterial

from aosr.gui.labels import (
    LISTENING_POINTS, LOW_FREQUENCY_AXES, ROOM_LENGTHS, SOURCE_MODELS, SPEAKERS, SPEAKER_SETUP, WALLS)
from aosr.reporting.display import FURNITURE_FIELDS, FURNITURE_KINDS, FURNITURE_MATERIALS, FURNITURE_ROOM_ANGLE
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
    **{f"speaker_setup.{key}": SPEAKER_SETUP[key] for key in ("kind", "mount", "representative", "cabinet")},
    "speaker_setup": SPEAKER_SETUP["speaker_setup"],
    **{f"speaker_setup.cabinet.{key}": SPEAKER_SETUP[key] for key in (
        "width_m", "depth_m", "height_m", "acoustic_center_behind_front_m", "acoustic_center_above_bottom_m")},
    "source_model": "聲源模型", "speakers": "喇叭", "receiver_set": "座位清單", "scene": "房間與材料",
    "receiver_set.points": "座位清單", "channel_group.feature_match_tolerance_hz": "峰谷配對容差",
    "scene.room_m": "房間長寬高", "scene.sound_speed_m_s": "聲速", "scene.density_kg_m3": "密度",
    "scene.surface_zones_by_wall": "六面表面分區",
    "scene.impedance_pa_s_per_m_by_wall": "六面阻抗", "scene.scattering_by_wall": "六面散射",
    "scene.reflection_order_k": "反射階數", "scene.low_frequency_axis": "低頻軸",
    # 產品預設指向讓喇叭對準主位：這一格就是主位的座標。
    "scene.source_model.aim_m": "喇叭對準點（主位）",
}
WALL_FIELDS = {"impedance_pa_s_per_m_by_wall": "阻抗", "scattering_by_wall": "散射", "surface_zones_by_wall": "材料構造"}
SEAT_FIELDS = {"importance": "重要度", "direction_relative_to_primary": "相對方向",
               "role": "角色", "receiver_id": "代號"}
# 認不得的路徑：認得的段換中文、順序號寫第幾個，其餘照原樣，用「›」隔開。
PATH_WORDS = {"scene": "房間與材料", "room_m": "房間", "speakers": "喇叭", "receiver_set": "座位清單",
              "points": "座位", "channel_group": "聲道組", "channels": "聲道",
              "comparisons": "聲道比較", "position_m": "座標", "source_model": "聲源模型",
              "pairs": "喇叭與座位", "speaker_setup": SPEAKER_SETUP["speaker_setup"],
              "cabinet": SPEAKER_SETUP["cabinet"], **WALL_FIELDS}
# 模型按 kind（種類）分開，重名欄位不能互相蓋掉。
MATERIAL_MODELS: dict[str, type[BaseModel]] = {
    str(model.model_fields["kind"].default): model
    for model in (*get_args(get_args(SurfaceMaterial)[0]), PorousLayer, AirLayer, BoardLayer)}
MATERIAL_WORDS = {kind: {name: (field.description or name).split("（", 1)[-1].removesuffix("）")
                       for name, field in model.model_fields.items()}
                  for kind, model in MATERIAL_MODELS.items()}
PATH_WORDS.update({name: label.split("：", 1)[0] for words in MATERIAL_WORDS.values() for name, label in words.items()
                   if name not in ("kind", "thickness_m", "points")})
PATH_WORDS.update({"kind": "種類代號", "thickness_m": "層厚度，m"})
PATH_WORDS.update({"material": "牆面材料", "area": "分區範圍", "assembly": "構造",
                   "air": "空氣層", "board": "薄板層", "porous": "多孔層"})


PATH_WORDS.update({name: (field.description or name).split("（", 1)[-1].removesuffix("）")
                   for name, field in MeasuredImpedancePoint.model_fields.items()})


def material_summary(material: SurfaceMaterial) -> str:
    """逐層白話只從所屬模型的描述取欄名、單位與種類中文。"""
    if isinstance(material, MeasuredImpedance):
        kind = material.kind
        name = _enum_word(MATERIAL_WORDS[kind]["kind"], kind)
        source = material_field_text(kind, "source_and_conditions").split("，", 1)[0]
        base = material_field_text(kind, "base_assembly").split("；", 1)[0]
        real = PATH_WORDS["real_pa_s_m"].split("，", 1)[0]
        imag = PATH_WORDS["imag_pa_s_m"].split("，", 1)[0]
        rows = "／".join(f"{point.frequency_hz!r} Hz：{real} {point.real_pa_s_m!r}、{imag} {point.imag_pa_s_m!r} Pa·s/m"
                        for point in material.points)
        return (f"{name}：{material.points[0].frequency_hz:g}–{material.points[-1].frequency_hz:g} Hz，"
                f"{len(material.points)} 列；{rows}；{source}：{material.source_and_conditions}；"
                f"{base}：{material_summary(material.base_assembly)}")
    return _model_summary(type(material), lambda name: getattr(material, name))


def _enum_word(description: str, value: object) -> str:
    """解析模型描述中的「代號 表示／是／為 中文」，不另建代號翻譯表。"""
    for match in re.finditer(r"(?:^|[：，])([^：，]+?)\s*(?:表示|是|為)\s*([^，；]+)", description):
        codes, word = match.group(1), match.group(2)
        if str(value) in {code.strip() for code in codes.split("、")}:
            return word.strip()
    raise ValueError(f"材料模型描述缺少 {value!r} 的中文說明")


def material_field_text(kind: str, name: str) -> str:
    """枚舉只取欄名，其餘保留所屬模型的中文欄名與單位。"""
    description = MATERIAL_WORDS[kind][name]
    return description.split("：", 1)[0] if "：" in description else description.split("，", 1)[0] if name == "kind" else description


def material_form_spec() -> dict[str, dict[str, object]]:
    """表單欄名、單位與選項全部執行期來自模型描述，不建立材料物件。"""
    specs: dict[str, dict[str, object]] = {}
    for kind, model in MATERIAL_MODELS.items():
        fields: dict[str, object] = {}
        for name, field in model.model_fields.items():
            description = MATERIAL_WORDS[kind][name]
            label, _, unit = material_field_text(kind, name).partition("，")
            choices = get_args(field.annotation) if str(field.annotation).startswith("typing.Literal[") else ()
            fields[name] = {"label": label, "unit": unit.split("；", 1)[0],
                            "options": {str(value): _enum_word(description, value) for value in choices},
                            "default": field.default if not field.is_required() else None}
        specs[kind] = {"label": _enum_word(MATERIAL_WORDS[kind]["kind"], kind), "fields": fields}
    return specs


def _plain_material_message(message: str) -> str:
    """核心原文只在顯示時替換代號；模型與其描述都不改。"""
    words: dict[str, str] = dict(WALLS)
    message = re.sub(r"(種類代號)\s+kind(?=（種類）)", r"\1", message)
    for kind, spec in material_form_spec().items():
        words[kind] = str(spec["label"])
        for name, field in cast(dict[str, dict[str, object]], spec["fields"]).items():
            words[name] = str(field["label"])
            words.update(cast(dict[str, str], field["options"]))
    for code in sorted(words, key=len, reverse=True):
        message = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(code)}(?![A-Za-z0-9_])", lambda _match: words[code], message)
    return message


def _model_summary(model: type[BaseModel], read: Callable[[str], object]) -> str:
    parts = []
    for name, field in model.model_fields.items():
        value = read(name)
        if value is None:
            continue
        description = (field.description or name).split("（", 1)[-1].removesuffix("）")
        if isinstance(value, tuple):
            parts.append("／".join(f"第 {index + 1} 層：{material_summary(item)}"
                                  for index, item in enumerate(value)))
        elif isinstance(value, str):
            word = _enum_word(description, value)
            parts.append(word if name == "kind" else f"{description.split('：', 1)[0]}：{word}")
        else:
            label, _, unit = description.partition("，")
            unit = unit.split("；", 1)[0]
            parts.append(f"{label} {str(value).removesuffix('.0')}" + (f" {unit}" if unit else ""))
    return "；".join(parts)


def fixed_material_summary(impedance: float) -> str:
    """舊固定值只借模型描述做顯示，材料出生地仍只有正式轉換入口。"""
    model = MATERIAL_MODELS["constant_impedance"]
    fields = model.model_fields
    return _model_summary(model, lambda name: impedance if fields[name].is_required() else fields[name].default)

FURNITURE_PATH_WORDS = {**{key: label for key, (label, _unit) in FURNITURE_FIELDS.items()},
                        "furniture_id": "代號", "material": "材質", "placement": "擺法"}
PLACEMENT_BRANCHES = frozenset({"ListenerPlacement", "RoomPlacement"})

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


def _material_path(rest: list[str], document: Document) -> str:
    """路徑有聯合分支時用該型別；缺分支則從原文件的材料或層取種類。"""
    kind = next((part for part in reversed(rest) if part in MATERIAL_MODELS), None)
    if kind is None:
        value: object = document.get("scene")
        for part in rest:
            if isinstance(value, dict):
                if part not in value:
                    break
                value = value[part]
            elif isinstance(value, (list, tuple)) and part.isdigit() and int(part) < len(value):
                value = value[int(part)]
            else:
                break
            # 種類代號不是字串（清單、字典）就不查名稱表：不能當鍵，查了會崩。
            if isinstance(value, dict) and isinstance(value.get("kind"), str) and value["kind"] in MATERIAL_MODELS:
                kind = cast(str, value["kind"])
    parts = []
    if "layers" in rest:
        tail = rest[rest.index("layers") + 1:]
        if tail and tail[0].isdigit():
            parts.append(f"第 {int(tail[0]) + 1} 層")
    leaf = rest[-1]
    if kind and leaf in MATERIAL_WORDS[kind]:
        parts.append(material_field_text(kind, leaf))
    elif leaf in MATERIAL_MODELS:
        parts.append(_enum_word(MATERIAL_WORDS[leaf]["kind"], leaf))
    elif leaf not in ("material", "layers") and not leaf.isdigit():
        parts.append(PATH_WORDS.get(leaf, leaf))
    return " › ".join(parts)


def _scene_field(rest: list[str], document: Document) -> str | None:
    if len(rest) >= 2 and rest[0] == "surface_zones_by_wall":
        wall = WALLS.get(rest[1], rest[1])
        tail = rest[2:]
        return f"{wall}材料構造" + (" › " + _material_path(rest, document) if tail else "")
    if len(rest) == 2 and rest[0] == "room_m":
        return ROOM_LENGTHS.get(rest[1], f"房間 › {rest[1]}")
    if len(rest) == 2 and rest[0] in WALL_FIELDS:
        return f"{WALLS.get(rest[1], f'{rest[1]} 牆')}{WALL_FIELDS[rest[0]]}"
    return None


def _known_prefixes(rest: list[str], codes: list[str]) -> list[tuple[str, list[str]]]:
    """路徑剩下的部分若以文件裡真的有的代號開頭（代號可含點），回每一種切法（代號, 後段）。"""
    joined = ".".join(rest)
    return [(code, joined[len(code) + 1:].split(".") if joined != code else [])
            for code in set(codes) if joined == code or joined.startswith(f"{code}.")]


def _tail_rank(tail: list[str], known: Callable[[str], bool]) -> int:
    """切法怎麼挑：後段是認得的欄位最先，後段空著其次，都不是最後；同一級再挑長的代號。
    代號跟欄位撞名時（喇叭 spk 與 spk.x，路徑 speakers.spk.x）才分得出是 spk 的 x 座標。"""
    return 0 if len(tail) == 1 and known(tail[0]) else 1 if not tail else 2


def _speaker_ids(document: Document) -> list[str]:
    speakers = document.get("speakers")
    return [code for code in speakers if isinstance(code, str)] if isinstance(speakers, dict) else []


def _seat_ids(document: Document) -> list[str]:
    return [code for item in _items(_child(document.get("receiver_set"), "points"))
            if isinstance(code := item.get("receiver_id"), str)]


def _speaker_field(document: Document, rest: list[str]) -> str | None:
    if not rest:
        return None
    speaker, tail = min(_known_prefixes(rest, _speaker_ids(document)),
                        key=lambda split: (_tail_rank(split[1], AXES.__contains__), -len(split[0])),
                        default=(rest[0], rest[1:]))
    name = speaker_name(document, speaker)
    if not tail:
        return name
    if len(tail) == 1:
        return f"{name} {tail[0]} 座標" if tail[0] in AXES else f"{name} › {tail[0]}"
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
    # 不用點切：代號可含點，pairs.spk.L.main 用點切會變成「喇叭 spk → 座位 L」；拿文件裡真的有的代號比對開頭。
    split = min(((speaker, seat, tail) for speaker, after in _known_prefixes(rest, _speaker_ids(document))
                 for seat, tail in _known_prefixes(after, _seat_ids(document))),
                key=lambda split: (_tail_rank(split[2][:1], _pair_side_field), -len(split[0]), -len(split[1])),
                default=(rest[0], rest[1], rest[2:]))
    speaker, seat = speaker_name(document, split[0]), seat_name(document, split[1])
    field = split[2][0] if split[2] else ""
    if field.startswith("source"):
        return speaker
    return seat if field == "receiver_m" else f"{speaker} → {seat}"


def _pair_side_field(field: str) -> bool:
    return field.startswith("source") or field == "receiver_m"


def _furniture_at(document: Document, index: int) -> Document:
    items = document.get("furniture")
    if isinstance(items, list) and 0 <= index < len(items) and isinstance(items[index], dict):
        return cast(Document, items[index])
    return {}


def _furniture_name(document: Document, index: int) -> str:
    item = _furniture_at(document, index)
    kind = item.get("kind")
    name = FURNITURE_KINDS.get(kind, "種類未明") if isinstance(kind, str) else "種類未明"
    identifier = item.get("furniture_id")
    identity = f"{name}，{identifier}" if isinstance(identifier, str) else name
    return f"第 {index + 1} 件家具（{identity}）"


def _furniture_fields(document: Document, *identifiers: str) -> tuple[str, ...]:
    items = document.get("furniture")
    names = tuple(_furniture_name(document, index) for index, item in enumerate(items)
                  if isinstance(item, dict) and item.get("furniture_id") in identifiers) if isinstance(items, list) else ()
    known = {item.get("furniture_id") for item in _items(items) if isinstance(item.get("furniture_id"), str)}
    return names + tuple(f"家具（{identifier}）" for identifier in identifiers if identifier not in known)


def _close_furniture_fields(match: re.Match[str], document: Document) -> tuple[str, ...]:
    # 代號本身可含「 與 」；每個分隔點都試，兩半都在文件裡才採用，否則維持原切法。
    known = {item.get("furniture_id") for item in _items(document.get("furniture"))
             if isinstance(item.get("furniture_id"), str)}
    identifiers = f"{match['first']} 與 {match['second']}"
    for separator in re.finditer(" 與 ", identifiers):
        first, second = identifiers[:separator.start()], identifiers[separator.end():]
        if first in known and second in known:
            return _furniture_fields(document, first, second)
    return _furniture_fields(document, match["first"], match["second"])


def _furniture_field(document: Document, rest: list[str]) -> str | None:
    if not rest or not rest[0].isdigit():
        return None
    index = int(rest[0])
    name = _furniture_name(document, index)
    branch = next((part for part in rest[1:] if part in PLACEMENT_BRANCHES), None)
    tail = [part for part in rest[1:] if part not in PLACEMENT_BRANCHES and part != "placement"]
    if not tail:
        return f"{name}・擺法" if "placement" in rest else name
    if tail[0] == "bottom_center_m":
        axis = tail[1] if len(tail) == 2 else ""
        label = f"底面中心 {AXES[int(axis)]}" if axis in ("0", "1", "2") else "底面中心"
    elif tail[0] == "yaw_deg" and (branch == "RoomPlacement" or
                                  (branch is None and _furniture_at(document, index).get("kind") == "ceiling_cloud")):
        label = FURNITURE_ROOM_ANGLE
    else:
        label = " › ".join(FURNITURE_PATH_WORDS.get(part, f"第 {int(part) + 1} 個" if part.isdigit() else part)
                           for part in tail)
    return f"{name}・{label}"


def _placement_branch_applies(path: str, document: Document) -> bool:
    parts = path.split(".")
    if len(parts) < 4 or parts[0] != "furniture" or not parts[1].isdigit() or parts[2] != "placement":
        return True
    kind = _furniture_at(document, int(parts[1])).get("kind")
    if not isinstance(kind, str) or kind not in FURNITURE_KINDS:
        return True
    wanted = "RoomPlacement" if kind == "ceiling_cloud" else "ListenerPlacement"
    return parts[3] not in PLACEMENT_BRANCHES or parts[3] == wanted


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
        found = _scene_field(rest, scheme)
    elif head == "speakers":
        found = _speaker_field(scheme, rest)
    elif head == "receiver_set":
        found = _seat_field(scheme, rest)
    elif head == "pairs":
        found = _pair_field(scheme, rest)
    elif head == "furniture":
        found = _furniture_field(scheme, rest)
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


# 喇叭類型與擺法的選項；desk、floor 這類字也出現在別的選項（家具種類），整組都是喇叭選項時才套喇叭的中文。
SPEAKER_SETUP_OPTIONS = frozenset({"bookshelf", "floorstanding", "stand", "desk", "floor"})


def _choices(match: re.Match[str]) -> str:
    values = _quoted(match["choices"])
    names = {**SOURCE_MODELS, **LOW_FREQUENCY_AXES, **FURNITURE_KINDS}
    if set(values) <= SPEAKER_SETUP_OPTIONS:
        names |= {value: SPEAKER_SETUP[value] for value in values}
    return "只能選" + "或".join(f"「{names.get(value, value)}」" for value in values)


def _height_message(match: re.Match[str]) -> str:
    # 喇叭名已在欄名；只在網頁重排公尺數字，驗證原文與技術明細完全不動。
    text = f"高度 {match['height']} m 跟擺法推出值不同：{match['reason']}"
    return re.sub(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?= m)",
                  lambda number: f"{float(number[0]):.12g}", text)


def _inside_fields(match: re.Match[str], document: Document) -> tuple[str, ...]:
    actor = (speaker_name(document, match["actor"]) if match["role"] == "喇叭"
             else seat_name(document, match["actor"]))
    return tuple(f"{actor} → {name}" for name in _furniture_fields(document, match["id"]))


def _rule(pattern: str, message: str | Callable[[re.Match[str]], str],
          fields: Callable[[re.Match[str], Document], tuple[str, ...]] | None = None) -> Rule:
    return Rule(re.compile(pattern, re.DOTALL), _fixed(message) if isinstance(message, str) else message,
                fields)


# 由上往下比，第一條對上的算數；「大於或等於」要排在「大於」前面。
RULES = (
    _rule(r"家具 (?P<id>.+) 的材質 (?P<material>\S+) 不適用 (?P<kind>\S+)",
          lambda match: f"材質「{FURNITURE_MATERIALS.get(match['material'], match['material'])}」"
                        f"不適用「{FURNITURE_KINDS.get(match['kind'], match['kind'])}」"),
    _rule(r"家具材質必須是五種材質之一：.+",
          "家具材質必須是五種材質之一：" + "／".join(FURNITURE_MATERIALS.values())),
    _rule(r"furniture_id 不可為空白", "家具代號不可空白"),
    _rule(r"furniture_id 不可重複：(?P<id>.+)", "家具代號不可重複",
          lambda match, document: _furniture_fields(document, match["id"])),
    _rule(r"家具 (?P<id>.+)：(?P<reason>天雲必須用房間座標|沙發、座椅、茶几、書桌必須跟著主位)",
          lambda match: match["reason"]),
    _rule(r"家具 (?P<id>.+)：(?P<reason>貼地家具的底面必須在地板接觸界線內|懸空家具的底面必須高於地板並超過接觸界線)",
          lambda match: match["reason"], lambda match, document: _furniture_fields(document, match["id"])),
    _rule(r"家具 (?P<id>.+)：(?P<reason>.+)",
          lambda match: match["reason"].removeprefix("家具"),
          lambda match, document: _furniture_fields(document, match["id"])),
    _rule(r"家具 (?P<id>.+) 超出房間接觸界線", "超出房間接觸界線",
          lambda match, document: _furniture_fields(document, match["id"])),
    _rule(r"家具 (?P<first>.+) 與 (?P<second>.+) 的間隙必須超過接觸界線", "兩件家具的間隙必須超過接觸界線",
          _close_furniture_fields),
    _rule(r"(?P<role>座位|喇叭) (?P<actor>.+) 在家具 (?P<id>.+) 的內部超過接觸界線",
          "在家具內部超過接觸界線", _inside_fields),
    _rule(r"喇叭放桌面必須剛好一件茶几或書桌，現在有 (?P<n>\d+) 件",
          lambda match: f"喇叭放桌面必須剛好一件茶几或書桌，現在有 {match['n']} 件",
          lambda _match, _document: (SPEAKER_SETUP["speaker_setup"],)),
    # 欄名從原句的代號取，不從路徑切：代號可含點，pairs.left.side.a 用點切會變成座位 side。
    _rule(r"不符合擺位要求：喇叭 (?P<speaker>.+?) 到座位 (?P<seat>.+?) 的直達路徑被家具 (?P<ids>.+) 擋住",
          lambda match: f"不符合擺位要求：直達路徑被家具 {match['ids']} 擋住",
          lambda match, document: (f"{speaker_name(document, match['speaker'])} → {seat_name(document, match['seat'])}",)),
    # 欄名從原句的代號取，不從路徑切（代號可含點）；網頁數字用十二位有效數字。
    _rule(r"喇叭 (?P<id>.+) 的高度 (?P<height>\S+) m 跟擺法推出值不同：(?P<reason>.+)", _height_message,
          lambda match, document: (f"{speaker_name(document, match['id'])} z 座標",)),
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
    _rule(r"方案必須有 left 與 right 兩個聲道角色", "方案必須有左、右兩個聲道",
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
    _rule(r"Input should be a valid boolean", "要填真假值"),
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
          "固定阻抗欄只收正實數；隨頻率變的牆請填構造，實測阻抗表還不收"),
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
    return ((field,) if field else ()), _plain_material_message(problem.message) if "surface_zones_by_wall" in problem.path else problem.message


def plan_problems(problems: Sequence[SchemeProblem], document: object,
                  plan: dict[str, object]) -> list[dict[str, object]]:
    """驗證可能先停在出界；補列圖上已標紅的直達，保留原本所有拒收理由。"""
    paths = cast(list[dict[str, object]], plan.get("blocked_paths", []))
    blocked = tuple(SchemeProblem(
        f"pairs.{path['speaker_id']}.{path['receiver_id']}",
        f"不符合擺位要求：喇叭 {path['speaker_id']} 到座位 {path['receiver_id']} 的直達路徑被家具 "
        f"{'、'.join(cast(tuple[str, ...], path['furniture_ids']))} 擋住") for path in paths)
    return plain_problems(tuple(dict.fromkeys((*problems, *blocked))), document)


def plain_problems(problems: Sequence[SchemeProblem], document: object) -> list[dict[str, object]]:
    """給網頁的問題清單：直達被擋每對一行，其餘同一句合併欄名；原路徑與原文另外留著。

    每一條是 {text: 直接印的一行, message: 白話, fields: 表單中文欄名, paths: 原本的欄位路徑,
    details: 原本的「路徑：訊息」}；頁面只印 text，技術細節要看時才用 paths／details。
    """
    scheme = _as_document(document)
    groups: dict[tuple[str, str], list[tuple[tuple[str, ...], SchemeProblem]]] = {}
    filtered: list[SchemeProblem] = []
    for problem in problems:
        if not _placement_branch_applies(problem.path, scheme):
            filtered.append(problem)
            continue
        fields, message = plain_problem(problem, scheme)
        # 直達被擋按每一對列出，不能把相同家具擋住的不同聆聽點合成一行。
        pair = problem.path if message.startswith("不符合擺位要求：直達路徑被家具 ") else ""
        groups.setdefault((message, pair), []).append((fields, problem))
    rows = [_merged(message, members) for (message, _), members in groups.items()]
    return _keep_filtered_details(rows, filtered)


def _keep_filtered_details(rows: list[dict[str, object]], filtered: list[SchemeProblem]) -> list[dict[str, object]]:
    if filtered and not rows:
        rows.append(_merged("", []))
    for problem in filtered:
        prefix = problem.path.split(".placement.", 1)[0] + "."
        row = next((row for row in rows if any(path.startswith(prefix)
                   for path in cast(list[str], row["paths"]))), rows[0])
        for key, value in (("paths", problem.path), ("details", str(problem))):
            values = cast(list[str], row[key])
            if value not in values:
                values.append(value)
    return rows


def _merged(message: str, members: list[tuple[tuple[str, ...], SchemeProblem]]) -> dict[str, object]:
    named = [field for fields, _ in members for field in fields]
    # 整份方案的問題跟有欄名的同一句合併：補寫「整份方案」，不讓它在那一行裡消失。
    if named and any(not fields for fields, _ in members):
        named.append(WHOLE_SCHEME_LABEL)
    fields = list(dict.fromkeys(named))
    return {"text": f"{'、'.join(fields)}：{message}" if fields else message, "message": message,
            "fields": fields, "paths": list(dict.fromkeys(problem.path for _, problem in members)),
            "details": list(dict.fromkeys(str(problem) for _, problem in members))}
