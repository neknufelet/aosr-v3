"""只判讀本批綱要考卷所用關鍵字；未知驗證關鍵字直接報錯，不冒充完整驗證器。"""
import re
from collections.abc import Callable, Mapping
from typing import cast

Schema = dict[str, object]
ANNOTATIONS = {"title", "description", "default", "discriminator", "$defs", "$schema",
               "quantity", "unit", "reference", "validity"}
KEYWORDS = {"$ref", "type", "const", "enum", "required", "properties", "additionalProperties",
            "items", "minItems", "maxItems", "minimum", "maximum", "exclusiveMinimum",
            "exclusiveMaximum", "anyOf", "oneOf", "allOf", "not", "if", "then", "else", "minLength", "maxLength", "pattern"}


def accepts(schema: Schema, value: object, root: Schema) -> bool:
    unknown = set(schema) - KEYWORDS - ANNOTATIONS - {key for key in schema if key.startswith("x-")}
    assert not unknown, f"判讀器沒有實作：{unknown}"
    if "$ref" in schema:
        target: object = root
        for part in str(schema["$ref"]).removeprefix("#/").split("/"):
            target = cast(Schema, target)[part]
        # 2020-12 的 $ref 與旁邊的關鍵字一起生效，不是只看引用目標。
        siblings = {key: item for key, item in schema.items() if key != "$ref"}
        return accepts(cast(Schema, target), value, root) and accepts(siblings, value, root)
    if not _branches(schema, value, root) or not _scalar(schema, value):
        return False
    if isinstance(value, Mapping):
        return _object(schema, cast(Schema, value), root)
    if isinstance(value, list):
        return _array(schema, value, root)
    return True


def _branches(schema: Schema, value: object, root: Schema) -> bool:
    for keyword in ("anyOf", "oneOf", "allOf"):
        if keyword not in schema:
            continue
        results = [accepts(branch, value, root) for branch in cast(list[Schema], schema[keyword])]
        if keyword == "anyOf" and not any(results):
            return False
        if keyword == "oneOf" and sum(results) != 1:
            return False
        if keyword == "allOf" and not all(results):
            return False
    if "not" in schema and accepts(cast(Schema, schema["not"]), value, root):
        return False
    if "if" in schema:
        branch = "then" if accepts(cast(Schema, schema["if"]), value, root) else "else"
        if branch in schema and not accepts(cast(Schema, schema[branch]), value, root):
            return False
    return True


def _scalar(schema: Schema, value: object) -> bool:
    types = {"object": isinstance(value, Mapping), "array": isinstance(value, list),
             "null": value is None, "string": isinstance(value, str),
             "number": isinstance(value, (int, float)) and not isinstance(value, bool)}
    if "type" in schema and not types[str(schema["type"])]:
        return False
    if "const" in schema and not _same(value, schema["const"]):
        return False
    if "enum" in schema and not any(_same(value, item) for item in cast(list[object], schema["enum"])):
        return False
    if isinstance(value, str):
        if "minLength" in schema and len(value) < cast(int, schema["minLength"]):
            return False
        if "maxLength" in schema and len(value) > cast(int, schema["maxLength"]):
            return False
        # JSON 綱要的 pattern 不錨定：字串裡任何位置找得到就算符合。
        if "pattern" in schema and re.search(str(schema["pattern"]), value) is None:
            return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        checks: tuple[tuple[str, Callable[[float], bool]], ...] = (
            ("minimum", lambda bound: value < bound), ("maximum", lambda bound: value > bound),
            ("exclusiveMinimum", lambda bound: value <= bound), ("exclusiveMaximum", lambda bound: value >= bound))
        for keyword, failed in checks:
            if keyword in schema and failed(cast(float, schema[keyword])):
                return False
    return True


def _same(value: object, expected: object) -> bool:
    """JSON 的真假值不是數字：Python 的 True == 1 不能讓 const／enum 放過。"""
    return isinstance(value, bool) is isinstance(expected, bool) and value == expected


def _object(schema: Schema, value: Schema, root: Schema) -> bool:
    if not set(cast(list[str], schema.get("required", []))) <= value.keys():
        return False
    properties = cast(Schema, schema.get("properties", {}))
    for name, item in value.items():
        rule = properties.get(name, schema.get("additionalProperties", True))
        if rule is False or isinstance(rule, dict) and not accepts(rule, item, root):
            return False
    return True


def _array(schema: Schema, value: list[object], root: Schema) -> bool:
    if "minItems" in schema and len(value) < cast(int, schema["minItems"]):
        return False
    if "maxItems" in schema and len(value) > cast(int, schema["maxItems"]):
        return False
    return all(accepts(cast(Schema, schema["items"]), item, root) for item in value) if "items" in schema else True
