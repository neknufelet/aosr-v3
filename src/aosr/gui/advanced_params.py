"""網頁進階參數清單的唯讀載入（#1343）：一條參數的中文名、預設、單位、範圍、說明與所屬頁面。

清單只有網頁在讀，所以住 gui 底下、跟這一支放一起（``advanced_params.toml``）：放進 ``aosr/config`` 會進計算指紋，
改一個網頁預設就讓所有舊結果被標成要重算。路徑必給（同 ``docs/decisions/config-loaders-keep-path-required.md``）。
預設值只住一處：寫 ``default`` 的這份就是家；寫 ``source`` 的家在引擎設定目錄另一份設定檔，``resolved_default``
讀的時候現抓、這裡不抄數字；兩者都沒有的是「沒有固定值」，``default_note`` 寫白話。
"""
from __future__ import annotations

import math
import re
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Value = bool | int | float | str
Kind = Literal["integer", "number", "boolean", "choice", "text"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Page(_Frozen):
    id: str = Field(pattern=r"^[a-z_]+$")
    name: str = Field(min_length=1)
    path: str = Field(pattern=r"^/[a-z_/-]*$")


class Source(_Frozen):
    """預設值的家：引擎設定目錄（``aosr/config/data``）哪一份設定檔、哪一格（點分路徑，整數段是陣列第幾個）。"""

    file: str = Field(pattern=r"^[a-z_]+\.toml$")
    key: str = Field(pattern=r"^[a-z_0-9]+(\.[a-z_0-9]+)*$")


class Choice(_Frozen):
    value: str = Field(min_length=1)
    label: str = Field(min_length=1)


class Param(_Frozen):
    key: str = Field(pattern=r"^[a-z_]+\.[a-z_]+$")
    page: str = Field(min_length=1)
    field: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    name: str = Field(min_length=1)
    unit: str
    kind: Kind
    default: Value | None = None
    source: Source | None = None
    default_note: str = ""
    min: int | float | None = None
    max: int | float | None = None
    min_open: bool = False
    max_open: bool = False
    choices: tuple[Choice, ...] = ()
    range_note: str = ""
    blurb: str = Field(min_length=1)

    @model_validator(mode="after")
    def _checked(self) -> Param:
        if self.default is not None and self.source is not None:
            raise ValueError(f"{self.key}：default 與 source 只能擇一，兩個都寫就是兩份數字")
        if self.default is None and self.source is None and not self.default_note:
            raise ValueError(f"{self.key}：沒有 default 也沒有 source 時要寫 default_note（預設是什麼）")
        if (self.kind == "choice") != bool(self.choices):
            raise ValueError(f"{self.key}：choices 只給 kind = \"choice\"，而且 choice 一定要有")
        if (self.min_open and self.min is None) or (self.max_open and self.max is None):
            raise ValueError(f"{self.key}：min_open／max_open 要配 min／max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"{self.key}：min 大於 max")
        if self.default is not None:
            check_value(self, self.default)
        return self


class AdvancedParams(_Frozen):
    page: tuple[Page, ...] = Field(min_length=1)
    param: tuple[Param, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> AdvancedParams:
        pages = [page.id for page in self.page]
        keys = [param.key for param in self.param]
        if len(set(pages)) != len(pages) or len(set(keys)) != len(keys):
            raise ValueError("頁面代號與參數 key 都不准重複")
        unknown = sorted({param.page for param in self.param} - set(pages))
        if unknown:
            raise ValueError(f"參數指到沒登記的頁面：{unknown}")
        return self

    def by_key(self, key: str) -> Param:
        for param in self.param:
            if param.key == key:
                return param
        raise KeyError(f"進階參數清單沒有 {key}")


def check_value(param: Param, value: object) -> None:
    """值的型別與範圍要對得上這一條的宣告；對不上就丟 ``ValueError``（清單與它指的設定檔不一致）。"""
    is_bool = isinstance(value, bool)
    if param.kind == "boolean" and not is_bool:
        raise ValueError(f"{param.key}：預設要是 true／false，收到 {value!r}")
    if param.kind == "integer" and (is_bool or not isinstance(value, int)):
        raise ValueError(f"{param.key}：預設要是整數，收到 {value!r}")
    if param.kind == "number" and (is_bool or not isinstance(value, int | float) or not math.isfinite(value)):
        raise ValueError(f"{param.key}：預設要是有限的數字，收到 {value!r}")
    if param.kind in ("choice", "text") and not isinstance(value, str):
        raise ValueError(f"{param.key}：預設要是文字，收到 {value!r}")
    if param.kind == "choice" and value not in {choice.value for choice in param.choices}:
        raise ValueError(f"{param.key}：預設 {value!r} 不在可選值裡")
    if isinstance(value, int | float) and not is_bool:
        low_bad = param.min is not None and (value <= param.min if param.min_open else value < param.min)
        high_bad = param.max is not None and (value >= param.max if param.max_open else value > param.max)
        if low_bad or high_bad:
            raise ValueError(f"{param.key}：預設 {value!r} 超出登記的範圍")


def load_advanced_params(path: str | Path) -> AdvancedParams:
    with Path(path).open("rb") as handle:
        return AdvancedParams.model_validate(tomllib.load(handle))


def _walk(document: object, key: str, where: str) -> object:
    node = document
    for part in key.split("."):
        if re.fullmatch(r"\d+", part) and isinstance(node, list) and int(part) < len(node):
            node = node[int(part)]
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise ValueError(f"{where} 找不到 {key}")
    return node


def resolved_default(param: Param, directory: str | Path) -> Value | None:
    """這一條實際的預設：寫在清單的直接回；指到別檔的去那一檔讀那一格（並核型別與範圍）；沒有固定值回 ``None``。"""
    if param.source is None:
        return param.default
    with (Path(directory) / param.source.file).open("rb") as handle:
        value = _walk(tomllib.load(handle), param.source.key, param.source.file)
    check_value(param, value)
    if not isinstance(value, bool | int | float | str):
        raise ValueError(f"{param.key}：{param.source.file} 的 {param.source.key} 不是單一個值")
    return value
