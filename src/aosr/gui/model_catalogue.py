"""吸音模型目錄的唯讀載入（#1343 追加）：aosr 實際在用的每個吸音模型、它的參數、有效範圍與文獻引用。

目錄只有網頁在讀，跟 ``advanced_params`` 一樣住 gui 底下（``model_catalogue.toml``），不進計算指紋與物理身分。
數字一個都不抄：參數的中文名、單位、可填範圍從 ``aosr.materials.construction_models`` 的欄位定義現讀；
Komatsu 的範圍與書目從範圍登記簿 ``porous_model_ranges.toml`` 現讀。repo 裡沒寫的書目欄位留空，頁面寫「repo 未列」。
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aosr.config.porous_model_ranges import KomatsuRanges, load_porous_model_ranges
from aosr.materials import construction_models

NO_RANGE: Final = "未列有效範圍"
NOT_LISTED: Final = "repo 未列"
RELATED_NOTE: Final = "相關文獻（外部對照報告用它比對；aosr 實作是否照它：未核對）"
BACKGROUND_NOTE: Final = "背景文獻（aosr 沒有實作這些模型）"
_AT = r"^[A-Za-z0-9_./-]+:[1-9][0-9]*$"
_UNIT = r"無量綱|比例|[A-Za-z0-9·/²³°%]+"
_CODE = r"^src/aosr/[a-z_/]+\.py::[A-Za-z_][A-Za-z0-9_]*$"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Mark(_Frozen):
    """一條出處：repo 根起算的「檔案:行」，加那一行一定含的字。"""

    at: str = Field(pattern=_AT)
    mark: str = Field(min_length=1)


class ModelRange(_Frozen):
    registry: str | None = Field(default=None, pattern=r"^komatsu$")
    note: str = ""
    at: str | None = Field(default=None, pattern=_AT)
    mark: str | None = None

    @model_validator(mode="after")
    def _sourced(self) -> ModelRange:
        if (self.registry is None) == (not self.note):
            raise ValueError("範圍要嘛指登記簿（registry），要嘛寫 note，只能擇一")
        if (self.at is None) != (self.mark is None):
            raise ValueError("範圍的 at 與 mark 同給同不給")
        if self.at is None and self.note != NO_RANGE:
            raise ValueError(f"沒有出處的範圍只准寫「{NO_RANGE}」")
        return self


class Citation(_Frozen):
    """文獻引用；repo 沒寫的欄位不填。``registry`` 指範圍登記簿那一節的書目與 DOI。"""

    authors: tuple[str, ...] = Field(min_length=1)
    year: int | None = None
    title: str | None = None
    venue: str | None = None
    doi: str | None = None
    registry: str | None = Field(default=None, pattern=r"^komatsu$")
    reference: str | None = None
    note: str = ""
    at: str = Field(pattern=_AT)

    @model_validator(mode="after")
    def _one_home(self) -> Citation:
        homes = (self.registry is not None) + (self.reference is not None)
        if homes > 1 or (homes and (self.title or self.venue or self.doi)):
            raise ValueError("書目從登記簿或核過的外部書目讀時，題名、期刊、DOI 不准再抄一份，兩處也只能擇一")
        if self.reference is not None and not self.note:
            raise ValueError("掛核過的外部書目要寫 note：書目核過不等於 aosr 照它實作")
        return self


class Reference(_Frozen):
    """核過的外部書目：書目核過，不代表 aosr 照它實作；怎麼掛在模型上由各欄位分開標。"""

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    text: str = Field(min_length=1)
    doi: str = Field(pattern=r"^https://doi\.org/10\.[0-9]{4,}/\S+$")
    verified: str = Field(min_length=1)


class Unimplemented(_Frozen):
    """aosr 沒有實作、只為了跟外部對照時知道出處而列的模型。"""

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str = Field(min_length=1)
    why: str = Field(min_length=1)
    references: tuple[str, ...] = Field(min_length=1)
    at: str = Field(pattern=_AT)
    mark: str = Field(min_length=1)


class Origin(_Frozen):
    text: str = Field(min_length=1)
    at: str = Field(pattern=_AT)
    mark: str = Field(min_length=1)


class Model(_Frozen):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str = Field(min_length=1)
    code: str = Field(pattern=_CODE)
    variant: str | None = None
    compute: tuple[str, ...] = Field(min_length=1)
    params: tuple[str, ...]
    blurb: str = Field(min_length=1)
    range: ModelRange
    citation: Citation | None = None
    related: tuple[str, ...] = ()
    background: tuple[str, ...] = ()
    origin: Origin

    @model_validator(mode="after")
    def _known(self) -> Model:
        for code in self.compute:
            if re.fullmatch(_CODE, code) is None:
                raise ValueError(f"{self.id}：compute {code!r} 不是「檔案::名字」")
        for param in self.params:
            field_property(param)
        return self


class NotModel(_Frozen):
    code: str = Field(pattern=_CODE)
    why: str = Field(min_length=1)


class ModelCatalogue(_Frozen):
    model: tuple[Model, ...] = Field(min_length=1)
    not_model: tuple[NotModel, ...] = ()
    reference: tuple[Reference, ...] = ()
    unimplemented: tuple[Unimplemented, ...] = ()

    @model_validator(mode="after")
    def _unique(self) -> ModelCatalogue:
        ids = [model.id for model in self.model]
        if len(set(ids)) != len(ids):
            raise ValueError("模型代號不准重複")
        known = {item.id for item in self.reference}
        if len(known) != len(self.reference):
            raise ValueError("外部書目代號不准重複")
        used = [ref for model in self.model for ref in (*model.related, *model.background)]
        used += [model.citation.reference for model in self.model if model.citation and model.citation.reference]
        used += [ref for item in self.unimplemented for ref in item.references]
        unknown = sorted(set(used) - known)
        if unknown:
            raise ValueError(f"指到沒登記的外部書目：{unknown}")
        return self


def load_model_catalogue(path: str | Path) -> ModelCatalogue:
    with Path(path).open("rb") as handle:
        return ModelCatalogue.model_validate(tomllib.load(handle))


def field_property(param: str) -> dict[str, object]:
    """「類別.欄位」在構造格式綱要裡的那一格；可為空的欄位取非空那一支。找不到丟 ``ValueError``。"""
    owner, _, name = param.partition(".")
    cls = getattr(construction_models, owner, None)
    if not (isinstance(cls, type) and issubclass(cls, BaseModel)) or name not in cls.model_fields:
        raise ValueError(f"參數 {param!r} 在 construction_models 找不到")
    prop: dict[str, object] = cls.model_json_schema()["properties"][name]
    branches = prop.get("anyOf")
    if isinstance(branches, list):
        concrete = [branch for branch in branches if isinstance(branch, dict) and branch.get("type") != "null"]
        prop = {**concrete[0], "description": prop.get("description", "")}
    return prop


def described(description: str) -> tuple[str, str]:
    """欄位說明「代號（中文名，單位；補充）」拆成中文名與單位；沒寫單位回空字串。"""
    inner = description.partition("（")[2]
    name = re.split(r"[，：；）]", inner, maxsplit=1)[0]
    rest = inner[len(name):]
    unit = re.split(r"[；：）]", rest[1:], maxsplit=1)[0] if rest.startswith("，") else ""
    return name, unit if re.fullmatch(_UNIT, unit) else ""


def _bounds(prop: dict[str, object]) -> str:
    choices = prop.get("enum")
    if isinstance(choices, list):
        return "／".join(str(value) for value in choices)
    parts = [f"{sign} {prop[key]}" for key, sign in
             (("exclusiveMinimum", "＞"), ("minimum", "≥"), ("exclusiveMaximum", "＜"), ("maximum", "≤")) if key in prop]
    return "、".join(parts)


def param_row(param: str) -> dict[str, str]:
    prop = field_property(param)
    name, unit = described(str(prop.get("description", "")))
    return {"key": param, "name": name, "unit": unit, "bounds": _bounds(prop)}


def _komatsu_range(ranges: KomatsuRanges) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, field in KomatsuRanges.model_fields.items():
        if key == "provenance":
            continue
        name, unit = described(field.description or "")
        value = getattr(ranges, key)
        rows.append({"name": name, "unit": unit, "value": list(value) if isinstance(value, tuple) else value})
    return rows


def _reference_row(reference: Reference) -> dict[str, str]:
    return {"id": reference.id, "text": reference.text, "doi": reference.doi, "verified": reference.verified}


def catalogue_listing(path: str | Path, ranges_path: str | Path) -> dict[str, list[dict[str, object]]]:
    """整份目錄，參數、Komatsu 範圍與書目已現讀解開；給 ``/api/params`` 的 ``models`` 與 ``unimplemented``。"""
    catalogue = load_model_catalogue(path)
    references = {item.id: item for item in catalogue.reference}
    komatsu = load_porous_model_ranges(ranges_path).komatsu
    out: list[dict[str, object]] = []
    for model in catalogue.model:
        row = model.model_dump(mode="json", exclude={"params", "range", "citation", "related", "background"})
        row["related"] = {"note": RELATED_NOTE, "items": [_reference_row(references[ref]) for ref in model.related]}
        row["background"] = {"note": BACKGROUND_NOTE, "items": [_reference_row(references[ref]) for ref in model.background]}
        row["params"] = [param_row(param) for param in model.params]
        span = model.range
        row["range"] = {
            "note": "原研究樣品與量測範圍，不是作者訂的有效界線" if span.registry else span.note,
            "rows": _komatsu_range(komatsu) if span.registry else [],
            "at": span.at,
        }
        cite = model.citation
        if cite is None:
            row["citation"] = None
        else:
            provenance = komatsu.provenance if cite.registry else None
            verified = references[cite.reference] if cite.reference else None
            row["citation"] = {
                "authors": list(cite.authors),
                "year": cite.year if cite.year is not None else NOT_LISTED,
                "reference": provenance.reference if provenance else verified.text if verified else
                " ".join(part for part in (cite.title, cite.venue) if part) or NOT_LISTED,
                "doi": provenance.doi if provenance else verified.doi if verified else cite.doi or NOT_LISTED,
                "verified": verified.verified if verified else None,
                "note": cite.note,
                "at": cite.at,
            }
        out.append(row)
    missing = [{**item.model_dump(mode="json", exclude={"references"}),
                "references": [_reference_row(references[ref]) for ref in item.references]} for item in catalogue.unimplemented]
    return {"models": out, "unimplemented": missing}
