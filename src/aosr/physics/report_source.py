"""報表聲源模型的輸入、求解身分與場景輸出契約。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from typing import Annotated, Literal, Self

from pydantic import Field, StrictFloat, WithJsonSchema, field_validator, model_validator

from aosr.config.directivity_defaults import PositiveStrictFloat, TwoParameterCurve
from aosr.geometry.shoebox import Point
from aosr.physics.report_facts import (
    EMPTY_FOR_OMNIDIRECTIONAL,
    NO_BASIS_TEXT,
    NOT_MEASURED,
    FactsModel,
    coordinate_object_facts,
    facts,
)
from aosr.physics.source_directivity import MODEL_VERSION, SourceModel, speaker_axis


class SourceModelKind(StrEnum):
    OMNIDIRECTIONAL = SourceModel.OMNIDIRECTIONAL.value
    ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1 = SourceModel.TWO_PARAMETER.value


OMNIDIRECTIONAL_STATUS = "全向點聲源。"
ANALYTIC_STATUS = "解析近似：水平面擬合、上下方向沿用同一條曲線、尚未獨立驗證。"


_CURVE_PAPER = "決策紙 source-directivity-two-parameter-axisymmetric-baseline 的兩參數曲線"
_PARAMETERS_REFERENCE = "六個曲線標量，見底下每一欄自己的單位與基準"


class SourceCurveParameters(FactsModel):
    """兩參數曲線的六個標量，逐欄帶四件事；值的規則只住在 :class:`TwoParameterCurve`（驗證時交給它），
    「大於 0」那條界限跟它共用同一個型別 ``PositiveStrictFloat``，匯出的 schema 才說得出同一條界限。"""

    beta_limit: StrictFloat = Field(
        json_schema_extra=facts("指向參數 β 的高頻極限", "1", _CURVE_PAPER, NOT_MEASURED)
    )
    beta_corner_hz: PositiveStrictFloat = Field(
        json_schema_extra=facts("β 曲線的轉折頻率", "Hz", _CURVE_PAPER, NOT_MEASURED)
    )
    beta_exponent: PositiveStrictFloat = Field(
        json_schema_extra=facts("β 曲線的斜率指數", "1", _CURVE_PAPER, NOT_MEASURED)
    )
    power_floor_limit_db: StrictFloat = Field(
        json_schema_extra=facts("功率下限的高頻極限", "dB", "相對於正前方的聲壓平方（功率）", NOT_MEASURED)
    )
    power_floor_corner_hz: PositiveStrictFloat = Field(
        json_schema_extra=facts("功率下限曲線的轉折頻率", "Hz", _CURVE_PAPER, NOT_MEASURED)
    )
    power_floor_exponent: PositiveStrictFloat = Field(
        json_schema_extra=facts("功率下限曲線的斜率指數", "1", _CURVE_PAPER, NOT_MEASURED)
    )

    @model_validator(mode="after")
    def _same_rules_as_the_registry(self) -> Self:
        TwoParameterCurve.model_validate(self.model_dump())
        return self

    def to_curve(self) -> TwoParameterCurve:
        return TwoParameterCurve.model_validate(self.model_dump())

    @classmethod
    def from_curve(cls, curve: TwoParameterCurve) -> Self:
        return cls.model_validate(curve.model_dump())


class OmnidirectionalInput(FactsModel):
    kind: Literal[SourceModelKind.OMNIDIRECTIONAL] = Field(
        json_schema_extra=facts("聲源模型種類", "1", NO_BASIS_TEXT, NOT_MEASURED)
    )


class AnalyticAxisymmetricInput(FactsModel):
    kind: Literal[SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1] = Field(
        json_schema_extra=facts("聲源模型種類", "1", NO_BASIS_TEXT, NOT_MEASURED)
    )
    parameters: SourceCurveParameters = Field(
        description="六個曲線標量全部明寫，不從登記簿補預設",
        json_schema_extra=facts("聲源模型參數", "1", _PARAMETERS_REFERENCE, NOT_MEASURED),
    )
    aim_m: Annotated[
        Point,
        WithJsonSchema(
            coordinate_object_facts("長度", "m", "房間角落為原點", names=("x", "y", "z"), positive=False)
        ),
    ] = Field(description="聲源的共同對準點（公尺）")

    @field_validator("aim_m", mode="before")
    @classmethod
    def _aim_is_a_strict_point(cls, value: object) -> object:
        names = ("x", "y", "z")
        if not isinstance(value, dict) or set(value) != set(names):
            raise ValueError("aim_m 必須只有 x、y、z 三個座標")
        if any(isinstance(value[name], bool) or not isinstance(value[name], int | float)
               or not math.isfinite(value[name]) for name in names):
            raise ValueError("aim_m 每個座標必須是有限數字")
        return value


SourceModelInput = Annotated[
    OmnidirectionalInput | AnalyticAxisymmetricInput, Field(discriminator="kind")
]


@dataclass(frozen=True)
class SourceModelSpec:
    kind: SourceModelKind
    parameters: TwoParameterCurve | None = None
    aim: Point | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SourceModelKind):
            raise ValueError("source_model.kind 必須是已登記的聲源模型種類")
        if self.kind == SourceModelKind.OMNIDIRECTIONAL:
            if self.parameters is not None or self.aim is not None:
                raise ValueError("全向聲源模型不能帶參數或對準點")
        elif not isinstance(self.parameters, TwoParameterCurve) or not isinstance(self.aim, Point):
            raise ValueError("解析近似聲源模型必須帶完整參數與對準點")


def source_model_spec(value: OmnidirectionalInput | AnalyticAxisymmetricInput) -> SourceModelSpec:
    """把已驗過的 JSON 模型變成物理層使用的凍結身分。"""
    if isinstance(value, OmnidirectionalInput):
        return SourceModelSpec(kind=value.kind)
    return SourceModelSpec(kind=value.kind, parameters=value.parameters.to_curve(), aim=value.aim_m)


def require_omnidirectional(spec: SourceModelSpec) -> None:
    """第二刀所有計算入口共用的 fail-closed 守門。"""
    if spec.kind != SourceModelKind.OMNIDIRECTIONAL:
        raise ValueError(f"聲源模型 {spec.kind.value} 第四刀才接上計算")


class SourceModelSection(FactsModel):
    kind: SourceModelKind = Field(
        json_schema_extra=facts("聲源模型種類", "1", NO_BASIS_TEXT, NOT_MEASURED)
    )
    model_version: str | None = Field(
        json_schema_extra=facts("模型版本", "1", NO_BASIS_TEXT, NOT_MEASURED)
    )
    parameters: SourceCurveParameters | None = Field(
        json_schema_extra=facts("聲源模型參數", "1", _PARAMETERS_REFERENCE + "；全向時是空的", NOT_MEASURED)
    )
    aim_m: Point | None = Field(
        json_schema_extra=facts("對準點", "m", "房間角落為原點", NOT_MEASURED)
    )
    axis_unit_vector: tuple[float, float, float] | None = Field(
        json_schema_extra=facts("軸線單位向量", "1", "房間座標；從聲源指向對準點", EMPTY_FOR_OMNIDIRECTIONAL)
    )
    verification_status: str = Field(
        json_schema_extra=facts("驗證狀態", "1", NO_BASIS_TEXT, NOT_MEASURED)
    )

    @model_validator(mode="after")
    def _shape_and_status(self) -> Self:
        details = (self.model_version, self.parameters, self.aim_m, self.axis_unit_vector)
        if self.kind == SourceModelKind.OMNIDIRECTIONAL:
            if any(value is not None for value in details) or self.verification_status != OMNIDIRECTIONAL_STATUS:
                raise ValueError("全向聲源模型的欄位或驗證狀態不符")
        elif any(value is None for value in details) or self.model_version != MODEL_VERSION or self.verification_status != ANALYTIC_STATUS:
            raise ValueError("解析近似聲源模型的欄位或驗證狀態不符")
        return self

    @classmethod
    def from_spec(cls, spec: SourceModelSpec, source: Point | None) -> Self:
        if spec.kind == SourceModelKind.OMNIDIRECTIONAL:
            return cls(kind=spec.kind, model_version=None, parameters=None, aim_m=None,
                       axis_unit_vector=None, verification_status=OMNIDIRECTIONAL_STATUS)
        if source is None or spec.aim is None or spec.parameters is None:
            raise ValueError("解析近似報表需要聲源位置、對準點與參數")
        return cls(kind=spec.kind, model_version=MODEL_VERSION,
                   parameters=SourceCurveParameters.from_curve(spec.parameters),
                   aim_m=spec.aim, axis_unit_vector=speaker_axis(source, spec.aim),
                   verification_status=ANALYTIC_STATUS)
