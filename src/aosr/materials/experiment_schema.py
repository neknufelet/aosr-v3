"""實驗參數層：API 進來的那一份 YAML 驗證成型別（schema，結構定義）。

這一層是 ``POST /jobs`` 的請求模型，所以每一個模型都 ``extra="forbid"``（多給一格就拒收）
——打錯的欄位名要當場紅，不是安靜地被忽略。

**類別的說明文字是契約的一部分**：pydantic 會把它寫進 ``model_json_schema()`` 的
``description``，那一格是凍結答案裡逐字比的東西，所以下面幾句英文原封不動；中文說明寫在
註解裡，不寫進說明文字。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, cast

import jax.numpy as jnp
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aosr.config.receiver_grid import GRID_MARGIN_M, GRID_N
from aosr.materials.response import FrequencyAxis


# 一個三維座標（房間幾何與收音／發聲位置都用它）。
class Vec3(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        """給求解器用的三元組（那一層收 tuple，不收模型）。"""
        return (self.x, self.y, self.z)


# 怎麼建那條分析用的頻率軸：三種模式，各自要的欄位不一樣。
class FrequencyAxisSpec(BaseModel):
    """How to build the analysis :class:`FrequencyAxis`."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["linear_hz", "third_octave", "custom"] = "third_octave"
    f_min_hz: float | None = Field(default=None, gt=0.0)
    f_max_hz: float | None = Field(default=None, gt=0.0)
    n_points: int | None = Field(default=None, gt=1)
    freqs_hz: list[float] | None = None

    @model_validator(mode="after")
    def _check(self) -> FrequencyAxisSpec:
        """哪幾格必填**看模式**：模式不合的組合在建構就擋掉，不要拖到 build 才炸。"""
        if self.mode == "custom":
            if not self.freqs_hz:
                raise ValueError("custom frequency axis requires non-empty freqs_hz")
        else:
            if self.f_min_hz is None or self.f_max_hz is None:
                raise ValueError(f"{self.mode} axis requires f_min_hz and f_max_hz")
            if self.f_max_hz <= self.f_min_hz:
                raise ValueError("f_max_hz must exceed f_min_hz")
            if self.mode == "linear_hz" and self.n_points is None:
                raise ValueError("linear_hz axis requires n_points")
        return self

    def build(self) -> FrequencyAxis:
        """把這一份規格變成真的軸。"""
        if self.mode == "custom":
            return self._build_custom()
        if self.mode == "linear_hz":
            # ``cast`` 只對型別檢查說話，執行期一個值都不碰：這個模型沒有 frozen 也沒有
            # validate_assignment，欄位建構後改得動，改壞了就要由下面那一步自己炸（字串
            # 交給 JAX 炸、None 交給 int()／linspace 炸）。在這裡先驗一次會把下游的
            # TypeError 換成這一層的 ValueError，也會把上一代收不下的輸入轉成收得下。
            freqs = jnp.linspace(
                cast(float, self.f_min_hz),
                cast(float, self.f_max_hz),
                int(cast(int, self.n_points)),
            )
            return FrequencyAxis.from_hz(freqs, resolution="linear_hz")
        return self._build_third_octave()

    def _build_custom(self) -> FrequencyAxis:
        """自訂那一支：**先驗再排序**。

        API 這一邊本來就收未排序的輸入（材料層那支驗證器要求遞增，所以不能拿原始清單去叫
        它）；非有限、非正、重複三種一律拒收——那三種排序完也還是壞的。
        """
        freqs_hz = self.freqs_hz
        field_name = "frequency_axis.freqs_hz"
        if not freqs_hz:
            raise ValueError(f"{field_name} must be non-empty")
        if any(not math.isfinite(freq) for freq in freqs_hz):
            raise ValueError(f"{field_name} must contain only finite frequencies")
        if any(freq <= 0.0 for freq in freqs_hz):
            raise ValueError(f"{field_name} must contain only positive frequencies")
        if len(set(freqs_hz)) != len(freqs_hz):
            raise ValueError(f"{field_name} must not contain duplicate frequencies")
        freqs = sorted(freqs_hz)
        return FrequencyAxis.from_hz(jnp.asarray(freqs), resolution="custom")

    def _build_third_octave(self) -> FrequencyAxis:
        """1/3 倍頻那一支：以 1 kHz 為基準的名義中心，只取落在範圍內的。"""
        centres = []
        # ``cast`` 同上：只給型別檢查看，執行期照樣讓除法自己炸。
        n_lo = math.ceil(3 * math.log2(cast(float, self.f_min_hz) / 1000.0))
        n_hi = math.floor(3 * math.log2(cast(float, self.f_max_hz) / 1000.0))
        for n in range(n_lo, n_hi + 1):
            centres.append(1000.0 * (2.0 ** (n / 3.0)))
        if not centres:
            raise ValueError("third_octave range contains no band centres")
        return FrequencyAxis.from_hz(jnp.asarray(centres), resolution="third_octave")


# 空間均勻度評分用的收音點網格。預設值直接依賴第 2 層那一份，不手抄。
class ReceiverGridSpec(BaseModel):
    """Receiver grid for spatial-deviation evaluation (design §12.2)."""

    model_config = ConfigDict(extra="forbid")
    n_points: int = Field(default=GRID_N * GRID_N, gt=0)
    wall_offset_m: float = Field(default=GRID_MARGIN_M, ge=0.0)


# 一個表面邊界（boundary，幾何上的一片面）配一個材料 id。
class SurfaceMaterial(BaseModel):
    """Map a stable surface boundary id to a material id (design §5, §10)."""

    model_config = ConfigDict(extra="forbid")
    boundary_id: str
    material_id: str


# 房間：幾何等級 0–4，等級 0/1 用的是長方體尺寸。
class RoomSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    geometry_level: int = Field(ge=0, le=4, description="design §5 geometry level")
    dims_m: Vec3 | None = Field(default=None, description="shoebox dimensions (Level 0/1)")


# 一整份實驗請求。
class ExperimentConfig(BaseModel):
    """A full experiment request (validated for ``POST /jobs``)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    frequency_axis: FrequencyAxisSpec
    room: RoomSpec
    source: Vec3
    listening_position: Vec3
    receiver_grid: ReceiverGridSpec = ReceiverGridSpec()
    surface_materials: list[SurfaceMaterial] = Field(default_factory=list)


def load_experiment(path: str | Path) -> ExperimentConfig:
    """讀一份 ``experiment.yaml`` 並驗證。

    ``path`` **必填**：沒有預設路徑，這一層不替呼叫端猜檔案住在哪裡（決策紙
    ``config-loaders-keep-path-required``）。
    """
    with Path(path).open() as f:
        data = yaml.safe_load(f)
    return ExperimentConfig(**data)
