"""材料對應層：一個材料一份 YAML，讀成註冊表（registry，材料的登記簿）。

M1 還沒有 TMM（transfer matrix method，傳遞矩陣法），所以一份材料檔直接寫它的表面阻抗：
``rigid``（剛性牆）、``constant_impedance``（絕對值，Pa·s/m）、或
``constant_normalized_impedance``（正規化值 ζ = Z/ρc，乘上呼叫端注入的 ``rho_c`` 還原）。

**類別的說明文字是契約的一部分**（會進 ``model_json_schema()`` 的 ``description``），所以
下面那一句英文原封不動；中文說明寫在註解裡。
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from aosr.materials.registry import MaterialRegistry, constant_impedance, rigid_wall
from aosr.materials.response import BoundaryModel, FrequencyAxis, MaterialResponse


# 一個複數（表面阻抗有實部與虛部；虛部不給就是 0，代表純電阻性的表面）。
class _ComplexSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    real: float
    imag: float = 0.0

    def as_complex(self) -> complex:
        """還原成 Python 的複數。"""
        return complex(self.real, self.imag)


class MaterialSpec(BaseModel):
    """Validated contents of one ``config/materials/<id>.yaml`` (M1 placeholder)."""

    model_config = ConfigDict(extra="forbid")

    material_id: str
    boundary_model: BoundaryModel = "local_impedance"
    is_locally_reacting: bool = True
    model: Literal["rigid", "constant_impedance", "constant_normalized_impedance"] = (
        "constant_normalized_impedance"
    )
    z_magnitude: float | None = None
    impedance_pa_s_m: _ComplexSpec | None = None
    normalized_impedance: _ComplexSpec | None = None

    @model_validator(mode="after")
    def _check(self) -> MaterialSpec:
        """選了哪一支模型，那一支要的值就必填——缺值要在建構時紅，不是 build 才紅。"""
        if self.model == "constant_impedance" and self.impedance_pa_s_m is None:
            raise ValueError("constant_impedance requires impedance_pa_s_m")
        if self.model == "constant_normalized_impedance" and self.normalized_impedance is None:
            raise ValueError("constant_normalized_impedance requires normalized_impedance")
        return self

    def build(self, freq_axis: FrequencyAxis, rho_c: float) -> MaterialResponse:
        """在指定的頻率軸上建出這份材料的回應。

        ``rho_c``（空氣的特性阻抗）**由呼叫端注入**，這一層不自己算——溫度一換那個數就變，
        算在這裡等於把環境假設藏進材料。
        """
        if self.model == "rigid":
            # 剛性牆走 rigid_wall() 自己的邊界設定：spec 上的 boundary_model 與
            # is_locally_reacting **不往下傳**（上一代就是這樣，要改得先有決策紙）。
            # ``z_magnitude or 1e8`` 連 0.0 也會落到預設值，負值則照收。
            return rigid_wall(
                freq_axis,
                material_id=self.material_id,
                z_magnitude=self.z_magnitude or 1e8,
            )
        return constant_impedance(
            freq_axis,
            self._surface_impedance(rho_c),
            material_id=self.material_id,
            boundary_model=self.boundary_model,
            is_locally_reacting=self.is_locally_reacting,
        )

    def _surface_impedance(self, rho_c: float) -> complex:
        """兩支非剛性模型的表面阻抗：絕對值直接用，正規化值乘上 ``rho_c``。"""
        if self.model == "constant_impedance":
            return self.impedance_pa_s_m.as_complex()  # type: ignore[union-attr]  # expires=2026-12-08 reason=驗證器保證這一格在這一支模型底下不是 None
        return self.normalized_impedance.as_complex() * rho_c  # type: ignore[union-attr]  # expires=2026-12-08 reason=同上


def load_materials(
    materials_dir: str | Path,
    freq_axis: FrequencyAxis,
    rho_c: float,
) -> MaterialRegistry:
    """掃 ``materials_dir`` 底下的 ``*.yaml``，建成一份填好的註冊表。

    只吃最上層的 ``*.yaml``（``.yml`` 與子目錄都不讀），順序照檔名排序——同一棵樹在誰的
    機器上跑出來的註冊順序都要一樣。目錄不存在就回一份空的註冊表：那是「還沒有材料」，
    不是錯誤。三個參數都必填，路徑不給預設（決策紙 ``config-loaders-keep-path-required``）。
    """
    materials_dir = Path(materials_dir)
    registry = MaterialRegistry()
    for path in sorted(materials_dir.glob("*.yaml")):
        with path.open() as f:
            data = yaml.safe_load(f)
        spec = MaterialSpec(**data)
        registry.register(spec.build(freq_axis, rho_c), source_kind="analytic")
    return registry
