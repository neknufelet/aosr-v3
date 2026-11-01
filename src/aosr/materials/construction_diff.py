"""可微構造入口（#561）：第三階段要梯度，另開一條路，既有三個入口一個位元都不變。

既有入口（construction.surface_impedance 等）擋自動微分：凍結格式用嚴格浮點欄位、
取樣軸拒收追蹤值、最後用 numpy（陣列函式庫）檢查並硬要雙精度。這裡分兩段：

1. 樣板（ConstructionTemplate）用具體值建：照原樣跑一次 surface_impedance，
   取得註記並讓既有的合法性檢查把關；同時列出這個構造有哪些連續欄位可當變數。
2. surface_impedance_from_parameters 拿一組參數（JAX 的 pytree，字典裝零維陣列）
   換掉樣板裡對應欄位，走同一支 construction._layer_matrix（層轉矩陣函式）與
   transfer_matrix.compute（傳遞矩陣連乘），可被 jax.jit、jax.grad、jax.vmap 包住。

求值不再做具體值檢查（範圍、非有限、負實部都不擋），合法範圍由呼叫端（第三階段的
搜尋）守；樣板上的註記只描述基準值。結果的精度跟著輸入走：雙精度輸入給 complex128，
單精度（float32）的參數、頻率與角度給 complex64；本模組不轉型、不讀寫 JAX 全域設定。
只收多層構造（Assembly）與貼在原牆上的構造（MountedOnWall，#990）：固定阻抗與實測阻抗表沒有可微的層參數，建樣板時拒收。
貼在原牆上的構造：原牆不是變數，它的阻抗是新增層的有限負載，盡量在求值當下用當下的空氣與頻率（與角度）現算，
垂直與各角度走同一條路：固定阻抗是常數；多層構造原牆走它自己沒有變數的樣板；實測表的表外頻點走底構造沒有變數的樣板。
只有實測表的表內值是建樣板時在樣板頻率軸上算好存起來的（表內對數內插不在可微路徑上，跟空氣無關）。
**已知限制**：原牆是實測表時，求值的頻率軸必須是建樣板那一條。具體值的軸逐點比、不同就拒收；
在 jax.jit 等追蹤之下軸的內容看不到，只比得了長度——同長的另一條軸會靜默用舊的表內值（考卷釘成看得見的事實）。
要換軸請重建樣板；呼叫端（第三階段搜尋）要保證建樣板與求值用同一條軸。固定阻抗與多層構造的原牆沒有這個限制。
樣板建立要走既有入口，所以要在雙精度已開的行程裡建；單精度是「同一個行程裡交單精度陣列」。
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.physics_constants import PhysicsConstants
from aosr.config.porous_model_ranges import PorousModelRanges
from aosr.materials import transfer_matrix as tmm
from aosr.materials.construction import _layer_matrix, _open_area, _outside_table, _table_impedance, surface_impedance
from aosr.materials.construction_models import AirLayer, Assembly, BoardLayer, ConstantImpedance, ConstructionNote, FabricLayer, Layer, MeasuredImpedance, MicroPerforatedLayer, MountedOnWall, PerforatedLayer, PorousLayer, SlottedLayer, SurfaceMaterial


# 每種層會被 _layer_matrix 讀進計算的連續欄位；欄位值是空的（沒填）就不當變數。
# 背面貼布的面密度只記錄、不進計算，所以不列；孔排法、多孔模型這類是離散選項，也不列。
_PANEL_FIELDS = ("thickness_m", "open_area", "surface_mass_kg_m2")
_CONTINUOUS_FIELDS: dict[type, tuple[str, ...]] = {
    PorousLayer: ("thickness_m", "flow_resistivity_pa_s_m2", "porosity", "tortuosity"),
    AirLayer: ("thickness_m",),
    BoardLayer: ("surface_mass_kg_m2", "resistance_pa_s_m"),
    FabricLayer: ("surface_mass_kg_m2", "flow_resistance_pa_s_m"),
    MicroPerforatedLayer: (*_PANEL_FIELDS, "hole_diameter_m", "hole_spacing_m", "hole_spacing_2_m"),
    PerforatedLayer: (*_PANEL_FIELDS, "hole_diameter_m", "hole_spacing_m", "hole_spacing_2_m",
                      "backing_fabric_flow_resistance_pa_s_m"),
    SlottedLayer: (*_PANEL_FIELDS, "slot_width_m", "slot_spacing_m", "backing_fabric_flow_resistance_pa_s_m"),
}
_BACKING_FIELD = "backing_fabric_flow_resistance_pa_s_m"


@dataclass(frozen=True)
class TemplateVariable:
    """一個變數：參數字典的鍵、它換掉第幾層（從房間側的一起算）的哪一格、基準值。"""

    name: str
    layer_index: int
    field: str
    baseline: float


@dataclass(frozen=True)
class ConstructionTemplate:
    """凍結構造加上「哪些連續欄位是變數」；基準值已用既有入口驗過、註記已取得。

    貼在原牆上的構造：原牆是多層構造時帶它沒有變數的樣板（``wall_template``）；是實測表時帶表內值
    （``wall_table``，樣板頻率軸 ``wall_frequencies_hz`` 上的對數內插）、表外頻點的索引（``wall_outside``）
    與底構造沒有變數的樣板（``wall_template``，表外有頻點才有）。固定阻抗原牆與多層構造這幾格都是空的。
    """

    material: Assembly | MountedOnWall
    variables: tuple[TemplateVariable, ...]
    notes: tuple[ConstructionNote, ...]
    wall_table: tuple[complex, ...] = ()
    wall_frequencies_hz: tuple[float, ...] = ()
    wall_template: ConstructionTemplate | None = None
    wall_outside: tuple[int, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(variable.name for variable in self.variables)

    def baseline_parameters(self) -> dict[str, jax.Array]:
        """基準值參數字典；陣列精度照 JAX 當下預設（雙精度行程給 float64）。"""
        return {variable.name: jnp.asarray(variable.baseline) for variable in self.variables}


def variable_name(layer_index: int, field: str) -> str:
    """參數字典的鍵：layer<層號>.<欄位名>，層號從房間側的一起算，跟註記的層編號同一套。"""
    return f"layer{layer_index}.{field}"


def available_variables(material: Assembly | MountedOnWall) -> tuple[TemplateVariable, ...]:
    """列出構造裡每一個有值、會進計算的連續欄位。"""
    variables = []
    for index, layer in enumerate(material.layers, start=1):
        for field in _CONTINUOUS_FIELDS[type(layer)]:
            value = getattr(layer, field)
            if value is not None:
                variables.append(TemplateVariable(variable_name(index, field), index, field, float(value)))
    return tuple(variables)


def build_template(material: SurfaceMaterial, frequencies_hz: jax.Array, *, air: PhysicsConstants,
                   ranges: PorousModelRanges, variables: tuple[str, ...] | None = None) -> ConstructionTemplate:
    """用具體值跑一次 surface_impedance 把關並取註記；variables 不給就是全部可用變數。

    variables 給了就只放那幾個，其他欄位固定在基準值；名字不在可用清單裡就拒收。
    貼在原牆上的構造：原牆不是變數；多層構造原牆與實測表的底構造另建沒有變數的樣板，實測表另存表內值。
    """
    if not isinstance(material, Assembly | MountedOnWall):
        raise ValueError(f"可微入口只收多層構造（assembly）或貼在原牆上的構造（mounted_on_wall），收到 {material.kind}；"
                         "固定阻抗與實測阻抗表沒有可微的層參數")
    _, notes = surface_impedance(material, frequencies_hz, air=air, ranges=ranges)
    available = available_variables(material)
    if variables is None:
        chosen = available
    else:
        known = {variable.name: variable for variable in available}
        unknown = [name for name in variables if name not in known]
        if unknown or len(set(variables)) != len(variables):
            raise ValueError(f"變數名不認識或重複：{unknown or list(variables)}；可用的是 {list(known)}")
        chosen = tuple(known[name] for name in variables)
    if isinstance(material, Assembly):
        return ConstructionTemplate(material, chosen, notes)
    wall = material.wall
    if isinstance(wall, ConstantImpedance):
        return ConstructionTemplate(material, chosen, notes)
    if isinstance(wall, Assembly):
        return ConstructionTemplate(material, chosen, notes,
                                    wall_template=build_template(wall, frequencies_hz, air=air, ranges=ranges, variables=()))
    axis = jnp.asarray(frequencies_hz)
    outside = tuple(int(index) for index in np.flatnonzero(_outside_table(wall, axis)))
    base = (build_template(wall.base_assembly, axis[np.asarray(outside)], air=air, ranges=ranges, variables=())
            if outside else None)
    return ConstructionTemplate(material, chosen, notes, tuple(complex(value) for value in np.asarray(_table_impedance(wall, axis))),
                                tuple(float(value) for value in np.asarray(axis, dtype=np.float64)), base, outside)


def _layer_with(layer: Layer, updates: dict[str, jax.Array]) -> Layer:
    """複製一層並換掉欄位，不重新驗證：追蹤中的陣列過不了嚴格浮點欄位，檢查已在樣板做過。"""
    return layer.model_copy(update=updates) if updates else layer


def _matrix(layer: Layer, frequencies: jax.Array, air: PhysicsConstants, angles: jax.Array | None) -> jax.Array:
    """走 _layer_matrix；只有背面貼布那一片自己乘，因為原式把它的精度寫死成雙精度複數。"""
    if not isinstance(layer, PerforatedLayer | SlottedLayer) or layer.backing_fabric_flow_resistance_pa_s_m is None:
        return _layer_matrix(layer, frequencies, air, angles)
    panel = _layer_matrix(layer.model_copy(update={_BACKING_FIELD: None}), frequencies, air, angles)
    # 同 construction._backed_sheet：板後串聯布面流阻除板開孔率；零加上去不改任何位元。
    impedance = jnp.zeros_like(frequencies) + 0j + jnp.asarray(layer.backing_fabric_flow_resistance_pa_s_m) / _open_area(layer)
    return panel @ tmm._sheet_layer(tmm._broadcast_angles(impedance, angles))


def surface_impedance_from_parameters(template: ConstructionTemplate, parameters: Mapping[str, jax.Array],
                                      frequencies_hz: jax.Array, *, air: PhysicsConstants,
                                      incidence_angles_deg: jax.Array | None = None) -> jax.Array:
    """參數換進樣板後的表面阻抗 [Pa·s/m]：垂直入射 (F,)，給角度 [度] 是 (F,Nθ)。

    parameters 的鍵要剛好等於 template.names，每個值是零維陣列（批次請用 jax.vmap）。
    """
    if set(parameters) != set(template.names):
        raise ValueError(f"參數鍵要剛好是 {list(template.names)}，收到 {sorted(parameters)}")
    for name, value in parameters.items():
        if jnp.ndim(value) != 0:
            raise ValueError(f"參數 {name} 要是零維陣列，收到形狀 {jnp.shape(value)}；批次請用 jax.vmap 包")
    frequencies = jnp.asarray(frequencies_hz)
    angles = None if incidence_angles_deg is None else jnp.asarray(incidence_angles_deg)
    updates: dict[int, dict[str, jax.Array]] = {}
    for variable in template.variables:
        updates.setdefault(variable.layer_index, {})[variable.field] = parameters[variable.name]
    matrices = [_matrix(_layer_with(layer, updates.get(index, {})), frequencies, air, angles)
                for index, layer in enumerate(template.material.layers, start=1)]
    if isinstance(template.material, MountedOnWall):
        load = _wall_load(template, frequencies, angles, air)
        return tmm.compute(matrices, air=air, load_impedance_pa_s_m=load, incidence_angles_deg=angles)[0]
    return tmm.compute(matrices, air=air, backing=template.material.backing, incidence_angles_deg=angles)[0]


def _wall_load(template: ConstructionTemplate, frequencies: jax.Array, angles: jax.Array | None,
               air: PhysicsConstants) -> jax.Array:
    """原牆負載（求值的複數精度）：固定阻抗與多層構造用當下的空氣、頻率與角度現算；實測表見模組說明的已知限制。"""
    assert isinstance(template.material, MountedOnWall)
    wall = template.material.wall
    dtype = jnp.result_type(frequencies.dtype, jnp.complex64)
    if isinstance(wall, ConstantImpedance):
        return tmm.constant_z(frequencies, jnp.asarray(wall.impedance_pa_s_m, dtype=dtype), incidence_angles_deg=angles)
    if isinstance(wall, Assembly):
        assert template.wall_template is not None
        return surface_impedance_from_parameters(template.wall_template, {}, frequencies, air=air, incidence_angles_deg=angles)
    expected = np.asarray(template.wall_frequencies_hz)
    if frequencies.ndim != 1 or frequencies.shape[0] != expected.shape[0] or (
            not isinstance(frequencies, jax.core.Tracer)
            and not np.array_equal(np.asarray(frequencies), expected.astype(np.asarray(frequencies).dtype))):
        raise ValueError("原牆是實測表時只能在建樣板的那一條頻率軸上求值（表內值只在那裡算過）；換軸請重建樣板")
    # 表內照局部反應攤開，表外同既有入口走底構造（用當下的空氣）。
    load = tmm._broadcast_angles(jnp.asarray(np.asarray(template.wall_table, dtype=np.complex128), dtype=dtype), angles)
    if template.wall_template is None:
        return load
    outside = np.asarray(template.wall_outside)
    base = surface_impedance_from_parameters(template.wall_template, {}, frequencies[outside], air=air, incidence_angles_deg=angles)
    return load.at[outside].set(base.astype(load.dtype))
