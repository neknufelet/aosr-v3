"""票 #134 第一候選（材料那一塊）的 case 表：**「有哪些 case」的唯一住處**。

形狀沿票 #127 那一張（``blueprint/config_cut1_cases.py``）的先例，理由一樣：答案檔住
``blueprint/``，而 ``identity-strings-generated`` 與 ``refs-and-links-resolve`` 都扣掉那
一層，所以**沒有任何機制保證它還有幾筆**。把「有哪些 case」搬進版控、當成產生器與考卷
共用的同一份清單：

* 產生器（``blueprint/generate_materials_cut1_answers.py``）只跑**這裡宣告的**，在唯讀的
  donor 工作樹上跑，把每一筆的實際結果寫進答案檔；
* 考卷（``tests/engine/``）拿同一份清單在新家跑一次，逐筆跟答案檔比，並且斷言 id 集合
  **等於**這裡宣告的集合（少一筆紅、多一筆也紅），id 還不准重複（票 #163 的洞）。

零件（型別、值的記號、步驟的小工廠、``resolve``）住 :mod:`blueprint.materials_cut1_steps`；
case 清單住 :mod:`blueprint.materials_cut1_response_cases` 與
:mod:`blueprint.materials_cut1_registry_cases`。拆成幾支是 ``style-guard`` 第⑤條的單檔
行數上限，拆法照模組走。

**這一支不准 import donor 或新家的任何東西，也不准 import JAX。** 真正碰模組的是
:mod:`blueprint.materials_cut1_probe`。

**``coverage`` 那一格是這張票的「沒有裁判就不准關票」。** 決策紙
``engine-not-imported-new-home-grows-block-by-block`` 寫著：一塊裡哪些公開函式沒有寫死
期望值的測試，要列出來、列著沒補完不准關票。這裡把它變成機器看得見的東西——**每一個公開
符號都要指得出是哪幾筆 case 在judging 它**，指不出來就紅（考卷
``tests/engine/test_materials_cut1_case_table.py``）。

**目前交四支**（freq_axis／response／source／registry）。另外兩支（``experiment_schema``／
``material_loader``）是同一張票的後段：位置留在 :data:`MODULES`
的註解裡，**今天沒有宣告任何 case，也就沒有任何裁判**——不是「已經驗過」。
"""
from __future__ import annotations

from typing import Final

from blueprint.materials_cut1_registry_cases import REGISTRY_CASES, SOURCE_CASES
from blueprint.materials_cut1_response_cases import RESPONSE_CASES
from blueprint.materials_cut1_steps import CaseEntry, ModuleSpec, float_sequence, resolve, size_of

__all__ = [
    "MODULES",
    "aliases_for",
    "case_id_list",
    "cases_for",
    "constant_id_list",
    "constants_for",
    "coverage_for",
    "declared_id_list",
    "declared_ids",
    "resolve",
]

# 每一支模組的公開符號 → 判它的那幾筆（case id、``const:<名字>``、``alias:<名字>``）。
# 符號清單逐筆對回 donor 各支的公開定義（``/tmp`` 那份 core-public-inventory 是同一份
# 名單的另一種寫法，但那份不在版控裡，所以這裡自己寫一份、由考卷對回新家真的有哪些公開
# 名字）。**一個符號指不出任何一筆，就是沒有裁判。**
FREQ_AXIS_CASES: Final[list[CaseEntry]] = [
    {
        "id": "freq_axis.FREQS_HZ.values",
        "steps": [float_sequence("freq_axis.FREQS_HZ", "FREQS_HZ", digits=2)],
        "report": ["FREQS_HZ"],
    },
    {
        "id": "freq_axis.FREQS_HZ_IDENTITY.values",
        "steps": [float_sequence("freq_axis.FREQS_HZ_IDENTITY", "FREQS_HZ_IDENTITY")],
        "report": ["FREQS_HZ_IDENTITY"],
    },
    {
        "id": "freq_axis.CHORAS_BANDS.values",
        "steps": [float_sequence("freq_axis.CHORAS_BANDS", "CHORAS_BANDS")],
        "report": ["CHORAS_BANDS"],
    },
    {
        "id": "freq_axis.axis.length",
        "steps": [
            float_sequence("freq_axis.FREQS_HZ", "axis", digits=2),
            size_of("axis", "length"),
        ],
        "report": ["length"],
    },
]

FREQ_AXIS_COVERAGE: Final[dict[str, list[str]]] = {
    "FREQS_HZ": ["freq_axis.FREQS_HZ.values", "freq_axis.axis.length"],
    "FREQS_HZ_IDENTITY": ["freq_axis.FREQS_HZ_IDENTITY.values"],
    "CHORAS_BANDS": ["freq_axis.CHORAS_BANDS.values"],
    "OCTAVE_BAND_CENTERS_HZ": ["const:OCTAVE_BAND_CENTERS_HZ"],
}

RESPONSE_COVERAGE: Final[dict[str, list[str]]] = {
    "BoundaryModel": ["alias:BoundaryModel"],
    "BOUNDARY_MODELS": ["const:BOUNDARY_MODELS"],
    "ResolutionMode": ["alias:ResolutionMode"],
    "RESOLUTION_MODES": ["const:RESOLUTION_MODES"],
    "MATERIAL_SCATTERING_DEFAULT_S": [
        "const:MATERIAL_SCATTERING_DEFAULT_S",
        "response.MaterialResponse.from_impedance.complex_with_default_scattering",
    ],
    "FrequencyAxis": [
        "response.FrequencyAxis.direct_construction_defaults",
        "response.FrequencyAxis.pytree_flatten_roundtrip",
    ],
    "FrequencyAxis.n_freq": [
        "response.FrequencyAxis.from_hz.three_floats",
        "response.FrequencyAxis.from_hz.single_point",
        "response.FrequencyAxis.from_hz.empty_axis_has_zero_n_freq",
    ],
    "FrequencyAxis.f_min": [
        "response.FrequencyAxis.from_hz.three_floats",
        "response.FrequencyAxis.from_hz.descending_input_is_not_sorted",
        "response.FrequencyAxis.from_hz.empty_axis_f_min_raises",
    ],
    "FrequencyAxis.f_max": [
        "response.FrequencyAxis.from_hz.three_floats",
        "response.FrequencyAxis.from_hz.descending_input_is_not_sorted",
    ],
    "FrequencyAxis.from_hz": [
        "response.FrequencyAxis.from_hz.three_floats",
        "response.FrequencyAxis.from_hz.integer_input_keeps_integer_dtype",
        "response.FrequencyAxis.from_hz.default_resolution_and_convention",
        "response.FrequencyAxis.from_hz.third_octave_resolution",
        "response.FrequencyAxis.from_hz.signature",
    ],
    "MaterialResponse": [
        "response.MaterialResponse.pytree_flatten_roundtrip",
        "response.MaterialResponse.pytree_flatten_without_scattering",
        "response.MaterialResponse.tree_map_doubles_leaves_only",
        "response.MaterialResponse.survives_a_jit_boundary",
    ],
    "MaterialResponse.n_freq": ["response.MaterialResponse.n_freq_matches_axis"],
    "MaterialResponse.from_impedance": [
        "response.MaterialResponse.from_impedance.complex_with_default_scattering",
        "response.MaterialResponse.from_impedance.scattering_none",
        "response.MaterialResponse.from_impedance.scattering_spectrum",
        "response.MaterialResponse.from_impedance.real_impedance_keeps_real_dtype",
        "response.MaterialResponse.from_impedance.boundary_model_is_not_validated",
        "response.MaterialResponse.from_impedance.scattering_wrong_shape_raises",
        "response.MaterialResponse.from_impedance.scattering_above_one_raises",
        "response.MaterialResponse.from_impedance.scattering_negative_raises",
        "response.MaterialResponse.from_impedance.scattering_nan_raises",
        "response.MaterialResponse.from_impedance.signature",
    ],
    "MaterialResponse.from_alpha": [
        "response.MaterialResponse.from_alpha.eight_octave_bands",
        "response.MaterialResponse.from_alpha.clamps_alpha_above_one",
        "response.MaterialResponse.from_alpha.mixed_out_of_range_alpha_inside_band_range",
        "response.MaterialResponse.from_alpha.negative_alpha_is_clamped_not_raised",
        "response.MaterialResponse.from_alpha.flat_extrapolation_outside_bands",
        "response.MaterialResponse.from_alpha.single_band_is_flat",
        "response.MaterialResponse.from_alpha.scattering_none",
        "response.MaterialResponse.from_alpha.length_mismatch_raises",
        "response.MaterialResponse.from_alpha.two_dimensional_input_raises",
        "response.MaterialResponse.from_alpha.empty_bands_raise",
        "response.MaterialResponse.from_alpha.non_finite_alpha_raises",
        "response.MaterialResponse.from_alpha.non_finite_band_raises",
        "response.MaterialResponse.from_alpha.non_ascending_bands_raise",
        "response.MaterialResponse.from_alpha.signature",
    ],
}

SOURCE_COVERAGE: Final[dict[str, list[str]]] = {
    "SourceKind": ["alias:SourceKind"],
    "SOURCE_KINDS": ["const:SOURCE_KINDS"],
    "MaterialEntry": [
        "source.MaterialEntry.defaults",
        "source.MaterialEntry.explicit_source_ref",
        "source.MaterialEntry.is_frozen",
        "source.MaterialEntry.signature",
    ],
    "is_aosr_material_id": [
        "source.is_aosr_material_id.with_prefix",
        "source.is_aosr_material_id.without_prefix",
        "source.is_aosr_material_id.empty_string",
        "source.is_aosr_material_id.prefix_only",
        "source.is_aosr_material_id.is_case_sensitive",
        "source.is_aosr_material_id.leading_space_is_not_a_prefix",
        "source.is_aosr_material_id.signature",
    ],
}

# 註冊表的 ``__init__``／``__contains__``／``__len__`` 也列在這裡：它們是呼叫端真的會用的
# 介面（``MaterialRegistry()``、``"m1" in reg``、``len(reg)``），只是「名字不以底線開頭」
# 那一種盤點法看不到它們（根對話的 independent-focus.md 指出來的）。
REGISTRY_COVERAGE: Final[dict[str, list[str]]] = {
    "MaterialRegistry": [
        "registry.MaterialRegistry.starts_empty",
        "registry.MaterialRegistry.contains_and_len",
    ],
    "MaterialRegistry.__init__": ["registry.MaterialRegistry.starts_empty"],
    "MaterialRegistry.__contains__": ["registry.MaterialRegistry.contains_and_len"],
    "MaterialRegistry.__len__": [
        "registry.MaterialRegistry.starts_empty",
        "registry.MaterialRegistry.contains_and_len",
    ],
    "MaterialRegistry.register": [
        "registry.MaterialRegistry.register.defaults_to_analytic",
        "registry.MaterialRegistry.register.explicit_kind_and_ref",
        "registry.MaterialRegistry.register.empty_material_id_raises",
        "registry.MaterialRegistry.register.unknown_source_kind_raises",
        "registry.MaterialRegistry.register.duplicate_without_overwrite_raises",
        "registry.MaterialRegistry.register.implicit_overwrite_inherits_kind_and_ref",
        "registry.MaterialRegistry.register.explicit_kind_drops_the_stale_ref",
        "registry.MaterialRegistry.register.signature",
    ],
    "MaterialRegistry.register_entry": [
        "registry.MaterialRegistry.register_entry.unknown_source_kind_raises",
        "registry.MaterialRegistry.register_entry.empty_material_id_raises",
        "registry.MaterialRegistry.register_entry.duplicate_without_overwrite_raises",
        "registry.MaterialRegistry.register_entry.overwrite_replaces",
    ],
    "MaterialRegistry.get": [
        "registry.MaterialRegistry.get.returns_the_response",
        "registry.MaterialRegistry.get.unknown_id_raises",
    ],
    "MaterialRegistry.entry": [
        "registry.MaterialRegistry.register.defaults_to_analytic",
        "registry.MaterialRegistry.entry.unknown_id_raises",
    ],
    "MaterialRegistry.source_kind": [
        "registry.MaterialRegistry.register.defaults_to_analytic",
        "registry.MaterialRegistry.source_kind.unknown_id_raises",
    ],
    "MaterialRegistry.ids": [
        "registry.MaterialRegistry.starts_empty",
        "registry.MaterialRegistry.ids.are_sorted",
    ],
    "rigid_wall": [
        "registry.rigid_wall.defaults",
        "registry.rigid_wall.custom_magnitude_and_id",
        "registry.rigid_wall.signature",
    ],
    "constant_impedance": [
        "registry.constant_impedance.complex_value",
        "registry.constant_impedance.metadata_overrides",
        "registry.constant_impedance.signature",
    ],
}

# 另外兩支（``experiment_schema``／``material_loader``）是同一張票的後段，今天不在表裡。
MODULES: Final[dict[str, ModuleSpec]] = {
    "freq_axis": {
        "donor": "lib.config.freq_axis",
        "engine": "aosr.materials.freq_axis",
        "file": "lib/config/freq_axis.py",
        "constants": ["OCTAVE_BAND_CENTERS_HZ"],
        "aliases": [],
        "coverage": FREQ_AXIS_COVERAGE,
        "cases": FREQ_AXIS_CASES,
    },
    "response": {
        "donor": "lib.materials.response",
        "engine": "aosr.materials.response",
        "file": "lib/materials/response.py",
        "constants": ["BOUNDARY_MODELS", "RESOLUTION_MODES", "MATERIAL_SCATTERING_DEFAULT_S"],
        "aliases": ["BoundaryModel", "ResolutionMode"],
        "coverage": RESPONSE_COVERAGE,
        "cases": RESPONSE_CASES,
    },
    "source": {
        "donor": "lib.materials.source",
        "engine": "aosr.materials.source",
        "file": "lib/materials/source.py",
        "constants": ["SOURCE_KINDS"],
        "aliases": ["SourceKind"],
        "coverage": SOURCE_COVERAGE,
        "cases": SOURCE_CASES,
    },
    "registry": {
        "donor": "lib.materials.registry",
        "engine": "aosr.materials.registry",
        "file": "lib/materials/registry.py",
        "constants": [],
        "aliases": [],
        "coverage": REGISTRY_COVERAGE,
        "cases": REGISTRY_CASES,
    },
}


def cases_for(module: str) -> list[CaseEntry]:
    """某一支模組宣告的每一筆 case。"""
    return MODULES[module]["cases"]


def constants_for(module: str) -> list[str]:
    """某一支模組宣告要凍結的公開常數。"""
    return MODULES[module]["constants"]


def aliases_for(module: str) -> list[str]:
    """某一支模組宣告要凍結的公開型別別名。"""
    return MODULES[module]["aliases"]


def coverage_for(module: str) -> dict[str, list[str]]:
    """某一支模組的「公開符號 → 判它的那幾筆」。"""
    return MODULES[module]["coverage"]


def case_id_list() -> list[str]:
    """每一筆 case 的 id，**一筆一格**（同一個 id 宣告兩次就出現兩次）。"""
    return [case["id"] for name in sorted(MODULES) for case in cases_for(name)]


def constant_id_list() -> list[str]:
    """每一個常數與型別別名的 id，一筆一格。"""
    out: list[str] = []
    for name in sorted(MODULES):
        out.extend(f"{name}.const.{item}" for item in constants_for(name))
        out.extend(f"{name}.alias.{item}" for item in aliases_for(name))
    return out


def declared_id_list() -> list[str]:
    """這張表宣告的每一個 id，一筆一格（case ＋ 常數 ＋ 別名）。"""
    return [*case_id_list(), *constant_id_list()]


def declared_ids() -> set[str]:
    """這張表宣告的 id 集合。"""
    return set(declared_id_list())
