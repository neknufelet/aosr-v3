"""票 #134 第二刀（繼承那三支）的 case 表：**「有哪些 case」的唯一住處**。

形狀沿第一刀那一張（:mod:`blueprint.materials_cut1_cases`），理由一樣：答案檔住
``blueprint/``，而 ``identity-strings-generated`` 與 ``refs-and-links-resolve`` 都扣掉那一層，
所以沒有任何機制保證它還有幾筆。把「有哪些 case」搬進版控、當成產生器與考卷共用的同一份
清單：產生器只跑這裡宣告的（在唯讀 donor 工作樹上），考卷拿同一份清單在新家跑一次、逐筆
比，並且斷言 id 集合**等於**這裡宣告的集合（少一筆紅、多一筆也紅），id 不准重複。

**這一刀交三支**：``freq_axis``／``experiment_schema``／``material_loader``——第一刀那張表
的檔頭把它們登記成「今天沒有宣告任何 case，也就沒有任何裁判」，這一張把那三支接起來。
第一刀那三支（``response``／``source``／``registry``）**不在這裡重判**：它們的答案已經在雲端
綠過，重判一次只會多一份會漂開的第二意見。

**``response`` 在這裡只是材料，不是被判的對象**（:data:`INGREDIENTS`）：載入器與軸規格都要
先有一條 ``FrequencyAxis`` 才跑得動，所以 case 會寫 ``response.FrequencyAxis.from_hz``。
那一支的公開行為由第一刀的考卷判，這裡只借它建東西——分成兩張表寫清楚，免得看起來像
「這一刀又判了它一次」。

**``coverage`` 那一格是這張票的「沒有裁判就不准關票」**：每一個公開符號都要指得出是哪幾筆
在判它，指不到就紅（考卷 ``tests/engine/test_materials_cut2_case_table.py``）。指得到的除了
case id、``const:<名字>``、``alias:<名字>``，這一刀多一種 ``probe:<名字>``——
:data:`PROBES` 那幾支是**另開一個直譯器**才問得出來的事（x64 開關），不是一筆 case。

**這一支不准 import donor 或新家的任何東西，也不准 import JAX。**
"""
from __future__ import annotations

from typing import Final

from blueprint.materials_cut1_steps import ModuleSpec, resolve
from blueprint.materials_cut2_freq_axis_cases import FREQ_AXIS_CASES
from blueprint.materials_cut2_loader_cases import LOADER_CASES
from blueprint.materials_cut2_schema_cases import SCHEMA_CASES
from blueprint.materials_cut2_steps import CaseEntry

__all__ = [
    "INGREDIENTS",
    "MODULES",
    "PROBES",
    "aliases_for",
    "case_id_list",
    "constant_id_list",
    "constants_for",
    "coverage_for",
    "declared_id_list",
    "declared_ids",
    "lookup_name",
    "resolve",
]

# 另開一個直譯器才問得出來的事。``x64_identity``：``FREQS_HZ_IDENTITY`` 被釘成 float32，
# 所以**不管載入的時候 jax_enable_x64 是開還是關**，那條 tuple 都要是同一組值。同一個行程
# 裡問不出這件事（常數在 import 的時候就算完了），所以產生器與考卷各自另起一個行程、把
# 開關打開再載入一次。
PROBES: Final[tuple[str, ...]] = ("x64_identity",)

FREQ_AXIS_COVERAGE: Final[dict[str, list[str]]] = {
    "FREQS_HZ": [
        "const:FREQS_HZ",
        "freq_axis.interp_to_bands.default_source_is_freqs_hz",
        "freq_axis.FREQ_AXIS.is_built_from_freqs_hz",
    ],
    "FREQS_HZ_IDENTITY": [
        "const:FREQS_HZ_IDENTITY",
        "freq_axis.FREQS_HZ_IDENTITY.differs_from_the_raw_array_values",
        "probe:x64_identity",
    ],
    "FREQ_AXIS": [
        "const:FREQ_AXIS",
        "freq_axis.FREQ_AXIS.fields_and_derived_values",
        "freq_axis.FREQ_AXIS.is_built_from_freqs_hz",
    ],
    "CHORAS_BANDS": [
        "const:CHORAS_BANDS",
        "freq_axis.interp_to_bands.default_target_is_choras_bands",
    ],
    "OCTAVE_BAND_CENTERS_HZ": ["const:OCTAVE_BAND_CENTERS_HZ"],
    "interp_to_bands": [
        "freq_axis.interp_to_bands.defaults",
        "freq_axis.interp_to_bands.default_source_is_freqs_hz",
        "freq_axis.interp_to_bands.default_target_is_choras_bands",
        "freq_axis.interp_to_bands.custom_source_and_target",
        "freq_axis.interp_to_bands.interpolates_in_log_space",
        "freq_axis.interp_to_bands.outside_the_source_range_is_flat",
        "freq_axis.interp_to_bands.integer_input_returns_floats",
        "freq_axis.interp_to_bands.empty_target_bands_give_an_empty_list",
        "freq_axis.interp_to_bands.length_mismatch_raises",
        "freq_axis.interp_to_bands.signature",
    ],
}

SCHEMA_COVERAGE: Final[dict[str, list[str]]] = {
    "Vec3": [
        "experiment_schema.Vec3.model_json_schema",
        "experiment_schema.Vec3.integer_input_is_coerced_to_float",
        "experiment_schema.Vec3.extra_field_is_rejected",
        "experiment_schema.Vec3.missing_field_is_rejected",
    ],
    "Vec3.as_tuple": [
        "experiment_schema.Vec3.as_tuple",
        "experiment_schema.Vec3.as_tuple.signature",
    ],
    "FrequencyAxisSpec": [
        "experiment_schema.FrequencyAxisSpec.model_json_schema",
        "experiment_schema.FrequencyAxisSpec.defaults_are_third_octave",
        "experiment_schema.FrequencyAxisSpec.custom_keeps_unsorted_input_on_the_spec",
        "experiment_schema.FrequencyAxisSpec.non_positive_f_min_is_rejected",
        "experiment_schema.FrequencyAxisSpec.n_points_must_exceed_one",
        "experiment_schema.FrequencyAxisSpec.unknown_mode_is_rejected",
        "experiment_schema.FrequencyAxisSpec.extra_field_is_rejected",
        "experiment_schema.FrequencyAxisSpec.third_octave_without_range_is_rejected",
        "experiment_schema.FrequencyAxisSpec.f_max_must_exceed_f_min",
        "experiment_schema.FrequencyAxisSpec.linear_hz_without_n_points_is_rejected",
        "experiment_schema.FrequencyAxisSpec.donor_test.empty_custom_list_is_rejected",
    ],
    "FrequencyAxisSpec.build": [
        "experiment_schema.FrequencyAxisSpec.build.signature",
        "experiment_schema.FrequencyAxisSpec.build.custom_single_point",
        "experiment_schema.FrequencyAxisSpec.build.custom_ignores_the_range_fields",
        "experiment_schema.FrequencyAxisSpec.build.linear_hz",
        "experiment_schema.FrequencyAxisSpec.build.third_octave",
        "experiment_schema.FrequencyAxisSpec.build.third_octave_without_any_centre_raises",
        "experiment_schema.FrequencyAxisSpec.donor_test.unsorted_custom_input_is_sorted",
        "experiment_schema.FrequencyAxisSpec.donor_test.build_rejects_nan",
        "experiment_schema.FrequencyAxisSpec.donor_test.build_rejects_infinity",
        "experiment_schema.FrequencyAxisSpec.donor_test.build_rejects_minus_infinity",
        "experiment_schema.FrequencyAxisSpec.donor_test.build_rejects_negative_frequency",
        "experiment_schema.FrequencyAxisSpec.donor_test.build_rejects_zero_frequency",
        "experiment_schema.FrequencyAxisSpec.donor_test.build_rejects_duplicate_frequency",
        "experiment_schema.FrequencyAxisSpec.build.linear_hz_with_n_points_cleared_raises",
        "experiment_schema.FrequencyAxisSpec.build.linear_hz_with_f_min_cleared_raises",
        "experiment_schema.FrequencyAxisSpec.build.third_octave_with_f_min_cleared_raises",
    ],
    "ReceiverGridSpec": [
        "experiment_schema.ReceiverGridSpec.model_json_schema",
        "experiment_schema.ReceiverGridSpec.defaults",
        "experiment_schema.ReceiverGridSpec.zero_points_is_rejected",
        "experiment_schema.ReceiverGridSpec.negative_wall_offset_is_rejected",
        "experiment_schema.ReceiverGridSpec.zero_wall_offset_is_allowed",
    ],
    "SurfaceMaterial": [
        "experiment_schema.SurfaceMaterial.model_json_schema",
        "experiment_schema.SurfaceMaterial.round_trip",
        "experiment_schema.SurfaceMaterial.missing_material_id_is_rejected",
    ],
    "RoomSpec": [
        "experiment_schema.RoomSpec.model_json_schema",
        "experiment_schema.RoomSpec.dims_are_optional",
        "experiment_schema.RoomSpec.nested_dims_are_a_vec3",
        "experiment_schema.RoomSpec.geometry_level_above_four_is_rejected",
        "experiment_schema.RoomSpec.negative_geometry_level_is_rejected",
    ],
    "ExperimentConfig": [
        "experiment_schema.ExperimentConfig.model_json_schema",
        "experiment_schema.ExperimentConfig.full_round_trip",
        "experiment_schema.ExperimentConfig.defaults_for_grid_and_materials",
        "experiment_schema.ExperimentConfig.extra_field_is_rejected",
        "experiment_schema.ExperimentConfig.missing_required_fields_are_rejected",
        "experiment_schema.ExperimentConfig.the_axis_spec_is_built_not_stored_as_a_dict",
    ],
    "load_experiment": [
        "experiment_schema.load_experiment.signature",
        "experiment_schema.load_experiment.reads_a_yaml_file",
        "experiment_schema.load_experiment.accepts_a_string_path",
        "experiment_schema.load_experiment.missing_file_raises",
        "experiment_schema.load_experiment.a_directory_raises",
        "experiment_schema.load_experiment.broken_yaml_raises",
        "experiment_schema.load_experiment.empty_file_raises",
        "experiment_schema.load_experiment.a_yaml_list_raises",
        "experiment_schema.load_experiment.an_invalid_experiment_is_rejected",
    ],
}

LOADER_COVERAGE: Final[dict[str, list[str]]] = {
    "MaterialSpec": [
        "material_loader.MaterialSpec.model_json_schema",
        "material_loader.MaterialSpec.defaults",
        "material_loader.MaterialSpec.missing_material_id_is_rejected",
        "material_loader.MaterialSpec.extra_field_is_rejected",
        "material_loader.MaterialSpec.unknown_model_is_rejected",
        "material_loader.MaterialSpec.normalized_model_needs_its_value",
        "material_loader.MaterialSpec.absolute_model_needs_its_value",
        "material_loader.MaterialSpec.complex_spec_rejects_extra_fields",
        "material_loader.MaterialSpec.complex_spec_needs_a_real_part",
    ],
    "MaterialSpec.build": [
        "material_loader.MaterialSpec.build.signature",
        "material_loader.MaterialSpec.build.normalized_multiplies_by_rho_c",
        "material_loader.MaterialSpec.build.normalized_imag_defaults_to_zero",
        "material_loader.MaterialSpec.build.absolute_ignores_rho_c",
        "material_loader.MaterialSpec.build.metadata_is_carried_through",
        "material_loader.MaterialSpec.build.rigid_without_magnitude_uses_the_default",
        "material_loader.MaterialSpec.build.rigid_zero_magnitude_becomes_the_default",
        "material_loader.MaterialSpec.build.rigid_negative_magnitude_is_kept",
        "material_loader.MaterialSpec.build.rigid_custom_magnitude_is_used",
        "material_loader.MaterialSpec.build.rigid_drops_the_boundary_overrides",
        "material_loader.MaterialSpec.build.rigid_ignores_the_impedance_fields",
    ],
    "load_materials": [
        "material_loader.load_materials.signature",
        "material_loader.load_materials.registers_one_material_per_yaml_file",
        "material_loader.load_materials.source_kind_is_analytic",
        "material_loader.load_materials.reads_files_in_sorted_order",
        "material_loader.load_materials.accepts_a_string_path",
        "material_loader.load_materials.an_empty_directory_gives_an_empty_registry",
        "material_loader.load_materials.a_missing_directory_gives_an_empty_registry",
        "material_loader.load_materials.only_reads_yaml_at_the_top_level",
        "material_loader.load_materials.duplicate_material_id_raises",
        "material_loader.load_materials.an_invalid_material_file_is_rejected",
        "material_loader.load_materials.broken_yaml_raises",
        "material_loader.load_materials.an_empty_yaml_file_raises",
        "material_loader.load_materials.a_rigid_file_builds_a_rigid_wall",
    ],
}

MODULES: Final[dict[str, ModuleSpec]] = {
    "freq_axis": {
        "donor": "lib.config.freq_axis",
        "engine": "aosr.materials.freq_axis",
        "file": "lib/config/freq_axis.py",
        "constants": [
            "FREQS_HZ",
            "FREQS_HZ_IDENTITY",
            "FREQ_AXIS",
            "CHORAS_BANDS",
            "OCTAVE_BAND_CENTERS_HZ",
        ],
        "aliases": [],
        "coverage": FREQ_AXIS_COVERAGE,
        "cases": FREQ_AXIS_CASES,
    },
    "experiment_schema": {
        "donor": "lib.config.experiment_schema",
        "engine": "aosr.materials.experiment_schema",
        "file": "lib/config/experiment_schema.py",
        "constants": [],
        "aliases": [],
        "coverage": SCHEMA_COVERAGE,
        "cases": SCHEMA_CASES,
    },
    "material_loader": {
        "donor": "lib.config.material_loader",
        "engine": "aosr.materials.material_loader",
        "file": "lib/config/material_loader.py",
        "constants": [],
        "aliases": [],
        "coverage": LOADER_COVERAGE,
        "cases": LOADER_CASES,
    },
}

# 只當材料用的模組（case 會叫它們建東西，但這一刀不判它們——第一刀已經判過）。
INGREDIENTS: Final[dict[str, dict[str, str]]] = {
    "response": {
        "donor": "lib.materials.response",
        "engine": "aosr.materials.response",
        "file": "lib/materials/response.py",
    },
}


def lookup_name(module: str, side: str) -> str:
    """表上的名字換成真的 import 路徑（``side`` 是 ``donor`` 或 ``engine``）。

    被判的那三支與只當材料的那幾支走同一支解析——case 表寫 ``response.FrequencyAxis`` 的
    時候，兩邊都要找得到同一個位置的東西。
    """
    if module in MODULES:
        return str(MODULES[module][side])  # type: ignore[literal-required]  # expires=2026-12-08 reason=side 是執行期才知道的鍵，TypedDict 的字面鍵型別在這裡幫不上忙
    if module in INGREDIENTS:
        return INGREDIENTS[module][side]
    raise KeyError(f"case 表上沒有 {module!r} 這一支（被判的與當材料的都沒有）")


def cases_for(module: str) -> list[CaseEntry]:
    """某一支模組宣告的每一筆 case。"""
    return MODULES[module]["cases"]


def constants_for(module: str) -> list[str]:
    """某一支模組宣告要凍結的公開常數。"""
    return MODULES[module]["constants"]


def aliases_for(module: str) -> list[str]:
    """某一支模組宣告要凍結的公開型別別名（這一刀三支都沒有）。"""
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
