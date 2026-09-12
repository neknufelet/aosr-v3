"""票 #134 第二刀：``material_loader`` 那一支的 case 清單。

這一支是**上一代 ``lib/config/material_loader.py`` 的公開行為**寫成的題目：``MaterialSpec``
（三支解析模型、複數阻抗、必填與守門）與 ``load_materials``（掃一個目錄、一個檔一個材料、
註冊進 ``MaterialRegistry``）。

**兩件上一代的怪脾氣照抄，不順手修**（這張票的合約是「跟上一代一致」，要改行為得先有一張
決策紙）：

* ``rigid`` 那一支走 ``self.z_magnitude or 1e8``——``None`` 用 1e8 是本意，**``0.0`` 也用
  1e8**（``or`` 把零當成假），負數則照收。三種各一筆。
* ``rigid`` 那一支**不把 ``boundary_model``／``is_locally_reacting`` 傳下去**：spec 上寫了
  也沒用，建出來的是 ``rigid_wall()`` 自己的預設。一筆 case 明著餵不一樣的值，把「沒傳」
  釘死。

檔案那幾筆都在探針現開的暫存目錄底下（不是真的 repo）；錯誤訊息裡的那一段絕對路徑由探針
換成 ``<TMP>``，其他一個字都不放寬。
"""
from __future__ import annotations

from typing import Final

from blueprint.materials_cut2_steps import (
    CaseEntry,
    Step,
    call,
    contains,
    false,
    flt,
    make_dir,
    make_file,
    method,
    obj,
    read,
    ref,
    seq,
    signature,
    size_of,
    strs,
    under,
    under_text,
)

# 三點頻率軸：跟第一刀那幾筆用的是同一組（100／200／400 Hz）。
AXIS3: Final[list[Step]] = [
    call(
        "response.FrequencyAxis.from_hz",
        "axis",
        freqs_hz=seq(flt(100.0), flt(200.0), flt(400.0)),
        resolution=strs("custom"),
    )
]

# 空氣的特性阻抗（上一代注入 ``rho_c`` 的那個數；正規化阻抗乘上它才是絕對阻抗）。
RHO_C: Final[float] = 415.0

NORMALIZED_YAML: Final[str] = """
material_id: m_norm
normalized_impedance: {real: 2.0, imag: 0.5}
"""
ABSOLUTE_YAML: Final[str] = """
material_id: m_abs
model: constant_impedance
impedance_pa_s_m: {real: 830.0, imag: -415.0}
"""
RIGID_YAML: Final[str] = """
material_id: m_rigid
model: rigid
"""


def _spec_case(case_id: str, steps: list[Step], report: list[str]) -> CaseEntry:
    """一筆 ``MaterialSpec`` 的 case（前面先建那條三點軸）。"""
    return {"id": case_id, "steps": [*AXIS3, *steps], "report": report}


LOADER_CASES: Final[list[CaseEntry]] = [
    {
        "id": "material_loader.MaterialSpec.model_json_schema",
        "steps": [call("material_loader.MaterialSpec.model_json_schema", "schema")],
        "report": ["schema"],
    },
    {
        "id": "material_loader.MaterialSpec.defaults",
        "steps": [
            call(
                "material_loader.MaterialSpec",
                "spec",
                material_id=strs("m1"),
                normalized_impedance=obj(real=flt(2.0)),
            ),
            method("spec", "model_dump", "dumped"),
        ],
        "report": ["dumped"],
    },
    {
        "id": "material_loader.MaterialSpec.missing_material_id_is_rejected",
        "steps": [call("material_loader.MaterialSpec", "spec", normalized_impedance=obj(real=flt(2.0)))],
        "report": ["spec"],
    },
    {
        "id": "material_loader.MaterialSpec.extra_field_is_rejected",
        "steps": [
            call(
                "material_loader.MaterialSpec",
                "spec",
                material_id=strs("m1"),
                normalized_impedance=obj(real=flt(2.0)),
                thickness_m=flt(0.05),
            )
        ],
        "report": ["spec"],
    },
    {
        "id": "material_loader.MaterialSpec.unknown_model_is_rejected",
        "steps": [call("material_loader.MaterialSpec", "spec", material_id=strs("m1"), model=strs("tmm"))],
        "report": ["spec"],
    },
    {
        "id": "material_loader.MaterialSpec.normalized_model_needs_its_value",
        "steps": [call("material_loader.MaterialSpec", "spec", material_id=strs("m1"))],
        "report": ["spec"],
    },
    {
        "id": "material_loader.MaterialSpec.absolute_model_needs_its_value",
        "steps": [
            call(
                "material_loader.MaterialSpec",
                "spec",
                material_id=strs("m1"),
                model=strs("constant_impedance"),
            )
        ],
        "report": ["spec"],
    },
    {
        "id": "material_loader.MaterialSpec.complex_spec_rejects_extra_fields",
        "steps": [
            call(
                "material_loader.MaterialSpec",
                "spec",
                material_id=strs("m1"),
                normalized_impedance=obj(real=flt(2.0), imag=flt(0.5), phase=flt(0.0)),
            )
        ],
        "report": ["spec"],
    },
    {
        "id": "material_loader.MaterialSpec.complex_spec_needs_a_real_part",
        "steps": [
            call(
                "material_loader.MaterialSpec",
                "spec",
                material_id=strs("m1"),
                normalized_impedance=obj(imag=flt(0.5)),
            )
        ],
        "report": ["spec"],
    },
    {
        "id": "material_loader.MaterialSpec.build.signature",
        "steps": [signature("material_loader.MaterialSpec.build", "sig")],
        "report": ["sig"],
    },
    _spec_case(
        "material_loader.MaterialSpec.build.normalized_multiplies_by_rho_c",
        [
            call(
                "material_loader.MaterialSpec",
                "spec",
                material_id=strs("m_norm"),
                normalized_impedance=obj(real=flt(2.0), imag=flt(0.5)),
            ),
            method("spec", "build", "mat", freq_axis=ref("axis"), rho_c=flt(RHO_C)),
        ],
        ["mat"],
    ),
    _spec_case(
        "material_loader.MaterialSpec.build.normalized_imag_defaults_to_zero",
        [
            call(
                "material_loader.MaterialSpec",
                "spec",
                material_id=strs("m_norm"),
                normalized_impedance=obj(real=flt(2.0)),
            ),
            method("spec", "build", "mat", freq_axis=ref("axis"), rho_c=flt(RHO_C)),
        ],
        ["mat"],
    ),
    _spec_case(
        "material_loader.MaterialSpec.build.absolute_ignores_rho_c",
        [
            call(
                "material_loader.MaterialSpec",
                "spec",
                material_id=strs("m_abs"),
                model=strs("constant_impedance"),
                impedance_pa_s_m=obj(real=flt(830.0), imag=flt(-415.0)),
            ),
            method("spec", "build", "mat", freq_axis=ref("axis"), rho_c=flt(RHO_C)),
        ],
        ["mat"],
    ),
    _spec_case(
        "material_loader.MaterialSpec.build.metadata_is_carried_through",
        [
            call(
                "material_loader.MaterialSpec",
                "spec",
                material_id=strs("m_meta"),
                boundary_model=strs("angle_dependent"),
                is_locally_reacting=false(),
                normalized_impedance=obj(real=flt(2.0)),
            ),
            method("spec", "build", "mat", freq_axis=ref("axis"), rho_c=flt(RHO_C)),
            read("mat", "boundary_model", "boundary_model"),
            read("mat", "is_locally_reacting", "is_locally_reacting"),
        ],
        ["boundary_model", "is_locally_reacting"],
    ),
    _spec_case(
        "material_loader.MaterialSpec.build.rigid_without_magnitude_uses_the_default",
        [
            call("material_loader.MaterialSpec", "spec", material_id=strs("m_rigid"), model=strs("rigid")),
            method("spec", "build", "mat", freq_axis=ref("axis"), rho_c=flt(RHO_C)),
        ],
        ["mat"],
    ),
    _spec_case(
        # 上一代的怪脾氣①：``z_magnitude=0.0`` 走 ``or`` 那一支，變成 1e8（**不是** 0）。
        "material_loader.MaterialSpec.build.rigid_zero_magnitude_becomes_the_default",
        [
            call(
                "material_loader.MaterialSpec",
                "spec",
                material_id=strs("m_rigid"),
                model=strs("rigid"),
                z_magnitude=flt(0.0),
            ),
            method("spec", "build", "mat", freq_axis=ref("axis"), rho_c=flt(RHO_C)),
        ],
        ["mat"],
    ),
    _spec_case(
        "material_loader.MaterialSpec.build.rigid_negative_magnitude_is_kept",
        [
            call(
                "material_loader.MaterialSpec",
                "spec",
                material_id=strs("m_rigid"),
                model=strs("rigid"),
                z_magnitude=flt(-2.5),
            ),
            method("spec", "build", "mat", freq_axis=ref("axis"), rho_c=flt(RHO_C)),
        ],
        ["mat"],
    ),
    _spec_case(
        "material_loader.MaterialSpec.build.rigid_custom_magnitude_is_used",
        [
            call(
                "material_loader.MaterialSpec",
                "spec",
                material_id=strs("m_rigid"),
                model=strs("rigid"),
                z_magnitude=flt(5000.0),
            ),
            method("spec", "build", "mat", freq_axis=ref("axis"), rho_c=flt(RHO_C)),
        ],
        ["mat"],
    ),
    _spec_case(
        # 上一代的怪脾氣②：``rigid`` 那一支不把 spec 上的 boundary_model／
        # is_locally_reacting 傳下去，建出來的是 rigid_wall() 自己的預設。
        "material_loader.MaterialSpec.build.rigid_drops_the_boundary_overrides",
        [
            call(
                "material_loader.MaterialSpec",
                "spec",
                material_id=strs("m_rigid"),
                model=strs("rigid"),
                boundary_model=strs("angle_dependent"),
                is_locally_reacting=false(),
            ),
            method("spec", "build", "mat", freq_axis=ref("axis"), rho_c=flt(RHO_C)),
            read("mat", "boundary_model", "boundary_model"),
            read("mat", "is_locally_reacting", "is_locally_reacting"),
        ],
        ["boundary_model", "is_locally_reacting"],
    ),
    _spec_case(
        "material_loader.MaterialSpec.build.rigid_ignores_the_impedance_fields",
        [
            call(
                "material_loader.MaterialSpec",
                "spec",
                material_id=strs("m_rigid"),
                model=strs("rigid"),
                impedance_pa_s_m=obj(real=flt(1.0), imag=flt(2.0)),
            ),
            method("spec", "build", "mat", freq_axis=ref("axis"), rho_c=flt(RHO_C)),
        ],
        ["mat"],
    ),
    # ── load_materials ──────────────────────────────────────────────────────
    {
        "id": "material_loader.load_materials.signature",
        "steps": [signature("material_loader.load_materials", "sig")],
        "report": ["sig"],
    },
    _spec_case(
        "material_loader.load_materials.registers_one_material_per_yaml_file",
        [
            make_dir("materials"),
            make_file("materials/b_abs.yaml", ABSOLUTE_YAML),
            make_file("materials/a_norm.yaml", NORMALIZED_YAML),
            call(
                "material_loader.load_materials",
                "reg",
                materials_dir=under("materials"),
                freq_axis=ref("axis"),
                rho_c=flt(RHO_C),
            ),
            method("reg", "ids", "ids"),
            size_of("reg", "count"),
            contains("reg", strs("m_norm"), "has_norm"),
            method("reg", "get", "norm", material_id=strs("m_norm")),
            method("reg", "get", "abs", material_id=strs("m_abs")),
        ],
        ["ids", "count", "has_norm", "norm", "abs"],
    ),
    _spec_case(
        "material_loader.load_materials.source_kind_is_analytic",
        [
            make_dir("materials"),
            make_file("materials/a_norm.yaml", NORMALIZED_YAML),
            call(
                "material_loader.load_materials",
                "reg",
                materials_dir=under("materials"),
                freq_axis=ref("axis"),
                rho_c=flt(RHO_C),
            ),
            method("reg", "source_kind", "kind", material_id=strs("m_norm")),
            method("reg", "entry", "entry", material_id=strs("m_norm")),
        ],
        ["kind", "entry"],
    ),
    _spec_case(
        # 掃的順序是 ``sorted(glob("*.yaml"))``：檔名 a 先於 b，跟寫檔的順序無關。
        "material_loader.load_materials.reads_files_in_sorted_order",
        [
            make_dir("materials"),
            make_file("materials/z_last.yaml", "material_id: z_last\nnormalized_impedance: {real: 1.0}\n"),
            make_file("materials/a_first.yaml", "material_id: a_first\nnormalized_impedance: {real: 1.0}\n"),
            make_file("materials/m_middle.yaml", "material_id: m_middle\nnormalized_impedance: {real: 1.0}\n"),
            call(
                "material_loader.load_materials",
                "reg",
                materials_dir=under("materials"),
                freq_axis=ref("axis"),
                rho_c=flt(RHO_C),
            ),
            method("reg", "ids", "ids"),
        ],
        ["ids"],
    ),
    _spec_case(
        "material_loader.load_materials.accepts_a_string_path",
        [
            make_dir("materials"),
            make_file("materials/a_norm.yaml", NORMALIZED_YAML),
            call(
                "material_loader.load_materials",
                "reg",
                materials_dir=under_text("materials"),
                freq_axis=ref("axis"),
                rho_c=flt(RHO_C),
            ),
            method("reg", "ids", "ids"),
        ],
        ["ids"],
    ),
    _spec_case(
        "material_loader.load_materials.an_empty_directory_gives_an_empty_registry",
        [
            make_dir("materials"),
            call(
                "material_loader.load_materials",
                "reg",
                materials_dir=under("materials"),
                freq_axis=ref("axis"),
                rho_c=flt(RHO_C),
            ),
            method("reg", "ids", "ids"),
            size_of("reg", "count"),
        ],
        ["ids", "count"],
    ),
    _spec_case(
        # 目錄不存在**不炸**，給一個空的註冊表（上一代的行為，照抄）。
        "material_loader.load_materials.a_missing_directory_gives_an_empty_registry",
        [
            call(
                "material_loader.load_materials",
                "reg",
                materials_dir=under("not-there"),
                freq_axis=ref("axis"),
                rho_c=flt(RHO_C),
            ),
            method("reg", "ids", "ids"),
            size_of("reg", "count"),
        ],
        ["ids", "count"],
    ),
    _spec_case(
        # ``*.yaml`` 是字面上的：``.yml``、``.txt``、子目錄底下的都不讀。
        "material_loader.load_materials.only_reads_yaml_at_the_top_level",
        [
            make_dir("materials"),
            make_file("materials/a_norm.yaml", NORMALIZED_YAML),
            make_file("materials/b_short.yml", "material_id: m_yml\nnormalized_impedance: {real: 1.0}\n"),
            make_file("materials/c_note.txt", "material_id: m_txt\n"),
            make_file("materials/nested/d_deep.yaml", "material_id: m_deep\nnormalized_impedance: {real: 1.0}\n"),
            call(
                "material_loader.load_materials",
                "reg",
                materials_dir=under("materials"),
                freq_axis=ref("axis"),
                rho_c=flt(RHO_C),
            ),
            method("reg", "ids", "ids"),
            size_of("reg", "count"),
        ],
        ["ids", "count"],
    ),
    _spec_case(
        # 兩個檔同一個 material_id → 註冊表的重複守門把它擋下來（訊息逐字比）。
        "material_loader.load_materials.duplicate_material_id_raises",
        [
            make_dir("materials"),
            make_file("materials/a_one.yaml", NORMALIZED_YAML),
            make_file("materials/b_two.yaml", NORMALIZED_YAML),
            call(
                "material_loader.load_materials",
                "reg",
                materials_dir=under("materials"),
                freq_axis=ref("axis"),
                rho_c=flt(RHO_C),
            ),
        ],
        ["reg"],
    ),
    _spec_case(
        "material_loader.load_materials.an_invalid_material_file_is_rejected",
        [
            make_dir("materials"),
            make_file("materials/a_bad.yaml", "material_id: m_bad\nmodel: constant_impedance\n"),
            call(
                "material_loader.load_materials",
                "reg",
                materials_dir=under("materials"),
                freq_axis=ref("axis"),
                rho_c=flt(RHO_C),
            ),
        ],
        ["reg"],
    ),
    _spec_case(
        "material_loader.load_materials.broken_yaml_raises",
        [
            make_dir("materials"),
            make_file("materials/a_broken.yaml", "material_id: [m_bad\n"),
            call(
                "material_loader.load_materials",
                "reg",
                materials_dir=under("materials"),
                freq_axis=ref("axis"),
                rho_c=flt(RHO_C),
            ),
        ],
        ["reg"],
    ),
    _spec_case(
        "material_loader.load_materials.an_empty_yaml_file_raises",
        [
            make_dir("materials"),
            make_file("materials/a_empty.yaml", ""),
            call(
                "material_loader.load_materials",
                "reg",
                materials_dir=under("materials"),
                freq_axis=ref("axis"),
                rho_c=flt(RHO_C),
            ),
        ],
        ["reg"],
    ),
    _spec_case(
        "material_loader.load_materials.a_rigid_file_builds_a_rigid_wall",
        [
            make_dir("materials"),
            make_file("materials/a_rigid.yaml", RIGID_YAML),
            call(
                "material_loader.load_materials",
                "reg",
                materials_dir=under("materials"),
                freq_axis=ref("axis"),
                rho_c=flt(RHO_C),
            ),
            method("reg", "get", "mat", material_id=strs("m_rigid")),
        ],
        ["mat"],
    ),
]
