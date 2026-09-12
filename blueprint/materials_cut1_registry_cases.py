"""票 #134 第一候選：``source`` 與 ``registry`` 那兩支的 case。

拆檔的理由同 :mod:`blueprint.materials_cut1_response_cases`；這兩支綁在一起是因為
註冊表那幾筆要用 ``source`` 的 ``MaterialEntry``，兩份清單改動起來本來就同進同出。
"""
from __future__ import annotations

from typing import Final

from blueprint.materials_cut1_steps import (
    CaseEntry,
    RESPONSE_META,
    axis3,
    call,
    cplx,
    contains,
    false,
    flatten,
    flt,
    impedance3,
    method,
    nan,
    none,
    plus_inf,
    read,
    reads,
    ref,
    report_names,
    seq,
    signature,
    size_of,
    strs,
    through_jit,
    tree_double,
    true,
    write,
)

SOURCE_CASES: Final[list[CaseEntry]] = [
    {
        "id": "source.is_aosr_material_id.with_prefix",
        "steps": [call("source.is_aosr_material_id", "out", material_id=strs("aosr:felt"))],
        "report": ["out"],
    },
    {
        "id": "source.is_aosr_material_id.without_prefix",
        "steps": [call("source.is_aosr_material_id", "out", material_id=strs("felt"))],
        "report": ["out"],
    },
    {
        "id": "source.is_aosr_material_id.empty_string",
        "steps": [call("source.is_aosr_material_id", "out", material_id=strs(""))],
        "report": ["out"],
    },
    {
        "id": "source.is_aosr_material_id.prefix_only",
        "steps": [call("source.is_aosr_material_id", "out", material_id=strs("aosr:"))],
        "report": ["out"],
    },
    {
        "id": "source.is_aosr_material_id.is_case_sensitive",
        "steps": [call("source.is_aosr_material_id", "out", material_id=strs("AOSR:felt"))],
        "report": ["out"],
    },
    {
        "id": "source.is_aosr_material_id.leading_space_is_not_a_prefix",
        "steps": [call("source.is_aosr_material_id", "out", material_id=strs(" aosr:felt"))],
        "report": ["out"],
    },
    {
        "id": "source.is_aosr_material_id.signature",
        "steps": [signature("source.is_aosr_material_id", "sig")],
        "report": ["sig"],
    },
    {
        "id": "source.MaterialEntry.defaults",
        "steps": [
            *axis3(),
            *impedance3(),
            call("source.MaterialEntry", "entry", response=ref("mat"), source_kind=strs("analytic")),
            *reads("entry", ["source_kind", "source_ref"], "e"),
            read("entry", "response", "e_response"),
            read("e_response", "material_id", "e_material_id"),
        ],
        "report": [*report_names("e", ["source_kind", "source_ref"]), "e_material_id"],
    },
    {
        "id": "source.MaterialEntry.explicit_source_ref",
        "steps": [
            *axis3(),
            *impedance3(),
            call(
                "source.MaterialEntry",
                "entry",
                response=ref("mat"),
                source_kind=strs("measured_alpha"),
                source_ref=strs("datasheet:acme-7"),
            ),
            *reads("entry", ["source_kind", "source_ref"], "e"),
        ],
        "report": report_names("e", ["source_kind", "source_ref"]),
    },
    {
        "id": "source.MaterialEntry.is_frozen",
        "steps": [
            *axis3(),
            *impedance3(),
            call("source.MaterialEntry", "entry", response=ref("mat"), source_kind=strs("analytic")),
            write("entry", "source_kind", strs("computed_tmm")),
            read("entry", "source_kind", "e_source_kind"),
        ],
        "report": ["e_source_kind"],
    },
    {
        "id": "source.MaterialEntry.signature",
        "steps": [signature("source.MaterialEntry", "sig")],
        "report": ["sig"],
    },
]

REGISTRY_CASES: Final[list[CaseEntry]] = [
    {
        "id": "registry.MaterialRegistry.starts_empty",
        "steps": [
            call("registry.MaterialRegistry", "reg"),
            method("reg", "ids", "ids"),
            size_of("reg", "size"),
        ],
        "report": ["ids", "size"],
    },
    {
        "id": "registry.MaterialRegistry.register.defaults_to_analytic",
        "steps": [
            *axis3(),
            *impedance3(),
            call("registry.MaterialRegistry", "reg"),
            method("reg", "register", "_", mat=ref("mat")),
            method("reg", "source_kind", "kind", material_id=strs("m1")),
            method("reg", "entry", "entry", material_id=strs("m1")),
            read("entry", "source_ref", "entry_source_ref"),
            method("reg", "ids", "ids"),
        ],
        "report": ["kind", "entry_source_ref", "ids"],
    },
    {
        "id": "registry.MaterialRegistry.register.explicit_kind_and_ref",
        "steps": [
            *axis3(),
            *impedance3(),
            call("registry.MaterialRegistry", "reg"),
            method(
                "reg",
                "register",
                "_",
                mat=ref("mat"),
                source_kind=strs("measured_alpha"),
                source_ref=strs("datasheet:acme-7"),
            ),
            method("reg", "entry", "entry", material_id=strs("m1")),
            *reads("entry", ["source_kind", "source_ref"], "e"),
        ],
        "report": report_names("e", ["source_kind", "source_ref"]),
    },
    {
        "id": "registry.MaterialRegistry.register.empty_material_id_raises",
        "steps": [
            *axis3(),
            *impedance3("mat", "axis", ""),
            call("registry.MaterialRegistry", "reg"),
            method("reg", "register", "_", mat=ref("mat")),
        ],
        "report": ["_"],
    },
    {
        "id": "registry.MaterialRegistry.register.unknown_source_kind_raises",
        "steps": [
            *axis3(),
            *impedance3(),
            call("registry.MaterialRegistry", "reg"),
            method("reg", "register", "_", mat=ref("mat"), source_kind=strs("guessed")),
        ],
        "report": ["_"],
    },
    {
        "id": "registry.MaterialRegistry.register.duplicate_without_overwrite_raises",
        "steps": [
            *axis3(),
            *impedance3(),
            call("registry.MaterialRegistry", "reg"),
            method("reg", "register", "_", mat=ref("mat")),
            method("reg", "register", "__", mat=ref("mat")),
        ],
        "report": ["__"],
    },
    {
        "id": "registry.MaterialRegistry.register.implicit_overwrite_inherits_kind_and_ref",
        "steps": [
            *axis3(),
            *impedance3(),
            call("registry.MaterialRegistry", "reg"),
            method(
                "reg",
                "register",
                "_",
                mat=ref("mat"),
                source_kind=strs("measured_alpha"),
                source_ref=strs("datasheet:acme-7"),
            ),
            method("reg", "register", "__", mat=ref("mat"), overwrite=true()),
            method("reg", "entry", "entry", material_id=strs("m1")),
            *reads("entry", ["source_kind", "source_ref"], "e"),
        ],
        "report": report_names("e", ["source_kind", "source_ref"]),
    },
    {
        "id": "registry.MaterialRegistry.register.explicit_kind_drops_the_stale_ref",
        "steps": [
            *axis3(),
            *impedance3(),
            call("registry.MaterialRegistry", "reg"),
            method(
                "reg",
                "register",
                "_",
                mat=ref("mat"),
                source_kind=strs("measured_alpha"),
                source_ref=strs("datasheet:acme-7"),
            ),
            method(
                "reg",
                "register",
                "__",
                mat=ref("mat"),
                source_kind=strs("computed_tmm"),
                overwrite=true(),
            ),
            method("reg", "entry", "entry", material_id=strs("m1")),
            *reads("entry", ["source_kind", "source_ref"], "e"),
        ],
        "report": report_names("e", ["source_kind", "source_ref"]),
    },
    {
        "id": "registry.MaterialRegistry.register.signature",
        "steps": [signature("registry.MaterialRegistry.register", "sig")],
        "report": ["sig"],
    },
    {
        "id": "registry.MaterialRegistry.register_entry.unknown_source_kind_raises",
        "steps": [
            *axis3(),
            *impedance3(),
            call("source.MaterialEntry", "entry", response=ref("mat"), source_kind=strs("guessed")),
            call("registry.MaterialRegistry", "reg"),
            method("reg", "register_entry", "_", entry=ref("entry")),
        ],
        "report": ["_"],
    },
    {
        "id": "registry.MaterialRegistry.register_entry.empty_material_id_raises",
        "steps": [
            *axis3(),
            *impedance3("mat", "axis", ""),
            call("source.MaterialEntry", "entry", response=ref("mat"), source_kind=strs("analytic")),
            call("registry.MaterialRegistry", "reg"),
            method("reg", "register_entry", "_", entry=ref("entry")),
        ],
        "report": ["_"],
    },
    {
        "id": "registry.MaterialRegistry.register_entry.duplicate_without_overwrite_raises",
        "steps": [
            *axis3(),
            *impedance3(),
            call("source.MaterialEntry", "entry", response=ref("mat"), source_kind=strs("analytic")),
            call("registry.MaterialRegistry", "reg"),
            method("reg", "register_entry", "_", entry=ref("entry")),
            method("reg", "register_entry", "__", entry=ref("entry")),
        ],
        "report": ["__"],
    },
    {
        "id": "registry.MaterialRegistry.register_entry.overwrite_replaces",
        "steps": [
            *axis3(),
            *impedance3(),
            call("source.MaterialEntry", "first", response=ref("mat"), source_kind=strs("analytic")),
            call(
                "source.MaterialEntry",
                "second",
                response=ref("mat"),
                source_kind=strs("computed_tmm"),
                source_ref=strs("tmm:run-3"),
            ),
            call("registry.MaterialRegistry", "reg"),
            method("reg", "register_entry", "_", entry=ref("first")),
            method("reg", "register_entry", "__", entry=ref("second"), overwrite=true()),
            method("reg", "entry", "got", material_id=strs("m1")),
            *reads("got", ["source_kind", "source_ref"], "e"),
        ],
        "report": report_names("e", ["source_kind", "source_ref"]),
    },
    {
        "id": "registry.MaterialRegistry.get.unknown_id_raises",
        "steps": [
            *axis3(),
            *impedance3(),
            call("registry.MaterialRegistry", "reg"),
            method("reg", "register", "_", mat=ref("mat")),
            method("reg", "get", "got", material_id=strs("nope")),
        ],
        "report": ["got"],
    },
    {
        "id": "registry.MaterialRegistry.get.returns_the_response",
        "steps": [
            *axis3(),
            *impedance3(),
            call("registry.MaterialRegistry", "reg"),
            method("reg", "register", "_", mat=ref("mat")),
            method("reg", "get", "got", material_id=strs("m1")),
            *reads("got", ["material_id", "Z_surface", "boundary_model"], "g"),
        ],
        "report": report_names("g", ["material_id", "Z_surface", "boundary_model"]),
    },
    {
        "id": "registry.MaterialRegistry.entry.unknown_id_raises",
        "steps": [
            call("registry.MaterialRegistry", "reg"),
            method("reg", "entry", "got", material_id=strs("nope")),
        ],
        "report": ["got"],
    },
    {
        "id": "registry.MaterialRegistry.source_kind.unknown_id_raises",
        "steps": [
            call("registry.MaterialRegistry", "reg"),
            method("reg", "source_kind", "got", material_id=strs("nope")),
        ],
        "report": ["got"],
    },
    {
        "id": "registry.MaterialRegistry.ids.are_sorted",
        "steps": [
            *axis3(),
            call("registry.rigid_wall", "wall_b", freq_axis=ref("axis"), material_id=strs("b_wall")),
            call("registry.rigid_wall", "wall_a", freq_axis=ref("axis"), material_id=strs("a_wall")),
            call("registry.MaterialRegistry", "reg"),
            method("reg", "register", "_", mat=ref("wall_b")),
            method("reg", "register", "__", mat=ref("wall_a")),
            method("reg", "ids", "ids"),
        ],
        "report": ["ids"],
    },
    {
        "id": "registry.MaterialRegistry.contains_and_len",
        "steps": [
            *axis3(),
            *impedance3(),
            call("registry.MaterialRegistry", "reg"),
            method("reg", "register", "_", mat=ref("mat")),
            contains("reg", strs("m1"), "has"),
            contains("reg", strs("nope"), "has_not"),
            size_of("reg", "size"),
        ],
        "report": ["has", "has_not", "size"],
    },
    {
        "id": "registry.rigid_wall.defaults",
        "steps": [
            *axis3(),
            call("registry.rigid_wall", "wall", freq_axis=ref("axis")),
            *reads("wall", [*RESPONSE_META, "n_freq"], "w"),
        ],
        "report": report_names("w", [*RESPONSE_META, "n_freq"]),
    },
    {
        "id": "registry.rigid_wall.custom_magnitude_and_id",
        "steps": [
            *axis3(),
            call(
                "registry.rigid_wall",
                "wall",
                freq_axis=ref("axis"),
                material_id=strs("aosr:wall"),
                z_magnitude=flt(4.2e5),
            ),
            *reads("wall", ["material_id", "Z_surface"], "w"),
        ],
        "report": report_names("w", ["material_id", "Z_surface"]),
    },
    {
        "id": "registry.rigid_wall.signature",
        "steps": [signature("registry.rigid_wall", "sig")],
        "report": ["sig"],
    },
    {
        "id": "registry.constant_impedance.complex_value",
        "steps": [
            *axis3(),
            call(
                "registry.constant_impedance",
                "flat",
                freq_axis=ref("axis"),
                z_value=cplx(800.0, -120.0),
                material_id=strs("flat"),
            ),
            *reads("flat", [*RESPONSE_META, "n_freq"], "c"),
        ],
        "report": report_names("c", [*RESPONSE_META, "n_freq"]),
    },
    {
        "id": "registry.constant_impedance.metadata_overrides",
        "steps": [
            *axis3(),
            call(
                "registry.constant_impedance",
                "flat",
                freq_axis=ref("axis"),
                z_value=cplx(415.0, 0.0),
                material_id=strs("flat"),
                boundary_model=strs("extended_reacting"),
                is_locally_reacting=false(),
            ),
            *reads("flat", ["boundary_model", "is_locally_reacting"], "c"),
        ],
        "report": report_names("c", ["boundary_model", "is_locally_reacting"]),
    },
    {
        "id": "registry.constant_impedance.signature",
        "steps": [signature("registry.constant_impedance", "sig")],
        "report": ["sig"],
    },
]

