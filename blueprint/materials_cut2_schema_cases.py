"""票 #134 第二刀：``experiment_schema`` 那一支的 case 清單。

這一支是**上一代 ``lib/config/experiment_schema.py`` 的公開行為**寫成的題目：五個 pydantic
模型（``Vec3``／``FrequencyAxisSpec``／``ReceiverGridSpec``／``SurfaceMaterial``／
``RoomSpec``／``ExperimentConfig``）加上載入器 ``load_experiment``。

**模型的契約有兩面，兩面都要有題。** 一面是 ``model_json_schema()``——欄位有哪些、哪些
必填、預設是什麼、多給一格收不收（``extra="forbid"`` 在 schema 裡是
``additionalProperties: false``），那一格整份記下來就凍住了整個形狀；另一面是**真的收一筆
輸入會怎樣**：合法的那一筆 dump 出什麼、不合法的那幾筆炸什麼（例外種類與訊息逐字比）。

**模型實例本身不進記號**：codec 認得 dataclass 與陣列，不認得 pydantic 的模型，所以合法
那幾筆一律 ``model_dump()`` 之後再記——dump 出來的是普通的表與純量，鍵與值都走同一套記號。
``.build()`` 回的是 ``FrequencyAxis``（flax 的 dataclass），那個 codec 認得，直接記。

**上一代那七題原封不動搬進來**：``tests/test_experiment_schema.py`` 的 nan／inf／負數／零／
重複／空清單／未排序，七題各自是下面的一筆 case（id 尾巴標了 ``donor_test``）。
"""
from __future__ import annotations

from typing import Final

from blueprint.materials_cut2_steps import (
    CaseEntry,
    call,
    flt,
    make_file,
    method,
    minus_inf,
    nan,
    none,
    obj,
    plus_inf,
    read,
    seq,
    signature,
    strs,
    under,
    under_text,
    write,
)

# 一份合法的實驗設定：下面好幾筆都拿它當底，只改其中一格。
_ROOM: Final[dict[str, object]] = obj(
    geometry_level=0,
    dims_m=obj(x=flt(5.0), y=flt(4.0), z=flt(3.0)),
)
_AXIS_CUSTOM: Final[dict[str, object]] = obj(
    mode=strs("custom"),
    freqs_hz=seq(flt(100.0), flt(200.0), flt(400.0)),
)

EXPERIMENT_YAML: Final[str] = """
name: cut2-experiment
frequency_axis:
  mode: custom
  freqs_hz: [100.0, 200.0, 400.0]
room:
  geometry_level: 1
  dims_m: {x: 5.0, y: 4.0, z: 3.0}
source: {x: 1.0, y: 1.0, z: 1.2}
listening_position: {x: 2.5, y: 2.0, z: 1.2}
surface_materials:
  - {boundary_id: wall_north, material_id: m_absorber}
"""

_SCHEMA_MODELS: Final[tuple[str, ...]] = (
    "Vec3",
    "FrequencyAxisSpec",
    "ReceiverGridSpec",
    "SurfaceMaterial",
    "RoomSpec",
    "ExperimentConfig",
)


def _json_schema_case(model: str) -> CaseEntry:
    """一個模型的 ``model_json_schema()`` 整份凍下來（欄位、必填、預設、多給一格收不收）。"""
    return {
        "id": f"experiment_schema.{model}.model_json_schema",
        "steps": [call(f"experiment_schema.{model}.model_json_schema", "schema")],
        "report": ["schema"],
    }


SCHEMA_CASES: Final[list[CaseEntry]] = [
    *[_json_schema_case(name) for name in _SCHEMA_MODELS],
    # ── Vec3 ────────────────────────────────────────────────────────────────
    {
        "id": "experiment_schema.Vec3.as_tuple",
        "steps": [
            call("experiment_schema.Vec3", "v", x=flt(1.5), y=flt(-2.0), z=flt(0.0)),
            method("v", "as_tuple", "out"),
            method("v", "model_dump", "dumped"),
        ],
        "report": ["out", "dumped"],
    },
    {
        "id": "experiment_schema.Vec3.as_tuple.signature",
        "steps": [signature("experiment_schema.Vec3.as_tuple", "sig")],
        "report": ["sig"],
    },
    {
        "id": "experiment_schema.Vec3.integer_input_is_coerced_to_float",
        "steps": [
            call("experiment_schema.Vec3", "v", x=1, y=2, z=3),
            method("v", "as_tuple", "out"),
        ],
        "report": ["out"],
    },
    {
        "id": "experiment_schema.Vec3.extra_field_is_rejected",
        "steps": [call("experiment_schema.Vec3", "v", x=flt(1.0), y=flt(1.0), z=flt(1.0), w=flt(1.0))],
        "report": ["v"],
    },
    {
        "id": "experiment_schema.Vec3.missing_field_is_rejected",
        "steps": [call("experiment_schema.Vec3", "v", x=flt(1.0), y=flt(1.0))],
        "report": ["v"],
    },
    # ── FrequencyAxisSpec：驗證器 ───────────────────────────────────────────
    {
        "id": "experiment_schema.FrequencyAxisSpec.defaults_are_third_octave",
        "steps": [
            call("experiment_schema.FrequencyAxisSpec", "spec", f_min_hz=flt(100.0), f_max_hz=flt(1000.0)),
            method("spec", "model_dump", "dumped"),
        ],
        "report": ["dumped"],
    },
    {
        "id": "experiment_schema.FrequencyAxisSpec.custom_keeps_unsorted_input_on_the_spec",
        "steps": [
            call(
                "experiment_schema.FrequencyAxisSpec",
                "spec",
                mode=strs("custom"),
                freqs_hz=seq(flt(400.0), flt(100.0), flt(200.0)),
            ),
            method("spec", "model_dump", "dumped"),
        ],
        "report": ["dumped"],
    },
    {
        "id": "experiment_schema.FrequencyAxisSpec.non_positive_f_min_is_rejected",
        "steps": [
            call(
                "experiment_schema.FrequencyAxisSpec",
                "spec",
                mode=strs("linear_hz"),
                f_min_hz=flt(0.0),
                f_max_hz=flt(100.0),
                n_points=5,
            )
        ],
        "report": ["spec"],
    },
    {
        "id": "experiment_schema.FrequencyAxisSpec.n_points_must_exceed_one",
        "steps": [
            call(
                "experiment_schema.FrequencyAxisSpec",
                "spec",
                mode=strs("linear_hz"),
                f_min_hz=flt(100.0),
                f_max_hz=flt(200.0),
                n_points=1,
            )
        ],
        "report": ["spec"],
    },
    {
        "id": "experiment_schema.FrequencyAxisSpec.unknown_mode_is_rejected",
        "steps": [call("experiment_schema.FrequencyAxisSpec", "spec", mode=strs("octave"))],
        "report": ["spec"],
    },
    {
        "id": "experiment_schema.FrequencyAxisSpec.extra_field_is_rejected",
        "steps": [
            call(
                "experiment_schema.FrequencyAxisSpec",
                "spec",
                mode=strs("custom"),
                freqs_hz=seq(flt(100.0)),
                weighting=strs("A"),
            )
        ],
        "report": ["spec"],
    },
    {
        "id": "experiment_schema.FrequencyAxisSpec.third_octave_without_range_is_rejected",
        "steps": [call("experiment_schema.FrequencyAxisSpec", "spec", f_min_hz=flt(100.0))],
        "report": ["spec"],
    },
    {
        "id": "experiment_schema.FrequencyAxisSpec.f_max_must_exceed_f_min",
        "steps": [
            call(
                "experiment_schema.FrequencyAxisSpec",
                "spec",
                f_min_hz=flt(500.0),
                f_max_hz=flt(500.0),
            )
        ],
        "report": ["spec"],
    },
    {
        "id": "experiment_schema.FrequencyAxisSpec.linear_hz_without_n_points_is_rejected",
        "steps": [
            call(
                "experiment_schema.FrequencyAxisSpec",
                "spec",
                mode=strs("linear_hz"),
                f_min_hz=flt(100.0),
                f_max_hz=flt(200.0),
            )
        ],
        "report": ["spec"],
    },
    {
        # 上一代 test_experiment_schema.py 第六題：空清單在**建構**就被擋（不是 build）。
        "id": "experiment_schema.FrequencyAxisSpec.donor_test.empty_custom_list_is_rejected",
        "steps": [
            call("experiment_schema.FrequencyAxisSpec", "spec", mode=strs("custom"), freqs_hz=seq())
        ],
        "report": ["spec"],
    },
    # ── FrequencyAxisSpec.build ─────────────────────────────────────────────
    {
        "id": "experiment_schema.FrequencyAxisSpec.build.signature",
        "steps": [signature("experiment_schema.FrequencyAxisSpec.build", "sig")],
        "report": ["sig"],
    },
    {
        # 上一代第七題：未排序的合法輸入會被**排序**之後才建軸。
        "id": "experiment_schema.FrequencyAxisSpec.donor_test.unsorted_custom_input_is_sorted",
        "steps": [
            call(
                "experiment_schema.FrequencyAxisSpec",
                "spec",
                mode=strs("custom"),
                freqs_hz=seq(flt(400.0), flt(100.0), flt(200.0)),
            ),
            method("spec", "build", "axis"),
            read("axis", "freqs_hz", "freqs_hz"),
            read("axis", "resolution", "resolution"),
            read("axis", "n_freq", "n_freq"),
            read("axis", "f_min", "f_min"),
            read("axis", "f_max", "f_max"),
        ],
        "report": ["freqs_hz", "resolution", "n_freq", "f_min", "f_max"],
    },
    {
        "id": "experiment_schema.FrequencyAxisSpec.build.custom_single_point",
        "steps": [
            call(
                "experiment_schema.FrequencyAxisSpec",
                "spec",
                mode=strs("custom"),
                freqs_hz=seq(flt(1000.0)),
            ),
            method("spec", "build", "axis"),
        ],
        "report": ["axis"],
    },
    {
        # custom 模式不看 f_min/f_max：給了也不影響（上一代就是這樣，不順手修）。
        "id": "experiment_schema.FrequencyAxisSpec.build.custom_ignores_the_range_fields",
        "steps": [
            call(
                "experiment_schema.FrequencyAxisSpec",
                "spec",
                mode=strs("custom"),
                freqs_hz=seq(flt(100.0), flt(200.0)),
                f_min_hz=flt(900.0),
                f_max_hz=flt(1000.0),
            ),
            method("spec", "build", "axis"),
        ],
        "report": ["axis"],
    },
    {
        "id": "experiment_schema.FrequencyAxisSpec.donor_test.build_rejects_nan",
        "steps": [
            call(
                "experiment_schema.FrequencyAxisSpec",
                "spec",
                mode=strs("custom"),
                freqs_hz=seq(flt(100.0), nan()),
            ),
            method("spec", "build", "axis"),
        ],
        "report": ["axis"],
    },
    {
        "id": "experiment_schema.FrequencyAxisSpec.donor_test.build_rejects_infinity",
        "steps": [
            call(
                "experiment_schema.FrequencyAxisSpec",
                "spec",
                mode=strs("custom"),
                freqs_hz=seq(flt(100.0), plus_inf()),
            ),
            method("spec", "build", "axis"),
        ],
        "report": ["axis"],
    },
    {
        "id": "experiment_schema.FrequencyAxisSpec.donor_test.build_rejects_minus_infinity",
        "steps": [
            call(
                "experiment_schema.FrequencyAxisSpec",
                "spec",
                mode=strs("custom"),
                freqs_hz=seq(flt(100.0), minus_inf()),
            ),
            method("spec", "build", "axis"),
        ],
        "report": ["axis"],
    },
    {
        "id": "experiment_schema.FrequencyAxisSpec.donor_test.build_rejects_negative_frequency",
        "steps": [
            call(
                "experiment_schema.FrequencyAxisSpec",
                "spec",
                mode=strs("custom"),
                freqs_hz=seq(flt(100.0), flt(-10.0)),
            ),
            method("spec", "build", "axis"),
        ],
        "report": ["axis"],
    },
    {
        "id": "experiment_schema.FrequencyAxisSpec.donor_test.build_rejects_zero_frequency",
        "steps": [
            call(
                "experiment_schema.FrequencyAxisSpec",
                "spec",
                mode=strs("custom"),
                freqs_hz=seq(flt(100.0), flt(0.0)),
            ),
            method("spec", "build", "axis"),
        ],
        "report": ["axis"],
    },
    {
        "id": "experiment_schema.FrequencyAxisSpec.donor_test.build_rejects_duplicate_frequency",
        "steps": [
            call(
                "experiment_schema.FrequencyAxisSpec",
                "spec",
                mode=strs("custom"),
                freqs_hz=seq(flt(100.0), flt(200.0), flt(100.0)),
            ),
            method("spec", "build", "axis"),
        ],
        "report": ["axis"],
    },
    {
        "id": "experiment_schema.FrequencyAxisSpec.build.linear_hz",
        "steps": [
            call(
                "experiment_schema.FrequencyAxisSpec",
                "spec",
                mode=strs("linear_hz"),
                f_min_hz=flt(100.0),
                f_max_hz=flt(500.0),
                n_points=5,
            ),
            method("spec", "build", "axis"),
        ],
        "report": ["axis"],
    },
    {
        "id": "experiment_schema.FrequencyAxisSpec.build.third_octave",
        "steps": [
            call(
                "experiment_schema.FrequencyAxisSpec",
                "spec",
                f_min_hz=flt(100.0),
                f_max_hz=flt(1000.0),
            ),
            method("spec", "build", "axis"),
        ],
        "report": ["axis"],
    },
    {
        # 範圍太窄，一個 1/3 倍頻中心都框不到 → 炸（訊息逐字比）。
        "id": "experiment_schema.FrequencyAxisSpec.build.third_octave_without_any_centre_raises",
        "steps": [
            call(
                "experiment_schema.FrequencyAxisSpec",
                "spec",
                f_min_hz=flt(1050.0),
                f_max_hz=flt(1100.0),
            ),
            method("spec", "build", "axis"),
        ],
        "report": ["axis"],
    },
    # 模型**沒有** frozen 也沒有 validate_assignment：建構之後欄位還改得動，改壞了由
    # 下游那一步自己炸。三筆各自把一格清成 None，看 build 在哪一步、炸什麼——把「驗證器只
    # 在建構時跑一次」這件事釘住，順便擋住「在 build 裡補一層新的守門」那種悄悄收緊。
    {
        "id": "experiment_schema.FrequencyAxisSpec.build.linear_hz_with_n_points_cleared_raises",
        "steps": [
            call(
                "experiment_schema.FrequencyAxisSpec",
                "spec",
                mode=strs("linear_hz"),
                f_min_hz=flt(100.0),
                f_max_hz=flt(500.0),
                n_points=5,
            ),
            write("spec", "n_points", none()),
            method("spec", "build", "axis"),
        ],
        "report": ["axis"],
    },
    {
        "id": "experiment_schema.FrequencyAxisSpec.build.linear_hz_with_f_min_cleared_raises",
        "steps": [
            call(
                "experiment_schema.FrequencyAxisSpec",
                "spec",
                mode=strs("linear_hz"),
                f_min_hz=flt(100.0),
                f_max_hz=flt(500.0),
                n_points=5,
            ),
            write("spec", "f_min_hz", none()),
            method("spec", "build", "axis"),
        ],
        "report": ["axis"],
    },
    {
        "id": "experiment_schema.FrequencyAxisSpec.build.third_octave_with_f_min_cleared_raises",
        "steps": [
            call(
                "experiment_schema.FrequencyAxisSpec",
                "spec",
                f_min_hz=flt(100.0),
                f_max_hz=flt(1000.0),
            ),
            write("spec", "f_min_hz", none()),
            method("spec", "build", "axis"),
        ],
        "report": ["axis"],
    },
    # ── ReceiverGridSpec ────────────────────────────────────────────────────
    {
        "id": "experiment_schema.ReceiverGridSpec.defaults",
        "steps": [
            call("experiment_schema.ReceiverGridSpec", "grid"),
            method("grid", "model_dump", "dumped"),
        ],
        "report": ["dumped"],
    },
    {
        "id": "experiment_schema.ReceiverGridSpec.zero_points_is_rejected",
        "steps": [call("experiment_schema.ReceiverGridSpec", "grid", n_points=0)],
        "report": ["grid"],
    },
    {
        "id": "experiment_schema.ReceiverGridSpec.negative_wall_offset_is_rejected",
        "steps": [call("experiment_schema.ReceiverGridSpec", "grid", wall_offset_m=flt(-0.1))],
        "report": ["grid"],
    },
    {
        "id": "experiment_schema.ReceiverGridSpec.zero_wall_offset_is_allowed",
        "steps": [
            call("experiment_schema.ReceiverGridSpec", "grid", n_points=9, wall_offset_m=flt(0.0)),
            method("grid", "model_dump", "dumped"),
        ],
        "report": ["dumped"],
    },
    # ── SurfaceMaterial ─────────────────────────────────────────────────────
    {
        "id": "experiment_schema.SurfaceMaterial.round_trip",
        "steps": [
            call(
                "experiment_schema.SurfaceMaterial",
                "sm",
                boundary_id=strs("wall_north"),
                material_id=strs("m_absorber"),
            ),
            method("sm", "model_dump", "dumped"),
        ],
        "report": ["dumped"],
    },
    {
        "id": "experiment_schema.SurfaceMaterial.missing_material_id_is_rejected",
        "steps": [call("experiment_schema.SurfaceMaterial", "sm", boundary_id=strs("wall_north"))],
        "report": ["sm"],
    },
    # ── RoomSpec ────────────────────────────────────────────────────────────
    {
        "id": "experiment_schema.RoomSpec.dims_are_optional",
        "steps": [
            call("experiment_schema.RoomSpec", "room", geometry_level=0),
            method("room", "model_dump", "dumped"),
        ],
        "report": ["dumped"],
    },
    {
        "id": "experiment_schema.RoomSpec.nested_dims_are_a_vec3",
        "steps": [
            call("experiment_schema.RoomSpec", "room", geometry_level=4, dims_m=obj(x=flt(5.0), y=flt(4.0), z=flt(3.0))),
            method("room", "model_dump", "dumped"),
            read("room", "dims_m", "dims"),
            method("dims", "as_tuple", "as_tuple"),
        ],
        "report": ["dumped", "as_tuple"],
    },
    {
        "id": "experiment_schema.RoomSpec.geometry_level_above_four_is_rejected",
        "steps": [call("experiment_schema.RoomSpec", "room", geometry_level=5)],
        "report": ["room"],
    },
    {
        "id": "experiment_schema.RoomSpec.negative_geometry_level_is_rejected",
        "steps": [call("experiment_schema.RoomSpec", "room", geometry_level=-1)],
        "report": ["room"],
    },
    # ── ExperimentConfig ────────────────────────────────────────────────────
    {
        "id": "experiment_schema.ExperimentConfig.full_round_trip",
        "steps": [
            call(
                "experiment_schema.ExperimentConfig",
                "cfg",
                name=strs("cut2"),
                frequency_axis=_AXIS_CUSTOM,
                room=_ROOM,
                source=obj(x=flt(1.0), y=flt(1.0), z=flt(1.2)),
                listening_position=obj(x=flt(2.5), y=flt(2.0), z=flt(1.2)),
                receiver_grid=obj(n_points=9, wall_offset_m=flt(0.25)),
                surface_materials=seq(obj(boundary_id=strs("wall_north"), material_id=strs("m1"))),
            ),
            method("cfg", "model_dump", "dumped"),
        ],
        "report": ["dumped"],
    },
    {
        # 沒給 receiver_grid 與 surface_materials 的時候，那兩格的預設是什麼。
        "id": "experiment_schema.ExperimentConfig.defaults_for_grid_and_materials",
        "steps": [
            call(
                "experiment_schema.ExperimentConfig",
                "cfg",
                name=strs("cut2"),
                frequency_axis=_AXIS_CUSTOM,
                room=_ROOM,
                source=obj(x=flt(1.0), y=flt(1.0), z=flt(1.2)),
                listening_position=obj(x=flt(2.5), y=flt(2.0), z=flt(1.2)),
            ),
            method("cfg", "model_dump", "dumped"),
            read("cfg", "surface_materials", "materials"),
        ],
        "report": ["dumped", "materials"],
    },
    {
        "id": "experiment_schema.ExperimentConfig.extra_field_is_rejected",
        "steps": [
            call(
                "experiment_schema.ExperimentConfig",
                "cfg",
                name=strs("cut2"),
                frequency_axis=_AXIS_CUSTOM,
                room=_ROOM,
                source=obj(x=flt(1.0), y=flt(1.0), z=flt(1.2)),
                listening_position=obj(x=flt(2.5), y=flt(2.0), z=flt(1.2)),
                rho_c=flt(415.0),
            )
        ],
        "report": ["cfg"],
    },
    {
        "id": "experiment_schema.ExperimentConfig.missing_required_fields_are_rejected",
        "steps": [call("experiment_schema.ExperimentConfig", "cfg", name=strs("cut2"))],
        "report": ["cfg"],
    },
    {
        "id": "experiment_schema.ExperimentConfig.the_axis_spec_is_built_not_stored_as_a_dict",
        "steps": [
            call(
                "experiment_schema.ExperimentConfig",
                "cfg",
                name=strs("cut2"),
                frequency_axis=_AXIS_CUSTOM,
                room=_ROOM,
                source=obj(x=flt(1.0), y=flt(1.0), z=flt(1.2)),
                listening_position=obj(x=flt(2.5), y=flt(2.0), z=flt(1.2)),
            ),
            read("cfg", "frequency_axis", "spec"),
            method("spec", "build", "axis"),
        ],
        "report": ["axis"],
    },
    # ── load_experiment ─────────────────────────────────────────────────────
    {
        "id": "experiment_schema.load_experiment.signature",
        "steps": [signature("experiment_schema.load_experiment", "sig")],
        "report": ["sig"],
    },
    {
        "id": "experiment_schema.load_experiment.reads_a_yaml_file",
        "steps": [
            make_file("experiment.yaml", EXPERIMENT_YAML),
            call("experiment_schema.load_experiment", "cfg", path=under("experiment.yaml")),
            method("cfg", "model_dump", "dumped"),
        ],
        "report": ["dumped"],
    },
    {
        # ``str`` 那一支路也要有人走過（簽章寫 ``str | Path``）。
        "id": "experiment_schema.load_experiment.accepts_a_string_path",
        "steps": [
            make_file("experiment.yaml", EXPERIMENT_YAML),
            call("experiment_schema.load_experiment", "cfg", path=under_text("experiment.yaml")),
            method("cfg", "model_dump", "dumped"),
        ],
        "report": ["dumped"],
    },
    {
        "id": "experiment_schema.load_experiment.missing_file_raises",
        "steps": [call("experiment_schema.load_experiment", "cfg", path=under("nope.yaml"))],
        "report": ["cfg"],
    },
    {
        "id": "experiment_schema.load_experiment.a_directory_raises",
        "steps": [call("experiment_schema.load_experiment", "cfg", path=under(""))],
        "report": ["cfg"],
    },
    {
        "id": "experiment_schema.load_experiment.broken_yaml_raises",
        "steps": [
            make_file("broken.yaml", "name: [cut2\n"),
            call("experiment_schema.load_experiment", "cfg", path=under("broken.yaml")),
        ],
        "report": ["cfg"],
    },
    {
        "id": "experiment_schema.load_experiment.empty_file_raises",
        "steps": [
            make_file("empty.yaml", ""),
            call("experiment_schema.load_experiment", "cfg", path=under("empty.yaml")),
        ],
        "report": ["cfg"],
    },
    {
        "id": "experiment_schema.load_experiment.a_yaml_list_raises",
        "steps": [
            make_file("list.yaml", "- 1\n- 2\n"),
            call("experiment_schema.load_experiment", "cfg", path=under("list.yaml")),
        ],
        "report": ["cfg"],
    },
    {
        "id": "experiment_schema.load_experiment.an_invalid_experiment_is_rejected",
        "steps": [
            make_file("bad.yaml", "name: cut2\nroom: {geometry_level: 9}\n"),
            call("experiment_schema.load_experiment", "cfg", path=under("bad.yaml")),
        ],
        "report": ["cfg"],
    },
]