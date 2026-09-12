"""票 #134 第一候選：``response`` 那一支的 case（一筆 case 的形狀見 :mod:`blueprint.materials_cut1_steps`）。

這一支只放清單，不放邏輯：「有哪些 case」是版控裡的單一來源，產生器與考卷都照它跑。
分成幾支檔是因為 ``style-guard`` 第⑤條有單檔行數上限——上限擋的是「一支檔大到看不完」，
拆法照模組走（一支模組一份清單），不是照行數硬切。
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

RESPONSE_CASES: Final[list[CaseEntry]] = [
    {
        "id": "response.FrequencyAxis.from_hz.three_floats",
        "steps": [
            *axis3(),
            *reads("axis", ["freqs_hz", "n_freq", "f_min", "f_max", "resolution", "convention"], "a"),
        ],
        "report": report_names("a", ["freqs_hz", "n_freq", "f_min", "f_max", "resolution", "convention"]),
    },
    {
        "id": "response.FrequencyAxis.from_hz.integer_input_keeps_integer_dtype",
        "steps": [
            call("response.FrequencyAxis.from_hz", "axis", freqs_hz=seq(100, 200)),
            *reads("axis", ["freqs_hz", "f_min", "f_max"], "a"),
        ],
        "report": report_names("a", ["freqs_hz", "f_min", "f_max"]),
    },
    {
        "id": "response.FrequencyAxis.from_hz.default_resolution_and_convention",
        "steps": [
            call("response.FrequencyAxis.from_hz", "axis", freqs_hz=seq(flt(20.0), flt(50.0), flt(500.0))),
            *reads("axis", ["resolution", "convention", "n_freq", "f_min", "f_max"], "a"),
        ],
        "report": report_names("a", ["resolution", "convention", "n_freq", "f_min", "f_max"]),
    },
    {
        "id": "response.FrequencyAxis.from_hz.third_octave_resolution",
        "steps": [
            call(
                "response.FrequencyAxis.from_hz",
                "axis",
                freqs_hz=seq(flt(125.0), flt(250.0)),
                resolution=strs("third_octave"),
            ),
            *reads("axis", ["resolution", "convention"], "a"),
        ],
        "report": report_names("a", ["resolution", "convention"]),
    },
    {
        "id": "response.FrequencyAxis.from_hz.single_point",
        "steps": [
            call("response.FrequencyAxis.from_hz", "axis", freqs_hz=seq(flt(1000.0))),
            *reads("axis", ["n_freq", "f_min", "f_max"], "a"),
        ],
        "report": report_names("a", ["n_freq", "f_min", "f_max"]),
    },
    {
        "id": "response.FrequencyAxis.from_hz.descending_input_is_not_sorted",
        "steps": [
            call("response.FrequencyAxis.from_hz", "axis", freqs_hz=seq(flt(400.0), flt(100.0))),
            *reads("axis", ["freqs_hz", "f_min", "f_max"], "a"),
        ],
        "report": report_names("a", ["freqs_hz", "f_min", "f_max"]),
    },
    {
        "id": "response.FrequencyAxis.from_hz.empty_axis_has_zero_n_freq",
        "steps": [
            call("response.FrequencyAxis.from_hz", "axis", freqs_hz=seq()),
            read("axis", "n_freq", "a_n_freq"),
        ],
        "report": ["a_n_freq"],
    },
    {
        "id": "response.FrequencyAxis.from_hz.empty_axis_f_min_raises",
        "steps": [
            call("response.FrequencyAxis.from_hz", "axis", freqs_hz=seq()),
            read("axis", "f_min", "a_f_min"),
        ],
        "report": ["a_f_min"],
    },
    {
        "id": "response.FrequencyAxis.direct_construction_defaults",
        "steps": [
            call("response.FrequencyAxis.from_hz", "seed", freqs_hz=seq(flt(100.0), flt(200.0))),
            read("seed", "freqs_hz", "raw"),
            call("response.FrequencyAxis", "axis", freqs_hz=ref("raw")),
            *reads("axis", ["resolution", "convention", "n_freq"], "a"),
        ],
        "report": report_names("a", ["resolution", "convention", "n_freq"]),
    },
    {
        "id": "response.FrequencyAxis.pytree_flatten_roundtrip",
        "steps": [*axis3(), flatten("axis", "tree")],
        "report": ["tree"],
    },
    {
        "id": "response.FrequencyAxis.from_hz.signature",
        "steps": [signature("response.FrequencyAxis.from_hz", "sig")],
        "report": ["sig"],
    },
    {
        "id": "response.MaterialResponse.from_impedance.complex_with_default_scattering",
        "steps": [*axis3(), *impedance3(), *reads("mat", [*RESPONSE_META, "n_freq"], "m")],
        "report": report_names("m", [*RESPONSE_META, "n_freq"]),
    },
    {
        "id": "response.MaterialResponse.from_impedance.scattering_none",
        "steps": [
            *axis3(),
            call(
                "response.MaterialResponse.from_impedance",
                "mat",
                Z_surface=seq(cplx(1.0, 0.5), cplx(2.0, -1.0), cplx(3.0, 0.0)),
                freq_axis=ref("axis"),
                material_id=strs("m1"),
                scattering_coeff=none(),
            ),
            *reads("mat", ["scattering_coeff", "boundary_model", "is_locally_reacting"], "m"),
        ],
        "report": report_names("m", ["scattering_coeff", "boundary_model", "is_locally_reacting"]),
    },
    {
        "id": "response.MaterialResponse.from_impedance.scattering_spectrum",
        "steps": [
            *axis3(),
            call(
                "response.MaterialResponse.from_impedance",
                "mat",
                Z_surface=seq(cplx(1.0, 0.0), cplx(2.0, 0.0), cplx(3.0, 0.0)),
                freq_axis=ref("axis"),
                material_id=strs("m1"),
                scattering_coeff=seq(flt(0.0), flt(0.5), flt(1.0)),
            ),
            read("mat", "scattering_coeff", "m_scattering_coeff"),
        ],
        "report": ["m_scattering_coeff"],
    },
    {
        "id": "response.MaterialResponse.from_impedance.real_impedance_keeps_real_dtype",
        "steps": [
            *axis3(),
            call(
                "response.MaterialResponse.from_impedance",
                "mat",
                Z_surface=seq(flt(1.0), flt(2.0), flt(3.0)),
                freq_axis=ref("axis"),
                material_id=strs("m1"),
            ),
            read("mat", "Z_surface", "m_Z_surface"),
        ],
        "report": ["m_Z_surface"],
    },
    {
        "id": "response.MaterialResponse.from_impedance.boundary_model_is_not_validated",
        "steps": [
            *axis3(),
            call(
                "response.MaterialResponse.from_impedance",
                "mat",
                Z_surface=seq(cplx(1.0, 0.0), cplx(2.0, 0.0), cplx(3.0, 0.0)),
                freq_axis=ref("axis"),
                material_id=strs("m1"),
                boundary_model=strs("unresolved"),
                is_locally_reacting=false(),
            ),
            *reads("mat", ["boundary_model", "is_locally_reacting"], "m"),
        ],
        "report": report_names("m", ["boundary_model", "is_locally_reacting"]),
    },
    {
        "id": "response.MaterialResponse.from_impedance.scattering_wrong_shape_raises",
        "steps": [
            *axis3(),
            call(
                "response.MaterialResponse.from_impedance",
                "mat",
                Z_surface=seq(cplx(1.0, 0.0), cplx(2.0, 0.0), cplx(3.0, 0.0)),
                freq_axis=ref("axis"),
                material_id=strs("m1"),
                scattering_coeff=seq(flt(0.1), flt(0.2)),
            ),
        ],
        "report": ["mat"],
    },
    {
        "id": "response.MaterialResponse.from_impedance.scattering_above_one_raises",
        "steps": [
            *axis3(),
            call(
                "response.MaterialResponse.from_impedance",
                "mat",
                Z_surface=seq(cplx(1.0, 0.0), cplx(2.0, 0.0), cplx(3.0, 0.0)),
                freq_axis=ref("axis"),
                material_id=strs("m1"),
                scattering_coeff=flt(1.5),
            ),
        ],
        "report": ["mat"],
    },
    {
        "id": "response.MaterialResponse.from_impedance.scattering_negative_raises",
        "steps": [
            *axis3(),
            call(
                "response.MaterialResponse.from_impedance",
                "mat",
                Z_surface=seq(cplx(1.0, 0.0), cplx(2.0, 0.0), cplx(3.0, 0.0)),
                freq_axis=ref("axis"),
                material_id=strs("m1"),
                scattering_coeff=flt(-0.1),
            ),
        ],
        "report": ["mat"],
    },
    {
        "id": "response.MaterialResponse.from_impedance.scattering_nan_raises",
        "steps": [
            *axis3(),
            call(
                "response.MaterialResponse.from_impedance",
                "mat",
                Z_surface=seq(cplx(1.0, 0.0), cplx(2.0, 0.0), cplx(3.0, 0.0)),
                freq_axis=ref("axis"),
                material_id=strs("m1"),
                scattering_coeff=seq(flt(0.1), nan(), flt(0.2)),
            ),
        ],
        "report": ["mat"],
    },
    {
        "id": "response.MaterialResponse.from_impedance.signature",
        "steps": [signature("response.MaterialResponse.from_impedance", "sig")],
        "report": ["sig"],
    },
    {
        "id": "response.MaterialResponse.pytree_flatten_roundtrip",
        "steps": [*axis3(), *impedance3(), flatten("mat", "tree")],
        "report": ["tree"],
    },
    {
        "id": "response.MaterialResponse.pytree_flatten_without_scattering",
        "steps": [
            *axis3(),
            call(
                "response.MaterialResponse.from_impedance",
                "mat",
                Z_surface=seq(cplx(1.0, 0.0), cplx(2.0, 0.0), cplx(3.0, 0.0)),
                freq_axis=ref("axis"),
                material_id=strs("m1"),
                scattering_coeff=none(),
            ),
            flatten("mat", "tree"),
        ],
        "report": ["tree"],
    },
    {
        "id": "response.MaterialResponse.tree_map_doubles_leaves_only",
        "steps": [*axis3(), *impedance3(), tree_double("mat", "doubled"), flatten("doubled", "tree")],
        "report": ["tree"],
    },
    {
        "id": "response.MaterialResponse.survives_a_jit_boundary",
        "steps": [*axis3(), *impedance3(), through_jit("mat", "out"), flatten("out", "tree")],
        "report": ["tree"],
    },
    {
        "id": "response.MaterialResponse.n_freq_matches_axis",
        "steps": [*axis3(), *impedance3(), read("mat", "n_freq", "m_n_freq")],
        "report": ["m_n_freq"],
    },
    {
        "id": "response.MaterialResponse.from_alpha.eight_octave_bands",
        "steps": [
            call(
                "response.FrequencyAxis.from_hz",
                "axis",
                freqs_hz=seq(flt(63.0), flt(500.0), flt(4000.0)),
            ),
            call(
                "response.MaterialResponse.from_alpha",
                "mat",
                alpha_bands=seq(flt(0.1), flt(0.3), flt(0.6), flt(0.8)),
                band_freqs=seq(flt(63.0), flt(250.0), flt(1000.0), flt(4000.0)),
                freq_axis=ref("axis"),
                rho_c=flt(415.0),
                material_id=strs("fixed_alpha"),
            ),
            *reads("mat", [*RESPONSE_META, "n_freq"], "m"),
        ],
        "report": report_names("m", [*RESPONSE_META, "n_freq"]),
    },
    {
        "id": "response.MaterialResponse.from_alpha.clamps_alpha_above_one",
        "steps": [
            call("response.FrequencyAxis.from_hz", "axis", freqs_hz=seq(flt(125.0), flt(1000.0))),
            call(
                "response.MaterialResponse.from_alpha",
                "mat",
                alpha_bands=seq(flt(1.2), flt(1.05)),
                band_freqs=seq(flt(125.0), flt(1000.0)),
                freq_axis=ref("axis"),
                rho_c=flt(415.0),
                material_id=strs("over_one"),
            ),
            read("mat", "Z_surface", "m_Z_surface"),
        ],
        "report": ["m_Z_surface"],
    },
    {
        # **這一筆補的是一個實測到的裁判缺口**（`independent-clamp-counterexample.json`）。
        # 把 `from_alpha` 裡「插值**之前**先把 α 夾進 [0, 1]」那一行拿掉，原本那 81 個
        # case／常數／別名**全部照樣綠**（`clamp-mutant-existing-cases.json` 實測 0 筆不同）：
        # 舊的 `clamps_alpha_above_one` 兩個頻帶都 >1、而且軸上的點就落在頻帶上，於是後面
        # `sqrt` 那道 [1e-12, 1.0] 的夾子把缺少前夾子這件事整個蓋掉了。
        # 要戳破它需要兩件事同時成立：**同一張表裡有超出兩邊的 α**（-0.1 與 1.2），而且
        # **軸上的點落在頻帶之間**（250 與 750 Hz 都在 125…1000 裡面）——這樣插值算出來的
        # 是「被汙染的鄰居」的加權，前夾子有沒有做就看得出來了。donor 實測：
        # 有前夾子 Z_real=[5732.8486328125, 1100.363037109375]，拿掉變
        # [7385.8623046875, 765.8731689453125]。
        "id": "response.MaterialResponse.from_alpha.mixed_out_of_range_alpha_inside_band_range",
        "steps": [
            call("response.FrequencyAxis.from_hz", "axis", freqs_hz=seq(flt(250.0), flt(750.0))),
            call(
                "response.MaterialResponse.from_alpha",
                "mat",
                alpha_bands=seq(flt(-0.1), flt(0.5), flt(1.2)),
                band_freqs=seq(flt(125.0), flt(500.0), flt(1000.0)),
                freq_axis=ref("axis"),
                rho_c=flt(411.6),
                material_id=strs("mixed_out_of_range"),
            ),
            *reads("mat", ["Z_surface", "n_freq", "scattering_coeff"], "m"),
        ],
        "report": report_names("m", ["Z_surface", "n_freq", "scattering_coeff"]),
    },
    {
        # 同一個缺口的另一邊：**只有下界被超出**，而且軸上的點在頻帶之間。少了前夾子，
        # 負的 α 會把插值結果拉到 0 以下，`sqrt` 那道夾子只擋得住「1-α 太小」那一側，
        # 擋不住「1-α 大於 1」這一側——所以這一筆咬的是上一筆咬不到的那個方向。
        "id": "response.MaterialResponse.from_alpha.negative_alpha_is_clamped_not_raised",
        "steps": [
            call("response.FrequencyAxis.from_hz", "axis", freqs_hz=seq(flt(300.0), flt(900.0))),
            call(
                "response.MaterialResponse.from_alpha",
                "mat",
                alpha_bands=seq(flt(-0.5), flt(0.2)),
                band_freqs=seq(flt(125.0), flt(1000.0)),
                freq_axis=ref("axis"),
                rho_c=flt(415.0),
                material_id=strs("negative_alpha"),
            ),
            read("mat", "Z_surface", "m_Z_surface"),
        ],
        "report": ["m_Z_surface"],
    },
    {
        "id": "response.MaterialResponse.from_alpha.flat_extrapolation_outside_bands",
        "steps": [
            call(
                "response.FrequencyAxis.from_hz",
                "axis",
                freqs_hz=seq(flt(20.0), flt(250.0), flt(16000.0)),
            ),
            call(
                "response.MaterialResponse.from_alpha",
                "mat",
                alpha_bands=seq(flt(0.2), flt(0.9)),
                band_freqs=seq(flt(125.0), flt(1000.0)),
                freq_axis=ref("axis"),
                rho_c=flt(415.0),
                material_id=strs("edges"),
            ),
            read("mat", "Z_surface", "m_Z_surface"),
        ],
        "report": ["m_Z_surface"],
    },
    {
        "id": "response.MaterialResponse.from_alpha.single_band_is_flat",
        "steps": [
            call("response.FrequencyAxis.from_hz", "axis", freqs_hz=seq(flt(63.0), flt(8000.0))),
            call(
                "response.MaterialResponse.from_alpha",
                "mat",
                alpha_bands=seq(flt(0.5)),
                band_freqs=seq(flt(1000.0)),
                freq_axis=ref("axis"),
                rho_c=flt(415.0),
                material_id=strs("one_band"),
            ),
            read("mat", "Z_surface", "m_Z_surface"),
        ],
        "report": ["m_Z_surface"],
    },
    {
        "id": "response.MaterialResponse.from_alpha.scattering_none",
        "steps": [
            call("response.FrequencyAxis.from_hz", "axis", freqs_hz=seq(flt(125.0), flt(1000.0))),
            call(
                "response.MaterialResponse.from_alpha",
                "mat",
                alpha_bands=seq(flt(0.2), flt(0.4)),
                band_freqs=seq(flt(125.0), flt(1000.0)),
                freq_axis=ref("axis"),
                rho_c=flt(415.0),
                material_id=strs("no_s"),
                scattering_coeff=none(),
            ),
            read("mat", "scattering_coeff", "m_scattering_coeff"),
        ],
        "report": ["m_scattering_coeff"],
    },
    {
        "id": "response.MaterialResponse.from_alpha.length_mismatch_raises",
        "steps": [
            call("response.FrequencyAxis.from_hz", "axis", freqs_hz=seq(flt(125.0), flt(1000.0))),
            call(
                "response.MaterialResponse.from_alpha",
                "mat",
                alpha_bands=seq(flt(0.2), flt(0.4), flt(0.6)),
                band_freqs=seq(flt(125.0), flt(1000.0)),
                freq_axis=ref("axis"),
                rho_c=flt(415.0),
                material_id=strs("bad"),
            ),
        ],
        "report": ["mat"],
    },
    {
        "id": "response.MaterialResponse.from_alpha.two_dimensional_input_raises",
        "steps": [
            call("response.FrequencyAxis.from_hz", "axis", freqs_hz=seq(flt(125.0), flt(1000.0))),
            call(
                "response.MaterialResponse.from_alpha",
                "mat",
                alpha_bands=seq(seq(flt(0.2), flt(0.4))),
                band_freqs=seq(seq(flt(125.0), flt(1000.0))),
                freq_axis=ref("axis"),
                rho_c=flt(415.0),
                material_id=strs("bad"),
            ),
        ],
        "report": ["mat"],
    },
    {
        "id": "response.MaterialResponse.from_alpha.empty_bands_raise",
        "steps": [
            call("response.FrequencyAxis.from_hz", "axis", freqs_hz=seq(flt(125.0), flt(1000.0))),
            call(
                "response.MaterialResponse.from_alpha",
                "mat",
                alpha_bands=seq(),
                band_freqs=seq(),
                freq_axis=ref("axis"),
                rho_c=flt(415.0),
                material_id=strs("bad"),
            ),
        ],
        "report": ["mat"],
    },
    {
        "id": "response.MaterialResponse.from_alpha.non_finite_alpha_raises",
        "steps": [
            call("response.FrequencyAxis.from_hz", "axis", freqs_hz=seq(flt(125.0), flt(1000.0))),
            call(
                "response.MaterialResponse.from_alpha",
                "mat",
                alpha_bands=seq(flt(0.2), nan()),
                band_freqs=seq(flt(125.0), flt(1000.0)),
                freq_axis=ref("axis"),
                rho_c=flt(415.0),
                material_id=strs("bad"),
            ),
        ],
        "report": ["mat"],
    },
    {
        "id": "response.MaterialResponse.from_alpha.non_finite_band_raises",
        "steps": [
            call("response.FrequencyAxis.from_hz", "axis", freqs_hz=seq(flt(125.0), flt(1000.0))),
            call(
                "response.MaterialResponse.from_alpha",
                "mat",
                alpha_bands=seq(flt(0.2), flt(0.4)),
                band_freqs=seq(flt(125.0), plus_inf()),
                freq_axis=ref("axis"),
                rho_c=flt(415.0),
                material_id=strs("bad"),
            ),
        ],
        "report": ["mat"],
    },
    {
        "id": "response.MaterialResponse.from_alpha.non_ascending_bands_raise",
        "steps": [
            call("response.FrequencyAxis.from_hz", "axis", freqs_hz=seq(flt(125.0), flt(1000.0))),
            call(
                "response.MaterialResponse.from_alpha",
                "mat",
                alpha_bands=seq(flt(0.2), flt(0.4)),
                band_freqs=seq(flt(1000.0), flt(125.0)),
                freq_axis=ref("axis"),
                rho_c=flt(415.0),
                material_id=strs("bad"),
            ),
        ],
        "report": ["mat"],
    },
    {
        "id": "response.MaterialResponse.from_alpha.signature",
        "steps": [signature("response.MaterialResponse.from_alpha", "sig")],
        "report": ["sig"],
    },
]

