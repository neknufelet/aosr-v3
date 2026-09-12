"""票 #134：**上一代那幾條考卷的承接**（一條一條對得回去）。

兩個來源，處置不一樣：

1. donor 的 ``test_response`` 那一份（4 題）——**整檔帶得走**的那一份。這裡逐題重寫成新家的
   import，**每一個數值與 pytree 的期待一個字都沒放鬆**。其中
   ``assert len(leaves) == 3`` 那一條是「把數量鎖死」的寫法（規矩卡
   ``assertions-not-pinned-to-counts`` 咬它），**不是直接刪掉**：改成逐片葉子的具名結構
   對照（哪一片是什麼、值多少、dtype 與 shape 是什麼），那比數量嚴格——多一片、少一片、
   換一片都紅，而且說得出是哪一片。
2. donor 的 ``test_m13_material_source`` 那一份——同檔還 import 了 geometry／physics，
   **整檔帶不走**，所以抽出它那幾條**直接驗 registry／source** 的斷言。根對話已在唯讀
   donor 實跑過那五支具名測試（5 passed、exit 0）。這裡承接的是它們的行為，其中
   ``registry.get()`` 回**同一個物件**（身分，不是等值）這一條，標準答案的編碼表達不了
   ——編碼比的是值，身分只有這裡驗得到。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from aosr.materials.registry import MaterialRegistry, constant_impedance, rigid_wall
from aosr.materials.response import FrequencyAxis, MaterialResponse
from aosr.materials.source import MaterialEntry


def _spectrum(mat: MaterialResponse) -> jax.Array:
    """收窄散射頻譜（靜態型別是 ``Array | None``；這幾題的對象一定有值）。"""
    spectrum = mat.scattering_coeff
    assert spectrum is not None, "這一份回應沒有散射頻譜——這一題挑錯對象"
    return spectrum


def _make() -> MaterialResponse:
    """donor ``test_response`` 的 ``_make()``，逐字同一組輸入。"""
    fa = FrequencyAxis.from_hz(jnp.array([100.0, 200.0, 400.0]), resolution="custom")
    impedance = jnp.array([1.0 + 0.5j, 2.0 - 1.0j, 3.0 + 0.0j])
    return MaterialResponse.from_impedance(
        impedance, fa, material_id="m1", boundary_model="local_impedance"
    )


def test_frequency_axis_properties() -> None:
    """donor 同名那一題：三個屬性的值逐個照抄。"""
    fa = FrequencyAxis.from_hz(jnp.array([20.0, 50.0, 500.0]))
    assert fa.n_freq == 3
    assert fa.f_min == 20.0
    assert fa.f_max == 500.0


def test_pytree_roundtrip_preserves_data_and_metadata() -> None:
    """donor 同名那一題，**把數量斷言換成逐片葉子的結構對照**。

    donor 寫的是 ``assert len(leaves) == 3``（Z_surface、freq_axis.freqs_hz、散射頻譜）。
    這裡改成「哪幾片、各是什麼形狀與 dtype、值是什麼」——比數量嚴格：少一片、多一片、
    順序換掉、dtype 掉精度，都會在這裡紅，而且訊息說得出是哪一片。
    """
    mat = _make()
    leaves, treedef = jax.tree_util.tree_flatten(mat)
    shapes = [(str(leaf.dtype), tuple(leaf.shape)) for leaf in leaves]
    assert shapes == [("complex64", (3,)), ("float32", (3,)), ("float32", (3,))]
    assert jnp.allclose(leaves[0], mat.Z_surface)
    assert jnp.allclose(leaves[1], mat.freq_axis.freqs_hz)
    assert jnp.allclose(leaves[2], _spectrum(mat))

    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.material_id == "m1"
    assert rebuilt.boundary_model == "local_impedance"
    assert rebuilt.is_locally_reacting is True
    assert jnp.allclose(rebuilt.Z_surface, mat.Z_surface)
    assert jnp.allclose(rebuilt.freq_axis.freqs_hz, mat.freq_axis.freqs_hz)
    assert jnp.allclose(_spectrum(rebuilt), _spectrum(mat))


def test_tree_map_transforms_leaves() -> None:
    """donor 同名那一題：葉子乘二、靜態中介資料不動。"""
    mat = _make()
    doubled = jax.tree_util.tree_map(lambda x: x * 2, mat)
    assert jnp.allclose(doubled.Z_surface, mat.Z_surface * 2)
    assert jnp.allclose(doubled.freq_axis.freqs_hz, mat.freq_axis.freqs_hz * 2)
    assert jnp.allclose(_spectrum(doubled), _spectrum(mat) * 2)
    assert doubled.material_id == "m1"


def test_n_freq_matches_axis() -> None:
    """donor 同名那一題。"""
    mat = _make()
    assert mat.n_freq == 3


# ── donor 的 test_m13_material_source 抽出來的那幾條 ────────────────────────
def _axis() -> FrequencyAxis:
    """兩點軸（下面幾條共用）。"""
    return FrequencyAxis.from_hz(jnp.array([125.0, 1000.0]))


def test_register_default_source_kind_is_analytic() -> None:
    """沒給出身的新材料預設是 ``analytic``，而且 ``source_ref`` 是空的。"""
    registry = MaterialRegistry()
    registry.register(rigid_wall(_axis(), material_id="wall"))
    assert registry.source_kind("wall") == "analytic"
    assert registry.entry("wall").source_ref is None


def test_provenance_round_trip_register_entry_and_source_kind() -> None:
    """``register_entry`` 放進去的出身，``entry``／``source_kind`` 要原樣拿得回來。"""
    response = constant_impedance(_axis(), 800 + 0j, material_id="felt")
    registry = MaterialRegistry()
    registry.register_entry(
        MaterialEntry(response=response, source_kind="measured_alpha", source_ref="datasheet:acme-7")
    )
    assert registry.source_kind("felt") == "measured_alpha"
    assert registry.entry("felt").source_ref == "datasheet:acme-7"
    assert registry.ids() == ["felt"]


def test_get_returns_the_same_response_object() -> None:
    """``get()`` 回的是**同一個物件**，不是一份拷貝。

    這一條標準答案驗不到：答案檔比的是值，身分（``is``）編不進 JSON。donor 那一支考卷
    有這一條，所以它留在這裡。
    """
    response = constant_impedance(_axis(), 800 + 0j, material_id="felt")
    registry = MaterialRegistry()
    registry.register(response, source_kind="computed_tmm")
    assert registry.get("felt") is response
    assert registry.entry("felt").response is response


def test_overwrite_without_source_kind_preserves_existing_provenance() -> None:
    """隱式覆寫：kind 與 ref 都留著（不准被悄悄洗成 ``analytic``）。"""
    response = constant_impedance(_axis(), 800 + 0j, material_id="felt")
    registry = MaterialRegistry()
    registry.register(response, source_kind="measured_alpha", source_ref="datasheet:acme-7")
    registry.register(response, overwrite=True)
    assert registry.source_kind("felt") == "measured_alpha"
    assert registry.entry("felt").source_ref == "datasheet:acme-7"


def test_explicit_source_kind_change_does_not_carry_stale_source_ref() -> None:
    """明著給 kind：舊的 ref **不繼承**（它描述的是舊出身）。"""
    response = constant_impedance(_axis(), 800 + 0j, material_id="felt")
    registry = MaterialRegistry()
    registry.register(response, source_kind="measured_alpha", source_ref="datasheet:acme-7")
    registry.register(response, source_kind="computed_tmm", overwrite=True)
    assert registry.source_kind("felt") == "computed_tmm"
    assert registry.entry("felt").source_ref is None


def test_registry_entry_overwrite_and_unknown_errors_regression() -> None:
    """重複 id 不給 ``overwrite`` 要 ``KeyError``；不存在的 id 也是，訊息帶已知清單。"""
    response = constant_impedance(_axis(), 800 + 0j, material_id="felt")
    registry = MaterialRegistry()
    registry.register_entry(MaterialEntry(response=response, source_kind="analytic"))
    with pytest.raises(KeyError, match="already registered"):
        registry.register_entry(MaterialEntry(response=response, source_kind="analytic"))
    registry.register_entry(
        MaterialEntry(response=response, source_kind="computed_tmm"), overwrite=True
    )
    assert registry.source_kind("felt") == "computed_tmm"
    with pytest.raises(KeyError, match="unknown material 'nope'"):
        registry.entry("nope")
