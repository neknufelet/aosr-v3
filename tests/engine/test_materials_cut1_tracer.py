"""票 #134：``MaterialResponse.__post_init__`` 那個 **tracer 分支**的正反例。

那段守門刻意分成兩半，理由不一樣（程式碼的說明寫在
:meth:`aosr.materials.response.MaterialResponse.__post_init__`）：

* **形狀**永遠驗——編譯期也驗得了（``shape`` 是靜態的）。
* **範圍與有限性**只在**不是 tracer**的時候驗。tracer 是 JAX 在編譯期用來代替真值的替身；
  要它交出具體的數字等於強迫當場算一次，本來合法的 pytree 變換會因此爆掉。

這一支把兩邊都釘住：eager 的非法散射**仍然**被擋（沒有因為多了 tracer 這條路就放寬），
合法的 pytree 穿得過 ``jit``，而且穿過去的時候**沒有人去 concretize 那些 traced 資料**。
標準答案那一邊蓋不到這一面（答案檔裡的每一筆都是 eager 跑出來的）。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from aosr.materials.response import FrequencyAxis, MaterialResponse


def _spectrum(mat: MaterialResponse) -> jax.Array:
    """收窄散射頻譜（靜態型別是 ``Array | None``；這幾題的對象一定有值）。"""
    spectrum = mat.scattering_coeff
    assert spectrum is not None, "這一份回應沒有散射頻譜——這一題挑錯對象"
    return spectrum


def _axis() -> FrequencyAxis:
    """三點軸。"""
    return FrequencyAxis.from_hz(jnp.array([100.0, 200.0, 400.0]))


def _impedance() -> jax.Array:
    """三點複數阻抗。"""
    return jnp.array([1.0 + 0.5j, 2.0 - 1.0j, 3.0 + 0.0j])


@pytest.mark.parametrize(
    ("name", "spectrum"),
    [
        ("大於 1", [0.1, 1.5, 0.2]),
        ("小於 0", [0.1, -0.2, 0.3]),
        ("不是有限值", [0.1, float("nan"), 0.3]),
    ],
)
def test_eager_still_rejects_an_illegal_spectrum(name: str, spectrum: list[float]) -> None:
    """eager 那條路上，非法的散射頻譜**照樣**被擋（`name` 是哪一種非法）。"""
    with pytest.raises(ValueError, match=r"scattering_coeff must be in \[0, 1\] and finite"):
        MaterialResponse.from_impedance(
            _impedance(),
            _axis(),
            material_id="m1",
            scattering_coeff=jnp.array(spectrum),
        )


def test_a_legal_response_survives_a_jit_boundary() -> None:
    """合法的 pytree 穿得過 ``jit``，穿完葉子與靜態中介資料都在。"""
    mat = MaterialResponse.from_impedance(_impedance(), _axis(), material_id="m1")

    @jax.jit
    def identity(value: MaterialResponse) -> MaterialResponse:
        return value

    out = identity(mat)
    assert out.material_id == "m1"
    assert out.boundary_model == "local_impedance"
    assert jnp.allclose(out.Z_surface, mat.Z_surface)
    assert jnp.allclose(_spectrum(out), _spectrum(mat))


def test_the_range_gate_does_not_concretize_traced_data() -> None:
    """編譯期重建這個容器時，範圍那一道**不准**去要具體的值。

    這一條是那個分支存在的理由本身：``jit`` 底下的葉子是 tracer，範圍檢查如果照跑，
    就會是 ``ConcretizationTypeError``。這裡在編譯期真的重建一次容器——會炸就當場紅。
    """

    @jax.jit
    def rebuild(spectrum: jax.Array) -> jax.Array:
        mat = MaterialResponse(
            Z_surface=_impedance(),
            freq_axis=_axis(),
            scattering_coeff=spectrum,
            material_id="traced",
        )
        return _spectrum(mat)

    # 值刻意落在合法範圍**外面**：eager 會被擋（上面那幾條），編譯期不准去看它的值，
    # 所以這一呼叫必須成功——它同時證明「範圍那一道真的沒有在編譯期跑」。
    out = rebuild(jnp.array([0.1, 1.5, 0.2]))
    assert tuple(out.shape) == (3,)
    assert str(out.dtype) == "float32"


def test_the_shape_gate_still_bites_under_jit() -> None:
    """形狀那一道**編譯期也要咬**：``shape`` 是靜態的，沒有理由放過它。"""

    @jax.jit
    def rebuild(spectrum: jax.Array) -> jax.Array:
        mat = MaterialResponse(
            Z_surface=_impedance(),
            freq_axis=_axis(),
            scattering_coeff=spectrum,
            material_id="traced",
        )
        return _spectrum(mat)

    with pytest.raises(ValueError, match=r"scattering_coeff must have shape \(3,\), got \(2,\)"):
        rebuild(jnp.array([0.1, 0.2]))
