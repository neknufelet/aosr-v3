"""綁定表（#1206 第 1 支，左右對稱的計算那半邊）考卷：成對兩片從開跑到結果檔逐位相等、成對算一個單位、空表逐位不變。

場景全部自編（``_material_gradient_ties_probe``）：八片，交給搜尋的是 y0@a、y0@b、yL@a 三片，y0@a 與 yL@a 成對。
θ 六格照面序：y0@a 厚度、空腔（0、1），y0@b（2、3），yL@a（4、5）；成對的格是 (0, 4) 與 (1, 5)。

- 父行程（單精度、合成目標，不開子行程）：對稱目標綁與不綁逐位相同、不對稱目標綁了每步相等而且單調、平均不是相加、
  開跑前每一種拒收、候選與換樣板成對動、預算照單位算、不變量在每一個檢查點都會擋。
- 雙精度子行程（探針，整條產品路徑）：綁的那一場兩本帳每一列成對相等、換樣板一對一起換、帳本重播與接續、
  竄改一個最小浮點單位後接續會拋錯；不綁的那一場四個檔的位元組等於加綁定表之前的主線。

新名字一律在各題裡面才匯入：改產品程式之前每一題各自紅，不是整檔收不進來。
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from aosr.config.material_gradient import MOUNT_ON_WALL, MaterialGradientSettings
from aosr.gradient import material_gradient as search
from aosr.gradient import material_gradient_ledger as ledger
from aosr.gradient import material_gradient_loop as loop
from aosr.gradient import run
from aosr.gradient import template_ledger as templates
from aosr.gradient import template_rounds as rounds
from aosr.gradient.fixed_surfaces import designs_with_max_depths, expanded_fixed_surfaces
from aosr.materials.construction_models import ConstantImpedance
from aosr.runtime import child_process_env, configure_jax
from aosr.scoring.gradient_objective_jax import GradientObjectiveArrays
from tests.engine import _material_gradient_ties_probe as probe
from tests.engine._cpu_pins import forward_cpu_pins
from tests.engine._scheme_cache import shared_json

Record = dict[str, object]
Ties = tuple[tuple[str, str], ...]
TIES: Ties = probe.TIES
CELLS = ((0, 4), (1, 5))
GAP50 = probe.GAP50
# 不給綁定表那一場（零目標、假時鐘）四個檔與資料夾名：改產品程式之前在主線 7d73a450 用同一支探針量的。
# 零目標的每一格都是精確的浮點數，不含求解，位元組跟處理器無關。
MAIN_PLAIN_ID = "2cd514b43a544bba9f235517b56485a2"
MAIN_PLAIN_DIGESTS = {
    "project.json": "bf188a0935359f21186e7d496231f3818ae5206390d63b27683d301fa5a5cd7a",
    "ledger.jsonl": "55b44499515ee10924aa66ba48033bdf07e18dae473ff16744edf8128c2557a8",
    "templates.jsonl": "7690d66ddc3a8031e3771c67872e3f7bc4194cd6bc698fc851798aa0d10c849a",
    "result.json": "0181ca21af5d91b9af08bba0f1758534340f11a07032495383fa1d80ef985b78"}


# ---- 共用小工具 ----

def _aux(value: jax.Array) -> GradientObjectiveArrays:
    return GradientObjectiveArrays(value[None], value[None], value, value, value, value, value, value)


def _quadratic(weights: Sequence[float], targets: Sequence[float], cross: float) -> search.ChainObjective:
    """Σ w·(θ − 目標)² ＋ cross·Σ θ·θ[對調]：對調是把成對兩片的格互換。權重與目標成對相同時，目標對「兩片互換」完全對稱。"""
    swap = np.asarray([4, 5, 2, 3, 0, 1])

    def objective(theta: jax.Array) -> tuple[jax.Array, GradientObjectiveArrays]:
        shift = theta - jnp.asarray(targets, dtype=theta.dtype)
        value = jnp.sum(jnp.asarray(weights, dtype=theta.dtype) * shift * shift) + cross * jnp.sum(theta * theta[swap])
        return value, _aux(value)

    return cast(search.ChainObjective, objective)


SYMMETRIC = ((3.0, 10.0, 5.0, 10.0, 3.0, 10.0), (0.2, 0.08, 0.2, 0.07, 0.2, 0.08))
LOPSIDED = ((3.0, 10.0, 5.0, 10.0, 7.0, 10.0), (0.2, 0.06, 0.2, 0.07, 0.12, 0.09))


def _settings() -> MaterialGradientSettings:
    """探針那份自編設定，下降多走幾步（這裡考的是每一步，不是停在哪）。"""
    document = cast(dict[str, dict[str, object]], dict(probe.SETTINGS))
    return MaterialGradientSettings.model_validate({**document, "descent": {**document["descent"], "max_steps": 8}})


def _start() -> NDArray[np.float32]:
    return search.baseline_theta(probe.designs()).astype(np.float32)


def _descend(objective: search.ChainObjective, cells: Sequence[tuple[int, int]]) -> search.DescentResult:
    designs = probe.designs()
    extra: Record = {"tied_cells": cells} if cells else {}
    return search.projected_descent(objective, _start(), search.design_bounds(designs), _settings().descent,
                                    limits=search.depth_limits(designs), **extra)  # type: ignore[arg-type]


def _snap_and_retune(objective: search.ChainObjective, theta: NDArray[np.float32],
                     ties: Ties) -> tuple[search.SnapResult, search.DescentResult]:
    designs = probe.designs()
    extra: Record = {"ties": ties} if ties else {}
    snapped = search.snap_to_specs(theta, designs, **extra)  # type: ignore[arg-type]
    return snapped, search.retune_cavities(objective, snapped.theta, designs, _settings(), **extra)  # type: ignore[arg-type]


def _paired(theta: Sequence[float]) -> bool:
    """成對的格逐位相等（比 float.hex，+0 與 −0 也分得出來）。"""
    return all(float(theta[first]).hex() == float(theta[second]).hex() for first, second in CELLS)


def _one_ulp_off(theta: NDArray[np.floating], cell: int = 5) -> NDArray[np.floating]:
    broken = np.asarray(theta).copy()
    broken[cell] = np.nextafter(broken[cell], broken.dtype.type(1.0))
    return broken


def _snapped() -> NDArray[np.float64]:
    designs = probe.designs()
    return np.asarray(search.snap_to_specs(search.baseline_theta(designs), designs).theta, dtype=np.float64)


def _trials(settings: MaterialGradientSettings) -> int:
    descent = settings.descent
    return sum(1 for power in range(200) if descent.initial_step * descent.backtrack_ratio ** power >= descent.min_step)


# ---- 1 對稱目標：綁與不綁逐位相同 ----

def test_symmetric_objective_tied_descent_is_bitwise_the_untied_one() -> None:
    """抓平均寫錯（兩個相同的值平均不等於自己）、綁定改到了別的格：對稱目標、對稱起點，綁與不綁每一步 θ 逐位相同，
    取規格與空腔重調之後也相同。"""
    objective = _quadratic(*SYMMETRIC, cross=2.0)
    free, tied = _descend(objective, ()), _descend(objective, CELLS)
    assert len(free.steps) > 2 and all(_paired(step.theta) for step in free.steps)
    assert [step.theta for step in tied.steps] == [step.theta for step in free.steps]
    assert tied.theta.tobytes() == free.theta.tobytes() and tied.stop_reason == free.stop_reason
    free_snap, free_retune = _snap_and_retune(objective, free.theta, ())
    tied_snap, tied_retune = _snap_and_retune(objective, tied.theta, TIES)
    assert tied_snap.theta.tobytes() == free_snap.theta.tobytes()
    assert len(free_retune.steps) > 1 and free_retune.steps[-1].theta != free_retune.steps[0].theta
    assert [step.theta for step in tied_retune.steps] == [step.theta for step in free_retune.steps]


# ---- 2 不對稱目標：不綁會分開，綁了每一步相等、值單調不增 ----

def test_lopsided_objective_splits_untied_and_stays_paired_and_monotone_tied() -> None:
    """抓綁定沒作用（或只綁了厚度沒綁空腔）、綁了之後線搜尋用錯梯度讓值上升。先確認這一題不綁真的會分開。"""
    objective = _quadratic(*LOPSIDED, cross=0.0)
    free, tied = _descend(objective, ()), _descend(objective, CELLS)
    assert free.theta[0] != free.theta[4] and free.theta[1] != free.theta[5]
    assert len(tied.steps) > 2 and tied.steps[-1].theta != tied.steps[0].theta
    assert all(_paired(step.theta) for step in tied.steps)
    values = [step.value for step in tied.steps]
    assert all(later <= earlier for earlier, later in zip(values, values[1:], strict=False))
    snapped, retune = _snap_and_retune(objective, tied.theta, TIES)
    assert _paired(snapped.theta) and all(_paired(step.theta) for step in retune.steps)
    assert len(retune.steps) > 1 and retune.steps[-1].theta[1] != retune.steps[0].theta[1]
    retuned = [step.value for step in retune.steps]
    assert all(later <= earlier for earlier, later in zip(retuned, retuned[1:], strict=False))
    free_snap, free_retune = _snap_and_retune(objective, free.theta, ())
    assert free_retune.theta[1] != free_retune.theta[5] and free_snap.theta[0] == free_snap.theta[4]


# ---- 3 平均不是相加 ----

def test_tied_step_uses_the_average_gradient_not_the_sum() -> None:
    """抓把平均寫成相加（步長變兩倍）：線性目標 3·θ₀＋1·θ₄，成對兩格的梯度是 3 與 1、平均 2。
    盒界放寬、不設總深度，第一個試步一定過（線性目標恰在 Armijo 線下），位移就是 起始步長×2，手算逐位相等；
    同一步的投影梯度範數是 √(2²＋2²)（不綁是 √(3²＋1²)）。"""
    slope = jnp.asarray([3.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=jnp.float32)

    def objective(theta: jax.Array) -> tuple[jax.Array, GradientObjectiveArrays]:
        value = jnp.sum(slope * theta)
        return value, _aux(value)

    settings = _settings().descent.model_copy(update={"max_steps": 1})
    start = np.full(6, 0.5, dtype=np.float32)
    bounds = (np.full(6, -10.0), np.full(6, 10.0))
    result = search.projected_descent(objective, start, bounds, settings, limits=(), tied_cells=((0, 4),))
    moved = np.float32(0.5) - np.float32(settings.initial_step) * np.float32(2.0)
    assert result.steps[1].step_size == settings.initial_step
    assert result.theta[0].tobytes() == moved.tobytes() == result.theta[4].tobytes()
    assert result.theta[[1, 2, 3, 5]].tobytes() == start[[1, 2, 3, 5]].tobytes()
    assert result.steps[0].gradient_norm == pytest.approx(math.sqrt(8.0), rel=1e-6)
    untied = search.projected_descent(objective, start, bounds, settings, limits=())
    assert untied.steps[0].gradient_norm == pytest.approx(math.sqrt(10.0), rel=1e-6)
    assert untied.theta[0] != untied.theta[4]


# ---- 4 開跑前拒收 ----

def _refused(ties: Ties, designs: Sequence[search.SurfaceDesign],
             allowed: dict[str, tuple[str, ...]] | None = None) -> str:
    from aosr.gradient.ties import check_ties_against_start

    with pytest.raises(ValueError) as caught:
        check_ties_against_start(ties, designs, allowed or {})
    return str(caught.value)


def _changed(name: str, **changes: object) -> tuple[search.SurfaceDesign, ...]:
    return tuple(dataclasses.replace(design, **changes) if design.surface_id == name else design  # type: ignore[arg-type]
                 for design in probe.designs())


def test_matching_pair_is_accepted_and_stored_in_one_order() -> None:
    """抓 A=B 與 B=A 存成兩種（同一張表兩個資料夾）；成對兩片完全相同時不拒收。"""
    from aosr.gradient.ties import check_ties_against_start, checked_ties, parse_ties

    assert parse_ties(["yL@a=y0@a"]) == parse_ties(["y0@a=yL@a"]) == TIES
    assert checked_ties([("yL", "y0"), ("xL", "x0")]) == (("x0", "xL"), ("y0", "yL"))
    assert parse_ties([]) == ()
    check_ties_against_start(TIES, probe.designs(), {})
    check_ties_against_start((), probe.designs(), {})


@pytest.mark.parametrize(("values", "words"), [
    (["y0@a"], "要寫成"), (["=yL@a"], "要寫成"), (["y0@a=yL@a=y0@b"], "要寫成"),
    (["y0@a=y0@a"], "兩片相同"), (["y0@a=yL@a", "yL@a=y0@a"], "給了兩次"),
    (["y0@a=yL@a", "y0@a=y0@b"], "y0@a 同時在兩對"), (["y0@a=wall9"], "面片代號不認識")])
def test_tie_flag_refuses_bad_shape_self_pair_repeat_and_shared_surface(values: list[str], words: str) -> None:
    """抓旗標格式錯、自己配自己、同一對給兩次、同一片在兩對裡還照收。"""
    from aosr.gradient.ties import parse_ties

    with pytest.raises(ValueError, match=words):
        parse_ties(values)


def test_tie_refuses_surface_that_is_not_in_this_search() -> None:
    assert "片 yL@c 不在這一場的面片清單" in _refused((("y0@a", "yL@c"),), probe.designs())


def test_tie_refuses_surface_not_handed_to_the_search() -> None:
    """交給搜尋的片＝不是固定面的片；yL@b 在這個場景是固定面。"""
    message = _refused((("y0@b", "yL@b"),), probe.designs())
    assert "綁定 y0@b=yL@b" in message and "yL@b 是固定面，不是交給搜尋的片" in message


def test_tie_refuses_different_family_and_start_template() -> None:
    family = _refused(TIES, run.designs_for((None, None, None, None, GAP50, GAP50, "glass_wool_24k_50mm", None),
                                            probe.settings(), kept=probe.kept_patches(), fixed_surfaces=probe.FIXED))
    assert "綁定 y0@a=yL@a" in family and "家族不同" in family and "glass_wool_24k" in family
    template = _refused(TIES, run.designs_for((None, None, None, None, GAP50, GAP50, "glass_wool_48k_50mm", None),
                                              probe.settings(), kept=probe.kept_patches(), fixed_surfaces=probe.FIXED))
    assert "綁定 y0@a=yL@a" in template and "起跑樣板不同" in template and "glass_wool_48k_50mm" in template


def test_tie_refuses_different_bounds_depth_cap_mount_and_allowed_list() -> None:
    """裝法與 θ 上下界從命令列做不出不同（一場一種裝法、界只跟家族走），所以直接改設計物件考這兩條。"""
    bounds = _refused(TIES, _changed("yL@a", lower=(0.05, 0.01)))
    assert "綁定 y0@a=yL@a" in bounds and "θ 上下界不同" in bounds
    capped = designs_with_max_depths(probe.designs(), probe.settings(), {"yL@a": 0.2})
    assert "深度上限不同" in _refused(TIES, capped)
    wall = ConstantImpedance(impedance_pa_s_m=31000.0)
    assert "裝法不同" in _refused(TIES, _changed("yL@a", mount=MOUNT_ON_WALL, backing_material=wall))
    listed = _refused(TIES, probe.designs(), {"yL@a": ("glass_wool_48k", "none")})
    assert "准用清單不同" in listed and "綁定 y0@a=yL@a" in listed
    same = {"y0@a": ("glass_wool_48k", "none"), "yL@a": ("glass_wool_48k", "none")}
    from aosr.gradient.ties import check_ties_against_start

    check_ties_against_start(TIES, probe.designs(), same)


def test_command_line_takes_the_tie_flag_and_refuses_before_loading_anything(tmp_path: Path) -> None:
    """抓旗標沒進說明、沒接到開跑：格式錯在讀方案之前就拒收（方案與替身的路徑根本不存在）。"""
    from aosr.gradient import cli

    parser = cli._parser()
    arguments = ["start", "--scheme", str(tmp_path / "none.json"), "--surrogate", str(tmp_path / "none.npz"),
                 "--presets", ",".join(["none"] * 6), "--purpose", "music", "--root", str(tmp_path / "root")]
    start = next(action for action in parser._actions if isinstance(action.choices, dict)).choices["start"]
    assert "--tie" in start.format_help() and "成對" in start.format_help()
    assert parser.parse_args(arguments).tie == []
    args = parser.parse_args([*arguments, "--tie", "y0@a=yL@a", "--tie", "y0@a=y0@b"])
    with pytest.raises(ValueError, match="y0@a 同時在兩對"):
        cli._start(args)
    assert not (tmp_path / "root").exists()


def test_start_designs_refuses_a_mismatched_tie() -> None:
    """抓開跑那一條路沒叫拒收：用自編分區方案走 ``start_designs``（開跑與建替身共用的那一支）。"""
    from aosr.gradient import cli
    from aosr.reporting.scheme import Scheme
    from tests.engine._directivity import DIRECTIVITY
    from tests.engine._furniture_cases import CAPABILITIES
    from tests.engine._surface_zone_cases import rectangle_document

    scheme = Scheme.model_validate(rectangle_document())
    names = [search_name for search_name in (design.surface_id for design in run.designs_for(
        (None,) * 21, probe.settings(), kept=run.scene_wall_materials(scheme, capabilities=CAPABILITIES,
                                                                      directivity=DIRECTIVITY)))]
    first, second = names[1], names[2]
    presets = tuple(GAP50 if name == first else "glass_wool_24k_50mm" if name == second else None for name in names)
    given = dict(mount="rigid", fixed={}, caps={}, allowed={}, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    with pytest.raises(ValueError, match=f"綁定 {first}={second}：家族不同"):
        cli.start_designs(scheme, probe.settings(), run.presets_table(), presets, ties=((first, second),), **given)
    same = tuple(GAP50 if name in (first, second) else None for name in names)
    cli.start_designs(scheme, probe.settings(), run.presets_table(), same, ties=((first, second),), **given)


# ---- 5 成對算一個單位：候選、跳點、換樣板、預算 ----

def test_pair_gives_one_set_of_candidates_and_jumps_together() -> None:
    """抓各片各出候選、跳點只跳一片：三片裡兩片成對，候選是兩個單位（代號用面序在前的 y0@a），不是三片。"""
    designs, settings, theta = probe.designs(), probe.settings(), _snapped()
    slope = np.asarray([-3.0, 0.0, -1.0, 0.0, -5.0, 0.0])
    free = loop.candidates_from_gradient(theta, slope, designs, settings)
    assert [item.candidate_id for item in free] == ["yL@a:thicker", "y0@a:thicker", "y0@b:thicker"]
    tied = loop.candidates_from_gradient(theta, slope, designs, settings, ties=TIES)
    assert [item.candidate_id for item in tied] == ["y0@a:thicker", "y0@b:thicker"]
    pair = tied[0]
    assert pair.delta[0] == pair.delta[4] == 0.1 - 0.05 and pair.delta[2] == 0.0
    assert pair.predicted_change == float(np.dot(slope, np.asarray(pair.delta)))
    jumped = loop.apply_candidate(theta, pair, designs, ties=TIES)
    assert _paired(jumped) and jumped[0] == 0.1 and jumped[2] == 0.05
    alone = loop.apply_candidate(theta, tied[1], designs, ties=TIES)
    assert _paired(alone) and alone[2] == 0.1 and alone[0] == 0.05


def test_round_budget_counts_units_not_surfaces() -> None:
    """抓預算照片數算：不開輪（剩 0 次）不求值，只看開輪前算的最壞求值數。兩個單位是 1＋3×(1＋K×試探數)，三片是 1＋4×(…)。"""
    def never(_: jax.Array) -> tuple[jax.Array, GradientObjectiveArrays]:
        raise AssertionError("這一題不該求值")

    designs, settings, theta = probe.designs(), probe.settings(), _snapped()
    segment = 1 + settings.retune.steps * _trials(settings)
    free = loop.confirmation_round(never, theta, designs, settings, budget_left=0)
    tied = loop.confirmation_round(never, theta, designs, settings, budget_left=0, ties=TIES)
    assert (free.bound, tied.bound) == (1 + 4 * segment, 1 + 3 * segment)
    assert not tied.opened and tied.evaluations == 0


def _context(ties: Ties) -> rounds.TemplateContext:
    return probe.problem("zero", ties).templates


def test_pair_is_one_template_unit_and_swaps_together() -> None:
    """抓換樣板各片各列、只換一片、預算照片數：成對只列面序在前那一片的目標；換過去兩片同一個樣板、θ 相等；
    下一輪最壞求值數照單位（6 個快速評分）不是照片（9 個）。"""
    designs, settings, theta = probe.designs(), probe.settings(), _snapped()
    surfaces, fixed = rounds.surfaces_of(designs), expanded_fixed_surfaces(probe.FIXED, probe.NAMES)
    free = rounds.alternatives(surfaces, settings, fixed)
    tied = rounds.alternatives(surfaces, settings, fixed, ties=TIES)
    assert len(free) == 9 and tied == tuple(item for item in free if item[0] != "yL@a")
    segment = 1 + settings.retune.steps * _trials(settings)
    assert rounds.next_round_bound(surfaces, settings, fixed) == 9 + 3 * segment
    assert rounds.next_round_bound(surfaces, settings, fixed, ties=TIES) == 6 + 3 * segment
    context = _context(TIES)
    swapped, moved = rounds.swap(designs, theta, "y0@a", "glass_wool_96k", context)
    assert [design.preset_id for design in swapped if design.surface_id in ("y0@a", "yL@a")] == ["glass_wool_96k_50mm"] * 2
    assert search.design_names(swapped) == ("y0@a.layer1.thickness_m", "y0@b.layer1.thickness_m",
                                            "y0@b.layer2.thickness_m", "yL@a.layer1.thickness_m")
    assert moved[0] == moved[3] == 0.05 and moved[1:3].tobytes() == theta[2:4].tobytes()
    bare, kept = rounds.swap(designs, theta, "y0@a", "none", context)
    assert [design.preset_id for design in bare if design.surface_id in ("y0@a", "yL@a")] == [None, None]
    assert kept.tobytes() == theta[2:4].tobytes()
    with pytest.raises(ValueError, match="yL@a 跟 y0@a 成對"):
        rounds.swap(designs, theta, "yL@a", "glass_wool_96k", context)


# ---- 8 不變量在產品程式裡擋 ----

def test_invariant_stops_descent_and_retune_that_start_one_ulp_apart() -> None:
    """抓不變量只寫在考卷裡：成對一格差一個最小浮點單位，下降與空腔重調一進來就拋錯。"""
    from aosr.gradient.ties import TieBroken

    designs, objective = probe.designs(), _quadratic(*LOPSIDED, cross=0.0)
    assert not issubclass(TieBroken, ValueError)
    with pytest.raises(TieBroken, match="yL@a"):
        search.projected_descent(objective, _one_ulp_off(_start()), search.design_bounds(designs), _settings().descent,
                                 limits=search.depth_limits(designs), tied_cells=CELLS)
    feasible = search.snap_to_specs(_start(), designs).theta
    with pytest.raises(TieBroken, match="成對"):
        search.retune_cavities(objective, _one_ulp_off(feasible), designs, _settings(), ties=TIES)
    search.retune_cavities(objective, _one_ulp_off(feasible), designs, _settings())


def test_invariant_stops_snap_jump_and_swap() -> None:
    """抓取規格、候選跳點、換樣板之後沒檢查：這三步各自交出成對不相等的 θ 就拋錯（跳點與換樣板動的是沒成對的那一片，
    成對那一對原本就差一個最小浮點單位）。"""
    from aosr.gradient.ties import TieBroken

    designs, settings = probe.designs(), probe.settings()
    broken = _one_ulp_off(_snapped())
    assert search.snap_to_specs(broken, designs).theta.tobytes() == broken.tobytes()
    with pytest.raises(TieBroken, match="取規格"):
        search.snap_to_specs(broken, designs, ties=TIES)
    candidate = next(item for item in loop.candidates_from_gradient(_snapped(), np.zeros(6), designs, settings)
                     if item.surface_id == "y0@b")
    with pytest.raises(TieBroken, match="候選跳點"):
        loop.apply_candidate(broken, candidate, designs, ties=TIES)
    with pytest.raises(TieBroken, match="換樣板"):
        rounds.swap(designs, broken, "y0@b", "glass_wool_96k", _context(TIES))
    rounds.swap(designs, broken, "y0@b", "glass_wool_96k", _context(()))


def test_invariant_stops_a_round_that_starts_one_ulp_apart() -> None:
    """抓接續時從帳本讀回的 θ 沒檢查：候選確認一輪一進來就拋錯，還沒求值。"""
    from aosr.gradient.ties import TieBroken

    def never(_: jax.Array) -> tuple[jax.Array, GradientObjectiveArrays]:
        raise AssertionError("這一題不該求值")

    with pytest.raises(TieBroken, match="候選確認"):
        loop.confirmation_round(never, _one_ulp_off(_snapped()), probe.designs(), probe.settings(), budget_left=10**6,
                                ties=TIES)


# ---- 身分與標頭 ----

def test_ties_enter_identity_and_both_headers_only_when_given() -> None:
    """抓綁定表沒進身分（綁與不綁撞同一個資料夾）、空表也寫鍵。"""
    assert probe.identity() == MAIN_PLAIN_ID
    assert probe.identity(TIES) != MAIN_PLAIN_ID
    plain, tied = probe.header(), probe.header(TIES)
    assert "ties" not in plain.model_dump(mode="json") and tied.model_dump(mode="json")["ties"] == [["y0@a", "yL@a"]]
    assert "ties" not in probe.project().model_dump(mode="json")
    assert probe.project(TIES).model_dump(mode="json")["ties"] == [["y0@a", "yL@a"]]
    template = run.template_header_for(tied, probe.settings(), probe.kept_patches(), base_ledger_sha256="e" * 64)
    assert template.model_dump(mode="json")["ties"] == [["y0@a", "yL@a"]]
    bare = run.template_header_for(plain, probe.settings(), probe.kept_patches(), base_ledger_sha256="e" * 64)
    assert "ties" not in bare.model_dump(mode="json")
    with pytest.raises(ValueError, match="綁定"):
        ledger.MaterialGradientHeader.model_validate({**plain.model_dump(), "ties": (("y0@a", "yL@c"),)})


def test_real_header_writer_takes_ties() -> None:
    """抓正式的標頭寫入端（``header_for``）沒收綁定表：六面整面的小替身，y0 與 yL 成對。"""
    from aosr.config.frequency_axis import FEM_LANE_FREQUENCIES_HZ
    from aosr.scoring.gradient_objective_jax import gradient_objective_constants
    from tests.engine._six_search_helpers import ledger_constants
    from tests.engine.test_material_gradient import _gradient_settings, _receivers
    from tests.engine.test_material_gradient_mount import _designs, _stand_in

    constants = ledger_constants(gradient_objective_constants(np.asarray(FEM_LANE_FREQUENCIES_HZ), _gradient_settings(),
                                                              _receivers(), 2))
    designs = _designs("rigid")
    plain = ledger.header_for(_stand_in(), designs, probe.settings(), objective_constants=constants)
    tied = ledger.header_for(_stand_in(), designs, probe.settings(), objective_constants=constants,
                             ties=(("x0", "xL"),))
    assert tied.ties == (("x0", "xL"),) and plain.ties == ()
    assert tied.model_copy(update={"ties": ()}) == plain


# ---- 雙精度子行程：整條產品路徑 ----

def _probe(directory: Path, part: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    env = forward_cpu_pins(child_process_env(threads=1))
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = ":".join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    result = subprocess.run([sys.executable, str(Path(probe.__file__)), str(directory), part], env=env,
                            cwd=directory, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    return (directory / "output.json").read_text(encoding="utf-8")


def _shared(factory: pytest.TempPathFactory, worker_id: str, part: str) -> tuple[Path, Record]:
    """探針每一段整場只跑一次，平行工人之間用檔案鎖共用。"""
    directory = (factory.mktemp(f"gradient-ties-{part}") if worker_id == "master"
                 else factory.getbasetemp().parent / f"gradient-ties-{part}-shared")
    text = shared_json(factory, worker_id, f"gradient-ties-v1-{part}", lambda: _probe(directory / "runs", part))
    return directory / "runs", cast(Record, json.loads(text))


@pytest.fixture(scope="module")
def plain(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> tuple[Path, Record]:
    """不給綁定表的那一場（零目標）；主線上原樣跑得動，是控制組。"""
    return _shared(tmp_path_factory, worker_id, "plain")


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> tuple[Path, Record]:
    """y0@a 與 yL@a 成對的那一場（不對稱二次式），含接續與竄改。"""
    return _shared(tmp_path_factory, worker_id, "tied")


def _rows(path: Path) -> list[Record]:
    return [json.loads(line) for line in path.read_bytes().splitlines()[1:]]


def _pair_values(row: Record, surface: str) -> list[str]:
    names, theta = cast(list[str], row["variable_names"]), cast(list[str], row["theta_hex"])
    return [value for name, value in zip(names, theta, strict=True) if name.split(".", 1)[0] == surface]


def test_empty_table_writes_the_same_bytes_as_before(plain: tuple[Path, Record]) -> None:
    """抓空表時也寫鍵、或加這一格改到了不綁的位元組：不給綁定表，資料夾名與四個檔跟主線 7d73a450 逐位相同。"""
    folder, output = plain[0] / "plain", cast(Record, plain[1]["plain"])
    assert output["gradient_id"] == MAIN_PLAIN_ID
    assert {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in MAIN_PLAIN_DIGESTS} == (
        MAIN_PLAIN_DIGESTS)
    for name in (run.PROJECT_FILE, run.LEDGER_FILE, run.TEMPLATES_FILE):
        assert "ties" not in json.loads((folder / name).read_bytes().splitlines()[0])


def test_saves_without_the_field_read_as_untied_and_resume(plain: tuple[Path, Record]) -> None:
    """抓缺欄位讀成「綁」：沒有這一格的 ``project.json`` 與兩本帳的標頭讀回都是不綁；截掉半輪接續不被拒、結果逐位相同。"""
    folder = plain[0] / "plain"
    assert run.read_project(folder).ties == ()
    assert ledger.read_ledger(folder / run.LEDGER_FILE)[0].ties == ()
    assert templates.read_template_ledger(folder / run.TEMPLATES_FILE)[0].ties == ()
    assert cast(Record, plain[1]["plain"])["resumed"] == {"rounds": True, "templates": True}


def test_tied_search_keeps_the_pair_equal_in_every_ledger_row(runs: tuple[Path, Record]) -> None:
    """抓某一段漏綁：庚二帳本每一列（下降、取規格、重調、跳點、候選重調、決定）成對的格逐位相等；下降段的值單調不增；
    候選代號只用面序在前那一片，而且那一對真的被接受跳過一次（兩片一起到 0.2）。"""
    rows = _rows(runs[0] / "tied" / run.LEDGER_FILE)
    assert {str(row["phase"]) for row in rows} == {"descent", "snap", "retune", "baseline_retune", "candidate_jump",
                                                    "candidate_retune", "round_decision"}
    assert all(_paired([float.fromhex(value) for value in cast(list[str], row["theta_hex"])]) for row in rows)
    descent = [cast(float, row["value"]) for row in rows if row["phase"] == "descent"]
    assert len(descent) > 1 and all(later <= earlier for earlier, later in zip(descent, descent[1:], strict=False))
    candidates = {str(row["candidate"]) for row in rows if row["candidate"] is not None}
    assert candidates and not any(name.startswith("yL@a") for name in candidates)
    accepted = [cast(Record, row["decision"])["accepted"] for row in rows if row["phase"] == "round_decision"]
    assert "y0@a:thicker" in accepted
    final = [float.fromhex(value) for value in cast(list[str], rows[-1]["theta_hex"])]
    assert final[0] == final[4] == 0.2


def test_tied_search_swaps_the_pair_together_and_replays(runs: tuple[Path, Record]) -> None:
    """抓換樣板只換一片、重播照片算：換樣板帳本每一列成對兩片同一個樣板、θ 相等；快速評分一輪 6 個（不綁是 9 個）、
    沒有 yL@a 自己的候選；那一對一起換成 96K 被接受；帶設定與起點重播得回來；結果檔成對兩筆數字相同。"""
    folder = runs[0] / "tied"
    rows = _rows(folder / run.TEMPLATES_FILE)
    for row in rows:
        presets = dict(cast(list[tuple[str, str | None]], [tuple(item) for item in cast(list[list[str]], row["surfaces"])]))
        assert presets["y0@a"] == presets["yL@a"]
        assert _pair_values(row, "y0@a") == _pair_values(row, "yL@a")
    first = [str(row["candidate"]) for row in rows if row["phase"] == "quick_score" and row["round"] == 0]
    assert len(first) == 6 and not any(name.startswith("yL@a") for name in first)
    decisions = [row for row in rows if row["phase"] == "template_decision"]
    assert cast(Record, decisions[0]["decision"])["accepted"] == "y0@a->glass_wool_96k"
    assert dict(cast(list[tuple[str, str]], [tuple(item) for item in cast(list[list[str]], decisions[0]["surfaces"])]))[
        "yL@a"] == "glass_wool_96k_50mm"
    header, base = ledger.read_ledger(folder / run.LEDGER_FILE)
    template_header, replayed = templates.read_template_ledger(
        folder / run.TEMPLATES_FILE, settings=probe.settings(), start=run.template_start(header, base))
    assert template_header.ties == TIES and len(replayed) == len(rows)
    assert run.read_project(folder).ties == TIES
    assert cast(Record, runs[1]["tied"])["resumed"] == {"rounds": True, "templates": True}
    result = run.parse_result((folder / run.RESULT_FILE).read_bytes())
    specs = {spec.surface_id: spec for spec in result.surfaces}
    assert len(specs) == len(probe.NAMES)
    left, right = specs["y0@a"], specs["yL@a"]
    assert (left.preset_id, left.thickness_m, left.cavity_m, left.material_digest) == (
        right.preset_id, right.thickness_m, right.cavity_m, right.material_digest)
    assert left.preset_id == "glass_wool_96k_50mm"


def test_template_replay_refuses_a_row_that_swaps_half_a_pair(runs: tuple[Path, Record], tmp_path: Path) -> None:
    """抓重播的「只換一個單位」退回「只換一片」：把一列快速評分改成只換 y0@a、yL@a 留在原樣板，重播要點名拒收。"""
    source = (runs[0] / "tied" / run.TEMPLATES_FILE).read_bytes().splitlines(keepends=True)
    index, row = next((index, json.loads(line)) for index, line in enumerate(source)
                      if index and json.loads(line)["candidate"] == "y0@a->glass_wool_96k")
    position = [item[0] for item in row["surfaces"]].index("yL@a")
    row["surfaces"][position][1] = GAP50
    row["variable_names"].append("yL@a.layer2.thickness_m")
    row["theta_hex"].append(0.05.hex())
    target = tmp_path / run.TEMPLATES_FILE
    target.write_bytes(b"".join(source[:index]) + (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    with pytest.raises(ValueError, match="不是起點配置只換那一個單位"):
        templates.read_template_ledger(target)


def test_tampered_ledger_stops_the_resume_with_a_non_zero_exit_error(runs: tuple[Path, Record]) -> None:
    """抓不變量只在開跑那一條路：帳本裡成對一格被改一個最小浮點單位，接續拋的是 ``TieBroken``（不是 ``ValueError``，
    子行程入口照一般錯誤回非零離開碼），帳本不動、不寫結果檔。"""
    broken = cast(Record, runs[1]["broken"])
    assert broken["error"] == "TieBroken" and broken["is_value_error"] is False
    assert "yL@a" in str(broken["message"]) and "成對" in str(broken["message"])
    assert broken["ledger_unchanged"] is True and broken["result_written"] is False
