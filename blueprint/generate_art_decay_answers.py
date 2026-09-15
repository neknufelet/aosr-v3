"""把凍結 donor 的晚期衰減 T20 跑成標準答案檔。

這支只在上一代環境執行；房間、材料、出身與環境記錄沿用
``generate_art_answers``，衰減曲線則逐式鏡像 donor 的
``art_wls_t20_t60``。每個實數同時保存十進位與可逐位還原的十六進位。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

from blueprint import generate_art_answers as base


ANSWER_SCHEMA: Final[int] = 1
MIN_WEIGHT_SOURCE: Final[str] = "lib/physics/art_kernel.py:209 _ART_WLS_MIN_WEIGHT"
LOG_FLOOR_SOURCE: Final[str] = "lib/physics/art_kernel.py:210 _ART_WLS_LOG_FLOOR_DB"
FIT_SOURCE: Final[str] = "lib/physics/art_kernel.py:249-318 art_wls_t20_t60"


def _decay_arrays(
    transfer: object,
    fe: object,
    areas: object,
    modules: dict[str, ModuleType],
) -> tuple[base._Array, base._Array, base._Array, base._Array]:
    """逐式鏡像 donor 的階數衰減、精確尾巴以前的曲線與擬合陣列。"""
    jax = modules["jax"]
    jnp = modules["jnp"]
    art = modules["art"]
    lane = modules["lane"]
    rpf, _squeezed = art._as_rpf(transfer)
    patch_count, _unused, frequency_count = rpf.shape
    area = jnp.asarray(areas, dtype=rpf.dtype)
    total_area = jnp.sum(area)
    area_weight = area / total_area
    energy = jnp.ones((patch_count, frequency_count), dtype=rpf.dtype)

    def step(term: object, _unused_step: None) -> tuple[object, object]:
        next_term = jnp.einsum("ijf,jf->if", rpf, term)
        mean_energy = jnp.einsum("p,pf->f", area_weight, next_term)
        return next_term, mean_energy

    _last, order_energy = jax.lax.scan(
        step, energy, None, length=int(lane.ART_NEUMANN_K_MAX)
    )
    rho, _vector = art.art_perron_eig(rpf, K=lane.ART_POWERITER_K)
    tail = order_energy[-1] * rho / (1.0 - rho)
    edc = jnp.cumsum(order_energy[::-1], axis=0)[::-1] + tail[None, :]
    ratio = edc / edc[0]
    tiny = jnp.asarray(jnp.finfo(rpf.dtype).tiny, dtype=rpf.dtype)
    safe_ratio = jnp.where(ratio > tiny, ratio, jnp.ones_like(ratio))
    level = jnp.where(
        ratio > tiny,
        10.0 * jnp.log10(safe_ratio),
        jnp.full_like(ratio, art._ART_WLS_LOG_FLOOR_DB),
    )
    order = jnp.arange(1, int(lane.ART_NEUMANN_K_MAX) + 1, dtype=rpf.dtype)
    time = (order / fe)[:, None]
    return rpf, level, time, rho


def _fit_arrays(
    level: object,
    time: object,
    modules: dict[str, ModuleType],
) -> tuple[base._Array, base._Array, base._Array, base._Array]:
    """依 donor 的 sigmoid 軟視窗作加權直線擬合。"""
    jax = modules["jax"]
    jnp = modules["jnp"]
    art = modules["art"]
    lane = modules["lane"]
    softness = lane.ART_WLS_WINDOW_SOFTNESS_DB
    weight = jax.nn.sigmoid(
        (lane.ART_WLS_T20_HI_DB - level) / softness
    ) * jax.nn.sigmoid((level - lane.ART_WLS_T20_LO_DB) / softness)
    weight_sum = jnp.sum(weight, axis=0)
    safe_weight = jnp.where(
        weight_sum > art._ART_WLS_MIN_WEIGHT,
        weight_sum,
        jnp.ones_like(weight_sum),
    )
    time_mean = jnp.sum(weight * time, axis=0) / safe_weight
    level_mean = jnp.sum(weight * level, axis=0) / safe_weight
    covariance = jnp.sum(
        weight * (time - time_mean[None, :]) * (level - level_mean[None, :]),
        axis=0,
    )
    variance = jnp.sum(weight * (time - time_mean[None, :]) ** 2, axis=0)
    safe_variance = jnp.where(variance > 0.0, variance, jnp.ones_like(variance))
    slope = jnp.where(variance > 0.0, covariance / safe_variance, jnp.zeros_like(covariance))
    valid = (weight_sum > art._ART_WLS_MIN_WEIGHT) & (slope < 0.0)
    safe_slope = jnp.where(slope < 0.0, slope, -jnp.ones_like(slope))
    return weight_sum, slope, valid, -60.0 / safe_slope


def _build_bands(case: str, modules: dict[str, ModuleType]) -> tuple[list[dict[str, object]], dict[str, str], int]:
    """跑指定材料並收成答案頻帶。"""
    jax = modules["jax"]
    np = modules["np"]
    lane = modules["lane"]
    response = modules["response"]
    art = modules["art"]
    m4 = modules["m4"]
    proxy = modules["proxy"]
    n_per_wall = int(lane.ART_N_PER_WALL_DEFAULT)
    frequency_axis = response.FrequencyAxis.from_hz(
        np.asarray(base.FREQUENCIES_HZ, dtype=float), resolution="custom"
    )
    geometry = proxy.ShoeboxGeometry.of(*base.ROOM)
    walls = base._wall_set(case, response, proxy, np, frequency_axis)
    token = m4.ValidatedMaterials.validate(walls, frequency_axis, rho_c=base.RHO_C)
    wall_alpha = proxy.material_alpha_matrix(
        token.materials, frequency_axis, base.RHO_C, validate=False
    )
    patches = art.art_patch_geometry(geometry, n_per_wall)
    form = art.art_form_factors(patches)
    transfer = art.art_transfer_operator(form, wall_alpha[patches.wall_index, :])
    fe = base.SOUND_SPEED * modules["jnp"].sum(geometry.wall_areas) / (4.0 * geometry.volume)
    rpf, level, time, _rho = _decay_arrays(transfer, fe, patches.areas, modules)
    weight_sum, slope, valid, t20 = _fit_arrays(level, time, modules)
    perron = art.art_perron_t60(rpf, fe=fe, K=lane.ART_POWERITER_K)
    official = art.art_t60(
        geometry,
        token.wall_set,
        frequency_axis,
        sound_speed=base.SOUND_SPEED,
        rho_c=base.RHO_C,
        n_per_wall=n_per_wall,
        validate_materials=False,
    )
    jax.block_until_ready((weight_sum, slope, valid, t20, perron, official))
    bands = []
    for index, frequency in enumerate(base.FREQUENCIES_HZ):
        bands.append(
            {
                "frequency_hz": base._real(frequency),
                "t20_s": base._real(official[index]),
                "fe_hz": base._real(fe),
                "slope_db_per_s": base._real(slope[index]),
                "soft_weight_sum": base._real(weight_sum[index]),
                "fell_back_to_perron": not bool(valid[index]),
                "perron_t60_s": base._real(perron[index]),
            }
        )
    dtypes = {
        "transfer_operator": str(transfer.dtype),
        "decay_level": str(level.dtype),
        "fit_slope": str(slope.dtype),
        "t20": str(official.dtype),
        "perron_t60": str(perron.dtype),
    }
    return bands, dtypes, n_per_wall


def build_payload(root: Path, case: str) -> dict[str, object]:
    """組成出身、環境、參數與逐頻帶衰減答案。"""
    modules = base._modules()
    bands, dtypes, n_per_wall = _build_bands(case, modules)
    env = base.environment()
    env["dtypes"] = dtypes
    parameters = base._parameters(case, modules, n_per_wall)
    parameters["ART_WLS_T20_HI_DB"] = {
        **base._real(modules["lane"].ART_WLS_T20_HI_DB),
        "source": "lib/config/art_lane.py:30 ART_WLS_T20_HI_DB",
    }
    parameters["ART_WLS_T20_LO_DB"] = {
        **base._real(modules["lane"].ART_WLS_T20_LO_DB),
        "source": "lib/config/art_lane.py:31 ART_WLS_T20_LO_DB",
    }
    parameters["ART_WLS_WINDOW_SOFTNESS_DB"] = {
        **base._real(modules["lane"].ART_WLS_WINDOW_SOFTNESS_DB),
        "source": "lib/config/art_lane.py:32 ART_WLS_WINDOW_SOFTNESS_DB",
    }
    parameters["ART_WLS_MIN_WEIGHT"] = {
        **base._real(modules["art"]._ART_WLS_MIN_WEIGHT),
        "source": MIN_WEIGHT_SOURCE,
    }
    parameters["ART_WLS_LOG_FLOOR_DB"] = {
        **base._real(modules["art"]._ART_WLS_LOG_FLOOR_DB),
        "source": LOG_FLOOR_SOURCE,
    }
    return {
        "schema": ANSWER_SCHEMA,
        "note": (
            "由 blueprint/generate_art_decay_answers.py 在唯讀凍結 donor 上產生，"
            f"不准手改；T20 與中介量逐式錨定 {FIT_SOURCE}。"
        ),
        "donor": base.donor_provenance(root),
        "env": env,
        "command": base.command_record(),
        "parameters": parameters,
        "bands": bands,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="產生凍結 donor 的 ART 晚期衰減答案")
    parser.add_argument("--case", required=True, choices=list(base.CASES))
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    payload = build_payload(base.donor_root_from_env(), args.case)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
