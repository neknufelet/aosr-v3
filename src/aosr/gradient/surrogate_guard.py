"""搜尋與終點共用的替身輸入及方案原牆檢查。"""
from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from aosr.gradient.boundary import design_patch_keys
from aosr.materials.construction_models import MeasuredImpedance, SurfaceMaterial
from aosr.physics.fem_reduced import FullOperators, ReducedModel
from aosr.physics.fem_reduced_boundary import PatchKeys, checked_patch_keys


def require_matching_sources(model: ReducedModel, source_ids: NDArray[np.str_]) -> None:
    """六類幾何的有序聲源識別碼必須與替身 RHS（右端聲源列）完全相同。"""
    if not np.array_equal(source_ids, np.asarray(model.source_names, dtype=np.str_)):
        raise ValueError(f'六類常數的聲源識別碼與替身不同（含順序）：常數 {source_ids.tolist()}，替身 {model.source_names}')


def require_matching_frequencies(model: ReducedModel, frequencies: Sequence[float] | NDArray[np.float64],
                                  expected: NDArray[np.float64]) -> None:
    """整條頻率軸必須逐點相同，不能只核點數或搜尋計分範圍。"""
    axis = np.asarray(frequencies, dtype=np.float64)
    if not np.array_equal(axis, model.frequencies_hz) or not np.array_equal(axis, expected):
        raise ValueError('頻率軸必須等於替身與六類常數的 FEM 頻率軸')


def require_supported_original_walls(kept: Iterable[SurfaceMaterial]) -> None:
    """六類終點必算方案原牆基準；裸實測表尚未接入，貼牆構造照常收。"""
    if any(isinstance(material, MeasuredImpedance) for material in kept):
        raise ValueError('方案原本的牆含裸實測材料：實測原牆的表內／表外吸音分流尚未接入六類終點；拒絕寫出定義不同的分數')


def require_matching_patches(model: ReducedModel | FullOperators, names: Sequence[str] | None = None, *,
                             keys: PatchKeys | None = None) -> None:
    try:
        wanted = keys if keys is not None else design_patch_keys(names) if names is not None else model.patch_keys
    except ValueError as error:
        raise ValueError(f"搜尋的面片清單必須等於替身的面片清單（含順序；整面照 Wall.all()）：{error}") from error
    count = len(model.boundary if isinstance(model, FullOperators) else model.matrices.boundary)
    if checked_patch_keys(model.patch_keys) != wanted or count != len(wanted):
        raise ValueError("搜尋的面片清單必須等於替身的面片清單（含順序；整面照 Wall.all()）")
