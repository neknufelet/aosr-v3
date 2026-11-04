"""搜尋與替身必須用同一份有序面片清單。"""
from collections.abc import Sequence

from aosr.gradient.boundary import design_patch_keys
from aosr.physics.fem_reduced import FullOperators, ReducedModel
from aosr.physics.fem_reduced_boundary import PatchKeys, checked_patch_keys


def require_matching_patches(model: ReducedModel | FullOperators, names: Sequence[str] | None = None, *,
                             keys: PatchKeys | None = None) -> None:
    try:
        wanted = keys if keys is not None else design_patch_keys(names) if names is not None else model.patch_keys
    except ValueError as error:
        raise ValueError(f"搜尋的面片清單必須等於替身的面片清單（含順序；整面照 Wall.all()）：{error}") from error
    count = len(model.boundary if isinstance(model, FullOperators) else model.matrices.boundary)
    if checked_patch_keys(model.patch_keys) != wanted or count != len(wanted):
        raise ValueError("搜尋的面片清單必須等於替身的面片清單（含順序；整面照 Wall.all()）")
