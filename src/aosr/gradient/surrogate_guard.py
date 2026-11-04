"""分區替身的搜尋接線尚未施工：所有搜尋入口共用明確的能力閘門（#561 第六步）。"""
from aosr.physics.fem_reduced import FullOperators, ReducedModel
from aosr.physics.fem_reduced_boundary import boundary_patches


def require_whole_surrogate(model: ReducedModel | FullOperators) -> None:
    if model.patch_keys != boundary_patches() or len(model.boundary if isinstance(model, FullOperators)
                                                    else model.matrices.boundary) != len(boundary_patches()):
        raise ValueError("搜尋還不支援分區替身（#561 第六步）")
