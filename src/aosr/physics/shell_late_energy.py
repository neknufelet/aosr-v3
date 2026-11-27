"""自訂房間（房間殼）的晚期混響：每一格面之間的能量交換，含造型遮擋。

**跟長方體那一路同一套算式。** :mod:`aosr.physics.late_energy` 是六面各切 n×n、中心點形狀因子；
這一支把外殼與造型的每個三角形切成小格（沿最長邊對切到不超過設定長度），形狀因子改用**多點積分**
（每格再切成子三角形、取各自中心），兩點之間被造型擋住的那一對不算。原始形狀因子之後的步驟——
``(I−R)y=E`` 直接解、逐階展開、面積加權平均吸音率的 Eyring
比值、尾巴——呼叫長方體那一支的同一份程式；逐列正規化與雙邊縮放那一段另抄一份（理由見
:func:`normalized_form_factors`），考卷逐位守著兩份一樣。

**看不到任何一格的小格**（例如貼牆板子背後那一塊牆、貼天花板的造型頂面）不跟任何人交換能量，
從交換與平均吸音率裡拿掉，面積記在 ``hidden_area_m2``。

**每間房算一次、每個候選只換吸音率。** :class:`ShellLateGeometry` 只有幾何（分格、形狀因子），
跟材料無關；:func:`solve_shell_late_energy_by_order` 每個候選只吃各圖層的吸音率。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aosr.config.shell_lane import SHELL_FORM_FACTOR_SPLITS, SHELL_GEOMETRY_TOL_M, SHELL_PATCH_EDGE_M
from aosr.geometry.shell_faces import (
    BoolArray, FloatArray, IntArray, Occluders, ShellFaces, segments_hit_triangles, triangle_areas,
)
from aosr.physics.late_energy import (
    LateEnergyOrderBand,
    LateEnergyOrderResult,
    _eyring_ratio,
    _raw_energy_and_orders,
    _tail_energy,
)

# 一次算幾對取樣點的形狀因子核（含遮擋）；只管記憶體，不影響答案。
_SAMPLE_PAIRS_PER_CHUNK = 200_000
# 交換矩陣一次帶幾個頻點進長方體那一支的直接解；只管記憶體（格數² × 頻點數），不影響答案。
_FREQUENCIES_PER_CHUNK = 8


def normalized_form_factors(raw: FloatArray, areas: FloatArray) -> FloatArray:
    """原始形狀因子逐列正規化，再對稱化成交換矩陣並雙邊縮放到列和等於面積（守互易與列和）。

    這一段逐字照 :func:`aosr.physics.late_energy._form_factors` 算完原始值之後的那一半；那一支在
    物理身分的程式摘要裡，抽成共用函式會讓長方體那一路的身分變、老闆存的結果要重跑，所以這裡另抄
    一份，由 ``tests/engine/test_shell_lane.py`` 拿長方體的中心點原始值餵兩邊、逐位比對守著不走樣。
    """
    raw_row_sum = np.sum(raw, axis=1)
    if np.any(raw_row_sum <= 0.0):
        raise ValueError("形狀因子有無法正規化的空列")
    original = raw / raw_row_sum[:, None]
    exchange = areas[:, None] * original
    balanced = (exchange + exchange.T) / 2.0
    deviation = float(np.max(np.abs(np.sum(balanced, axis=1) - areas) / areas))
    while True:
        row_sum = np.sum(balanced, axis=1)
        if np.any(row_sum <= 0.0):
            raise ValueError("交換矩陣有無法平衡的空列")
        scale = np.sqrt(areas / row_sum)
        candidate = balanced * np.multiply.outer(scale, scale)
        candidate_deviation = float(np.max(np.abs(np.sum(candidate, axis=1) - areas) / areas))
        if not candidate_deviation < deviation:
            break
        balanced = candidate
        deviation = candidate_deviation
    return np.asarray(balanced / areas[:, None], dtype=np.float64)


@dataclass(frozen=True)
class ShellLateGeometry:
    """晚期那一路只看幾何的部分：每一格的圖層、面積、中心與交換後的形狀因子。"""

    layers: tuple[str, ...]
    patch_layer: IntArray
    areas: FloatArray
    centroids: FloatArray
    form_factors: FloatArray
    hidden_area_m2: float

    @property
    def patch_count(self) -> int:
        """參與交換的格數。"""
        return int(self.areas.shape[0])

    def layer_areas(self) -> FloatArray:
        """每個圖層參與交換的面積（被蓋住、看不到的不算），順序同 ``layers``。"""
        return np.asarray(np.bincount(self.patch_layer, weights=self.areas, minlength=len(self.layers)),
                          dtype=np.float64)


def split_longest_edge(corners: FloatArray, owners: IntArray, max_edge_m: float) -> tuple[FloatArray, IntArray]:
    """沿最長邊對切，直到每個三角形的最長邊不超過 ``max_edge_m``；頂點順序（法線方向）不變。"""
    if not max_edge_m > 0.0:
        raise ValueError("分格長度必須是正數")
    done_corners: list[FloatArray] = []
    done_owners: list[IntArray] = []
    while len(corners):
        # 第 k 條邊是頂點 k → k+1。
        edges = np.linalg.norm(np.roll(corners, -1, axis=1) - corners, axis=2)
        longest = np.argmax(edges, axis=1)
        short = edges[np.arange(len(corners)), longest] <= max_edge_m
        done_corners.append(corners[short])
        done_owners.append(owners[short])
        long = np.flatnonzero(~short)
        turn = (longest[long, None] + np.arange(3)[None, :]) % 3
        rolled = corners[long[:, None], turn]
        middle = (rolled[:, 0] + rolled[:, 1]) / 2.0
        first = np.stack((rolled[:, 0], middle, rolled[:, 2]), axis=1)
        second = np.stack((middle, rolled[:, 1], rolled[:, 2]), axis=1)
        corners = np.concatenate((first, second))
        owners = np.concatenate((owners[long], owners[long]))
    return np.concatenate(done_corners), np.concatenate(done_owners)


def _midpoint_samples(corners: FloatArray, splits: int) -> FloatArray:
    """每個三角形切成 4^splits 個子三角形（取中點），回 (格數, 4^splits, 3) 的子三角形中心。"""
    pieces = corners[:, None, :, :]
    for _round in range(splits):
        a, b, c = pieces[:, :, 0], pieces[:, :, 1], pieces[:, :, 2]
        ab, bc, ca = (a + b) / 2.0, (b + c) / 2.0, (c + a) / 2.0
        pieces = np.concatenate((np.stack((a, ab, ca), axis=2), np.stack((ab, b, bc), axis=2),
                                 np.stack((ca, bc, c), axis=2), np.stack((ab, bc, ca), axis=2)), axis=1)
    return np.asarray(pieces.mean(axis=2), dtype=np.float64)


def _blocked_samples(points: FloatArray, a: IntArray, b: IntArray, live: IntArray, pair_box: FloatArray,
                     per_pair: int, planes: tuple[IntArray, IntArray], occluders: Occluders, tol_m: float) -> BoolArray:
    """``live`` 那幾對取樣點的連線被不被擋。

    先用「兩格合起來的外接盒」跟每一組擋的三角形的外接盒比：兩格之間的連線全都在這個盒子裡，
    盒子碰不到的那一組一定擋不到，就不必一條一條比（答案不變、只是快）。兩格自己所在的平面不拿來擋。
    """
    blocked = np.zeros(len(live), dtype=np.bool_)
    pair_of = live // per_pair
    first, second = planes
    for group, box, plane in zip(occluders.groups, occluders.boxes, occluders.planes, strict=True):
        overlap = ((pair_box[:, 0] <= box[1] + tol_m) & (pair_box[:, 1] >= box[0] - tol_m)).all(axis=1)
        test = np.flatnonzero(overlap[pair_of] & ~blocked & (first[pair_of] != plane) & (second[pair_of] != plane))
        if len(test):
            blocked[test] = segments_hit_triangles(points[a[live[test]]], points[b[live[test]]], group, tol_m)
    return blocked


def _raw_form_factors(corners: FloatArray, samples: FloatArray, normals: FloatArray, areas: FloatArray,
                      patch_planes: IntArray, occluders: Occluders, tol_m: float) -> tuple[FloatArray, BoolArray]:
    """多點積分的原始形狀因子：``F_ij ≈ Σ_a Σ_b (1/n)·(A_j/n)·cosθ_a·cosθ_b/(π r²)``，被擋的那一對不算；
    另回每個取樣點有沒有看到任何一格（(格數, n)），用來把格子被蓋住的那一份面積扣掉。"""
    count, per_patch, _ = samples.shape
    points = samples.reshape(-1, 3)
    owner = np.repeat(np.arange(count), per_patch)
    point_normals = normals[owner]
    weight = np.repeat(areas / per_patch, per_patch) / per_patch
    patch_box = np.stack((corners.min(axis=1), corners.max(axis=1)), axis=1)
    raw = np.zeros((count, count), dtype=np.float64)
    sees = np.zeros(count * per_patch, dtype=np.bool_)
    pair_i, pair_j = np.nonzero(normals @ normals.T < 1.0 - 1e-12)
    rows = np.repeat(np.arange(per_patch), per_patch)
    cols = np.tile(np.arange(per_patch), per_patch)
    step = max(1, _SAMPLE_PAIRS_PER_CHUNK // (per_patch * per_patch))
    for begin in range(0, len(pair_i), step):
        pi, pj = pair_i[begin:begin + step], pair_j[begin:begin + step]
        a = (pi[:, None] * per_patch + rows[None, :]).reshape(-1)
        b = (pj[:, None] * per_patch + cols[None, :]).reshape(-1)
        delta = points[b] - points[a]
        r2 = np.einsum("ij,ij->i", delta, delta)
        cos_a = np.einsum("ij,ij->i", point_normals[a], delta)
        cos_b = -np.einsum("ij,ij->i", point_normals[b], delta)
        kernel = np.where((cos_a > 0) & (cos_b > 0) & (r2 > 0), cos_a * cos_b / np.where(r2 > 0, r2 * r2, 1.0), 0.0)
        live = np.flatnonzero(kernel > 0)
        if occluders.groups and len(live):
            pair_box = np.stack((np.minimum(patch_box[pi, 0], patch_box[pj, 0]),
                                 np.maximum(patch_box[pi, 1], patch_box[pj, 1])), axis=1)
            hidden = _blocked_samples(points, a, b, live, pair_box, per_patch * per_patch,
                                      (patch_planes[pi], patch_planes[pj]), occluders, tol_m)
            kernel[live[hidden]] = 0.0
        sees[a[kernel > 0]] = True
        contribution = (kernel * weight[b] / np.pi).reshape(len(pi), -1).sum(axis=1)
        np.add.at(raw, (pi, pj), contribution)
    return raw, sees.reshape(count, per_patch)


def _patch_normals(corners: FloatArray) -> FloatArray:
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return np.asarray(cross / np.linalg.norm(cross, axis=1)[:, None], dtype=np.float64)


def shell_late_geometry(
    faces: ShellFaces,
    *,
    patch_edge_m: float = SHELL_PATCH_EDGE_M,
    splits: int = SHELL_FORM_FACTOR_SPLITS,
    tol_m: float = SHELL_GEOMETRY_TOL_M,
) -> ShellLateGeometry:
    """把房間殼切格、算形狀因子（多點積分＋遮擋）；每間房算一次。"""
    if splits < 0:
        raise ValueError("多點積分的對切輪數不可為負")
    corners, owners = split_longest_edge(faces.corners, np.arange(len(faces.corners), dtype=np.int64), patch_edge_m)
    areas = triangle_areas(corners)
    raw, sees = _raw_form_factors(corners, _midpoint_samples(corners, splits), _patch_normals(corners), areas,
                                  faces.plane_of[owners], faces.occluders, tol_m)
    # 一格只算看得到的那一份面積（取樣點看得到的比例）：貼牆的板子只蓋住半格，被蓋住的那半格不算進吸音與交換。
    visible_areas = areas * sees.mean(axis=1)
    seen = (raw.sum(axis=1) > 0.0) & (visible_areas > 0.0)
    hidden = float(areas.sum() - visible_areas[seen].sum())
    raw = raw[np.ix_(seen, seen)]
    kept_areas = visible_areas[seen]
    return ShellLateGeometry(
        layers=faces.layers,
        patch_layer=np.asarray(faces.layer_of[owners[seen]], dtype=np.int64),
        areas=kept_areas,
        centroids=np.asarray(corners[seen].mean(axis=1), dtype=np.float64),
        form_factors=normalized_form_factors(raw, kept_areas),
        hidden_area_m2=hidden,
    )


def solve_shell_late_energy_by_order(
    geometry: ShellLateGeometry,
    absorption: NDArray[np.float64],
    frequencies_hz: tuple[float, ...],
    *,
    max_order: int,
) -> LateEnergyOrderResult:
    """各圖層的無規入射吸音率（圖層數 × 頻點數）→ 逐階晚期能量（同長方體那一支的型別與算式）。"""
    if max_order < 1:
        raise ValueError(f"max_order 必須至少是 1，收到 {max_order}")
    if absorption.shape != (len(geometry.layers), len(frequencies_hz)):
        raise ValueError("吸音率表要是（材料圖層數 × 頻點數）")
    if not np.isfinite(absorption).all() or bool(((absorption < 0) | (absorption > 1)).any()):
        raise ValueError("無規入射吸音率必須有限且落在 [0,1]")
    alpha_patch = absorption[geometry.patch_layer]
    totals: list[FloatArray] = []
    orders: list[FloatArray] = []
    for begin in range(0, len(frequencies_hz), _FREQUENCIES_PER_CHUNK):
        chunk = alpha_patch[:, begin:begin + _FREQUENCIES_PER_CHUNK]
        transfer = np.asarray((1.0 - chunk)[:, None, :] * geometry.form_factors[:, :, None], dtype=np.float64)
        total, by_order = _raw_energy_and_orders(transfer, geometry.areas, max_order)
        totals.append(total)
        orders.append(by_order)
    raw_total, raw_by_order = np.concatenate(totals), np.concatenate(orders)
    alpha_bar = geometry.areas @ alpha_patch / float(geometry.areas.sum())
    bands = []
    for index, frequency in enumerate(frequencies_hz):
        ratio = _eyring_ratio(float(alpha_bar[index]))
        total_energy = float(raw_total[index]) * ratio
        by_order_energy = tuple(float(value) * ratio for value in raw_by_order[index])
        bands.append(LateEnergyOrderBand(
            frequency_hz=frequency,
            energy_by_order=by_order_energy,
            tail_energy=_tail_energy(total_energy, by_order_energy, frequency),
            late_reverberant_energy=total_energy,
        ))
    return LateEnergyOrderResult(max_order=max_order, bands=tuple(bands))
