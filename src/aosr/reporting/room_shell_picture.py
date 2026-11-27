"""自訂房間的剖視圖：把一堆帶圖層的三角形畫成一張 SVG（網頁上傳頁與背景計算共用）。

**怎麼剖。** 每個三角形帶一個「朝向空氣那一側」的法線；只畫從鏡頭看得到空氣那一側的面。
從房外斜上方看進去時，靠鏡頭的那幾面牆是背對鏡頭的，自然就不畫，看得到房內的地板、
遠側的牆、天雲與凸出的板子。遠的先畫、近的後畫（畫家演算法），不用 3D 函式庫。
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

_PALETTE = ("#c9c9c9", "#8a5a44", "#d9b38c", "#e07b39", "#f2efe6", "#3b7dd8",
            "#5fb36b", "#9b59b6", "#7fd3e6", "#b5651d", "#d64550", "#2e8b8b")
_AZIMUTH_DEG = -50.0
_ELEVATION_DEG = 32.0


def layer_color(index: int) -> str:
    """圖層的固定顏色；網頁的圖層清單用同一個，對得上圖。"""
    return _PALETTE[index % len(_PALETTE)]


def _shade(color: str, factor: float) -> str:
    red, green, blue = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgb({int(red * factor)},{int(green * factor)},{int(blue * factor)})"


def cutaway_svg(corners: NDArray[np.float64], air_normals: NDArray[np.float64], layers: Sequence[int],
                *, width: int = 720, height: int = 520) -> str:
    """``corners`` (N, 3, 3) 三角形角點；``air_normals`` (N, 3) 朝空氣的法線；``layers`` 逐列的圖層位置。"""
    azimuth, elevation = math.radians(_AZIMUTH_DEG), math.radians(_ELEVATION_DEG)
    view = np.array([math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth),
                     math.sin(elevation)])
    right = np.cross([0.0, 0.0, 1.0], view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    lengths = np.linalg.norm(air_normals, axis=1)
    facing = np.einsum("ij,j->i", air_normals, view) / np.where(lengths > 0, lengths, 1.0)
    shown = np.flatnonzero(facing > 0)
    if not len(shown):
        return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"></svg>'
    order = shown[np.argsort(np.einsum("ijk,k->i", corners[shown], view) / 3.0, kind="stable")]
    xs, ys = corners @ right, corners @ up
    pad = 12.0
    span_x = float(xs[shown].max() - xs[shown].min()) or 1.0
    span_y = float(ys[shown].max() - ys[shown].min()) or 1.0
    scale = min((width - 2 * pad) / span_x, (height - 2 * pad) / span_y)
    x0, y0 = float(xs[shown].min()), float(ys[shown].min())
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">']
    for index in order:
        points = " ".join(f"{pad + (x - x0) * scale:.1f},{height - pad - (y - y0) * scale:.1f}"
                          for x, y in zip(xs[index], ys[index], strict=True))
        fill = _shade(layer_color(int(layers[index])), 0.55 + 0.45 * float(facing[index]))
        parts.append(f'<polygon points="{points}" fill="{fill}" stroke="#333" stroke-width="0.4"/>')
    parts.append("</svg>")
    return "".join(parts)

