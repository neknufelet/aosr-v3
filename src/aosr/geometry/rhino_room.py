"""讀 Rhino 標準檔（.3dm）成房間殼（票 #1299）。

**標準檔長什麼樣。** 上層圖層名固定：``ROOM`` 放外殼的面、``TREAT`` 放房內造型（每個
物件一個封閉實體）、``SPEAKER`` 與 ``SEAT`` 放喇叭與座位元件（Rhino 的 block 插入）、
``REF`` 與 ``00_NOTES`` 不讀。``ROOM`` 與 ``TREAT`` 底下一個子圖層＝一種材料，子圖層名
就是材料圖層名。元件的原點是聲學中心（喇叭）或雙耳中間（座位），元件的 +Y 軸是朝向。

**這一版只讀網格（Mesh）。** SketchUp 轉檔出來的標準檔全部是網格；直接在 Rhino 畫的
曲面（NURBS 曲面、多重曲面）這一版讀不了，會列成「讀不了」擋下來並講怎麼轉，不會靜靜跳過。

讀檔失敗（檔案太大、不是 .3dm、讀不出來）丟 :class:`RhinoRoomError`，訊息是白話。
"""
from __future__ import annotations

from pathlib import Path

import rhino3dm

from aosr.geometry.room_shell import (
    SEAT_KINDS, SPEAKER_KINDS, LayerTriangle, Placement, RoomShell, Solid, Vec3,
)


SHELL_LAYER = "ROOM"
SOLID_LAYER = "TREAT"
SPEAKER_LAYER = "SPEAKER"
SEAT_LAYER = "SEAT"
UNREAD_LAYERS: tuple[str, ...] = ("REF", "00_NOTES")


class RhinoRoomError(ValueError):
    """整份檔讀不了；訊息是給人看的白話。"""


def _point(vertex: rhino3dm.Point3d) -> Vec3:
    return (float(vertex.X), float(vertex.Y), float(vertex.Z))


def mesh_triangles(mesh: rhino3dm.Mesh, layer: str) -> tuple[LayerTriangle, ...]:
    """網格的每個面拆成三角形；四邊形照 0-1-2、0-2-3 拆，頂點順序保留法線方向。"""
    # 用雙精度取點：``Vertices[i]`` 回的是單精度，離原點遠時會差到焊點距離的量級。
    points = [_point(mesh.Vertices.Point3dAt(i)) for i in range(len(mesh.Vertices))]
    triangles = []
    for index in range(len(mesh.Faces)):
        a, b, c, d = mesh.Faces[index]
        triangles.append(LayerTriangle(layer, points[a], points[b], points[c]))
        if c != d:
            triangles.append(LayerTriangle(layer, points[a], points[c], points[d]))
    return tuple(triangles)


def _split_layer(full_path: str) -> tuple[str, str]:
    """``ROOM::Wall_乳膠漆`` → (``ROOM``, ``Wall_乳膠漆``)；沒有子圖層時材料圖層是空字串。"""
    top, _, rest = full_path.partition("::")
    return top, rest


def _placement(model: rhino3dm.File3dm, item: rhino3dm.InstanceReference,
               kinds: tuple[str, ...]) -> Placement | None:
    definition = model.InstanceDefinitions.FindId(item.ParentIdefId)
    name = "" if definition is None else definition.Name
    kind = next((k for k in kinds if name == k or name.startswith(f"{k}_")), None)
    if kind is None:
        return None
    x = item.Xform
    return Placement(kind, (x.M03, x.M13, x.M23), (x.M01, x.M11, x.M21))


def _object_label(obj: rhino3dm.File3dmObject, full_path: str) -> str:
    return obj.Attributes.Name or full_path


def read_rhino_room(path: Path, max_file_bytes: int) -> RoomShell:
    """讀一份 Rhino 標準檔；讀不了的物件記進 ``skipped`` 或擋下來的說明裡。"""
    size = path.stat().st_size
    if size > max_file_bytes:
        raise RhinoRoomError(f"檔案 {size / 1e6:.1f} MB，超過上限 {max_file_bytes / 1e6:.0f} MB")
    model = rhino3dm.File3dm.Read(str(path))
    if model is None:
        raise RhinoRoomError("讀不出這份檔，請確認是 Rhino 的 .3dm 檔")
    shell: list[LayerTriangle] = []
    solids: list[Solid] = []
    speakers: list[Placement] = []
    seats: list[Placement] = []
    skipped: list[str] = []
    unreadable: list[str] = []
    for obj in model.Objects:
        if obj.Attributes.Mode == rhino3dm.ObjectMode.InstanceDefinitionObject:
            # 元件定義裡面的物件（喇叭外形、標示線）不是場景裡的東西，座標也是元件自己的局部座標。
            continue
        full_path = model.Layers[obj.Attributes.LayerIndex].FullPath
        top, material = _split_layer(full_path)
        geometry = obj.Geometry
        if top in (SHELL_LAYER, SOLID_LAYER):
            if not isinstance(geometry, rhino3dm.Mesh):
                unreadable.append(f"{_object_label(obj, full_path)}（{type(geometry).__name__}，這一版只讀網格，"
                                  "請在 Rhino 用 Mesh 指令轉成網格再存）")
            elif len(geometry.Faces) == 0:
                # 零面的網格也能存進 .3dm；拆成三角形前先記下物件，外殼或造型才不會靜靜消失。
                unreadable.append(f"{_object_label(obj, full_path)}（這個網格沒有面，請在 Rhino 補上網格面）")
            elif top == SHELL_LAYER:
                shell.extend(mesh_triangles(geometry, material))
            else:
                solids.append(Solid(_object_label(obj, full_path), mesh_triangles(geometry, material)))
        elif top in (SPEAKER_LAYER, SEAT_LAYER) and isinstance(geometry, rhino3dm.InstanceReference):
            kinds = SPEAKER_KINDS if top == SPEAKER_LAYER else SEAT_KINDS
            found = _placement(model, geometry, kinds)
            if found is None:
                unreadable.append(f"{full_path} 圖層裡認不出種類的元件（元件名要用 "
                                  f"{'、'.join(kinds)} 開頭）")
            elif top == SPEAKER_LAYER:
                speakers.append(found)
            else:
                seats.append(found)
        elif top in (SPEAKER_LAYER, SEAT_LAYER):
            unreadable.append(f"{full_path} 圖層上不是元件的物件（{type(geometry).__name__}），"
                              "喇叭與座位要用元件放")
        elif top not in UNREAD_LAYERS:
            skipped.append(f"不在固定圖層裡的物件（{full_path}）")
    units = "m" if model.Settings.ModelUnitSystem == rhino3dm.UnitSystem.Meters else model.Settings.ModelUnitSystem.name
    return RoomShell(units, tuple(shell), tuple(solids), tuple(speakers), tuple(seats),
                     tuple(skipped), tuple(unreadable))
