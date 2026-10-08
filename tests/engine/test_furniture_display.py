"""中文名稱逐值比對；答案來自家具決策紙第 3、16、17 條及施工單房間軸命名。"""
from aosr.config.furniture_materials import load_furniture_materials
from aosr.config.furniture_materials import FurnitureMaterialName
from typing import get_args
from aosr.config.paths import config_path
from aosr.geometry.furniture import FaceDirection, FurnitureKind
from aosr.reporting import display


def test_every_furniture_kind_material_and_face_has_chinese() -> None:
    kinds = {"sofa": "沙發", "chair": "座椅", "coffee_table": "茶几",
             "desk": "書桌", "ceiling_cloud": "天雲"}
    materials = {"fabric": "布面", "leather": "皮面", "wood": "木質",
                 "glass": "玻璃", "absorptive_cloud": "吸音天雲"}
    faces = {"top": "頂面", "bottom": "底面", "+x": "朝 +x 的面",
             "-x": "朝 -x 的面", "+y": "朝 +y 的面", "-y": "朝 -y 的面"}
    assert {kind.value: display.FURNITURE_KINDS[kind.value] for kind in FurnitureKind} == kinds
    registry = load_furniture_materials(config_path("furniture_materials.toml"))
    assert set(get_args(FurnitureMaterialName)) == materials.keys()
    assert {name: display.FURNITURE_MATERIALS[name] for name, _ in registry.materials()} == materials
    assert {face.value: display.FURNITURE_FACES[face.value] for face in FaceDirection} == faces
