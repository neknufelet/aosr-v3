"""既有固定阻抗案例顯式提供材料；不依賴產品的下游退路。"""
from collections.abc import Mapping
from aosr.geometry.shoebox import Wall
from aosr.materials.construction_models import ConstantImpedance, SurfaceMaterial


def fixed_materials(walls: Mapping[Wall, object]) -> dict[Wall, SurfaceMaterial]:
    """壞實數格的拒收題保留獨立合法材料，讓入口原有實數驗證負責拒收。"""
    return {wall: ConstantImpedance(impedance_pa_s_m=(value if isinstance(value, (int, float))
                                                   and not isinstance(value, bool) and value > 0. else 900.))
            for wall, value in walls.items()}
