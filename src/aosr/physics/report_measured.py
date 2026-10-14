"""實測表兩端的接縫資料；獨立端點求值，只報告、不訂警告門檻。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import Field

from aosr.materials.catalog_absorption import complex_random_incidence_absorption
from aosr.materials.construction_models import MeasuredImpedance
from aosr.physics.report_facts import FactsModel, NOT_MEASURED, facts
from aosr.physics.report_surfaces import WholeWallZone
from aosr.physics.wall_surfaces import assembly_endpoint_responses


class SeamValues(FactsModel):
    """同一端點的一組阻抗與兩種入射吸音率；數字不含兩組之差。"""

    real_pa_s_m: float = Field(ge=0, description="端點阻抗實部，Pa·s/m", json_schema_extra=facts("阻抗實部", "Pa*s/m", "絕對值", "可估"))
    imag_pa_s_m: float = Field(description="端點阻抗虛部，Pa·s/m；正時間指數", json_schema_extra=facts("阻抗虛部", "Pa*s/m", "絕對值", "可估"))
    normal_absorption: float = Field(description="端點垂直入射吸音率", json_schema_extra=facts("吸音率", "1", "入射能量", "可估"))
    random_absorption: float = Field(description="端點無規入射吸音率；各組使用自己的算法", json_schema_extra=facts("吸音率", "1", "入射能量", "可估"))


class MeasuredSeam(FactsModel):
    """表的精確端點，兩組數字各自保存。"""

    frequency_hz: float = Field(gt=0, description="實測表的精確端點頻率，Hz", json_schema_extra=facts("頻率", "Hz", "絕對值", NOT_MEASURED))
    measured: SeamValues = Field(description="實測表值；假設與角度無關", json_schema_extra=facts("實測端點組", "1", "沒有基準（端點資料容器）", NOT_MEASURED))
    base_assembly: SeamValues = Field(description="底構造端點值；保留自己的角度響應", json_schema_extra=facts("底構造端點組", "1", "沒有基準（端點資料容器）", NOT_MEASURED))


def _values(z: complex, alpha_random: float, rho_c: float) -> SeamValues:
    return SeamValues(real_pa_s_m=z.real, imag_pa_s_m=z.imag,
        normal_absorption=1 - abs((z - rho_c) / (z + rho_c)) ** 2, random_absorption=alpha_random)


def measured_seams(zones: Mapping[str, Sequence[WholeWallZone]] | None, *, density_kg_m3: float,
                   sound_speed_m_s: float) -> dict[str, tuple[MeasuredSeam, ...]] | None:
    """直接取表端點，不借三路頻率軸；空氣使用當次場景的密度與聲速。"""
    walls = {wall: entries[0].material for wall, entries in (zones or {}).items()
             if isinstance(entries[0].material, MeasuredImpedance)}
    if not walls:
        return None
    result = {}
    for wall, material in walls.items():
        assert isinstance(material, MeasuredImpedance)
        points = (material.points[0], material.points[-1])
        try:
            normal, diffuse, rho_c = assembly_endpoint_responses(material.base_assembly,
                tuple(point.frequency_hz for point in points), density_kg_m3=density_kg_m3, sound_speed_m_s=sound_speed_m_s)
        except ValueError as exc:
            # 表蓋住整條三路軸時底構造只在這裡求值；錯誤要跟三路一樣說出是哪面牆、哪張表。
            raise ValueError(f"{wall}：{material.coverage_text()}的接縫；底構造{exc}") from exc
        result[wall] = tuple(MeasuredSeam(frequency_hz=point.frequency_hz,
            measured=_values(complex(point.real_pa_s_m, point.imag_pa_s_m),
                complex_random_incidence_absorption(complex(point.real_pa_s_m, point.imag_pa_s_m) / rho_c), rho_c),
            base_assembly=_values(z, alpha, rho_c))
            for point, z, alpha in zip(points, normal, diffuse, strict=True))
    return result
