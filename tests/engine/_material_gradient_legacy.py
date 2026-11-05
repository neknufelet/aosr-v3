"""歷史位元組控制組使用當時的家族；不重錄經過數值運算的固定答案。"""
from aosr.config.material_gradient import MaterialGradientSettings


def legacy_family_settings(base: MaterialGradientSettings) -> MaterialGradientSettings:
    document = base.model_dump()
    for index, family in enumerate(document["family"]):
        family.update(thicknesses_m=(0.05, 0.1) if index == 0 else (0.05,),
                      cavity_range_m=(0.0, 0.15), max_total_depth_m=0.15)
    return MaterialGradientSettings.model_validate(document)
