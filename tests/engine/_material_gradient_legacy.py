"""歷史位元組控制組使用當時的家族；不重錄經過數值運算的固定答案。"""
from aosr.config.material_gradient import MaterialGradientSettings


def legacy_family_settings(base: MaterialGradientSettings) -> MaterialGradientSettings:
    document = base.model_dump()
    for index, family in enumerate(document["family"]):
        family.update(thicknesses_m=(0.05, 0.1) if index == 0 else (0.05,),
                      cavity_range_m=(0.0, 0.15), max_total_depth_m=0.15)
    return MaterialGradientSettings.model_validate(document)


def legacy_descent_settings(base: MaterialGradientSettings) -> MaterialGradientSettings:
    """舊小房的投影遮罩控制明給停止門檻 0.5，與現行預設分開。"""
    return legacy_stopping_settings(legacy_family_settings(base))


def legacy_stopping_settings(base: MaterialGradientSettings) -> MaterialGradientSettings:
    """舊低頻唯讀標頭保留當時的 0.5 與設定指紋，不重錄答案。"""
    return base.model_copy(update={'descent': base.descent.model_copy(update={'gradient_norm_stop': 0.5})})
