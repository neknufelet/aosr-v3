"""構造資料考卷：抓默默丟欄位、缺必填、未知分辨代號與不合法數值。"""
from __future__ import annotations

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from aosr.materials.construction_models import AirLayer, Assembly, BoardLayer, ConstantImpedance, PorousLayer, SurfaceMaterial


def _porous(**changes: object) -> dict[str, object]:
    return {"kind": "porous", "thickness_m": 0.05, "flow_resistivity_pa_s_m2": 19100.,
            "material_class": "fibrous", **changes}


@pytest.mark.parametrize("model,changes", [
    ("komatsu", {"porosity": 0.9}), ("komatsu", {"tortuosity": 1.1}),
    ("miki", {"porosity": 0.9}), ("miki", {"tortuosity": 1.1}),
    ("komatsu", {"porosity": None}), ("miki", {"tortuosity": None}),
    ("wilson_2015", {}), ("wilson_2015", {"porosity": 0.9}),
    ("wilson_2015", {"tortuosity": 1.1}),
    ("mechel_1976", {}), ("mechel_1976", {"porosity": 0.9, "tortuosity": 1.1}),
    ("mechel_1976", {"porosity": 0.9, "tortuosity": None}),
    ("mechel_1976", {"porosity": 1.01}), ("db", {}),
])
def test_unused_or_missing_model_parameters_rejected(model: str, changes: dict[str, object]) -> None:
    """抓模型忽略明給欄位、補上未提供的必填參數或接受未准用模型。"""
    with pytest.raises(ValidationError):
        PorousLayer.model_validate(_porous(model=model, **changes))


@pytest.mark.parametrize("changes", [
    {"kind": "unknown"}, {"extra": 1}, {"material_class": "wood"},
    {"thickness_m": 0}, {"flow_resistivity_pa_s_m2": -1}, {"porosity": float("nan")},
    {"thickness_m": float("inf")}, {"flow_resistivity_pa_s_m2": float("nan")},
])
def test_invalid_porous_data_rejected(changes: dict[str, object]) -> None:
    """抓多孔層分辨代號、類別、額外欄位或有限正數守門失效。"""
    with pytest.raises(ValidationError):
        PorousLayer.model_validate(_porous(**changes))


def test_required_fields_and_discriminators() -> None:
    """抓材料類別、背牆或非空層清單偷偷給預設，及表面材料分辨欄位失效。"""
    missing_class = _porous()
    missing_class.pop("material_class")
    with pytest.raises(ValidationError):
        PorousLayer.model_validate(missing_class)
    for data in ({"kind": "assembly", "layers": [_porous()]},
                 {"kind": "assembly", "layers": [], "backing": "rigid"},
                 {"kind": "assembly", "layers": [_porous()], "backing": "radiation"},
                 {"kind": "assembly", "layers": [{"kind": "unknown"}], "backing": "rigid"},
                 {"kind": "unknown"}, {"kind": "constant_impedance", "impedance_pa_s_m": 300., "extra": 1}):
        with pytest.raises(ValidationError):
            TypeAdapter(SurfaceMaterial).validate_python(data)
    assembly: SurfaceMaterial = TypeAdapter(SurfaceMaterial).validate_python({"kind": "assembly", "layers": [_porous()], "backing": "air"})
    assert isinstance(assembly, Assembly)
    assert isinstance(assembly.layers[0], PorousLayer)
    assert assembly.layers[0].model == "komatsu"


@pytest.mark.parametrize("kind,fields", [
    ("air", {"thickness_m": 0}), ("air", {"thickness_m": float("nan")}),
    ("air", {"thickness_m": float("inf")}), ("air", {"thickness_m": 0.03, "extra": 1}),
    ("board", {"surface_mass_kg_m2": 0, "resistance_pa_s_m": 0}),
    ("board", {"surface_mass_kg_m2": 2, "resistance_pa_s_m": -1}),
    ("board", {"surface_mass_kg_m2": float("inf"), "resistance_pa_s_m": 0}),
    ("board", {"surface_mass_kg_m2": 2, "resistance_pa_s_m": float("nan")}),
    ("constant_impedance", {"impedance_pa_s_m": 0}),
    ("constant_impedance", {"impedance_pa_s_m": float("inf")}),
    ("constant_impedance", {"impedance_pa_s_m": 300 + 1j}),
])
def test_other_layers_and_constant_reject_bad_numbers(kind: str, fields: dict[str, object]) -> None:
    """抓空氣、板與固定阻抗收下零厚度、負阻力、非有限值或複數常數。"""
    records: dict[str, type[BaseModel]] = {"air": AirLayer, "board": BoardLayer, "constant_impedance": ConstantImpedance}
    record = records[kind]
    with pytest.raises(ValidationError):
        record.model_validate({"kind": kind, **fields})


def test_legal_parameter_domains_and_immutable_layers() -> None:
    """抓擅自收緊曲折度或孔隙率，以及凍結構造仍可改內部層。"""
    wilson = PorousLayer.model_validate(_porous(model="wilson_2015", porosity=1.2, tortuosity=0.5))
    mechel = PorousLayer.model_validate(_porous(model="mechel_1976", porosity=1.))
    assembly = Assembly(layers=(wilson, mechel), backing="rigid")
    assert isinstance(assembly.layers, tuple)
    with pytest.raises(ValidationError):
        wilson.thickness_m = 0.06
    with pytest.raises(ValidationError):
        assembly.backing = "air"
    assert BoardLayer(surface_mass_kg_m2=2., resistance_pa_s_m=0.).resistance_pa_s_m == 0
