"""材料頁的唯讀元件目錄。樣板依層的物性合併，尺寸與出處只作展示。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import cast

from aosr.config.material_gradient import MaterialGradientSettings
from aosr.config.paths import config_path
from aosr.config.porous_model_ranges import KomatsuRanges, load_porous_model_ranges
from aosr.gui.problem_text import MATERIAL_MODELS, _enum_word, kind_word, material_form_spec
from aosr.materials.construction_models import Assembly, Layer, PorousLayer
from aosr.materials.library import PROVENANCE_LABELS, builtin_entries, fixed_surface


_HOLES = ("thickness_m", "hole_diameter_m", "hole_spacing_m", "hole_pattern", "hole_spacing_2_m", "open_area", "surface_mass_kg_m2")
_DIMENSIONS = {
    "porous": ("thickness_m",), "fabric": (), "micro_perforated": _HOLES, "perforated": _HOLES,
    "slotted": ("thickness_m", "slot_width_m", "slot_spacing_m", "open_area", "surface_mass_kg_m2"),
    "board": (), "air": ("thickness_m",),
}
Value = float | str
Fields = dict[str, dict[str, object]]
Identity = tuple[tuple[str, Value], ...]


def _number(value: float, scale: int = 1) -> str:
    """由十進位公尺／比例換算顯示值，不讓二進位尾數或收位改變尺寸。"""
    return format((Decimal(str(value)) * scale).normalize(), ",f")


def _word(fields: Fields, key: str, value: Value) -> str:
    spec = fields[key]
    unit = str(spec["unit"])
    if isinstance(value, str):
        shown = cast(dict[str, str], spec["options"])[value]
    elif unit == "m":
        shown, unit = _number(value, 1000), "mm"
    elif key == "open_area":
        return f"{spec['label']} {_number(value, 100)}%"
    else:
        shown = _number(value)
    return f"{spec['label']} {shown}" + (f" {unit}" if unit else "")


@dataclass
class Element:
    kind: str
    name: str
    parameters: list[str]
    material_class: str | None = None
    range_title: str = "還沒填範圍"
    ranges: list[str] = field(default_factory=list)
    spec_title: str = "還沒填規格"
    families: dict[str, tuple[float, ...]] = field(default_factory=dict)
    template_note: str = "內建樣板："
    templates: dict[str, list[str]] = field(default_factory=dict)
    provenance: list[str] = field(default_factory=list)
    flow_sources: dict[str, list[str]] = field(default_factory=dict)

    def payload(self) -> dict[str, object]:
        return asdict(self) | {"flow_sources": [f"流阻：{source}（{'、'.join(labels)}）" for source, labels in self.flow_sources.items()]}


def _research_word(key: str, value: str, fallback: str) -> str:
    description = KomatsuRanges.model_fields[key].description or ""
    try:
        return _enum_word(description.removesuffix("）"), value)
    except ValueError:
        return f"{value}（{fallback}）"


def _set_range(row: Element, identity: dict[str, Value], research: KomatsuRanges) -> None:
    if identity.get("model") == "komatsu":
        row.range_title = "原研究量過的範圍（不是準確性界線）"
        row.ranges = [
            f"小松研究樣品表的流阻 {_number(research.sample_flow_resistivity_min_pa_s_m2)}–{_number(research.sample_flow_resistivity_max_pa_s_m2)} Pa·s/m²",
            f"小松研究量測方法所列頻率 {_number(research.measured_frequency_min_hz)}–{_number(research.measured_frequency_max_hz)} Hz",
        ]
        if not row.families:
            category = _research_word("fitted_material_class", research.fitted_material_class, "研究樣品類別")
            words = [_research_word("fitted_materials", value, f"研究樣品，屬於{category}") for value in research.fitted_materials]
            row.ranges.append(f"原研究擬合的是{'與'.join(words)}，這一筆不在其中")
    elif row.kind in ("micro_perforated", "perforated"):
        attr, sign = ("le", "≤") if row.kind == "micro_perforated" else ("ge", "≥")
        bounds = MATERIAL_MODELS[row.kind].model_fields["hole_diameter_m"].metadata
        edge, = [getattr(bound, attr) for bound in bounds if hasattr(bound, attr)]
        row.range_title = "材料欄位的孔徑限制"
        row.ranges = [f"{kind_word(row.kind)}孔徑 {sign} {_number(edge, 1000)} mm"]


def _element(kind: str, identity: Identity, fields: Fields) -> Element:
    props = dict(identity)
    name = kind_word(kind)
    for key in ("flow_resistivity_pa_s_m2", "flow_resistance_pa_s_m"):
        if key in props:
            name += "　" + _word(fields, key, props[key])
    if "backing_fabric_surface_mass_kg_m2" in props:
        name += "（背面貼布）"
    row = Element(kind, name, [_word(fields, k, v) for k, v in identity])
    if "material_class" in props:
        row.material_class = str(props["material_class"])
    return row


def _record_layer(row: Element, layer: Layer, fields: Fields, label: str, provenance: str) -> None:
    data = layer.model_dump()
    sizes = row.templates.setdefault(label, [])
    size = "、".join(_word(fields, k, data[k]) for k in _DIMENSIONS[row.kind] if k in data)
    if size and size not in sizes:
        sizes.append(size)
    if provenance not in row.provenance:
        row.provenance.append(provenance)
    if isinstance(layer, PorousLayer):
        options = cast(dict[str, str], fields["flow_resistivity_source"]["options"])
        labels = row.flow_sources.setdefault(options[layer.flow_resistivity_source], [])
        if label not in labels:
            labels.append(label)


def _distinguish_names(rows: list[Element]) -> None:
    """同一流阻的不同棉／布或不同貼布：名字再帶上該組不同的物性。"""
    groups: dict[str, list[Element]] = defaultdict(list)
    for row in rows:
        groups[row.name].append(row)
    for peers in groups.values():
        if len(peers) > 1:
            common = set.intersection(*(set(row.parameters) for row in peers))
            for row in peers:
                extra = [p for p in row.parameters if p not in common]
                if extra:
                    row.name += "　" + "、".join(extra)


def material_elements(settings: MaterialGradientSettings) -> dict[str, object]:
    """依指定種類、首次遇到的樣板排序。家族厚度只給其樣板中同身分的多孔層。"""
    form = material_form_spec()
    research = load_porous_model_ranges(config_path("porous_model_ranges.toml")).komatsu
    groups: dict[str, dict[Identity, Element]] = {kind: {} for kind in _DIMENSIONS}
    entries = builtin_entries()
    family_ids = {family.id for family in settings.family}
    family_labels = {entry.id: entry.label.partition("（")[0] for entry in entries if entry.id in family_ids}
    template_families = {family.template_preset: family for family in settings.family}
    for entry in entries:
        surface = fixed_surface(entry)
        if not isinstance(surface, Assembly):
            continue
        family = template_families.get(entry.id)
        for layer in surface.layers:
            kind: str = layer.kind
            fields = cast(Fields, form[kind]["fields"])
            props = layer.model_dump(exclude={"kind", "flow_resistivity_source", *_DIMENSIONS[kind]})
            identity = tuple(props.items())
            if identity not in groups[kind]:
                groups[kind][identity] = _element(kind, identity, fields)
            row = groups[kind][identity]
            _record_layer(row, layer, fields, entry.label, entry.provenance)
            if isinstance(layer, PorousLayer) and family is not None:
                if family.id in family_labels:
                    flow = _word(fields, "flow_resistivity_pa_s_m2", layer.flow_resistivity_pa_s_m2)
                    row.name = f"{family_labels[family.id]}　{flow}"
                row.spec_title = "搜尋現在用的合法厚度（工程暫定）"
                row.families[family.id] = family.thicknesses_m

    rows: list[Element] = []
    for kind, members in groups.items():
        if not members:
            fields = cast(Fields, form[kind]["fields"])
            row = _element(kind, (), fields)
            row.template_note = "內建樣板沒有這一種"
            row.parameters = [str(spec["label"]) for key, spec in fields.items() if key not in ("kind", "flow_resistivity_source")]
            members[()] = row
        for identity, row in members.items():
            _set_range(row, dict(identity), research)
            rows.append(row)
    _distinguish_names(rows)
    return {"elements": [row.payload() for row in rows], "family_labels": family_labels, "provenance_labels": PROVENANCE_LABELS}
