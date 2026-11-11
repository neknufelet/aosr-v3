"""第三階段梯度搜尋吃材料庫（#1082 段 2-3a）：使用者的可最佳化材料接成搜尋家族；不載入 JAX。

內建材料就是產品設定檔本身（``builtin_entries`` 由 ``construction_presets.toml`` 與 ``material_gradient.toml`` 生成），
所以搜尋設定與構造預設清單照設定檔原樣讀，這裡只把**使用者**材料庫裡的可最佳化材料一筆一個家族**接在後面**：
沒有使用者可最佳化材料時，設定、清單、指紋與資料夾名一個位元都不動。

一筆使用者可最佳化材料 → 一個家族：
- 家族代號＝材料代號；它唯一的構造預設代號也是材料代號（材料庫已保證代號跟內建的都不撞），構造是
  :func:`~aosr.materials.library.template_assembly`（初值排法、剛背）——同一份當起跑樣板也當換樣板構造。
- 合法厚度：能調的是產品厚度；不能調的只有初值那一個。空腔範圍照材料；沒有空腔這一格是 (0, 0)。
- 總深度上限：材料自己寫了就用它；沒寫用層自己排得出的最深（多孔最厚＋空腔最深）。個案上限跟它取嚴是 2-4 的事。
- 空腔初值空白或 0：樣板沒有空氣層，這一場搜尋**不調它的空腔**（跟內建 24K、96K 同一條規則），:func:`cavity_notes` 照實寫出來。
- 使用者材料不准掛構造預設（``preset_members``／``template_preset`` 只給內建材料對應舊家族用）：填了就拒收、點名哪一格；
  填的代號不存在另外講明。

固定材料只能整塊選或不選，不能當搜尋家族微調：起跑或網頁選的家族填了固定材料的代號就拒收。
固定材料當換樣板候選、每區准用清單、實測固定材料進搜尋在 2-3b：准用清單點名的固定材料用 ``fixed:<代號>`` 這個目標
整塊換上（:func:`fixed_candidates`），使用者的那幾筆也進快照（v2）。

快照：搜尋資料夾存一份 ``materials.json``（只放接成家族的那幾筆，正規化 JSON），整份 SHA-256 進資料夾名與 ``project.json``；
接續、報告、終點一律讀資料夾裡那份，不讀網頁資料夾裡現在的材料庫。
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, NamedTuple, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aosr.config.material_gradient import FamilySettings, MaterialGradientSettings
from aosr.materials.library import FixedEntry, LibraryEntry, OptimizableEntry, load_user_library, template_assembly

SNAPSHOT_VERSION: Final = "aosr.material_gradient_materials.v1"
# v2（#1082 段 2-3b）：快照裡另有准用清單點名的使用者固定材料；只在有那種材料時寫，沒有照舊寫 v1（逐位不變）。
SNAPSHOT_VERSION_WITH_FIXED: Final = "aosr.material_gradient_materials.v2"
SNAPSHOT_FILE: Final = "materials.json"


class PresetMaterial(Protocol):
    """搜尋要的構造預設只讀兩格：代號與材料內文（:class:`~aosr.config.construction_presets.ConstructionPreset` 也合）。"""

    @property
    def id(self) -> str: ...

    @property
    def material(self) -> dict[str, object]: ...


class LibraryPreset(BaseModel):
    """使用者可最佳化材料生成的構造：代號＝材料代號、材料＝初值排法（剛背多層構造）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    material: dict[str, object]


class MaterialsRefused(ValueError):
    """材料庫接不進搜尋；訊息是給人看的白話，點名哪一筆、哪一格。"""


class SearchMaterials(NamedTuple):
    """這一場搜尋的材料：搜尋設定（設定檔的家族在前、使用者的接在後）、構造清單（同樣接在後）、接進來的使用者材料。"""

    settings: MaterialGradientSettings
    presets: tuple[PresetMaterial, ...]
    entries: tuple[OptimizableEntry, ...]


class MaterialsSnapshot(BaseModel):
    """搜尋資料夾裡的 ``materials.json``（v1：只有接成家族的可最佳化材料，2-3a 寫的舊資料夾照讀）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    format_version: Literal["aosr.material_gradient_materials.v1"]
    materials: tuple[OptimizableEntry, ...]


class MaterialsSnapshotWithFixed(BaseModel):
    """v2：接成家族的可最佳化材料（在前、順序照材料庫）＋准用清單點名的使用者固定材料（在後、照代號排）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    format_version: Literal["aosr.material_gradient_materials.v2"]
    materials: tuple[OptimizableEntry, ...]
    fixed: tuple[FixedEntry, ...] = Field(min_length=1)


class FixedCandidate(BaseModel):
    """准用清單裡一個固定材料目標的構造：代號是目標字串 ``fixed:<材料代號>``、材料是那份完整牆面材料（物理格式）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    material: dict[str, object]


def _name(entry: LibraryEntry) -> str:
    return f"材料「{entry.label}」（{entry.id}）"


def read_user_entries(path: Path) -> tuple[LibraryEntry, ...]:
    """使用者材料庫；檔案不在是空的，讀不進來照講原因拒收（不當成空的）。"""
    try:
        return load_user_library(path)
    except (OSError, ValueError, ValidationError) as error:
        raise MaterialsRefused(f"使用者材料庫 {path} 讀不進來，搜尋不開：{error}") from error


def _family(entry: OptimizableEntry, known_presets: frozenset[str]) -> tuple[FamilySettings, LibraryPreset]:
    for field, ids in (("preset_members（屬於這組的構造預設）", entry.preset_members),
                       ("template_preset（換樣板構造）", () if entry.template_preset is None else (entry.template_preset,))):
        missing = [item for item in ids if item not in known_presets]
        if missing:
            raise MaterialsRefused(f"{_name(entry)}的 {field} 填了不存在的構造預設代號 {missing}，搜尋不開")
        if ids:
            raise MaterialsRefused(f"{_name(entry)}的 {field} 有填：使用者材料不能掛構造預設（這兩格只給內建材料對應舊家族用），"
                                   "請清掉這一格再開搜尋")
    porous, cavity = entry.porous_thickness, entry.cavity
    thicknesses = porous.choices_m if porous.choices_m is not None else (float(entry.porous.thickness_m),)
    initial_cavity = None if cavity is None or not cavity.initial_m else float(cavity.initial_m)
    depth = entry.max_total_depth_m if entry.max_total_depth_m is not None else entry.depth_bounds()[1]
    preset = library_preset(entry)
    try:
        family = FamilySettings(id=entry.id, presets=(entry.id,), thicknesses_m=tuple(float(value) for value in thicknesses),
                                cavity_range_m=(0.0, 0.0) if cavity is None else (float(cavity.range_m[0]), float(cavity.range_m[1])),
                                max_total_depth_m=float(depth), template_preset=entry.id,
                                initial_thickness_m=float(entry.porous.thickness_m), initial_cavity_m=initial_cavity)
    except ValidationError as error:
        raise MaterialsRefused(f"{_name(entry)}接不成搜尋家族：{error}") from error
    return family, preset


def library_preset(entry: OptimizableEntry) -> LibraryPreset:
    """一筆使用者可最佳化材料的構造：代號＝材料代號、材料＝初值排法（剛背）。"""
    return LibraryPreset(id=entry.id, material=template_assembly(entry).model_dump(mode="json"))


def with_user_presets(presets: Sequence[PresetMaterial], entries: Sequence[OptimizableEntry]) -> tuple[PresetMaterial, ...]:
    """只要構造清單的讀者（例如重建目標時設定已另外核過）：產品構造預設在前、使用者材料的構造接在後，不讀搜尋設定。"""
    return (*presets, *(library_preset(entry) for entry in entries))


def with_user_materials(settings: MaterialGradientSettings, presets: Sequence[PresetMaterial],
                        entries: Sequence[LibraryEntry]) -> SearchMaterials:
    """設定檔的家族與構造清單原樣在前，使用者的可最佳化材料一筆一個家族接在後；固定材料不接（不能微調）。"""
    chosen = tuple(entry for entry in entries if isinstance(entry, OptimizableEntry))
    if not chosen:
        return SearchMaterials(settings, tuple(presets), ())
    known = frozenset(preset.id for preset in presets)
    built = [_family(entry, known) for entry in chosen]
    merged = MaterialGradientSettings(descent=settings.descent, retune=settings.retune, candidates=settings.candidates,
                                      templates=settings.templates, export=settings.export,
                                      family=(*settings.family, *(family for family, _ in built)))
    return SearchMaterials(merged, (*presets, *(preset for _, preset in built)), chosen)


def cavity_notes(materials: SearchMaterials) -> tuple[str, ...]:
    """空腔標成能調、但初值空白或 0 的使用者材料：樣板沒有空氣層，這一場不調它的空腔，照實講。"""
    return tuple(f"{_name(entry)}的空腔初值是空的（或 0）：樣板沒有空氣層，這一場搜尋不調它的空腔"
                 "（跟內建 24K、96K 同一條規則）；要調空腔請給一個大於 0 的初值"
                 for entry in materials.entries
                 if entry.cavity is not None and entry.cavity.adjustable and not entry.cavity.initial_m)


def refuse_fixed_as_family(ids: Sequence[str | None], entries: Sequence[LibraryEntry]) -> None:
    """起跑配置或選的家族填了使用者固定材料的代號就拒收：固定材料只能整塊選或不選，不能微調。"""
    fixed = {entry.id: entry for entry in entries if isinstance(entry, FixedEntry)}
    for item in ids:
        if item is not None and item in fixed:
            raise MaterialsRefused(f"{_name(fixed[item])}是固定材料：搜尋只能整塊選或不選、不能微調，"
                                   "不能拿來當搜尋的家族或起跑樣板（要整塊用就把那一區設成固定）")


def snapshot_bytes(entries: Sequence[OptimizableEntry], fixed: Sequence[FixedEntry] = ()) -> bytes:
    """快照的正規化 JSON（鍵排序、兩格縮排、結尾換行）；材料順序照材料庫（家族順序也照它）。
    沒有固定材料寫 v1（跟 2-3a 逐位相同），有就寫 v2（固定材料照代號排）。"""
    document: BaseModel = (MaterialsSnapshot(format_version=SNAPSHOT_VERSION, materials=tuple(entries)) if not fixed else
                           MaterialsSnapshotWithFixed(format_version=SNAPSHOT_VERSION_WITH_FIXED, materials=tuple(entries),
                                                      fixed=tuple(sorted(fixed, key=lambda entry: entry.id))))
    return (json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


class Snapshot(NamedTuple):
    """讀回的快照：接成家族的可最佳化材料、准用清單點名的使用者固定材料（v1 沒有，是空的）。"""

    materials: tuple[OptimizableEntry, ...]
    fixed: tuple[FixedEntry, ...]


def read_snapshot_entries(path: Path, sha256: str) -> Snapshot:
    """搜尋資料夾裡的材料快照：整份 SHA-256 要等於開跑時記的，不同就拒收（不接續、不報告）；v1、v2 都讀。"""
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ValueError(f"材料快照 {path} 讀不到（{type(error).__name__}），拒收") from error
    if hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError(f"材料快照 {path} 跟開跑時不同（SHA-256 變了），拒收")
    version = json.loads(content).get("format_version")
    if version == SNAPSHOT_VERSION_WITH_FIXED:
        document = MaterialsSnapshotWithFixed.model_validate_json(content)
        return Snapshot(document.materials, document.fixed)
    return Snapshot(MaterialsSnapshot.model_validate_json(content).materials, ())


def read_snapshot(path: Path, sha256: str) -> tuple[OptimizableEntry, ...]:
    """快照裡接成家族的可最佳化材料（:func:`read_snapshot_entries` 的前半）。"""
    return read_snapshot_entries(path, sha256).materials


def fixed_candidates(targets: Sequence[str], presets: Sequence[PresetMaterial],
                     user_fixed: Sequence[FixedEntry]) -> tuple[FixedCandidate, ...]:
    """准用清單裡的 ``fixed:<代號>`` → 那份固定材料的構造（照目標字串排、不重複）。

    代號先找使用者的固定材料（材料庫已保證跟內建不撞），再找產品構造預設（內建的固定材料就是它們，
    :func:`~aosr.materials.library.builtin_entries`）；都沒有、或指到使用者的可最佳化材料就拒收、點名哪一個。
    """
    from aosr.config.material_gradient import fixed_target_id

    users = {entry.id: entry for entry in user_fixed}
    builtin = {preset.id: preset for preset in presets}
    found: dict[str, FixedCandidate] = {}
    for target in sorted(set(targets)):
        material_id = fixed_target_id(target)
        if material_id in users:
            material = users[material_id].surface_material().model_dump(mode="json")
        elif material_id in builtin and not isinstance(builtin[material_id], LibraryPreset):
            material = dict(builtin[material_id].material)
        else:
            raise MaterialsRefused(f"准用清單的固定材料 {material_id} 不在材料庫（內建構造預設或使用者的固定材料）裡，搜尋不開")
        found[target] = FixedCandidate(id=target, material=material)
    return tuple(found.values())


def user_fixed_entries(ids: Sequence[str], entries: Sequence[LibraryEntry]) -> tuple[FixedEntry, ...]:
    """准用清單點名、而且是使用者固定材料的那幾筆（照代號排）；快照要存它們，網頁之後改材料庫不影響這一場。"""
    wanted = set(ids)
    return tuple(sorted((entry for entry in entries if isinstance(entry, FixedEntry) and entry.id in wanted),
                        key=lambda entry: entry.id))
