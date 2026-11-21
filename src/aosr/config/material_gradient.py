"""第三階段牆面材料梯度搜尋設定的唯讀載入（#561）：連續下降、重調、候選確認迴圈、換樣板、樣板家族。

路徑必給（``docs/decisions/config-loaders-keep-path-required.md``）；正式檔是
``config_path("material_gradient.toml")``。數字全是工程暫定，依據紙
``stage-three-gradient-objective-low-frequency-lane.md`` 代價段；這裡只守形狀與彼此一致，不給預設值；
數字欄位一律嚴格：整數欄只收整數，浮點欄只收寫成浮點的數（字串、整數都拒收）。

兩個指紋（庚三，#561）：:attr:`MaterialGradientSettings.fingerprint` 只算庚二以前就有的欄位（不含 ``[templates]``
與家族的換樣板三欄），值跟加欄之前逐位相同——庚二帳本標頭、``project.json`` 與資料夾名都綁它，換樣板設定改了
不讓已收尾的庚二帳本變成接續不上；:attr:`MaterialGradientSettings.full_fingerprint` 算全部欄位，綁在換樣板帳本的標頭。
"""
from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StrictInt, StrictStr, model_validator


_FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
# 「不貼」（保留方案原本那面的牆）的保留目標代號：換樣板的目標是家族代號或它，所以家族不准叫這個名字。唯一一份來源。
NONE_TARGET: Final = "none"
# 安裝方式（#990）：每次搜尋一個值（命令列 ``--mount``、``project.json``、兩本帳本標頭）。rigid＝樣板照原樣（剛背）；
# on_wall＝貼在原牆上（新增層背後接方案原本那面牆）。唯一一份來源。
Mount = Literal["rigid", "on_wall"]
MOUNT_RIGID: Final = "rigid"
MOUNT_ON_WALL: Final = "on_wall"
MOUNTS: Final[tuple[Mount, ...]] = (MOUNT_RIGID, MOUNT_ON_WALL)
# 固定面的保留字：那一面（或那一區）用方案檔自己寫的材料、不帶變數、不換樣板（#1093）。含連字號，構造預設與
# 使用者材料庫的 id（小寫英數與底線）撞不到；住設定層是為了讓網頁不必 import 會拉 JAX 的搜尋模組。
SCHEME_MATERIAL: Final = "scheme-material"
# 換樣板的「整塊換成某個固定材料」目標（#1082 段 2-3b）：``fixed:<材料代號>``，只從每區准用清單進來（沒有清單的區照舊
# 只換家族與不貼）。冒號不在材料代號允許的字元裡，跟家族代號、構造預設代號、不貼都撞不到；面的配置裡也用同一個字串。
FIXED_TARGET_PREFIX: Final = "fixed:"


def is_fixed_target(value: str | None) -> bool:
    """是不是「整塊換成固定材料」的目標字串。"""
    return value is not None and value.startswith(FIXED_TARGET_PREFIX)


def fixed_target(material_id: str) -> str:
    return f"{FIXED_TARGET_PREFIX}{material_id}"


def fixed_target_id(target: str) -> str:
    """``fixed:<代號>`` 的代號；不是這個形狀就拒收。"""
    if not is_fixed_target(target) or not target[len(FIXED_TARGET_PREFIX):]:
        raise ValueError(f"{target!r} 不是 {FIXED_TARGET_PREFIX}<材料代號>")
    return target[len(FIXED_TARGET_PREFIX):]


def _exact_float(value: object) -> object:
    """浮點欄只收浮點數：字串 "0.05"、整數 1、布林都拒收（pydantic 的嚴格浮點仍會收整數）。"""
    if type(value) is not float:
        raise ValueError(f"必須寫成浮點數（帶小數點），收到 {type(value).__name__} {value!r}")
    return value


ExactFloat = Annotated[float, BeforeValidator(_exact_float)]


class DescentSettings(BaseModel):
    """投影梯度下降＋回溯線搜尋（Armijo）的步數、步長與兩個停止值。"""

    model_config = _FROZEN

    max_steps: StrictInt = Field(ge=1)
    initial_step: ExactFloat = Field(gt=0.0)
    backtrack_ratio: ExactFloat = Field(gt=0.0, lt=1.0)
    armijo_constant: ExactFloat = Field(gt=0.0, lt=1.0)
    gradient_norm_stop: ExactFloat = Field(gt=0.0)
    min_step: ExactFloat = Field(gt=0.0)

    @model_validator(mode="after")
    def _min_below_initial(self) -> Self:
        if self.min_step >= self.initial_step:
            raise ValueError("min_step 必須小於 initial_step")
        return self


class RetuneSettings(BaseModel):
    """取規格後空腔聯合重調的步數上限（線搜尋退到步長下限就提早停）。"""

    model_config = _FROZEN

    steps: StrictInt = Field(ge=1)


class CandidateSettings(BaseModel):
    """候選確認迴圈：每輪最多確認幾個候選、改善門檻（合分絕對差）、最多幾輪、求值次數上限。

    求值次數算次數不算秒數：值＋梯度算一次，線搜尋的每一個試探都算一次。求值次數上限限的是帳本記的求值次數（不含前段、不含接續時丟掉的半輪與重播）；機器實際算的次數可以更多。
    """

    model_config = _FROZEN

    max_per_round: StrictInt = Field(ge=1)
    improvement_threshold: ExactFloat = Field(gt=0.0)
    max_rounds: StrictInt = Field(ge=1)
    max_evaluations: StrictInt = Field(ge=1)


class TemplateSettings(BaseModel):
    """換樣板的候選確認迴圈（庚三）：每輪最多幾個單牆替換進重調、最多幾輪、求值次數上限、總時間上限（秒）。

    總時間只在一輪寫完決定列之後比帳本記的各輪秒數加總，不看牆鐘；改善門檻沿用 ``[candidates]``。
    """

    model_config = _FROZEN

    max_per_round: StrictInt = Field(ge=1)
    max_rounds: StrictInt = Field(ge=1)
    max_evaluations: StrictInt = Field(ge=1)
    max_seconds: StrictInt = Field(ge=1)


def centimeters(value: float) -> str:
    """範圍那幾句給人看的數字（#1199）：公尺換成公分、照 ``:g`` 去掉多餘的零（第 2 支的範圍格子是公分）。
    住設定層（#1234）：材料庫接成搜尋家族那一步（``materials`` 層）寫空腔下限那一句也用這一支。"""
    return f"{value * 100:g}"


class ExportSettings(BaseModel):
    """終點把六面規格匯出成材料：空腔小於 ``min_cavity_m``（公尺）就整層拿掉、不放空氣層。"""

    model_config = _FROZEN

    min_cavity_m: ExactFloat = Field(gt=0.0)

    def cavity_floor_problem(self, low_m: float) -> str | None:
        """空腔範圍的下限合不合匯出規則（#1234）：0（可以沒有空腔）或至少 ``min_cavity_m`` 才收，介於中間回一句白話（公分）。

        下限介於中間時，搜尋落在下限附近、匯出照規則整層拿掉，交出去的構造違反範圍；下限 0 本來就是「可以沒有空腔」，
        拿掉不違反。判法只有這一份：分區範圍（``depth_fit.range_problem``）、材料庫接成家族（``search_library``）、
        材料頁存檔、搜尋設定的家族都叫這裡；句子只講空腔本身，哪一區、哪一份材料由呼叫端加。"""
        if not 0.0 < low_m < self.min_cavity_m:
            return None
        floor = centimeters(self.min_cavity_m)
        return (f"空腔下限 {centimeters(low_m)} cm 要寫 0（可以沒有空腔）或至少 {floor} cm，"
                f"因為比 {floor} cm 薄的空腔匯出時會整層拿掉")


class FamilySettings(BaseModel):
    """一個樣板家族：哪些 preset、合法棉板厚度、空腔範圍、每面總深度上限（公尺），與換樣板用的構造和合法初值。

    ``template_preset`` 要在 ``presets`` 裡；``initial_thickness_m`` 要恰是合法厚度；``initial_cavity_m`` 有寫就不可為負、
    厚度加它不超過總深度上限（雙精度算，跟 ``is_feasible`` 同一個判法；在不在空腔範圍內由換樣板那一步的盒界核）。樣板沒有空氣層的家族不寫空腔初值（缺欄是 None，
    這一格是「沒有這一層」，不是預設值）；樣板有沒有空氣層要對照構造預設清單，由換樣板那一步核。
    """

    model_config = _FROZEN

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    presets: tuple[StrictStr, ...] = Field(min_length=1)
    thicknesses_m: tuple[ExactFloat, ...] = Field(min_length=1)
    cavity_range_m: tuple[ExactFloat, ExactFloat]
    max_total_depth_m: ExactFloat = Field(gt=0.0)
    template_preset: StrictStr
    initial_thickness_m: ExactFloat
    initial_cavity_m: ExactFloat | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.id == NONE_TARGET:
            raise ValueError(f"家族代號不能用 {NONE_TARGET}：{NONE_TARGET} 是「不貼」的保留代號，家族不能用")
        if len(set(self.presets)) != len(self.presets):
            raise ValueError(f"家族 {self.id} 的 preset 重複")
        thicknesses = self.thicknesses_m
        if any(value <= 0.0 for value in thicknesses) or list(thicknesses) != sorted(set(thicknesses)):
            raise ValueError(f"家族 {self.id} 的合法厚度必須是正數、由薄到厚、不重複")
        low, high = self.cavity_range_m
        if not 0.0 <= low <= high:
            raise ValueError(f"家族 {self.id} 的空腔範圍必須 0 ≤ 下限 ≤ 上限")
        if thicknesses[0] > self.max_total_depth_m:
            raise ValueError(f"家族 {self.id} 最薄的合法厚度已超過總深度上限，取規格無處可退")
        if self.template_preset not in self.presets:
            raise ValueError(f"家族 {self.id} 的換樣板構造 {self.template_preset} 不在它的 preset 清單裡")
        if self.initial_thickness_m not in thicknesses:
            raise ValueError(f"家族 {self.id} 的換樣板初始厚度 {self.initial_thickness_m} 不是合法厚度")
        cavity = self.initial_cavity_m
        if cavity is not None and cavity < 0.0:
            raise ValueError(f"家族 {self.id} 的換樣板初始空腔 {cavity} 不可為負")
        if self.initial_thickness_m + (0.0 if cavity is None else cavity) > self.max_total_depth_m:
            raise ValueError(f"家族 {self.id} 的換樣板初值厚度加空腔超過總深度上限")
        return self


class VprFamilySettings(BaseModel):
    """可調 VPR 的家族（#1200 第 1 支）：板重範圍（kg/m²）與可空的板重規格、背襯合法厚度、固定前層深度、總深度上限（公尺）與初值。

    跟 :class:`FamilySettings` 分開、也**不進** :class:`MaterialGradientSettings` 的家族清單：舊家族表示不出板重這個變數，
    加欄會動到每一個舊家族的指紋；搜尋那一層接上之前（第 2 支）沒有人讀它。內容由材料庫那一筆驗過（板重與厚度的範圍、
    初值、總深），這裡只存數字；板重或背襯不調時，範圍兩端相等、合法厚度只有初值那一個。總深＝固定前層＋板重除以鋼板密度＋背襯厚度；``template_preset`` 是初值排法那份構造的代號。
    """

    model_config = _FROZEN

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    plate_density_kg_m3: ExactFloat = Field(gt=0.0)
    plate_mass_range_kg_m2: tuple[ExactFloat, ExactFloat]
    plate_mass_choices_kg_m2: tuple[ExactFloat, ...] | None = None
    initial_plate_mass_kg_m2: ExactFloat = Field(gt=0.0)
    backing_thicknesses_m: tuple[ExactFloat, ...] = Field(min_length=1)
    initial_backing_thickness_m: ExactFloat = Field(gt=0.0)
    front_depth_m: ExactFloat = Field(ge=0.0)
    max_total_depth_m: ExactFloat = Field(gt=0.0)
    template_preset: StrictStr


class MaterialGradientSettings(BaseModel):
    """六節：下降、重調、候選確認、換樣板、匯出、家族；一個 preset 最多屬於一個家族。"""

    model_config = _FROZEN

    descent: DescentSettings
    retune: RetuneSettings
    candidates: CandidateSettings
    templates: TemplateSettings
    export: ExportSettings
    family: tuple[FamilySettings, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _distinct_families(self) -> Self:
        ids = [item.id for item in self.family]
        presets = [preset for item in self.family for preset in item.presets]
        if len(set(ids)) != len(ids) or len(set(presets)) != len(presets):
            raise ValueError("家族代號不可重複，一個 preset 最多屬於一個家族")
        return self

    @model_validator(mode="after")
    def _cavity_floors(self) -> Self:
        """每個家族的空腔下限照匯出規則核（:meth:`ExportSettings.cavity_floor_problem`，#1234）：家族的範圍跟最小空腔
        只在這一份設定裡碰得到面，設定檔的家族在這裡擋；使用者材料接成家族那一步先點名材料擋過。"""
        for item in self.family:
            problem = self.export.cavity_floor_problem(item.cavity_range_m[0])
            if problem is not None:
                raise ValueError(f"家族 {item.id} 的{problem}")
        return self

    def family_of(self, preset_id: str) -> FamilySettings | None:
        """這個 preset 屬於哪個家族；不在任何家族回 None。"""
        return next((item for item in self.family if preset_id in item.presets), None)

    def family_by_id(self, family_id: str) -> FamilySettings:
        """家族代號 → 家族；不認識就拒收。"""
        found = next((item for item in self.family if item.id == family_id), None)
        if found is None:
            raise ValueError(f"家族代號不認識：{family_id}")
        return found

    @property
    def fingerprint(self) -> str:
        """庚二以前的欄位（扣掉 ``[templates]`` 與家族的換樣板三欄）驗證後內容的正規化 JSON SHA-256 十六進位指紋，
        寫進庚二步驟帳本標頭與 ``project.json``；跟加欄之前同一份設定算出來逐位相同。"""
        content = self.model_dump(mode="json", exclude={"templates"})
        content["family"] = [{key: value for key, value in family.items() if key not in _TEMPLATE_FIELDS}
                             for family in content["family"]]
        return _digest(content)

    @property
    def full_fingerprint(self) -> str:
        """全部欄位（含 ``[templates]`` 與家族的換樣板三欄）的指紋，寫進換樣板帳本標頭。"""
        return _digest(self.model_dump(mode="json"))


_TEMPLATE_FIELDS: Final = frozenset(("template_preset", "initial_thickness_m", "initial_cavity_m"))


def _digest(content: object) -> str:
    text = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_material_gradient_settings(path: str | Path) -> MaterialGradientSettings:
    """讀 ``[descent]``、``[retune]``、``[candidates]``、``[templates]``、``[export]``、``[[family]]``；缺節、多欄或數字不合就報錯。"""
    with Path(path).open("rb") as config_file:
        data = tomllib.load(config_file)
    return MaterialGradientSettings.model_validate(data)


def with_max_steps(settings: MaterialGradientSettings, max_steps: int) -> MaterialGradientSettings:
    """開跑時另外指定的往下走步數上限（#1073）：其餘照原設定；步數照 :class:`DescentSettings` 的檢查（整數、至少 1）。"""
    descent = DescentSettings.model_validate({**settings.descent.model_dump(), "max_steps": max_steps})
    return settings.model_copy(update={"descent": descent})
