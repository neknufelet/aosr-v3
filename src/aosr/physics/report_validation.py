"""報表數值、牆面與錯誤訊息的驗證；不載入報表模型。"""
from __future__ import annotations

import math
from typing import Final

from pydantic import ValidationError, ValidationInfo

from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.geometry.shoebox import Wall
from aosr.physics.report_facts import POSITIVE_EXCLUSIVE_MINIMUM, SCATTERING_MINIMUM, SCATTERING_MAXIMUM


# 複數與逐頻阻抗這兩條材料形式的能力表代號；拒收訊息從那兩條 unsupported 的 note 讀。
UNSUPPORTED_MATERIALS: Final[tuple[str, ...]] = (
    "complex_impedance_by_wall",
    "frequency_dependent_impedance",
)



CAPABILITY_ENTRY: Final[str] = "three_lane_report"



class FieldValueError(ValueError):
    """驗證訊息保留原文，另帶機器可讀的欄位路徑。"""

    def __init__(self, field: str, message: str, *, pair_specific: bool = False) -> None:
        super().__init__(message)
        self.field = field
        self.pair_specific = pair_specific



class ReportInputError(ValueError):
    """對舊呼叫者仍是 ValueError；方案層另讀結構化欄位。"""

    def __init__(self, message: str, issues: tuple[tuple[str, str, bool], ...]) -> None:
        super().__init__(message)
        self.issues = issues



PER_REPORT_INPUT_FIELDS: Final[tuple[str, ...]] = ("source_m", "receiver_m")



def _number_is_finite(value: object) -> bool:
    """有限數字；布林不算數字（布林是 int 的子類，會靜靜變成 1.0）。"""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(value)



def _checked_number(value: object, where: str) -> float:
    """照命令列原本的寫法驗一格數字，訊息一字不變。"""
    if not _number_is_finite(value):
        raise ValueError(f"{where} 必須是有限數字")
    assert isinstance(value, int | float)
    return float(value)



class WiringError(RuntimeError):
    """呼叫端把這一層接錯了（不是使用者的輸入不好）。

    這一種錯誤不准被 :func:`load_input_document` 包成「輸入不合 ReportInput」——
    那會把「程式接線壞了」報成「你的輸入不好」，看訊息的人會去改 JSON 而不是改呼叫端。
    """



def _table_of(info: ValidationInfo) -> CapabilityTable:
    """從驗證脈絡拿這一次要用的能力表；沒帶就大聲炸，不偷偷載第二張。

    :func:`load_input_document` 在 ``model_validate(context=…)`` 裡把表放進來，
    所以命令列 ``--capabilities`` 指定的那一張就是拒收訊息讀的那一張。

    拿不到表是**接線錯誤**不是壞輸入，所以丟 :class:`WiringError`
    （``load_input_document`` 只把 :class:`~pydantic.ValidationError` 收成人話，
    這一種照原樣往上冒）。
    """
    table = (info.context or {}).get("table")
    if not isinstance(table, CapabilityTable):
        raise WiringError(
            "輸入驗證沒有帶著能力表：load_input_document 要收呼叫端指定的那一張"
        )
    return table



def _directivity_of(info: ValidationInfo) -> DirectivityDefaults:
    """可調範圍只從呼叫端的同一份驗證脈絡取得。"""
    directivity = (info.context or {}).get("directivity")
    if not isinstance(directivity, DirectivityDefaults):
        raise WiringError("輸入驗證沒有帶著指向性登記簿：load_input_document 必須由呼叫端給 directivity")
    return directivity



def _checked_mapping(value: object, where: str) -> dict[str, object]:
    """照命令列原本的寫法驗一層 JSON 物件，訊息一字不變。"""
    if not isinstance(value, dict):
        raise ValueError(f"{where} 必須是 JSON 物件")
    return {str(key): item for key, item in value.items()}



def _wall_numbers(
    value: object,
    *,
    where: str,
    allow_zero: bool,
    table: CapabilityTable,
) -> dict[str, float]:
    """六面牆各一格數字；阻抗要正、散射落在 [0,1]，複數或逐頻形式當場拒收。

    拒收訊息的那一句 hint 只從呼叫端傳進來的**那一張**表算（命令列走
    ``--capabilities``，考卷走它自己指定的檔）——同一個地方算一次，不在這裡再拼第二份。
    """
    fields = _checked_mapping(value, where)
    names = [wall.wall_name() for wall in Wall.all()]
    unknown = sorted(key for key in fields if key not in names)
    if unknown:
        raise ValueError(
            f"{where} 多了不認識的牆名 {unknown}；這一格只收 {names} 這六面。"
            "多打的那一個被靜靜忽略，改的人會以為改生效了，所以這裡直接擋"
        )
    result: dict[str, float] = {}
    for wall in Wall.all():
        name = wall.wall_name()
        cell_where = f"{where}.{name}"
        cell = fields.get(name)
        if isinstance(cell, dict | list | tuple):
            hint = (
                unsupported_materials_hint(table)
                if where == "impedance_pa_s_per_m_by_wall"
                else "散射係數只收一個實數"
            )
            raise FieldValueError(cell_where, f"{cell_where}：{hint}")
        try:
            number = _checked_number(cell, cell_where)
        except ValueError as exc:
            raise FieldValueError(cell_where, str(exc)) from exc
        if not allow_zero:
            if number <= POSITIVE_EXCLUSIVE_MINIMUM:
                raise FieldValueError(cell_where,
                    f"{cell_where} 必須是正實數阻抗；{unsupported_materials_hint(table)}"
                )
        elif not SCATTERING_MINIMUM <= number <= SCATTERING_MAXIMUM:
            raise FieldValueError(cell_where, f"{cell_where} 必須落在 [0,1]（散射係數）")
        result[name] = number
    return result



def _reason_or_value(value: float | None, reason: str | None, *, where: str) -> None:
    """值空要帶原因；有值不准帶原因。

    「帶原因」是去掉前後空白之後還有字：空字串與整串空白讀起來跟沒有原因一樣，
    契約放它過等於允許一份說不出為什麼的空值。
    """
    if value is None and (reason is None or not reason.strip()):
        raise ValueError(f"{where} 是空的就必須帶不可估原因（空字串與空白不算原因）")
    if value is not None and reason is not None:
        raise ValueError(f"{where} 有值就不准再給不可估原因（兩個只准出現一個）")



def unsupported_materials_hint(table: CapabilityTable) -> str:
    """複數與逐頻阻抗的拒收訊息來源：**呼叫端那一張**能力表上兩條 unsupported 的 ``note``。

    字串本身不寫死在程式裡，命令列與輸入模型走同一條路：命令列把 ``--capabilities``
    那一張表傳進來，這裡只讀它，不自己去載第二張（兩個來源就會對不上）。
    """
    notes = [
        item.note
        for item in table.for_entry(CAPABILITY_ENTRY).capability
        if item.materials in UNSUPPORTED_MATERIALS and item.status == "unsupported"
    ]
    if not notes:
        raise ValueError(
            f"能力表 {CAPABILITY_ENTRY} 沒有複數或逐頻阻抗的 unsupported 條目"
        )
    return " ".join(dict.fromkeys(notes))



def _hint_rejection(exc: ValidationError) -> str:
    """把 Pydantic 的多行 dump 收成一句人話：哪一欄錯、錯在哪。

    只留 Pydantic 自己指名的 ``loc`` 與 ``msg``；官網網址那一行是錯誤物件的說明文字
    （``str(exc)`` 的尾巴），不是給人看的判決，命令列那一層不收。訊息裡的換行也壓成
    一個空格——那一層印的是**一行**。

    欄名只說一次：模型的驗證器自己就在訊息開頭寫了一次欄位路徑
    （``impedance_pa_s_per_m_by_wall.floor 必須是正實數阻抗``、``sound_speed_m_s 必須是
    有限正數``），而 ``loc`` 指的是同一格。兩者指的是同一個前綴時，``loc`` 那一份收掉
    ——``loc`` 比訊息更細（``impedance_pa_s_per_m_by_wall`` 對
    ``impedance_pa_s_per_m_by_wall.floor``）時留訊息那一份，因為它多說了是哪一面牆。
    """
    lines: list[str] = []
    for error in exc.errors():
        message = _one_line(str(error["msg"]).removeprefix("Value error, "))
        cells = ".".join(str(cell) for cell in error["loc"])
        if cells and (message == cells or message.startswith(f"{cells}.")):
            # 訊息自己就從欄位路徑講起：它比 loc 更細（多說了是哪一面牆），留它。
            lines.append(message)
        elif cells and message.startswith(f"{cells} "):
            # loc 與訊息開頭同名（``room_m：room_m 必須是 JSON 物件``）：說一次就夠。
            lines.append(message)
        elif cells:
            lines.append(f"{cells}：{message}")
        else:
            lines.append(f"輸入：{message}")
    return "；".join(lines) if lines else "輸入不合契約但沒有可讀的欄位路徑"



def _one_line(text: str) -> str:
    """把一段話壓成一行（換行與連續空白收成一個空格）。"""
    return " ".join(text.split())



def _rejected(exc: ValidationError) -> ReportInputError:
    """模型的驗證失敗 → 一句人話的 :class:`ValueError`（命令列只看得見這一種形狀）。"""
    issues: list[tuple[str, str, bool]] = []
    for error in exc.errors():
        message = _one_line(str(error["msg"]).removeprefix("Value error, "))
        context = error.get("ctx")
        cause = context.get("error") if isinstance(context, dict) else None
        path = cause.field if isinstance(cause, FieldValueError) else ".".join(map(str, error["loc"]))
        specific = (cause.pair_specific if isinstance(cause, FieldValueError)
                    else path.split(".", 1)[0] in PER_REPORT_INPUT_FIELDS)
        issues.append((path, message, specific))
    return ReportInputError(f"輸入不合 ReportInput：{_hint_rejection(exc)}", tuple(issues))

