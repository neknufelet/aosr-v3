"""票 #134：**「插值前先夾 α」那一行的牙**（突變證明）。

這一支在補一個**實測到的裁判缺口**，不是在懷疑產品：把 ``from_alpha`` 裡
``alpha = np.clip(alpha, 0.0, 1.0)``（插值**之前**那一道夾子）拿掉，原本那 81 個
case／常數／別名**全部照樣綠**（外部實測 0 筆不同）。原因是舊的那一筆
``clamps_alpha_above_one`` 兩個頻帶都 >1、軸上的點又剛好落在頻帶上，於是後面 ``sqrt``
那道 ``[1e-12, 1.0]`` 的夾子把「少了前夾子」整件事蓋掉了。

所以這裡做兩件事：

1. 把那一行**真的拿掉**（在一份**外部的副本**上，工作樹裡的產品一個字都不動），
2. 證明新加的那兩筆 case 在突變體上**真的紅**，而且在真的產品上是綠的。

沒有第 2 步，新加的 case 只是「多兩題會過的題目」；有了它，那兩題才是牙。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from blueprint import materials_cut1_probe as probe
from tests.engine import _materials_answers as answers

# 這一支咬的是哪一行：插值**之前**那道夾子。原文在產品裡只准出現一次——出現零次或兩次
# 就當場炸（那代表程式改過形狀，這個突變已經不是原本那個突變了）。
CLAMP_LINE = "alpha = np.clip(alpha, 0.0, 1.0)"
MUTANT_LINE = "alpha = alpha  # 突變體：拿掉插值前的夾子"

# 突變體暫時掛進 sys.modules 用的名字（掛完立刻拔掉）。
MUTANT_NAME = "aosr_materials_response_clamp_mutant"

# 新加的那兩筆（它們存在的理由就是要咬住上面那一行）。
CLAMP_CASE_IDS = [
    "response.MaterialResponse.from_alpha.mixed_out_of_range_alpha_inside_band_range",
    "response.MaterialResponse.from_alpha.negative_alpha_is_clamped_not_raised",
]


def _product_source() -> tuple[Path, str]:
    """產品那一支的位置與原始碼（從這支考卷往上兩層是 repo 根）。"""
    path = Path(__file__).resolve().parents[2] / "src" / "aosr" / "materials" / "response.py"
    return path, path.read_text(encoding="utf-8")


def _load_mutant() -> ModuleType:
    """把「拿掉前夾子」的那一版載入成一個**另外的**模組（工作樹的檔不動）。

    就地 ``exec`` 一份改過的原始碼，不寫任何檔案，**也不動真的那一支**（它還在
    ``sys.modules`` 裡，下面那幾條要拿它當對照）。

    突變體要**暫時**用自己的名字掛進 ``sys.modules``：``dataclasses`` 在處理一個類別的
    時候會回頭去 ``sys.modules[cls.__module__]`` 找那個模組（找不到就是
    ``AttributeError: '__dict__'``，第一版實測就是這個），掛完立刻拔掉。
    """
    path, source = _product_source()
    if source.count(CLAMP_LINE) != 1:
        raise AssertionError(
            f"產品裡的夾子那一行出現 {source.count(CLAMP_LINE)} 次，不是 1 次"
            "——這個突變已經不是原本那個突變了，先看產品改成什麼樣子"
        )
    spec = importlib.util.spec_from_file_location(MUTANT_NAME, path)
    if spec is None:
        raise AssertionError(f"載不動 {path}")
    mutant = importlib.util.module_from_spec(spec)
    code = compile(source.replace(CLAMP_LINE, MUTANT_LINE), str(path), "exec")
    sys.modules[MUTANT_NAME] = mutant
    try:
        exec(code, mutant.__dict__)  # noqa: S102  # expires=2026-12-08 reason=這就是突變體本身：一份只活在這支考卷裡的產品副本，來源是版控裡那一支檔
    finally:
        sys.modules.pop(MUTANT_NAME, None)
    return mutant


def _mutant_lookup(name: str) -> ModuleType:
    """跟正式考卷同一個位置的解析器，只有 ``response`` 換成突變體。"""
    if name == "response":
        return _load_mutant()
    return answers.engine_lookup(name)


def test_the_mutant_really_is_a_different_program() -> None:
    """突變體真的被改到了（不然下面每一條都在跟自己比）。"""
    _path, source = _product_source()
    assert CLAMP_LINE in source, "產品裡沒有那一行——這一支考卷失去對象"
    mutant = _load_mutant()
    assert mutant is not sys.modules.get("aosr.materials.response")
    assert MUTANT_NAME not in sys.modules, "突變體沒有從 sys.modules 拔掉——它會汙染別的題"


@pytest.mark.parametrize("case_id", CLAMP_CASE_IDS)
def test_the_real_product_passes_this_case(case_id: str) -> None:
    """控制組的右腳：真的產品在這一筆上跟 donor 逐格相同。"""
    actual = answers.run_here(answers.case_by_id(case_id))
    gaps = answers.differences(actual, answers.result_of(case_id), case_id)
    assert gaps == [], "真的產品在這一筆上就對不上：\n" + "\n".join(gaps)


@pytest.mark.parametrize("case_id", CLAMP_CASE_IDS)
def test_removing_the_pre_interpolation_clamp_makes_this_case_red(case_id: str) -> None:
    """控制組的左腳：拿掉前夾子，這一筆**必須**紅。

    這是新加那兩筆存在的理由：少了它們，同一個突變在整份考卷上一題都不紅（外部實測
    81 項全綠）。
    """
    mutated = probe.run_case(_mutant_lookup, answers.case_by_id(case_id))
    gaps = answers.differences(mutated, answers.result_of(case_id), case_id)
    assert gaps != [], "拿掉插值前的夾子之後這一筆竟然還是綠的——這一筆咬不到那一行"


def test_the_mutant_still_passes_the_case_that_could_not_catch_it() -> None:
    """反面對照：舊的 ``clamps_alpha_above_one`` 在同一個突變上**照樣綠**。

    這一條不是在挑舊 case 的毛病，是把「為什麼要新加那兩筆」釘成一條會紅的斷言：哪天有人
    把新加的那兩筆刪掉、以為舊的那一筆就夠，這裡會說「它不夠」。
    """
    case_id = "response.MaterialResponse.from_alpha.clamps_alpha_above_one"
    mutated = probe.run_case(_mutant_lookup, answers.case_by_id(case_id))
    assert answers.differences(mutated, answers.result_of(case_id)) == [], (
        "舊那一筆現在咬得住這個突變了——那新加的兩筆的理由要重寫"
    )
