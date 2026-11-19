"""「鋼板直接貼在棉上」的構造（#1208 第 1 支）函式層考卷：判斷看構造不看名字、界線從設定檔讀、開跑計畫拒收、
材料庫頁與分區頁的介面回覆多哪一格、網頁行程照樣沒有 JAX。

資料全是自編的：房間是左右對稱考卷那間假房間（5.0 × 4.0 × 2.7 m，``_zone_symmetry_room``），黏住鋼板的背襯用
``_bonded_backing_cases`` 的原始欄位，材料代號與名字是這一檔自己取的。
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from pydantic import TypeAdapter

from aosr.config.paths import config_path
from aosr.gui import reference_only
from aosr.materials.construction_models import SurfaceMaterial
from aosr.materials.library import LibraryEntry, builtin_entries
from tests.engine._bonded_backing_cases import BONDED_VPR, MELAMINE_50, MELAMINE_100, MELAMINE_BONDED, SLOTTED_FRONT, STEEL_1MM, STEEL_2_5MM

Doc = dict[str, object]
_MATERIAL: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
_ENTRY: TypeAdapter[LibraryEntry] = TypeAdapter(LibraryEntry)
_SOURCE = "合成：考卷自己編的數字"
PLAIN: Doc = {"kind": "assembly", "backing": "rigid", "layers": [STEEL_2_5MM, MELAMINE_100]}  # 同兩層，背襯沒黏住
FRONTED: Doc = {"kind": "assembly", "backing": "rigid", "layers": [SLOTTED_FRONT, MELAMINE_50, STEEL_1MM, MELAMINE_BONDED]}
WALL: Doc = {"kind": "constant_impedance", "impedance_pa_s_m": 71500.0}
WOOL_ONLY: Doc = {"kind": "assembly", "backing": "rigid", "layers": [MELAMINE_50]}
_ROWS = [{"frequency_hz": 110.0, "real_pa_s_m": 930.5, "imag_pa_s_m": -260.5}, {"frequency_hz": 410.0, "real_pa_s_m": 705.5, "imag_pa_s_m": -90.5}]


def _table(base: Doc) -> Doc:
    return {"kind": "measured_impedance", "points": _ROWS, "base_assembly": base, "source_and_conditions": "合成表，沒有量測條件"}


def _mounted(layers: list[Doc], wall: Doc) -> Doc:
    return {"kind": "mounted_on_wall", "layers": layers, "wall": wall}


def _fixed(key: str, label: str, material: Doc) -> Doc:
    return {"id": key, "label": label, "provenance": "model", "source": _SOURCE, "links": [], "notes": [], "kind": "fixed",
            "material": material}


HITS: dict[str, Doc] = {
    "多層構造：鋼板後面黏住的棉": BONDED_VPR,
    "多層構造：開槽板與棉在前、黏住的棉在後": FRONTED,
    "貼在原牆上：貼上去的幾層裡有": _mounted(cast(list[Doc], BONDED_VPR["layers"]), WALL),
    "貼在原牆上：原牆本身是這種多層構造": _mounted([MELAMINE_50], BONDED_VPR),
    "貼在原牆上：原牆是實測表、底構造是這種": _mounted([MELAMINE_50], _table(BONDED_VPR)),
    "實測表：底構造是這種": _table(BONDED_VPR),
}
MISSES: dict[str, Doc] = {
    "多層構造：同樣的鋼板與棉、沒黏住": PLAIN,
    "多層構造：只有棉": WOOL_ONLY,
    "貼在原牆上：貼的與原牆都不是": _mounted(cast(list[Doc], PLAIN["layers"]), PLAIN),
    "貼在原牆上：原牆是實測表、底構造不是": _mounted([MELAMINE_50], _table(PLAIN)),
    "實測表：底構造不是": _table(PLAIN),
    "固定阻抗": WALL,
}


@pytest.mark.parametrize("name", HITS)
def test_a_construction_with_a_backing_glued_to_the_plate_is_reference_only(name: str) -> None:
    assert reference_only.reference_only(_MATERIAL.validate_python(HITS[name])) is True


@pytest.mark.parametrize("name", MISSES)
def test_a_construction_without_a_glued_backing_is_not(name: str) -> None:
    assert reference_only.reference_only(_MATERIAL.validate_python(MISSES[name])) is False


def test_it_reads_the_layers_and_never_the_name() -> None:
    """名字叫 VPR、層不是：不算；名字完全沒提、層是：算。材料庫的實測表（虛部可以沒有的那一種）看底構造。"""
    named = _ENTRY.validate_python(_fixed("own_named_vpr", "VPR 鋼板共振吸音（其實沒黏）", PLAIN))
    quiet = _ENTRY.validate_python(_fixed("own_quiet_panel", "自編牆板", BONDED_VPR))
    rows = [{"frequency_hz": 110.0, "real_pa_s_m": 930.5}, {"frequency_hz": 410.0, "real_pa_s_m": 705.5}]
    table = _ENTRY.validate_python(_fixed("own_table_panel", "自編表列板", {**_table(BONDED_VPR), "points": rows}))
    assert [reference_only.entry_reference_only(entry) for entry in (named, quiet, table)] == [False, True, True]
    assert reference_only.listing_block((named, quiet, table))["ids"] == ["own_quiet_panel", "own_table_panel"]


def test_only_fixed_entries_are_judged_and_no_builtin_material_is_one() -> None:
    """可最佳化材料與可調 VPR 不歸這一支（程式最佳化本來就不收可調 VPR，#1200）；內建材料一筆都不是。"""
    from tests.engine.test_material_library_vpr import VPR, _vpr

    tunable = _ENTRY.validate_python(_vpr())
    assert tunable.id == VPR and tunable.kind == "tunable_layers"
    assert reference_only.entry_reference_only(tunable) is False
    assert reference_only.listing_block(builtin_entries())["ids"] == []


def _band_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, upper: str) -> None:
    """登記簿抄一份到暫存資料夾、把第一階段計分範圍的上界換掉，讓這一支讀那一份。"""
    text = config_path("quality_targets.toml").read_text(encoding="utf-8")
    old = 'key = "stage_one_low_frequency.scored_range_hz"\nvalue = [20.0, 300.0]'
    assert text.count(old) == 1
    changed = tmp_path / "quality_targets.toml"
    changed.write_text(text.replace(old, old.replace("300.0", upper)), encoding="utf-8")
    monkeypatch.setattr(reference_only, "config_path", lambda name: changed if name == "quality_targets.toml" else config_path(name))


def test_the_band_comes_from_the_registry_and_every_sentence_follows_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert reference_only.band_hz() == 300.0
    assert reference_only.words() == {
        "band_hz": 300.0,
        "sentence": "VPR（鋼板共振吸音）這類「鋼板直接貼在棉上」的構造：300 Hz 以下的數字只能參考。",
        "band_words": "300 Hz 以下只能參考",
        "map_words": ["300 Hz 以下", "數字只能參考"],
        "why": "程式算這類構造用的是簡化算法（一維、鋼板當成整塊一起動）。跟 Fraunhofer 的實測比，算出來的共振偏高、頻寬偏窄。"
               "數字沒有改，照舊算；「指定貼一種」照舊能選、能貼，「由程式最佳化」不會挑它。",
        "why_only": "程式算這類構造用的是簡化算法（一維、鋼板當成整塊一起動）。跟 Fraunhofer 的實測比，算出來的共振偏高、頻寬偏窄。"
                    "數字沒有改，照舊算。",
        "kind_words": "；這一筆 300 Hz 以下的數字只能參考，所以「由程式最佳化」不會挑，要用就在分區頁「指定貼一種」。",
        "search_why": "這類「鋼板直接貼在棉上」的構造 300 Hz 以下的數字只能參考，所以不進程式最佳化；要用，把這一區改成「指定貼一種」。",
        "listed_why": "這一區的清單裡本來有它，程式不會挑。這類「鋼板直接貼在棉上」的構造 300 Hz 以下的數字只能參考，"
                      "所以不進程式最佳化；要用，把這一區改成「指定貼一種」。",
    }
    assert reference_only.plan_refused("前牆「z1」", "自編牆板") == (
        "前牆「z1」的准用清單裡有自編牆板：這類「鋼板直接貼在棉上」的構造 300 Hz 以下的數字只能參考，所以不進程式最佳化；"
        "請在分區頁取消它，或把這一區改成「指定貼一種」")
    before = reference_only.words()
    _band_file(tmp_path, monkeypatch, "262.5")
    after = reference_only.words()
    assert after["band_hz"] == 262.5 and after["map_words"] == ["262.5 Hz 以下", "數字只能參考"]
    for key in ("sentence", "band_words", "kind_words", "search_why", "listed_why"):
        assert after[key] == cast(str, before[key]).replace("300 Hz", "262.5 Hz") != before[key]
    assert (after["why"], after["why_only"]) == (before["why"], before["why_only"])  # 這兩句沒有界線
    assert "262.5 Hz 以下" in reference_only.plan_refused("前牆「z1」", "自編牆板")
