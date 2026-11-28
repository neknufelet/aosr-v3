"""「鋼板直接貼在棉上」的構造（#1208 第 1 支）函式層考卷：判斷看構造不看名字、界線從設定檔讀、開跑計畫拒收、
材料庫頁與分區頁的介面回覆多哪一格、網頁行程照樣沒有 JAX。

資料全是自編的：房間是左右對稱考卷那間假房間（5.0 × 4.0 × 2.7 m，``_zone_symmetry_room``），黏住鋼板的背襯用
``_bonded_backing_cases`` 的原始欄位，材料代號與名字是這一檔自己取的。
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from pydantic import TypeAdapter
from starlette.testclient import TestClient

from aosr.config.construction_presets import load_construction_presets
from aosr.config.material_gradient import load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.gui import reference_only, zone_limits
from aosr.gui.app import GuiSettings, create_app
from aosr.gui.gradient_jobs import GradientJobs, plan_fixed
from aosr.gui.gradient_plan import GradientRequest, PlanRefused, SearchPlan, library_depth_targets, plan_search, start_arguments
from aosr.gui.material_editor import user_library_path
from aosr.gui.zone_design import DesignSettings
from aosr.gui.zone_limits import DepthTargets
from aosr.materials.construction_models import SurfaceMaterial
from aosr.materials.library import LibraryEntry, builtin_entries, dump_user_library
from aosr.reporting.scheme import Scheme
from aosr.runtime import child_process_env
from tests.engine._construction_cases import construction
from tests.engine._cpu_pins import forward_cpu_pins
from tests.engine._construction_preset_answers import FITTED_VPR_IDS
from tests.engine._bonded_backing_cases import BONDED_VPR, MELAMINE_50, MELAMINE_100, MELAMINE_BONDED, SLOTTED_FRONT, STEEL_1MM, STEEL_2_5MM
from tests.engine._zone_symmetry_room import FRONT_PAIR, NAME, SIDE, symmetry_document
from tests.engine.test_gui_gradient_jobs import FakeLauncher

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
    assert cast(Doc, reference_only.listing_block((named, quiet, table)))["ids"] == ["own_quiet_panel", "own_table_panel"]


def test_only_fixed_entries_are_judged_and_the_fitted_1mm_preset_is_flagged() -> None:
    """可最佳化材料與可調 VPR 不歸這一支（程式最佳化本來就不收可調 VPR，#1200）；本次收錄的 1 毫米擬合預設命中。"""
    from tests.engine.test_material_library_vpr import VPR, _vpr

    tunable = _ENTRY.validate_python(_vpr())
    assert tunable.id == VPR and tunable.kind == "tunable_layers"
    assert reference_only.entry_reference_only(tunable) is False
    assert cast(Doc, reference_only.listing_block(builtin_entries()))["ids"] == list(FITTED_VPR_IDS)


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
        "map_line": "300 Hz 以下數字只能參考",
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


# ---- 開跑計畫拒收 ----

SETTINGS = load_material_gradient_settings(config_path("material_gradient.toml"))
HIT, PLAIN_ID, HIT_LABEL = "own_quiet_panel", "own_plain_panel", "自編牆板"
USER: list[Doc] = [_fixed(HIT, HIT_LABEL, BONDED_VPR), _fixed(PLAIN_ID, "自編對照板", PLAIN)]
ENTRIES = tuple(_ENTRY.validate_python(entry) for entry in USER)
REFUSED = ("前牆「a」的准用清單裡有自編牆板：這類「鋼板直接貼在棉上」的構造 300 Hz 以下的數字只能參考，所以不進程式最佳化；"
           "請在分區頁取消它，或把這一區改成「指定貼一種」")


def _design(allowed: list[str], **fields: object) -> DesignSettings:
    """前牆（x0）切兩塊，a 那一塊由程式最佳化、清單照給；其餘照預設（指定貼、不動）。"""
    return DesignSettings.model_validate({"schema_version": "aosr.zone_design.v1", "scheme_id": NAME,
                                          "zones": {"x0": {"a": {"mode": "search", "allowed": allowed, **fields}}}})


def _plan(allowed: list[str], **fields: object) -> SearchPlan:
    fixed_ids = frozenset((*(preset.id for preset in load_construction_presets(config_path("construction_presets.toml"))),
                           *(entry.id for entry in ENTRIES)))
    return plan_search(Scheme.model_validate(symmetry_document({"x0": FRONT_PAIR})), _design(allowed, **fields), GradientRequest(),
                       SETTINGS, fixed_ids, depth_targets=library_depth_targets(SETTINGS, ENTRIES))


def test_a_list_with_such_a_construction_is_refused_by_name_and_opens_without_it() -> None:
    with pytest.raises(PlanRefused) as caught:
        _plan(["glass_wool_48k", HIT, "none"])
    assert str(caught.value) == REFUSED
    opened = _plan(["glass_wool_48k", "none"])
    assert opened.allowed == {"x0@a": ("glass_wool_48k", "none")}


def test_an_ordinary_fixed_product_with_the_same_plate_and_foam_still_goes_through() -> None:
    """對照：同樣的鋼板與棉、只是背襯沒黏住的固定成品照舊進清單，開跑參數照舊帶 ``fixed:<代號>``。"""
    plan = _plan(["glass_wool_48k", PLAIN_ID, "none"])
    assert plan.allowed == {"x0@a": ("glass_wool_48k", f"fixed:{PLAIN_ID}", "none")}
    arguments = start_arguments(plan)
    assert arguments[arguments.index("--allowed") + 1] == f"x0@a=glass_wool_48k,fixed:{PLAIN_ID},none"


@pytest.mark.parametrize("allowed", [[HIT, "glass_wool_48k", "none"], [HIT, "none"], [HIT]], ids=["with_wool", "with_none", "alone"])
def test_one_that_also_does_not_fit_the_depth_is_still_refused_with_this_sentence(allowed: list[str]) -> None:
    """命中的那一筆同時放不下深度（背襯 10 cm、上限 5 cm）：照樣是這一句，不是先被當成放不下默默拿掉，
    也不是「一組都放不下」「只放了固定成品」。對照：同一個上限、清單拿掉它（還有可調的）就開得了。"""
    with pytest.raises(PlanRefused) as caught:
        _plan(allowed, max_depth_m=0.05)
    assert str(caught.value) == REFUSED
    assert _plan(["glass_wool_48k", "none"], max_depth_m=0.05).allowed == {"x0@a": ("glass_wool_48k", "none")}


def test_alone_in_the_list_it_is_this_sentence_and_not_the_only_fixed_products_one() -> None:
    with pytest.raises(PlanRefused) as caught:
        _plan([HIT])
    assert str(caught.value) == REFUSED
    with pytest.raises(PlanRefused, match="只放了固定成品"):
        _plan([PLAIN_ID])


def test_with_the_judgment_switched_off_the_same_list_plans_as_any_fixed_product(monkeypatch: pytest.MonkeyPatch) -> None:
    """控制組（同一跑裡比）：把判斷換成永遠不命中，同一份清單排出來的計畫就是一般固定成品那一條路——
    多的只有那一句拒收，計畫其他地方沒有被這一支動到。"""
    monkeypatch.setattr(zone_limits, "reference_only", lambda material: False)
    plan = _plan(["glass_wool_48k", HIT, "none"])
    assert plan.allowed == {"x0@a": ("glass_wool_48k", f"fixed:{HIT}", "none")}
    assert plan.model_dump(mode="json", exclude={"allowed"}) == _plan(["glass_wool_48k", PLAIN_ID, "none"]).model_dump(
        mode="json", exclude={"allowed"})


def _write_user_library(data_dir: Path, entries: tuple[LibraryEntry, ...] = ENTRIES) -> Path:
    """用產品自己的寫法存（跟材料頁存出來的同一種排法），「存回去位元組不變」才比得出來。"""
    path = user_library_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_user_library(entries), encoding="utf-8")
    return path


def test_the_web_plan_reads_the_user_library_and_refuses_the_same_way(tmp_path: Path) -> None:
    """網頁那一層（預覽、開跑、建替身、分區頁配對共用的 ``GradientJobs.plan``）：使用者材料庫裡的那一筆照樣被認出來，
    名字照材料庫顯示名。"""
    _write_user_library(tmp_path)
    jobs = GradientJobs(tmp_path, launcher=FakeLauncher(), engine_commit="c" * 40, capabilities=config_path("capabilities.toml"),
                        scheme_lock=threading.RLock())
    scheme = Scheme.model_validate(symmetry_document({"x0": FRONT_PAIR}))
    with pytest.raises(PlanRefused) as caught:
        jobs.plan(scheme, _design(["glass_wool_48k", HIT, "none"]), GradientRequest())
    assert str(caught.value) == REFUSED
    assert jobs.plan(scheme, _design(["glass_wool_48k", PLAIN_ID, "none"]), GradientRequest())[0].allowed == {
        "x0@a": ("glass_wool_48k", f"fixed:{PLAIN_ID}", "none")}


# ---- 材料庫頁的曲線介面 ----

# 主線（de6f12b2）畫得出曲線時回覆的鍵：沒命中的材料一個都不准多、不准少。
MAIN_KEYS = {"status", "basis", "basis_code", "basis_detail", "provenance", "source", "links", "arrangement", "range_scope",
             "frequencies_hz", "absorption", "measured_range_hz", "spans", "warnings", "axis_text"}
# 冷子行程（正式開機設定：雙精度）：每一份材料打三次曲線介面，照常、把判斷換成永遠不命中、把界線換成讀不出來，三份回覆都交回來。
_CURVE_TWICE = """
import json, sys
from pathlib import Path
from aosr.runtime import configure_calculation_jax
configure_calculation_jax()
from starlette.testclient import TestClient
from aosr.gui import material_curves, reference_only
from aosr.gui.app import GuiSettings, create_app


def broken_band():
    raise ValueError("界線讀不出來（考卷植的）")


cases = json.load(sys.stdin)
out = {}
with TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=Path(sys.argv[1]))), base_url="http://localhost") as client:
    for name, document in cases.items():
        real = client.post("/api/material-curve", json=document).json()
        judge = material_curves.entry_reference_only
        material_curves.entry_reference_only = lambda entry: False
        try:
            off = client.post("/api/material-curve", json=document).json()
        finally:
            material_curves.entry_reference_only = judge
        # 第三次：界線換成讀不出來（修補第 7 條）。沒命中的材料照樣畫得出來；命中的會炸，記下來當對照。
        band = reference_only.band_hz
        reference_only.band_hz = broken_band
        try:
            noband = client.post("/api/material-curve", json=document).json()
        except ValueError as error:
            noband = {"raised": str(error)}
        finally:
            reference_only.band_hz = band
        out[name] = {"real": real, "off": off, "noband": noband}
json.dump(out, sys.stdout)
"""


@pytest.fixture(scope="module")
def curves(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Doc]]:
    from tests.engine.test_material_library_vpr import _vpr

    folder = tmp_path_factory.mktemp("vpr-curves")
    cases = {"hit": USER[0], "fronted": _fixed("own_fronted_panel", "自編前置板", FRONTED), "table": _fixed("own_table", "自編表", _table(BONDED_VPR)),
             "plain": USER[1], "tunable": _vpr(), "unfinished": {"kind": "fixed", "id": "own_unfinished"}}
    env = forward_cpu_pins(child_process_env())
    env["TMPDIR"] = str(folder)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    done = subprocess.run([sys.executable, "-c", _CURVE_TWICE, str(folder)], input=json.dumps(cases), env=env, cwd=folder,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    return cast(dict[str, dict[str, Doc]], json.loads(done.stdout))


@pytest.mark.parametrize("name", ["hit", "fronted", "table"])
def test_a_hit_curve_reply_is_the_ordinary_reply_plus_one_field(curves: dict[str, dict[str, Doc]], name: str) -> None:
    """命中：回覆多一格（界線與每一句字）；拿掉那一格，等於同一行程裡判斷換成永遠不命中的回覆——曲線的數字沒有被動到。"""
    real, off = curves[name]["real"], curves[name]["off"]
    assert real["status"] == "drawn" and set(off) == MAIN_KEYS
    assert real["reference_only"] == reference_only.words()
    assert {key: value for key, value in real.items() if key != "reference_only"} == off


@pytest.mark.parametrize("name", ["plain", "tunable"])
def test_a_curve_reply_for_anything_else_has_exactly_the_keys_it_had(curves: dict[str, dict[str, Doc]], name: str) -> None:
    """沒命中（同樣的鋼板與棉但沒黏住、可調 VPR）：鍵集合跟主線一樣，判斷開或關回覆一模一樣。"""
    real, off = curves[name]["real"], curves[name]["off"]
    assert real["status"] == "drawn" and set(real) == MAIN_KEYS
    assert real == off


def test_an_unfinished_form_gets_no_field(curves: dict[str, dict[str, Doc]]) -> None:
    real = curves["unfinished"]["real"]
    assert real["status"] == "incomplete" and set(real) == {"status", "message", "range_scope"}


# ---- 分區頁拿的兩個介面：材料庫清單、限制檢查 ----

# 主線（de6f12b2）材料庫清單回覆的鍵與每一筆的鍵：沒有這類構造時一個都不准多。
LISTING_KEYS = {"materials", "user_file_problem", "material_form", "required_fields", "layer_kinds", "fixed_kinds", "provenance_labels",
                "entry_kind_labels", "surface_groups", "surface_group_short"}
ROW_KEYS = {"entry", "builtin", "basis", "placement"}


def _client(data_dir: Path) -> TestClient:
    return TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=data_dir)), base_url="http://127.0.0.1")


def _without_fitted_builtins(monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.gui import material_editor

    entries = tuple(entry for entry in builtin_entries() if entry.id not in FITTED_VPR_IDS)
    monkeypatch.setattr(material_editor, "builtin_entries", lambda: entries)


def test_the_library_listing_names_the_hits_beside_the_rows_and_never_inside_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """材料庫清單（分區頁與材料頁共用）：命中的代號與字放在旁邊一格；每一筆材料那一列的鍵、材料內文都跟沒有這一支時一樣
    （分區頁會把材料那一筆原樣送回曲線介面，材料頁會把它存回材料庫檔）。材料庫裡一筆都沒有時，回覆的鍵跟主線一樣。"""
    _without_fitted_builtins(monkeypatch)
    plain_dir, hit_dir = tmp_path / "plain", tmp_path / "hit"
    _write_user_library(plain_dir, ENTRIES[1:])
    _write_user_library(hit_dir)
    plain, hit = _client(plain_dir).get("/api/materials").json(), _client(hit_dir).get("/api/materials").json()
    assert set(plain) == LISTING_KEYS and set(hit) == LISTING_KEYS | {"reference_only"}
    assert hit["reference_only"] == {"ids": [HIT], **reference_only.words(), "reset_blocked": {
        HIT: "其他牆面的清單裡有自編牆板（不進程式最佳化）：先在「其他牆面」取消它，才能改回跟其他牆面一樣。"}}
    rows = {row["entry"]["id"]: row for row in hit["materials"]}
    assert all(set(row) == ROW_KEYS for row in hit["materials"])
    assert rows[HIT]["entry"] == ENTRIES[0].model_dump(mode="json")
    # 兩份回覆除了多的那一筆材料與那一格，其餘一模一樣。
    assert {key: value for key, value in hit.items() if key not in ("materials", "reference_only")} == {
        key: value for key, value in plain.items() if key != "materials"}
    assert [row for row in hit["materials"] if row["entry"]["id"] != HIT] == plain["materials"]


def test_saving_every_listed_row_back_leaves_the_library_file_byte_identical(tmp_path: Path) -> None:
    """材料頁的做法：清單拿到的那一筆原樣存回去。有命中那一筆的材料庫，每一筆使用者材料都存回一次，材料庫檔逐位不變
    （旗標要是塞進每一筆材料裡，這裡不是存不進去就是檔案變了）。"""
    path = _write_user_library(tmp_path)
    before = path.read_bytes()
    client = _client(tmp_path)
    rows = [row for row in client.get("/api/materials").json()["materials"] if not row["builtin"]]
    assert [row["entry"]["id"] for row in rows] == [HIT, PLAIN_ID]
    for row in rows:
        reply = client.put(f"/api/materials/{row['entry']['id']}", json=row["entry"])
        assert reply.status_code == 200, reply.text
    assert path.read_bytes() == before


def _limits(client: TestClient, allowed: list[str], design: Doc | None = None) -> Doc:
    body = {"scheme": symmetry_document({"x0": FRONT_PAIR}), "design": design or _design(allowed).model_dump(mode="json")}
    reply = client.post("/api/zone-limits", json=body)
    assert reply.status_code == 200, reply.text
    return cast(Doc, reply.json())


def test_the_limits_route_hands_the_page_the_refusal_sentence_only_when_there_is_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """分區頁每次改動都會問的限制檢查：准用清單裡有命中的那一筆，多回一格（哪一面、哪一區、哪一筆、開跑會拒收的那一句）；
    沒有就不多這一格。控制組（同一跑）：把判斷換成永遠不命中，同一份清單的回覆就是拿掉那一格的回覆。"""
    _write_user_library(tmp_path)
    client = _client(tmp_path)
    hit = _limits(client, ["glass_wool_48k", HIT, "none"])
    assert hit["reference_only"] == [{"id": HIT, "zones": [{"wall": "x0", "zone": "a"}], "text": REFUSED}]
    assert "reference_only" not in _limits(client, ["glass_wool_48k", PLAIN_ID, "none"])
    assert "reference_only" not in _limits(client, ["glass_wool_48k", "none"])
    monkeypatch.setattr(zone_limits, "reference_only", lambda material: False)
    off = _limits(client, ["glass_wool_48k", HIT, "none"])
    # 配對那一格（symmetry）會抄開跑拒收的那一句，另外比；其餘逐鍵相同。
    assert {key: value for key, value in hit.items() if key not in ("reference_only", "symmetry")} == {
        key: value for key, value in off.items() if key != "symmetry"}


_NO_JAX = """
import json, sys, threading
from pathlib import Path
from starlette.testclient import TestClient
from aosr.config.paths import config_path
from aosr.gui.app import GuiSettings, create_app
from aosr.gui.gradient_jobs import GradientJobs, plan_fixed
from aosr.gui.gradient_plan import GradientRequest, PlanRefused
from aosr.reporting.scheme import Scheme
from tests.engine._zone_symmetry_room import FRONT_PAIR, symmetry_document
from tests.engine.test_gui_gradient_jobs import FakeLauncher
from tests.engine._construction_preset_answers import FITTED_VPR_IDS
from tests.engine.test_gui_vpr_reference import HIT, REFUSED, _design, _write_user_library

assert "jax" not in sys.modules, "載入期就有 JAX（另一題在守）"
data = Path(sys.argv[1])
_write_user_library(data)
document = symmetry_document({"x0": FRONT_PAIR})
design = _design(["glass_wool_48k", HIT, "none"])
client = TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=data)), base_url="http://127.0.0.1")
assert client.get("/").status_code == 200
listing = client.get("/api/materials").json()
assert listing["reference_only"]["ids"] == [*FITTED_VPR_IDS, HIT], listing.get("reference_only")
limits = client.post("/api/zone-limits", json={"scheme": document, "design": design.model_dump(mode="json")}).json()
assert limits["reference_only"][0]["text"] == REFUSED, limits.get("reference_only")
after_page = "jax" in sys.modules
jobs = GradientJobs(data, launcher=FakeLauncher(), engine_commit="c" * 40, capabilities=config_path("capabilities.toml"),
                    scheme_lock=threading.RLock())
try:
    jobs.plan(Scheme.model_validate(document), design, GradientRequest())
except PlanRefused as refused:
    assert str(refused) == REFUSED, str(refused)
else:
    raise AssertionError("計畫沒有拒收")
print(f"分區頁兩個介面後 jax={after_page} 計畫拒收後 jax={'jax' in sys.modules}")
raise SystemExit(1 if "jax" in sys.modules else 0)
"""


def test_the_web_process_still_has_no_jax_after_the_zone_page_asks_about_such_a_construction(tmp_path: Path) -> None:
    """照 #1213 那一題的寫法：另開乾淨行程，開首頁、拿材料庫清單（有命中的那一格）、問一次限制檢查（回拒收那一句）、
    排一次會被拒收的計畫，之後 ``sys.modules`` 裡不准有 ``jax``。"""
    env = forward_cpu_pins(child_process_env())
    env["TMPDIR"] = str(tmp_path)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    done = subprocess.run([sys.executable, "-c", _NO_JAX, str(tmp_path)], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout[-300:] + done.stderr[-1500:]


# ---- 修補第 1 輪：一組一句、順序、沒命中不讀界線、幾道守門 ----

SECOND, SECOND_LABEL = "own_second_panel", "自編第二塊板"
TWO_HITS = (*ENTRIES, _ENTRY.validate_python(_fixed(SECOND, SECOND_LABEL, FRONTED)))


def _zones_design(zones: dict[str, dict[str, list[str]]]) -> Doc:
    return {"schema_version": "aosr.zone_design.v1", "scheme_id": NAME,
            "zones": {wall: {zone: {"mode": "search", "allowed": allowed} for zone, allowed in keys.items()} for wall, keys in zones.items()}}


def _sentence(zones: str, label: str = HIT_LABEL) -> str:
    return (f"{zones}的准用清單裡有{label}：這類「鋼板直接貼在棉上」的構造 300 Hz 以下的數字只能參考，所以不進程式最佳化；"
            "請在分區頁取消它，或把這一區改成「指定貼一種」")


def test_zones_sharing_one_list_get_one_sentence_naming_them_all(tmp_path: Path) -> None:
    """好幾區共用同一份清單（排序後一樣就算，寫的先後不同也算）：一組一句，區名用「、」接，照方案裡那一面的區序
    （b 在 a 前面）；清單不一樣的區自己一句。只有一區時逐字等於原本那一句。"""
    _write_user_library(tmp_path)
    client = _client(tmp_path)
    shared = _limits(client, [], _zones_design({"x0": {"a": ["glass_wool_48k", HIT, "none"], "b": ["none", HIT, "glass_wool_48k"]}}))
    assert shared["reference_only"] == [{"id": HIT, "zones": [{"wall": "x0", "zone": "b"}, {"wall": "x0", "zone": "a"}],
                                         "text": _sentence("前牆「b」、前牆「a」")}]
    apart = _limits(client, [], _zones_design({"x0": {"a": ["glass_wool_48k", HIT, "none"], "b": [HIT, "none"]}}))
    assert apart["reference_only"] == [{"id": HIT, "zones": [{"wall": "x0", "zone": "b"}], "text": _sentence("前牆「b」")},
                                        {"id": HIT, "zones": [{"wall": "x0", "zone": "a"}], "text": REFUSED}]
    assert _sentence("前牆「a」") == REFUSED


def test_the_order_of_zones_in_the_design_file_changes_neither_the_sentences_nor_which_one_the_plan_refuses_with(tmp_path: Path) -> None:
    """兩區各有一筆不同的命中材料：設計檔裡兩區寫的先後對調，限制檢查回的句子與順序都不變（照方案的區序，b 先）；
    開跑計畫拒收寫的就是限制檢查的第一句。"""
    _write_user_library(tmp_path, TWO_HITS)
    client = _client(tmp_path)
    jobs = GradientJobs(tmp_path, launcher=FakeLauncher(), engine_commit="c" * 40, capabilities=config_path("capabilities.toml"),
                        scheme_lock=threading.RLock())
    scheme = Scheme.model_validate(symmetry_document({"x0": FRONT_PAIR}))
    lists = {"a": ["glass_wool_48k", HIT, "none"], "b": ["glass_wool_48k", SECOND, "none"]}
    expected = [{"id": SECOND, "zones": [{"wall": "x0", "zone": "b"}], "text": _sentence("前牆「b」", SECOND_LABEL)},
                {"id": HIT, "zones": [{"wall": "x0", "zone": "a"}], "text": REFUSED}]
    for order in (("a", "b"), ("b", "a")):
        design = _zones_design({"x0": {zone: lists[zone] for zone in order}})
        assert list(cast(Doc, cast(Doc, design["zones"])["x0"])) == list(order)
        assert _limits(client, [], design)["reference_only"] == expected, order
        with pytest.raises(PlanRefused) as caught:
            jobs.plan(scheme, DesignSettings.model_validate(design), GradientRequest())
        assert str(caught.value) == expected[0]["text"], order


def _no_band(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> float:
        raise ValueError("界線讀不出來（考卷植的）")
    monkeypatch.setattr(reference_only, "band_hz", broken)


def test_a_library_without_such_a_construction_never_reads_the_band(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """沒有命中的材料庫不靠界線：把界線換成讀不出來，材料庫清單、限制檢查、曲線三個介面照樣 200，回覆跟界線讀得出來時
    （同一跑先打一次）一模一樣，鍵也就是主線那幾個。有命中的材料庫同樣的植法會炸（對照：這一題真的植到了）。"""
    _without_fitted_builtins(monkeypatch)
    _write_user_library(tmp_path, ENTRIES[1:])
    client = _client(tmp_path)
    design = _zones_design({"x0": {"a": ["glass_wool_48k", PLAIN_ID, "none"]}})
    curve = USER[1]

    def replies() -> list[object]:
        listing = client.get("/api/materials")
        limits = client.post("/api/zone-limits", json={"scheme": symmetry_document({"x0": FRONT_PAIR}), "design": design})
        drawn = client.post("/api/material-curve", json=curve)
        assert (listing.status_code, limits.status_code, drawn.status_code) == (200, 200, 200), (listing.text[:200], limits.text[:200], drawn.text[:200])
        return [listing.json(), limits.json(), drawn.json()]

    before = replies()
    _no_band(monkeypatch)
    after = replies()
    assert after == before
    assert set(cast(Doc, after[0])) == LISTING_KEYS and "reference_only" not in cast(Doc, after[1]) and "reference_only" not in cast(Doc, after[2])
    # 這個行程沒開雙精度，曲線那一支走不到畫得出來的路；畫得出來的那一趟在冷子行程那一題（下一題）。
    hit_dir = tmp_path / "hit"
    _write_user_library(hit_dir)
    with pytest.raises(ValueError, match="界線讀不出來"):
        _client(hit_dir).get("/api/materials")


def _registry(monkeypatch: pytest.MonkeyPatch, sections: list[object]) -> None:
    class Registry:
        stage_one_sections = sections
    monkeypatch.setattr(reference_only, "load_quality_targets", lambda path: Registry())


class _Section:
    def __init__(self, value: object) -> None:
        self._value = value

    def entry(self, key: str) -> object:
        assert key == reference_only.SCORED_RANGE_KEY
        return self._value


def _setting(value: object) -> object:
    real = load_quality_targets(config_path("quality_targets.toml")).stage_one_sections[0].entry(reference_only.SCORED_RANGE_KEY)
    return real.model_copy(update={"value": value})


def test_two_sections_with_different_upper_edges_are_an_error_and_not_a_pick(monkeypatch: pytest.MonkeyPatch) -> None:
    """品質登記簿第一階段有兩段、上界不一樣：材料沒有用途可問，不挑一個，報錯並列出讀到的值。兩段上界一樣就照讀。"""
    _registry(monkeypatch, [_Section(_setting((20.0, 300.0))), _Section(_setting((20.0, 250.0)))])
    with pytest.raises(ValueError, match=r"上界要剛好一個值，讀到 \[250\.0, 300\.0\]"):
        reference_only.band_hz()
    _registry(monkeypatch, [_Section(_setting((20.0, 250.0))), _Section(_setting((30.0, 250.0)))])
    assert reference_only.band_hz() == 250.0


@pytest.mark.parametrize("value", [300.0, (20.0, 300.0, 400.0), None], ids=["one_number", "three_numbers", "not_a_setting"])
def test_a_scored_range_that_is_not_a_range_is_an_error(monkeypatch: pytest.MonkeyPatch, value: object) -> None:
    """那一格不是一段頻率範圍（一個數、三個數、根本不是設定值）：報錯，不猜上界。"""
    _registry(monkeypatch, [_Section(None if value is None else _setting(value))])
    with pytest.raises(TypeError, match="不是一段頻率範圍"):
        reference_only.band_hz()


def test_a_design_that_does_not_validate_gets_the_ordinary_limits_reply_without_the_field(tmp_path: Path) -> None:
    """設計設定格式不對（不是由程式最佳化的區給了清單，清單裡還是命中那一筆）：限制檢查照樣 200，沒有那一格
    （格式不對這件事限制檢查自己會講）。"""
    _write_user_library(tmp_path)
    design: Doc = {"schema_version": "aosr.zone_design.v1", "scheme_id": NAME,
                   "zones": {"x0": {"a": {"mode": "fixed", "allowed": ["glass_wool_48k", HIT]}}}}
    reply = _limits(_client(tmp_path), [], design)
    assert "reference_only" not in reply and "zones" in reply


def test_a_user_library_that_cannot_be_read_gets_the_ordinary_limits_reply_without_the_field(tmp_path: Path) -> None:
    """使用者材料庫讀不進來、又有一區寫了清單：限制檢查照樣 200，沒有那一格（材料庫讀不進來限制檢查自己有一條紅）。"""
    path = user_library_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("這不是材料庫 [[[\n", encoding="utf-8")
    reply = _limits(_client(tmp_path), ["glass_wool_48k", HIT, "none"])
    assert "reference_only" not in reply and "zones" in reply


def test_without_any_list_the_user_library_is_not_read_for_this(tmp_path: Path) -> None:
    """沒有任何一區寫准用清單：這一格不為了自己去讀材料庫（分區頁每次改動都會問限制檢查；讀材料庫要開檔、驗每一筆）。
    有一區寫了清單才讀，而且只讀一次。"""
    calls: list[int] = []

    def provider() -> DepthTargets:
        calls.append(1)
        return library_depth_targets(SETTINGS, ENTRIES)

    scheme = symmetry_document({"x0": FRONT_PAIR})
    none_listed: Doc = {"schema_version": "aosr.zone_design.v1", "scheme_id": NAME, "zones": {"x0": {"a": {"mode": "search"}}}}
    assert zone_limits._draft_reference_only(scheme, none_listed, provider) == [] and calls == []
    listed = _design(["glass_wool_48k", HIT, "none"]).model_dump(mode="json")
    assert [row["text"] for row in zone_limits._draft_reference_only(scheme, listed, provider)] == [REFUSED]
    assert calls == [1]


@pytest.mark.parametrize("name", ["plain", "tunable"])
def test_a_drawn_curve_for_anything_else_never_reads_the_band(curves: dict[str, dict[str, Doc]], name: str) -> None:
    """修補第 7 條的曲線那一段（冷子行程、雙精度，真的畫得出來）：界線讀不出來時，沒命中的材料回覆跟平常一模一樣、
    鍵就是主線那幾個。對照：命中的那一筆同樣的植法會炸（這一題真的植到了）。"""
    real, noband = curves[name]["real"], curves[name]["noband"]
    assert noband == real and noband["status"] == "drawn" and set(noband) == MAIN_KEYS
    assert curves["hit"]["noband"] == {"raised": "界線讀不出來（考卷植的）"}


def test_zones_are_named_in_the_order_of_the_command_line_patches_across_walls(tmp_path: Path) -> None:
    """跨面的順序：句子裡的區、組與組的先後，都照命令列面片的順序（``surface_ids(scheme_patch_keys(…))``，天花在牆前面、
    同一面照方案的區序），不照設計檔寫的先後。限制檢查自己逐面逐區走的順序（``_zone_order``）跟那一支逐片相同
    （方案轉 90 度、轉 180 度也一樣）。"""
    from aosr.gradient.boundary import scheme_patch_keys, surface_ids
    from tests.engine._zone_symmetry_room import CENTRED

    cuts = {"x0": FRONT_PAIR, "y0": SIDE, "yL": SIDE, "ceiling": CENTRED}
    for kwargs in ({"turned": False, "flipped": False}, {"turned": True, "flipped": False}, {"turned": False, "flipped": True}):
        document = Scheme.model_validate(symmetry_document(cuts, turned=kwargs["turned"], flipped=kwargs["flipped"])).model_dump(mode="json")
        walked = [wall if zone == "whole_wall" else f"{wall}@{zone}" for wall, zone in zone_limits._zone_order(document)]
        assert walked == list(surface_ids(scheme_patch_keys(Scheme.model_validate(document)))), kwargs
    _write_user_library(tmp_path)
    shared = ["glass_wool_48k", HIT, "none"]
    # 設計檔故意倒著寫：右牆、前牆 a、天花、前牆 b。
    design = _zones_design({"yL": {"s": shared}, "x0": {"a": shared, "b": [HIT, "none"]}, "ceiling": {"c": shared}})
    reply = _client(tmp_path).post("/api/zone-limits", json={"scheme": symmetry_document(cuts), "design": design})
    assert reply.status_code == 200
    groups = cast(list[Doc], reply.json()["reference_only"])
    assert [group["zones"] for group in groups] == [
        [{"wall": "ceiling", "zone": "c"}, {"wall": "x0", "zone": "a"}, {"wall": "yL", "zone": "s"}], [{"wall": "x0", "zone": "b"}]]
    assert groups[0]["text"] == _sentence("天花「c」、前牆「a」、右牆「s」") and groups[1]["text"] == _sentence("前牆「b」")


# ---- 修補 2：方案裡沒有的區不進分組；壞草稿哪些區算在方案裡 ----

def _reply(tmp_path: Path, scheme: Doc, design: Doc) -> Doc:
    reply = _client(tmp_path).post("/api/zone-limits", json={"scheme": scheme, "design": design})
    assert reply.status_code == 200, reply.text
    return cast(Doc, reply.json())


def _named(reply: Doc) -> list[list[str]]:
    return [[f"{zone['wall']}/{zone['zone']}" for zone in cast(list[Doc], group["zones"])] for group in cast(list[Doc], reply.get("reference_only", []))]


def _walls(document: Doc) -> Doc:
    return cast(Doc, cast(Doc, document["scene"])["surface_zones_by_wall"])


def test_a_design_zone_the_scheme_does_not_have_is_in_no_group_and_no_sentence(tmp_path: Path) -> None:
    """設計設定裡有一區方案沒有（改過方案後留下的舊區）、跟方案裡兩區共用一份有命中的清單：那一組只點名方案裡的兩區，
    句子裡沒有它。只有那一區有命中：沒有那一格。"""
    _write_user_library(tmp_path)
    shared = ["glass_wool_48k", HIT, "none"]
    scheme = symmetry_document({"x0": FRONT_PAIR})
    reply = _reply(tmp_path, scheme, _zones_design({"x0": {"ghost": shared, "a": shared, "b": shared}}))
    assert reply["reference_only"] == [{"id": HIT, "zones": [{"wall": "x0", "zone": "b"}, {"wall": "x0", "zone": "a"}],
                                        "text": _sentence("前牆「b」、前牆「a」")}]
    alone = _reply(tmp_path, scheme, _zones_design({"x0": {"ghost": shared, "a": ["glass_wool_48k", "none"]}}))
    assert "reference_only" not in alone


def test_a_plan_for_a_design_whose_only_such_list_is_on_a_missing_zone_is_refused_as_it_would_be_without_the_judge(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """控制組：只有方案沒有的那一區有命中的清單。開跑計畫本來就先因為設計設定跟方案對不上拒收——結果跟把判斷換成
    「永遠不命中」時一模一樣（不是這一類的拒收句）。"""
    _write_user_library(tmp_path)
    jobs = GradientJobs(tmp_path, launcher=FakeLauncher(), engine_commit="c" * 40, capabilities=config_path("capabilities.toml"),
                        scheme_lock=threading.RLock())
    scheme = Scheme.model_validate(symmetry_document({"x0": FRONT_PAIR}))
    design = DesignSettings.model_validate(_zones_design({"x0": {"ghost": ["glass_wool_48k", HIT, "none"], "a": ["glass_wool_48k", "none"]}}))

    def outcome() -> str:
        with pytest.raises(PlanRefused) as caught:
            jobs.plan(scheme, design, GradientRequest())
        return str(caught.value)

    real = outcome()
    monkeypatch.setattr(zone_limits, "reference_only", lambda material: False)
    assert real == outcome() and real.startswith("設計設定跟方案對不上（") and "鋼板直接貼在棉上" not in real


def test_a_plan_without_any_design_gets_the_ordinary_refusal() -> None:
    """沒有設計設定（每一區都是固定的）：這一類的檢查沒有清單可看，開跑計畫照舊回「沒有任何一區交給搜尋」那一句。"""
    with pytest.raises(PlanRefused) as caught:
        plan_search(Scheme.model_validate(symmetry_document({"x0": FRONT_PAIR})), None, GradientRequest(), SETTINGS,
                    depth_targets=library_depth_targets(SETTINGS, ENTRIES))
    assert str(caught.value).startswith("沒有任何一區交給搜尋")


def test_two_hits_in_one_list_come_in_the_order_the_first_zone_wrote_them(tmp_path: Path) -> None:
    """同一份清單有兩筆命中：兩組的區相同，先後照第一區（方案的區序，b）那份清單裡寫的先後——不照代號、
    不照另一區怎麼寫、不照設計檔裡區的先後。"""
    _write_user_library(tmp_path, TWO_HITS)
    assert SECOND > HIT  # 照代號排的話命中那一筆在前；b 那一份把第二塊寫在前面
    lists = {"b": [SECOND, HIT, "none"], "a": [HIT, "none", SECOND]}
    zones = [{"wall": "x0", "zone": "b"}, {"wall": "x0", "zone": "a"}]
    for order in (("a", "b"), ("b", "a")):
        reply = _reply(tmp_path, symmetry_document({"x0": FRONT_PAIR}), _zones_design({"x0": {zone: lists[zone] for zone in order}}))
        assert reply["reference_only"] == [
            {"id": SECOND, "zones": zones, "text": _sentence("前牆「b」、前牆「a」", SECOND_LABEL)},
            {"id": HIT, "zones": zones, "text": _sentence("前牆「b」、前牆「a」")}], order


@pytest.mark.parametrize("broken", ["wall_is_a_number", "zone_is_a_string"])
def test_a_half_broken_draft_names_only_the_zones_it_still_has(tmp_path: Path, broken: str) -> None:
    """壞草稿照樣 200。左牆的區清單是數字 → 左牆沒有區；左牆有一區是字串 → 那一區不算。命中的清單放在好的前牆：
    有那一格、只點名前牆那兩區（壞的那一面寫的清單不進來）；命中的清單只在壞的那一面：沒有那一格。"""
    _write_user_library(tmp_path)
    scheme = symmetry_document({"x0": FRONT_PAIR, "y0": SIDE})
    _walls(scheme)["y0"] = 7 if broken == "wall_is_a_number" else ["s"]
    shared = ["glass_wool_48k", HIT, "none"]
    both = _reply(tmp_path, scheme, _zones_design({"y0": {"s": shared, "whole_wall": shared}, "x0": {"a": shared, "b": shared}}))
    assert _named(both) == [["x0/b", "x0/a"]]
    assert cast(list[Doc], both["reference_only"])[0]["text"] == _sentence("前牆「b」、前牆「a」")
    only_bad = _reply(tmp_path, scheme, _zones_design({"y0": {"s": shared, "whole_wall": shared}, "x0": {"a": ["glass_wool_48k", "none"]}}))
    assert "reference_only" not in only_bad


def test_a_wall_with_an_empty_zone_list_is_one_whole_wall_zone_in_face_order(tmp_path: Path) -> None:
    """某一面的區清單是空的 → 整面一區：左牆（y0）整面、右牆（yL）整面各一份不同的命中清單，組的先後照面序（左牆先），
    設計檔裡先寫右牆也一樣。"""
    _write_user_library(tmp_path)
    scheme = symmetry_document({"x0": FRONT_PAIR})
    _walls(scheme)["y0"] = []
    design = _zones_design({"yL": {"whole_wall": [HIT, "none"]}, "y0": {"whole_wall": ["glass_wool_48k", HIT, "none"]}})
    assert _named(_reply(tmp_path, scheme, design)) == [["y0/whole_wall"], ["yL/whole_wall"]]


@pytest.mark.parametrize("scene", [None, {}], ids=["scene_is_null", "scene_is_empty"])
def test_a_draft_without_a_scene_still_gets_a_reply(tmp_path: Path, scene: object) -> None:
    """``scene`` 是 null 或空的草稿（沒有哪一面寫了區 → 每一面整面一區）：照樣 200，命中的清單照樣點名。"""
    _write_user_library(tmp_path)
    reply = _reply(tmp_path, {"scene": scene}, _zones_design({"x0": {"whole_wall": ["glass_wool_48k", HIT, "none"]}}))
    assert _named(reply) == [["x0/whole_wall"]]


def test_a_rectangle_without_a_text_id_is_not_a_zone() -> None:
    """矩形的代號不是字串 → 那一區不算（其他區照算）；同一個代號寫兩次只算一區；沒寫的面整面一區。"""
    front = [{"area": "rectangle", "id": 5}, {"area": "rectangle", "id": "a"}, {"area": "rectangle", "id": "a"}]
    order = zone_limits._zone_order({"scene": {"surface_zones_by_wall": {"x0": front}}})
    assert order == [("floor", "whole_wall"), ("ceiling", "whole_wall"), ("x0", "a"), ("xL", "whole_wall"), ("y0", "whole_wall"), ("yL", "whole_wall")]


@pytest.mark.parametrize("backing", ["rigid", "zone_wall"])
def test_web_plan_allows_standalone_bonded_construction_and_explains_wall_dependency(tmp_path: Path, backing: str) -> None:
    entry = _ENTRY.validate_python(construction(backing, layers=BONDED_VPR["layers"], product=True))
    _write_user_library(tmp_path, (entry,))
    jobs = GradientJobs(tmp_path, launcher=FakeLauncher(), engine_commit="c" * 40, capabilities=config_path("capabilities.toml"),
                        scheme_lock=threading.RLock())
    scheme = Scheme.model_validate(symmetry_document({"x0": FRONT_PAIR}))
    design = _design(["glass_wool_48k", entry.id, "none"])
    if backing == "zone_wall":
        with pytest.raises(PlanRefused, match="現在還不能整塊當候選.*分區頁直接指定"):
            jobs.plan(scheme, design, GradientRequest())
    else:
        assert jobs.plan(scheme, design, GradientRequest())[0].allowed == {
            "x0@a": ("glass_wool_48k", f"fixed:{entry.id}", "none")}


def test_plan_fixed_translates_zone_wall_refusal(tmp_path: Path) -> None:
    entry = _ENTRY.validate_python(construction("zone_wall"))
    jobs = GradientJobs(tmp_path, launcher=FakeLauncher(), engine_commit="c" * 40, capabilities=config_path("capabilities.toml"),
                        scheme_lock=threading.RLock())
    plan, materials, _ = jobs.plan(Scheme.model_validate(symmetry_document({"x0": FRONT_PAIR})),
                                    _design(["glass_wool_48k", "none"]), GradientRequest())
    listed = plan.model_copy(update={"allowed": {"x0@a": (f"fixed:{entry.id}",)}})
    with pytest.raises(PlanRefused, match="背後要接這一區的原牆"):
        plan_fixed(listed, materials, (entry,))


def test_bonded_construction_in_allowed_list_keeps_warning_and_candidate(tmp_path: Path) -> None:
    entry = _ENTRY.validate_python(construction(layers=BONDED_VPR["layers"]))
    _write_user_library(tmp_path, (entry,))
    with _client(tmp_path) as client:
        warning = client.get("/api/materials").json()["reference_only"]
        assert entry.id in warning["ids"] and "只能參考" in warning["sentence"]
        assert entry.id not in warning["reset_blocked"]
        assert "reference_only" not in _limits(client, ["glass_wool_48k", entry.id, "none"])
    jobs = GradientJobs(tmp_path, launcher=FakeLauncher(), engine_commit="c" * 40, capabilities=config_path("capabilities.toml"),
                        scheme_lock=threading.RLock())
    design = _design(["glass_wool_48k", entry.id, "none"])
    plan, materials, entries = jobs.plan(Scheme.model_validate(symmetry_document({"x0": FRONT_PAIR})), design, GradientRequest())
    candidate, = plan_fixed(plan, materials, entries)
    assert candidate.target == f"fixed:{entry.id}"
