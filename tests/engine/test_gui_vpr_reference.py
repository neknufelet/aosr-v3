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
from pathlib import Path
from typing import cast

import pytest
from pydantic import TypeAdapter
from starlette.testclient import TestClient

from aosr.config.construction_presets import load_construction_presets
from aosr.config.material_gradient import load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.gui import reference_only, zone_limits
from aosr.gui.app import GuiSettings, create_app
from aosr.gui.gradient_jobs import GradientJobs
from aosr.gui.gradient_plan import GradientRequest, PlanRefused, SearchPlan, library_depth_targets, plan_search, start_arguments
from aosr.gui.material_editor import user_library_path
from aosr.gui.zone_design import DesignSettings
from aosr.materials.construction_models import SurfaceMaterial
from aosr.materials.library import LibraryEntry, builtin_entries, dump_user_library
from aosr.reporting.scheme import Scheme
from aosr.runtime import child_process_env
from tests.engine._cpu_pins import forward_cpu_pins
from tests.engine._bonded_backing_cases import BONDED_VPR, MELAMINE_50, MELAMINE_100, MELAMINE_BONDED, SLOTTED_FRONT, STEEL_1MM, STEEL_2_5MM
from tests.engine._zone_symmetry_room import FRONT_PAIR, NAME, symmetry_document
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
# 冷子行程（正式開機設定：雙精度）：每一份材料打兩次曲線介面，一次照常、一次把判斷換成永遠不命中，兩份回覆都交回來。
_CURVE_TWICE = """
import json, sys
from pathlib import Path
from aosr.runtime import configure_calculation_jax
configure_calculation_jax()
from starlette.testclient import TestClient
from aosr.gui import material_curves
from aosr.gui.app import GuiSettings, create_app

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
        out[name] = {"real": real, "off": off}
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


def test_the_library_listing_names_the_hits_beside_the_rows_and_never_inside_them(tmp_path: Path) -> None:
    """材料庫清單（分區頁與材料頁共用）：命中的代號與字放在旁邊一格；每一筆材料那一列的鍵、材料內文都跟沒有這一支時一樣
    （分區頁會把材料那一筆原樣送回曲線介面，材料頁會把它存回材料庫檔）。材料庫裡一筆都沒有時，回覆的鍵跟主線一樣。"""
    plain_dir, hit_dir = tmp_path / "plain", tmp_path / "hit"
    _write_user_library(plain_dir, ENTRIES[1:])
    _write_user_library(hit_dir)
    plain, hit = _client(plain_dir).get("/api/materials").json(), _client(hit_dir).get("/api/materials").json()
    assert set(plain) == LISTING_KEYS and set(hit) == LISTING_KEYS | {"reference_only"}
    assert hit["reference_only"] == {"ids": [HIT], **reference_only.words()}
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


def _limits(client: TestClient, allowed: list[str]) -> Doc:
    body = {"scheme": symmetry_document({"x0": FRONT_PAIR}), "design": _design(allowed).model_dump(mode="json")}
    reply = client.post("/api/zone-limits", json=body)
    assert reply.status_code == 200, reply.text
    return cast(Doc, reply.json())


def test_the_limits_route_hands_the_page_the_refusal_sentence_only_when_there_is_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """分區頁每次改動都會問的限制檢查：准用清單裡有命中的那一筆，多回一格（哪一面、哪一區、哪一筆、開跑會拒收的那一句）；
    沒有就不多這一格。控制組（同一跑）：把判斷換成永遠不命中，同一份清單的回覆就是拿掉那一格的回覆。"""
    _write_user_library(tmp_path)
    client = _client(tmp_path)
    hit = _limits(client, ["glass_wool_48k", HIT, "none"])
    assert hit["reference_only"] == [{"wall": "x0", "zone": "a", "id": HIT, "text": REFUSED}]
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
from aosr.gui.gradient_jobs import GradientJobs
from aosr.gui.gradient_plan import GradientRequest, PlanRefused
from aosr.reporting.scheme import Scheme
from tests.engine._zone_symmetry_room import FRONT_PAIR, symmetry_document
from tests.engine.test_gui_gradient_jobs import FakeLauncher
from tests.engine.test_gui_vpr_reference import HIT, REFUSED, _design, _write_user_library

assert "jax" not in sys.modules, "載入期就有 JAX（另一題在守）"
data = Path(sys.argv[1])
_write_user_library(data)
document = symmetry_document({"x0": FRONT_PAIR})
design = _design(["glass_wool_48k", HIT, "none"])
client = TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=data)), base_url="http://127.0.0.1")
assert client.get("/").status_code == 200
listing = client.get("/api/materials").json()
assert listing["reference_only"]["ids"] == [HIT], listing.get("reference_only")
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
