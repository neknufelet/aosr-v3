"""票 #134 第二刀：**探針只准寫在自己那一筆的暫存根底下**（正反控制）。

這一支守的是獨立審查當場重現出來的一個洞：檔案那幾種步驟原本寫成 ``root / name``，而
``Path("/a") / "/etc/x"`` 會**把左邊整個丟掉**——case 表給一個絕對路徑，檔就真的寫到根
外面去了，而且 ``run_case`` 照樣回一筆看起來很正常的結果。``..`` 同理。

**正控制**（先證明正常的那一條路真的會寫進去）：相對名字寫得成、位置在根底下。
**反控制**（再證明擋得住）：絕對路徑、``..`` 爬出去、符號連結指到外面，三種都當場炸
:class:`~blueprint.materials_cut2_probe.CaseTableError`，而且**外面那個哨兵檔沒有被建出來**。

**炸的種類是這一支的重點。** 框架失效不准被收成 ``raised``——那一格是要跟新家逐字比的
標準答案，把「我沒跑成」寫進去就是一個只會回綠的裁判。所以反控制除了「有炸」，還要證明
它**不是**一筆 ``raised`` 答案。

外面那一塊用 pytest 的 ``tmp_path``（不是真的 repo），暫存根由探針自己每一筆現開。
"""
from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

from blueprint import materials_cut2_probe as probe
from blueprint import materials_cut2_steps as steps

# 控制組要用的那個相對名字。用 :class:`Path` 接起來而不是寫成一個帶斜線的字面值：那種
# 字面值會被「引用要解析得到」那張卡當成指向這棵樹裡某個檔的引用，而它指的是暫存根底下
# 那一份。
MATERIALS_DIR: str = "materials"
MATERIAL_FILE: str = Path(MATERIALS_DIR, "a.yaml").as_posix()

# 這幾筆控制用的 case 一支模組都不碰（只有檔案步驟），所以解析器永遠不該被叫到。
NO_MODULES: dict[str, str] = {}


def _never_looked_up(name: str) -> ModuleType:
    """控制組的 case 不該解析任何模組——真的被叫到就是這一支測試寫錯了。"""
    raise AssertionError(f"控制組的 case 不該去拿模組，卻被要求 {name!r}")


def _case(case_id: str, made: list[steps.Step], report: list[str]) -> steps.CaseEntry:
    """組一筆只做檔案動作的 case。"""
    return {"id": case_id, "steps": made, "report": report}


def test_a_relative_name_really_writes_inside_the_case_root() -> None:
    """正控制：相對名字那一條路是通的（不通的話下面的反控制證不了任何事）。

    ``under_str`` 在這一筆兼當「寫進去的位置」的證人：探針把暫存根換成 ``<TMP>`` 之後，
    那個字串就是「根底下的哪一個位置」——它要看得出檔名，而且**不帶**任何真路徑。
    """
    case = _case(
        "control.relative_name_is_written",
        [
            steps.make_dir(MATERIALS_DIR),
            steps.make_file(MATERIAL_FILE, "material_id: m1\n"),
            steps.call("builtins.open", "handle", file=steps.under_text(MATERIAL_FILE)),
            steps.method("handle", "read", "text"),
            steps.method("handle", "close", "closed"),
        ],
        ["text"],
    )
    result = probe.run_case(_builtins_lookup, case, {"builtins": "builtins"})
    values = result.get("values")
    assert isinstance(values, dict), f"正控制沒跑完：{result!r}"
    assert values["text"] == {"kind": "str", "v": "material_id: m1\n"}


def _builtins_lookup(name: str) -> ModuleType:
    """正控制用得到的唯一一支模組（讀回剛剛寫進去的那個檔）。"""
    if name != "builtins":
        raise AssertionError(f"正控制只會用 builtins，卻被要求 {name!r}")
    import builtins

    return builtins


@pytest.mark.parametrize("verb", ["file", "mkdir"])
def test_an_absolute_target_is_refused_and_nothing_is_written_outside(
    verb: str, tmp_path: Path
) -> None:
    """反控制①：絕對路徑當場炸，外面那個哨兵一個字都沒被寫出來。"""
    sentinel = tmp_path / "escaped-sentinel.txt"
    made = (
        [steps.make_file(str(sentinel), "escaped\n")]
        if verb == "file"
        else [steps.make_dir(str(sentinel))]
    )
    with pytest.raises(probe.CaseTableError):
        probe.run_case(_never_looked_up, _case(f"control.absolute_{verb}", made, []), NO_MODULES)
    assert not sentinel.exists(), "守門沒擋住：檔真的被寫到暫存根外面去了"


@pytest.mark.parametrize("verb", ["file", "mkdir"])
def test_climbing_out_with_dot_dot_is_refused(verb: str) -> None:
    """反控制②：``..`` 爬出去也當場炸（相對寫法一樣逃得出去）。"""
    name = Path("..", "escaped-by-climbing.txt").as_posix()
    made = [steps.make_file(name, "escaped\n")] if verb == "file" else [steps.make_dir(name)]
    with pytest.raises(probe.CaseTableError):
        probe.run_case(_never_looked_up, _case(f"control.climb_{verb}", made, []), NO_MODULES)


@pytest.mark.parametrize("tag", ["under", "under_str"])
def test_an_escaping_value_is_refused_before_the_product_ever_sees_it(
    tag: str, tmp_path: Path
) -> None:
    """反控制③：值那一邊（``under``／``under_str``）也走同一支守門。

    這一條擋的是另一條路：步驟沒寫檔，但把一個根外面的路徑**當參數餵給產品**——那一筆
    跑出來的 ``raised`` 會是「產品讀不到那個檔」，看起來像行為，其實是 case 表在對根外面
    的東西下手。
    """
    outside = tmp_path / "outside.yaml"
    outside.write_text("material_id: m_outside\n", encoding="utf-8")
    value = steps.under(str(outside)) if tag == "under" else steps.under_text(str(outside))
    case = _case(
        f"control.escaping_value_{tag}",
        [steps.call("builtins.open", "handle", file=value)],
        ["handle"],
    )
    with pytest.raises(probe.CaseTableError):
        probe.run_case(_builtins_lookup, case, {"builtins": "builtins"})


def test_a_symlink_that_points_outside_is_refused(tmp_path: Path) -> None:
    """反控制④：解析完才算數——根底下的符號連結指到外面，一樣炸。

    這一條直接餵 :func:`~blueprint.materials_cut2_probe.confined`：符號連結要先存在才測得
    到它，而 case 表沒有「做一個符號連結」那種步驟（也不打算有）。
    """
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside)
    with pytest.raises(probe.CaseTableError):
        probe.confined(root.resolve(), Path("link", "x.yaml").as_posix())


def test_the_case_root_itself_is_still_allowed(tmp_path: Path) -> None:
    """空名字接出來就是根自己，那是合法的（``load_experiment`` 那一筆要餵一個目錄）。"""
    root = (tmp_path / "root").resolve()
    root.mkdir()
    assert probe.confined(root, "") == root
    assert probe.confined(root, MATERIAL_FILE) == root / "materials" / "a.yaml"


def test_a_framework_failure_is_not_recorded_as_a_donor_answer(tmp_path: Path) -> None:
    """守門炸出來的東西**不准**長成一筆 ``raised`` 標準答案。

    這一條是上面幾條的重點：如果 :class:`CaseTableError` 被 ``run_case`` 的那一層收掉，
    產生器就會把「我沒跑成」寫成一筆看起來很正常的答案，而考卷會照著它比——一個只會回綠
    的裁判。所以這裡要的不只是「有炸」，是**炸出函式外面**。
    """
    sentinel = tmp_path / "never-written.txt"
    case = _case("control.not_an_answer", [steps.make_file(str(sentinel), "x\n")], [])
    with pytest.raises(probe.CaseTableError) as caught:
        probe.run_case(_never_looked_up, case, NO_MODULES)
    assert "暫存根" in str(caught.value)
    assert not sentinel.exists()
